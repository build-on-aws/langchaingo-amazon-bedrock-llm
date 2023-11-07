package claude

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/schema"
)

var ErrEmptyResponse = errors.New("empty response")

type LLM struct {
	CallbacksHandler        callbacks.Handler
	brc                     *bedrockruntime.Client
	useHumanAssistantPrompt bool
	modelID                 string
}

var (
	_ llms.LLM           = (*LLM)(nil)
	_ llms.LanguageModel = (*LLM)(nil)
)

var (
	ErrMissingRegion = errors.New("empty region")
)

func New(region string, options ...ConfigOption) (*LLM, error) {

	if region == "" {
		return nil, ErrMissingRegion
	}

	claudeLLM := &LLM{useHumanAssistantPrompt: true, modelID: claudeV2ModelID}

	opts := &ConfigOptions{}
	for _, opt := range options {
		opt(opts)
	}

	if opts.BedrockRuntimeClient == nil {
		cfg, err := config.LoadDefaultConfig(context.Background(), config.WithRegion(region))
		if err != nil {
			return nil, err
		}

		claudeLLM.brc = bedrockruntime.NewFromConfig(cfg)
	} else {
		claudeLLM.brc = opts.BedrockRuntimeClient
	}

	if opts.DontUseHumanAssistantPrompt {
		claudeLLM.useHumanAssistantPrompt = false
	}

	if opts.ModelID != "" {
		claudeLLM.modelID = opts.ModelID
	}

	return claudeLLM, nil
}

func (o *LLM) Call(ctx context.Context, prompt string, options ...llms.CallOption) (string, error) {
	r, err := o.Generate(ctx, []string{prompt}, options...)
	if err != nil {
		return "", err
	}
	if len(r) == 0 {
		return "", ErrEmptyResponse
	}
	return r[0].Text, nil
}

const (
	claudePromptFormat = "\n\nHuman:%s\n\nAssistant:"
	claudeV2ModelID    = "anthropic.claude-v2" //https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html
)

func (o *LLM) Generate(ctx context.Context, prompts []string, options ...llms.CallOption) ([]*llms.Generation, error) {
	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMStart(ctx, prompts)
	}

	opts := &llms.CallOptions{}
	for _, opt := range options {
		opt(opts)
	}

	payload := Request{
		//Prompt: fmt.Sprintf(claudePromptFormat, prompts[0]),
		MaxTokensToSample: opts.MaxTokens,
		Temperature:       opts.Temperature,
		TopK:              opts.TopK,
		TopP:              opts.TopP,
		StopSequences:     opts.StopWords,
	}

	if o.useHumanAssistantPrompt {
		payload.Prompt = fmt.Sprintf(claudePromptFormat, prompts[0])
	} else {
		payload.Prompt = prompts[0]
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	var resp Response

	if opts.StreamingFunc != nil {

		resp, err = o.invokeAsyncAndGetResponse(payloadBytes, opts.StreamingFunc)
		if err != nil {
			return nil, err
		}

	} else {
		resp, err = o.invokeAndGetResponse(payloadBytes)
		if err != nil {
			return nil, err
		}
	}

	generations := []*llms.Generation{
		{Text: resp.Completion},
	}

	if o.CallbacksHandler != nil {
		o.CallbacksHandler.HandleLLMEnd(ctx, llms.LLMResult{Generations: [][]*llms.Generation{generations}})
	}
	return generations, nil
}

func (o *LLM) GeneratePrompt(ctx context.Context, prompts []schema.PromptValue, options ...llms.CallOption) (llms.LLMResult, error) {
	return llms.GeneratePrompt(ctx, o, prompts, options...)
}

func (o *LLM) GetNumTokens(text string) int {
	return llms.CountTokens("gpt4", text)
}

func (o *LLM) invokeAndGetResponse(payloadBytes []byte) (Response, error) {

	output, err := o.brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		Body:        payloadBytes,
		ModelId:     aws.String(o.modelID),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		return Response{}, err
	}

	var resp Response

	err = json.Unmarshal(output.Body, &resp)

	if err != nil {
		return Response{}, err
	}

	return resp, nil
}

func (o *LLM) invokeAsyncAndGetResponse(payloadBytes []byte, handler func(ctx context.Context, chunk []byte) error) (Response, error) {

	output, err := o.brc.InvokeModelWithResponseStream(context.Background(), &bedrockruntime.InvokeModelWithResponseStreamInput{
		Body:        payloadBytes,
		ModelId:     aws.String(o.modelID),
		ContentType: aws.String("application/json"),
	})

	if err != nil {
		return Response{}, err
	}

	var resp Response

	resp, err = ProcessStreamingOutput(output, handler)

	if err != nil {
		return Response{}, err
	}

	return resp, nil
}

func ProcessStreamingOutput(output *bedrockruntime.InvokeModelWithResponseStreamOutput, handler func(ctx context.Context, chunk []byte) error) (Response, error) {

	var combinedResult string
	resp := Response{}

	for event := range output.GetStream().Events() {
		switch v := event.(type) {
		case *types.ResponseStreamMemberChunk:

			var resp Response
			err := json.NewDecoder(bytes.NewReader(v.Value.Bytes)).Decode(&resp)
			if err != nil {
				return resp, err
			}

			handler(context.Background(), []byte(resp.Completion))
			combinedResult += resp.Completion

		case *types.UnknownUnionMember:
			fmt.Println("unknown tag:", v.Tag)

		default:
			fmt.Println("union is nil or unknown type")
		}
	}

	resp.Completion = combinedResult

	return resp, nil
}
