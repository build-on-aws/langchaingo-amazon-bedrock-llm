package main

import (
	"context"
	"fmt"
	"log"

	"github.com/build-on-aws/langchaingo-amazon-bedrock-llm/claude"
	"github.com/tmc/langchaingo/llms"
)

func main() {
	llm, err := claude.New("us-east-1")

	if err != nil {
		log.Fatal(err)
	}

	//text generation
	input := `<paragraph> 
	"In 1758, the Swedish botanist and zoologist Carl Linnaeus published in his Systema Naturae, the two-word naming of species (binomial nomenclature). Canis is the Latin word meaning "dog", and under this genus, he listed the domestic dog, the wolf, and the golden jackal."
	</paragraph>
	
	Please rewrite the above paragraph to make it understandable to a 5th grader.`

	_, err = llm.Call(context.Background(), input, llms.WithMaxTokens(2048), llms.WithTemperature(0.5), llms.WithTopK(250), llms.WithStreamingFunc(func(ctx context.Context, chunk []byte) error {
		fmt.Print(string(chunk))
		return nil
	}))

	if err != nil {
		log.Fatal(err)
	}
}
