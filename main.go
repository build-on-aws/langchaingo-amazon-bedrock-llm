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

	//code generation
	input := "Write a program to compute factorial in Go:"
	opt := llms.WithMaxTokens(2048)

	output, err := llm.Call(context.Background(), input, opt)

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("==== Code generation task output ====\n\n", output)

	//information extraction
	input = `<directory>
	Phone directory:
	John Latrabe, 800-232-1995, john909709@geemail.com
	Josie Lana, 800-759-2905, josie@josielananier.com
	Keven Stevens, 800-980-7000, drkevin22@geemail.com 
	Phone directory will be kept up to date by the HR manager."
	<directory>
	
	Please output the email addresses within the directory, one per line, in the order in which they appear within the text. If there are no email addresses in the text, output "N/A".`

	output, err = llm.Call(context.Background(), input, llms.WithMaxTokens(2048), llms.WithTemperature(0.5), llms.WithTopK(250), llms.WithTopP(1))

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("==== Information extraction task output ====\n\n", output)

	//question answering

	input = `You are a customer service agent tasked with classifying emails by type. Please output your answer and then justify your classification. How would you categorize this email? 
	<email>
	Can I use my Mixmaster 4000 to mix paint, or is it only meant for mixing food?
	</email> 
	
	The categories are: 
	(A) Pre-sale question 
	(B) Broken or defective item 
	(C) Billing question 
	(D) Other (please explain)`

	output, err = llm.Call(context.Background(), input, llms.WithMaxTokens(2048), llms.WithTemperature(0.5), llms.WithTopK(250), llms.WithTopP(1))

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("==== Question answering task output ====\n\n", output)

}
