// build example

package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/sns"
)

var clientCfg ClientConfig
var topicARN string

func init() {
	clientCfg.SetupFlags("", flag.CommandLine)

	flag.CommandLine.StringVar(&topicARN, "topic", "",
		"The topic `ARN` to send messages to")
}

func main() {
	if err := flag.CommandLine.Parse(os.Args[1:]); err != nil {
		flag.CommandLine.PrintDefaults()
		exitErrorf(err, "failed to parse CLI commands")
	}
	if len(topicARN) == 0 {
		flag.CommandLine.PrintDefaults()
		exitErrorf(errors.New("topic ARN required"), "")
	}

	httpClient := NewClient(clientCfg)
	sess, err := session.NewSession(&aws.Config{
		HTTPClient: httpClient,
	})
	if err != nil {
		exitErrorf(err, "failed to load config")
	}

	// Start making the requests.
	svc := sns.New(sess)
	ctx := context.Background()

	fmt.Printf("Message: ")

	// Scan messages from the input with newline separators.
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		trace, err := publishMessage(ctx, svc, topicARN, scanner.Text())
		if err != nil {
			fmt.Fprintf(os.Stderr, "failed to publish message, %v\n", err)
		}
		log.Println(trace)

		fmt.Println()
		fmt.Printf("Message: ")
	}
	if err := scanner.Err(); err != nil {
		fmt.Fprintf(os.Stderr, "failed to read input, %v", err)
	}
}

// publishMessage will send the message to the SNS topic returning an request
// trace for metrics.
func publishMessage(ctx context.Context, svc *sns.SNS, topic, msg string) (*RequestTrace, error) {
	trace := &RequestTrace{}

	_, err := svc.PublishWithContext(ctx, &sns.PublishInput{
		TopicArn: &topic,
		Message:  &msg,
	}, trace.TraceRequest)
	if err != nil {
		return trace, err
	}

	return trace, nil
}

func exitErrorf(err error, msg string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, "FAILED: %v\n"+msg+"\n", append([]interface{}{err}, args...)...)
	os.Exit(1)
}
