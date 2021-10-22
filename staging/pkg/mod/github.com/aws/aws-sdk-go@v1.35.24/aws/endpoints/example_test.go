package endpoints_test

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/sqs"
)

func ExampleEnumPartitions() {
	resolver := endpoints.DefaultResolver()
	partitions := resolver.(endpoints.EnumPartitions).Partitions()

	for _, p := range partitions {
		fmt.Println("Regions for", p.ID())
		for id := range p.Regions() {
			fmt.Println("*", id)
		}

		fmt.Println("Services for", p.ID())
		for id := range p.Services() {
			fmt.Println("*", id)
		}
	}
}

func ExampleResolverFunc() {
	myCustomResolver := func(service, region string, optFns ...func(*endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
		if service == endpoints.S3ServiceID {
			return endpoints.ResolvedEndpoint{
				URL:           "s3.custom.endpoint.com",
				SigningRegion: "custom-signing-region",
			}, nil
		}

		return endpoints.DefaultResolver().EndpointFor(service, region, optFns...)
	}

	sess := session.Must(session.NewSession(&aws.Config{
		Region:           aws.String("us-west-2"),
		EndpointResolver: endpoints.ResolverFunc(myCustomResolver),
	}))

	// Create the S3 service client with the shared session. This will
	// automatically use the S3 custom endpoint configured in the custom
	// endpoint resolver wrapping the default endpoint resolver.
	s3Svc := s3.New(sess)
	// Operation calls will be made to the custom endpoint.
	s3Svc.GetObject(&s3.GetObjectInput{
		Bucket: aws.String("myBucket"),
		Key:    aws.String("myObjectKey"),
	})

	// Create the SQS service client with the shared session. This will
	// fallback to the default endpoint resolver because the customization
	// passes any non S3 service endpoint resolve to the default resolver.
	sqsSvc := sqs.New(sess)
	// Operation calls will be made to the default endpoint for SQS for the
	// region configured.
	sqsSvc.ReceiveMessage(&sqs.ReceiveMessageInput{
		QueueUrl: aws.String("my-queue-url"),
	})
}
