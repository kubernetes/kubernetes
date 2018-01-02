// +build example

package main

import (
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/dynamodb"
	"github.com/aws/aws-sdk-go/service/s3"
	"github.com/aws/aws-sdk-go/service/sqs"
)

func main() {
	defaultResolver := endpoints.DefaultResolver()
	s3CustResolverFn := func(service, region string, optFns ...func(*endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
		if service == "s3" {
			return endpoints.ResolvedEndpoint{
				URL:           "s3.custom.endpoint.com",
				SigningRegion: "custom-signing-region",
			}, nil
		}

		return defaultResolver.EndpointFor(service, region, optFns...)
	}
	sess := session.Must(session.NewSessionWithOptions(session.Options{
		Config: aws.Config{
			Region:           aws.String("us-west-2"),
			EndpointResolver: endpoints.ResolverFunc(s3CustResolverFn),
		},
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

	// Create a DynamoDB service client that will use a custom endpoint
	// resolver that overrides the shared session's. This is useful when
	// custom endpoints are generated, or multiple endpoints are switched on
	// by a region value.
	ddbCustResolverFn := func(service, region string, optFns ...func(*endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
		return endpoints.ResolvedEndpoint{
			URL:           "dynamodb.custom.endpoint",
			SigningRegion: "custom-signing-region",
		}, nil
	}
	ddbSvc := dynamodb.New(sess, &aws.Config{
		EndpointResolver: endpoints.ResolverFunc(ddbCustResolverFn),
	})
	// Operation calls will be made to the custom endpoint set in the
	// ddCustResolverFn.
	ddbSvc.ListTables(&dynamodb.ListTablesInput{})

	// Setting Config's Endpoint will override the EndpointResolver. Forcing
	// the service clien to make all operation to the endpoint specified
	// the in the config.
	ddbSvcLocal := dynamodb.New(sess, &aws.Config{
		Endpoint: aws.String("http://localhost:8088"),
	})
	ddbSvcLocal.ListTables(&dynamodb.ListTablesInput{})
}
