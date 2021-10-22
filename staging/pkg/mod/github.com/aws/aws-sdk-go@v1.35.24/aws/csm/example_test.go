package csm_test

import (
	"fmt"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/csm"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/s3"
)

func ExampleStart() {
	r, err := csm.Start("clientID", ":31000")
	if err != nil {
		panic(fmt.Errorf("failed starting CSM:  %v", err))
	}

	sess, err := session.NewSession(&aws.Config{})
	if err != nil {
		panic(fmt.Errorf("failed loading session: %v", err))
	}

	r.InjectHandlers(&sess.Handlers)

	client := s3.New(sess)
	client.GetObject(&s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	// Pauses monitoring
	r.Pause()
	client.GetObject(&s3.GetObjectInput{
		Bucket: aws.String("bucket"),
		Key:    aws.String("key"),
	})

	// Resume monitoring
	r.Continue()
}
