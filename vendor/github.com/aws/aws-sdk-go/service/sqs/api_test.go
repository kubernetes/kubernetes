// +build integration

package sqs_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/sqs"
)

func TestFlattenedTraits(t *testing.T) {
	s := sqs.New(unit.Session)
	_, err := s.DeleteMessageBatch(&sqs.DeleteMessageBatchInput{
		QueueURL: aws.String("QUEUE"),
		Entries: []*sqs.DeleteMessageBatchRequestEntry{
			{
				ID:            aws.String("TEST"),
				ReceiptHandle: aws.String("RECEIPT"),
			},
		},
	})

	if err == nil {
		t.Fatalf("expect error, got nil")
	}
	if e, a := "InvalidAddress", err.Code(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if e, a := "The address QUEUE is not valid for this endpoint.", err.Message(); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}
