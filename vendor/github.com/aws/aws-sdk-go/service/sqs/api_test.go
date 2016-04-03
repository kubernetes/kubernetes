// +build integration

package sqs_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/sqs"
)

func TestFlattenedTraits(t *testing.T) {
	s := sqs.New(session.New())
	_, err := s.DeleteMessageBatch(&sqs.DeleteMessageBatchInput{
		QueueURL: aws.String("QUEUE"),
		Entries: []*sqs.DeleteMessageBatchRequestEntry{
			{
				ID:            aws.String("TEST"),
				ReceiptHandle: aws.String("RECEIPT"),
			},
		},
	})

	assert.Error(t, err)
	assert.Equal(t, "InvalidAddress", err.Code())
	assert.Equal(t, "The address QUEUE is not valid for this endpoint.", err.Message())
}
