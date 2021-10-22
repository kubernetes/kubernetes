// +build integration

package s3control_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/service/s3control"
)

func TestInteg_PublicAccessBlock(t *testing.T) {
	_, err := svc.GetPublicAccessBlock(&s3control.GetPublicAccessBlockInput{
		AccountId: aws.String(accountID),
	})
	if err != nil {
		aerr := err.(awserr.RequestFailure)
		// Only no such configuration is valid error to receive.
		if e, a := s3control.ErrCodeNoSuchPublicAccessBlockConfiguration, aerr.Code(); e != a {
			t.Fatalf("expected no error, or no such configuration, got %v", err)
		}
	}

	_, err = svc.PutPublicAccessBlock(&s3control.PutPublicAccessBlockInput{
		AccountId: aws.String(accountID),
		PublicAccessBlockConfiguration: &s3control.PublicAccessBlockConfiguration{
			IgnorePublicAcls: aws.Bool(true),
		},
	})
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	_, err = svc.DeletePublicAccessBlock(&s3control.DeletePublicAccessBlockInput{
		AccountId: aws.String(accountID),
	})
	if err != nil {
		t.Fatalf("expect no error, got %v", err)
	}
}
