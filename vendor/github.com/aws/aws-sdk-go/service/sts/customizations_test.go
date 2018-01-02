package sts_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/sts"
)

var svc = sts.New(unit.Session, &aws.Config{
	Region: aws.String("mock-region"),
})

func TestUnsignedRequest_AssumeRoleWithSAML(t *testing.T) {
	req, _ := svc.AssumeRoleWithSAMLRequest(&sts.AssumeRoleWithSAMLInput{
		PrincipalArn:  aws.String("ARN01234567890123456789"),
		RoleArn:       aws.String("ARN01234567890123456789"),
		SAMLAssertion: aws.String("ASSERT"),
	})

	err := req.Sign()
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
	if e, a := "", req.HTTPRequest.Header.Get("Authorization"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestUnsignedRequest_AssumeRoleWithWebIdentity(t *testing.T) {
	req, _ := svc.AssumeRoleWithWebIdentityRequest(&sts.AssumeRoleWithWebIdentityInput{
		RoleArn:          aws.String("ARN01234567890123456789"),
		RoleSessionName:  aws.String("SESSION"),
		WebIdentityToken: aws.String("TOKEN"),
	})

	err := req.Sign()
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
	if e, a := "", req.HTTPRequest.Header.Get("Authorization"); e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}
