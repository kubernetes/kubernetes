package sts_test

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/request"
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

func TestSTSCustomRetryErrorCodes(t *testing.T) {
	svc := sts.New(unit.Session, &aws.Config{
		MaxRetries: aws.Int(1),
	})
	svc.Handlers.Validate.Clear()

	const xmlErr = `<ErrorResponse><Error><Code>%s</Code><Message>some error message</Message></Error></ErrorResponse>`
	var reqCount int
	resps := []*http.Response{
		{
			StatusCode: 400,
			Header:     http.Header{},
			Body: ioutil.NopCloser(bytes.NewReader(
				[]byte(fmt.Sprintf(xmlErr, sts.ErrCodeIDPCommunicationErrorException)),
			)),
		},
		{
			StatusCode: 200,
			Header:     http.Header{},
			Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
		},
	}

	req, _ := svc.AssumeRoleWithWebIdentityRequest(&sts.AssumeRoleWithWebIdentityInput{})
	req.Handlers.Send.Swap(corehandlers.SendHandler.Name, request.NamedHandler{
		Name: "custom send handler",
		Fn: func(r *request.Request) {
			r.HTTPResponse = resps[reqCount]
			reqCount++
		},
	})

	if err := req.Send(); err != nil {
		t.Fatalf("expect no error, got %v", err)
	}

	if e, a := 2, reqCount; e != a {
		t.Errorf("expect %v requests, got %v", e, a)
	}
}
