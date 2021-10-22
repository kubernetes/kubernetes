package iotdataplane_test

import (
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/iotdataplane"
)

func TestRequireEndpointIfRegionProvided(t *testing.T) {
	svc := iotdataplane.New(unit.Session, &aws.Config{
		Region:                 aws.String("mock-region"),
		DisableParamValidation: aws.Bool(true),
	})
	req, _ := svc.GetThingShadowRequest(nil)
	err := req.Build()

	if e, a := "", svc.Endpoint; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if err == nil {
		t.Errorf("expect error, got none")
	}
	if e, a := aws.ErrMissingEndpoint, err; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestRequireEndpointIfNoRegionProvided(t *testing.T) {
	svc := iotdataplane.New(unit.Session, &aws.Config{
		DisableParamValidation: aws.Bool(true),
	})
	fmt.Println(svc.ClientInfo.SigningRegion)

	req, _ := svc.GetThingShadowRequest(nil)
	err := req.Build()

	if e, a := "", svc.Endpoint; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if err == nil {
		t.Errorf("expect error, got none")
	}
	if e, a := aws.ErrMissingEndpoint, err; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
}

func TestRequireEndpointUsed(t *testing.T) {
	svc := iotdataplane.New(unit.Session, &aws.Config{
		Region:                 aws.String("mock-region"),
		DisableParamValidation: aws.Bool(true),
		Endpoint:               aws.String("https://endpoint"),
	})
	req, _ := svc.GetThingShadowRequest(nil)
	err := req.Build()

	if e, a := "https://endpoint", svc.Endpoint; e != a {
		t.Errorf("expect %v, got %v", e, a)
	}
	if err != nil {
		t.Errorf("expect no error, got %v", err)
	}
}
