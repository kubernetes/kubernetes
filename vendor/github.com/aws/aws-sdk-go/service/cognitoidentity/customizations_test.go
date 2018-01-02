package cognitoidentity_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/cognitoidentity"
)

var svc = cognitoidentity.New(unit.Session)

func TestUnsignedRequest_GetID(t *testing.T) {
	req, _ := svc.GetIdRequest(&cognitoidentity.GetIdInput{
		IdentityPoolId: aws.String("IdentityPoolId"),
	})

	err := req.Sign()
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if e, a := "", req.HTTPRequest.Header.Get("Authorization"); e != a {
		t.Errorf("expected empty value '%v', but received, %v", e, a)
	}
}

func TestUnsignedRequest_GetOpenIDToken(t *testing.T) {
	req, _ := svc.GetOpenIdTokenRequest(&cognitoidentity.GetOpenIdTokenInput{
		IdentityId: aws.String("IdentityId"),
	})

	err := req.Sign()
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if e, a := "", req.HTTPRequest.Header.Get("Authorization"); e != a {
		t.Errorf("expected empty value '%v', but received, %v", e, a)
	}
}

func TestUnsignedRequest_GetCredentialsForIdentity(t *testing.T) {
	req, _ := svc.GetCredentialsForIdentityRequest(&cognitoidentity.GetCredentialsForIdentityInput{
		IdentityId: aws.String("IdentityId"),
	})

	err := req.Sign()
	if err != nil {
		t.Errorf("expected no error, but received %v", err)
	}

	if e, a := "", req.HTTPRequest.Header.Get("Authorization"); e != a {
		t.Errorf("expected empty value '%v', but received, %v", e, a)
	}
}
