// +build go1.8

package cognitoidentity_test

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting/unit"
	"github.com/aws/aws-sdk-go/service/cognitoidentity"
)

var svc = cognitoidentity.New(unit.Session)

func TestUnsignedRequests(t *testing.T) {
	cases := map[string]struct {
		ReqFn func() *request.Request
	}{
		"GetId": {
			ReqFn: func() *request.Request {
				req, _ := svc.GetIdRequest(&cognitoidentity.GetIdInput{
					IdentityPoolId: aws.String("IdentityPoolId"),
				})
				return req
			},
		},
		"GetOpenIdToken": {
			ReqFn: func() *request.Request {
				req, _ := svc.GetOpenIdTokenRequest(&cognitoidentity.GetOpenIdTokenInput{
					IdentityId: aws.String("IdentityId"),
				})
				return req
			},
		},
		"UnlinkIdentity": {
			ReqFn: func() *request.Request {
				req, _ := svc.UnlinkIdentityRequest(&cognitoidentity.UnlinkIdentityInput{
					IdentityId:     aws.String("IdentityId"),
					Logins:         map[string]*string{},
					LoginsToRemove: []*string{},
				})
				return req
			},
		},
		"GetCredentialsForIdentity": {
			ReqFn: func() *request.Request {
				req, _ := svc.GetCredentialsForIdentityRequest(&cognitoidentity.GetCredentialsForIdentityInput{
					IdentityId: aws.String("IdentityId"),
				})
				return req
			},
		},
	}

	for cn, c := range cases {
		t.Run(cn, func(t *testing.T) {
			req := c.ReqFn()
			err := req.Sign()
			if err != nil {
				t.Errorf("expected no error, but received %v", err)
			}

			if e, a := credentials.AnonymousCredentials, req.Config.Credentials; e != a {
				t.Errorf("expect request to use anonymous credentias, %v", a)
			}

			if e, a := "", req.HTTPRequest.Header.Get("Authorization"); e != a {
				t.Errorf("expected empty value '%v', but received, %v", e, a)
			}
		})
	}
}
