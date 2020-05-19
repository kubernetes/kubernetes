/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package eks

import (
	"encoding/base64"
	"fmt"
	"net/http"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/credentials/stscreds"
	"github.com/aws/aws-sdk-go/aws/session"
	v4 "github.com/aws/aws-sdk-go/aws/signer/v4"
	"k8s.io/klog/v2"

	"k8s.io/client-go/rest"
)

func init() {
	if err := rest.RegisterAuthProviderPlugin("eks", newEKSAuthProvider); err != nil {
		klog.Fatalf("Failed to register eks auth plugin: %v", err)
	}
}

// eksAuthProvider is an auth provider plugin that uses AWS credentials to provide
// tokens for kubectl to authenticate itself to an EKS apiserver. A sample json config
// is provided below with all recognized options described.
//
// {
//   'auth-provider': {
//     # Required
//     "name": "eks",
//
//     "config": {
//			# Required - The EKS cluster name, as configured in AWS
//			"clusterName": "",
//			# Required - AWS access key id that is used to authenticate to the cluster
//			"accessKeyId": "",
//			# Required - AWS secret access key that is used to authenticate to the cluster
//			"secretAccessKey": "",
//			# Optional - If set, this auth plugin assumes this role and authenticates with it
//			"roleARN": "",
//     }
//   }
// }
//
type eksAuthProvider struct {
	tokenSource *eksTokenSource
}

const (
	accessKeyIdField     = "accessKeyId"
	secretAccessKeyField = "secretAccessKey"
	clusterNameField     = "clusterName"
	roleARNField         = "roleARN"
)

// Authenticating to an EKS cluster requires to send a presigned STS url as the token.
// The signed URL is very short lived, and so is not persisted.
func newEKSAuthProvider(_ string, config map[string]string, _ rest.AuthProviderConfigPersister) (rest.AuthProvider, error) {
	var accessKeyId, secretAccessKey, clusterName, roleARN string

	requiredFields := map[string]*string{
		clusterNameField:     &clusterName,
		accessKeyIdField:     &accessKeyId,
		secretAccessKeyField: &secretAccessKey,
	}

	for key, strPtr := range requiredFields {
		val, ok := config[key]
		if !ok {
			klog.Errorf("Failed to find required: '%s' key in auth provider config", key)
			return nil, fmt.Errorf("failed to find required: '%s' key in auth provider config", key)
		}

		*strPtr = val
	}

	roleARN, _ = config[roleARNField]
	creds, err := getAWSCreds(accessKeyId, secretAccessKey, roleARN)
	if creds == nil || err != nil {
		klog.Errorf("Failed getting AWS creds: %v", err)
		return nil, fmt.Errorf("failed getting AWS creds: %w", err)
	}

	tokenSource := eksTokenSource{
		creds:       creds,
		clusterName: clusterName,
	}

	return &eksAuthProvider{tokenSource: &tokenSource}, nil
}

func getAWSCreds(accessKeyId, secretAccessKey, roleARN string) (*credentials.Credentials, error) {
	sessionToken := ""
	creds := credentials.NewStaticCredentials(accessKeyId, secretAccessKey, sessionToken)

	if roleARN != "" {
		awsSession, err := session.NewSession(&aws.Config{Credentials: creds})
		if awsSession == nil || err != nil {
			return nil, fmt.Errorf("failed creating session: %v", err)
		}

		creds = stscreds.NewCredentials(awsSession, roleARN)
	}

	return creds, nil
}

var _ rest.AuthProvider = (*eksAuthProvider)(nil)

// WrapTransport will add an authorization middleware to a given round-tripper
func (a *eksAuthProvider) WrapTransport(roundTripper http.RoundTripper) http.RoundTripper {
	return &eksRoundTripper{roundTripper: roundTripper, tokenSource: a.tokenSource}
}

// Login - not implemented in this context
func (a *eksAuthProvider) Login() error {
	// Irrelevant
	return nil
}

type eksTokenSource struct {
	creds       *credentials.Credentials
	clusterName string
}

const (
	stsUrlToSign = "https://sts.amazonaws.com/?Action=GetCallerIdentity&Version=2011-06-15"
	headerToSign = "x-k8s-aws-id"

	// STS presigned urls must be scoped to us-east-1
	stsRegion                 = "us-east-1"
	stsServiceName            = "sts"
	authorizationHeaderPrefix = "k8s-aws-v1."
)

// Token will retrieve the bearer token that should be passed to an EKS cluster
// Specifically, this is a presigned URL to STS/GetCallerIdentity, which the in-cluster authorization layer
// later executes, in order to map the current identity to k8s RBAC policies, defined in the aws-auth config-map.
// This will be invoked every time - there's no need to cache this, as the presigned URL is very short-lived,
// and there's no I/O made in calculating the token.
func (t *eksTokenSource) Token() (string, error) {
	signer := v4.NewSigner(t.creds)

	req, err := http.NewRequest(http.MethodGet, stsUrlToSign, nil)

	if req == nil || err != nil {
		klog.Errorf("Failed creating presign url request: %v", err)
		return "", fmt.Errorf("failed creating presign url request: %w", err)
	}

	req.Header.Add(headerToSign, t.clusterName)

	_, err = signer.Presign(req, nil, stsServiceName, stsRegion, time.Minute, time.Now())
	if err != nil {
		klog.Errorf("Failed presigning url: %v", err)
		return "", fmt.Errorf("failed presigning url: %v", err)
	}

	encodedUrl := base64.RawURLEncoding.EncodeToString([]byte(req.URL.String()))
	return authorizationHeaderPrefix + encodedUrl, nil
}

type eksRoundTripper struct {
	roundTripper http.RoundTripper
	tokenSource  *eksTokenSource
}

var _ http.RoundTripper = (*eksRoundTripper)(nil)

const (
	authorizationHeader = "Authorization"
	tokenType           = "Bearer"
)

// RoundTrip will set the Authorization header with an STS presigned URL
func (r *eksRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get(authorizationHeader)) != 0 {
		return r.roundTripper.RoundTrip(req)
	}

	token, err := r.tokenSource.Token()
	if err != nil {
		klog.Errorf("Failed acquiring token for authorization header: %v", err)
		return nil, fmt.Errorf("acquiring a token for authorization header: %w", err)
	}

	// clone the request in order to avoid modifying the headers of the original request
	req2 := new(http.Request)
	*req2 = *req
	req2.Header = make(http.Header, len(req.Header))
	for k, s := range req.Header {
		req2.Header[k] = append([]string(nil), s...)
	}

	req2.Header.Set(authorizationHeader, tokenType+" "+token)
	return r.roundTripper.RoundTrip(req2)
}
