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
	"net/url"
	"strings"
	"testing"
)

func TestEKSAuthProvider(t *testing.T) {
	t.Run("validate against invalid configurations", func(t *testing.T) {
		vectors := []struct {
			cfg           map[string]string
			expectedError string
		}{
			{
				cfg: map[string]string{
					// Missing clusterName
					accessKeyIdField:     "Access Key",
					secretAccessKeyField: "Secret Access Key",
				},
				expectedError: fmt.Sprintf("failed to find required: '%s' key in auth provider config", clusterNameField),
			},
			{
				cfg: map[string]string{
					clusterNameField: "Cluster Name",
					// Missing accessKeyId
					secretAccessKeyField: "Secret Access Key",
				},
				expectedError: fmt.Sprintf("failed to find required: '%s' key in auth provider config", accessKeyIdField),
			},
			{
				cfg: map[string]string{
					clusterNameField: "Cluster Name",
					accessKeyIdField: "Access Key ID",
					// Missing secretAccessKey
				},
				expectedError: fmt.Sprintf("failed to find required: '%s' key in auth provider config", secretAccessKeyField),
			},
		}

		for _, v := range vectors {
			_, err := newEKSAuthProvider("", v.cfg, nil)
			if err == nil {
				t.Errorf("cfg %v should fail but succeeded", v.cfg)
			}

			if err != nil && !strings.Contains(err.Error(), v.expectedError) {
				t.Errorf("cfg %v should fail with message containing '%s'. actual: '%s'", v.cfg, v.expectedError, err)
			}
		}
	})

	t.Run("it should return non-nil provider in happy cases", func(t *testing.T) {
		accessKey := "AAAAAAAAAAAAAAAAAAAA"

		vectors := []struct {
			cfg map[string]string
		}{
			{
				cfg: map[string]string{
					clusterNameField:     "Cluster Name",
					accessKeyIdField:     accessKey,
					secretAccessKeyField: "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
				},
			},
		}

		expectedQueryParams := map[string]string{
			"X-Amz-Algorithm":     "AWS4-HMAC-SHA256",
			"X-Amz-Credential":    fmt.Sprintf("%s/20200519/us-east-1/sts/aws4_request", accessKey),
			"X-Amz-Expires":       "60",
			"X-Amz-SignedHeaders": "host;x-k8s-aws-id",
		}

		for _, v := range vectors {
			provider, err := newEKSAuthProvider("", v.cfg, nil)
			if err != nil {
				t.Errorf("newEKSAuthProvider should not fail with '%s'", err)
			}

			if provider == nil {
				t.Fatalf("newEKSAuthProvider should return non-nil provider")
			}

			eksProvider := provider.(*eksAuthProvider)
			if eksProvider == nil {
				t.Fatalf("newEKSAuthProvider should return an instance of type eksAuthProvider, actual: %T", provider)
			}

			ts := eksProvider.tokenSource
			if ts == nil {
				t.Fatalf("eks token source should be non-nil")
			}

			expectedClusterName := v.cfg[clusterNameField]
			if ts.clusterName != expectedClusterName {
				t.Fatalf("unexpected cluster name, expected: %s, actual: %s", expectedClusterName, ts.clusterName)
			}

			token, err := ts.Token()
			if err != nil {
				t.Fatalf("unexpected failure for Token()")
			}

			if token == "" {
				t.Fatal("unexpected blank token from EKS auth provider")
			}

			if !strings.HasPrefix(token, authorizationHeaderPrefix) {
				t.Fatalf("unexpected token: %s, expected the prefix: %s", token, authorizationHeaderPrefix)
			}

			token = strings.TrimPrefix(token, authorizationHeaderPrefix)
			if token == "" {
				t.Fatalf("unexpected token: %s, expected a continuation post prefix", token)
			}

			decoded, err := base64.RawURLEncoding.DecodeString(token)
			if err != nil {
				t.Fatalf("error %v: failed decoding token: %s", err, decoded)
			}

			if len(decoded) == 0 {
				t.Fatalf("failed decoding token: %s, expected non blank decoded", decoded)
			}

			decodedUrl := string(decoded)
			if !strings.HasPrefix(decodedUrl, stsUrlToSign) {
				t.Fatalf("unexpected token: %s, expected sts URL as a prefix: %s", decodedUrl, stsUrlToSign)
			}

			parsedUrl, err := url.Parse(decodedUrl)
			if parsedUrl == nil || err != nil {
				t.Fatalf("unexpected token: %s, expected it to be a valid STS URL", decodedUrl)
			}

			queryParams := parsedUrl.Query()

			for key, val := range expectedQueryParams {
				fromUrl, ok := queryParams[key]
				if !ok {
					t.Fatalf("token does not contain required query param: %s", key)
				}

				if len(fromUrl) == 0 {
					t.Fatalf("expected query param: %s to contain at least one value, but got none", key)
				}

				if fromUrl[0] != val {
					t.Fatalf("expected query param: %s to be equal to: %s. actual value: %s", key, val, fromUrl[0])
				}
			}
		}
	})
}
