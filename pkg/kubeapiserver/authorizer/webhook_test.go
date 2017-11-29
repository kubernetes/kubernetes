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

package authorizer

import (
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"strings"
	"testing"
	"text/template"
	"time"

	authorization "k8s.io/api/authorization/v1beta1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/plugin/pkg/authorizer/webhook"
)

// kubeConfigData is formatted into kubeConfigTmpl below.
type kubeConfigData struct {
	Server                    string
	CAFilePath                string
	ClientCertificateFilePath string
	ClientKeyFilePath         string
}

// kubeConfigTmpl is a template kubeconfig file content, as used by the webhook
// authorizer subsystem.
const kubeConfigTmpl = `clusters:
- name: name-of-remote-authz-service
  cluster:
    certificate-authority: {{ .CAFilePath }}
    server: {{ .Server }}
users:
- name: name-of-api-server
  user:
    # In our test, it is OK that these two remain unfilled.
    client-certificate:
    client-key:
current-context: webhook
contexts:
- context:
    cluster: name-of-remote-authz-service
    user: name-of-api-server
  name: webhook
`

// must panics if it is passed a non-nil error.  Useful for short-circuiting
// error handling where we don't expect actual errors to occur.
func must(i interface{}, err error) interface{} {
	if err != nil {
		panic(fmt.Sprintf("unexpected error: %v", err))
	}
	return i
}

// noerror is a one-argument must (see above).
func noerror(err error) {
	if err != nil {
		panic(fmt.Sprintf("unexpected error: %v", err))
	}
}

func TestNewWebhookAuthorizer(t *testing.T) {
	tt := []struct {
		// This is appended to the TLS server base URL to form final URL.  We
		// can't set a full URL here since the address of the TLS server will
		// vary per run.
		UrlPath string

		// The authz request to send to the authorizer.
		Request authorizer.AttributesRecord

		// The full content of the fake authorizer's response.
		VerdictResponse authorization.SubjectAccessReview

		// The URL that the fake server responded for must have this suffix.
		// For example, if the URL is "https://somewhere:port/someurl", then
		// we expect ExpectedPath == "/someurl".
		ExpectedPath string

		// Whether the request was authorized or not.
		Verdict bool

		// What was the reason given for the verdict by the authz server.
		Reason string

		// If set, this is the error we expect.  Its error text must appear in
		// the error text of the actual error.
		Error error
	}{
		{
			// Test that the authz verdict is correctly propagated back to the
			// authorizer.
			UrlPath:      "/authorize",
			ExpectedPath: "/authorize",
			Request: authorizer.AttributesRecord{
				User: &user.DefaultInfo{
					Name: "jane",
				},
				Verb:            "GET",
				Resource:        "somepod",
				Namespace:       "somenamespace",
				APIGroup:        "someapigroup",
				Path:            "somepath",
				ResourceRequest: true,
			},
			VerdictResponse: authorization.SubjectAccessReview{
				Status: authorization.SubjectAccessReviewStatus{
					Allowed: false,
					Reason:  "No authz for you!",
				},
			},
			Verdict: false,
			Reason:  "No authz for you!",
		},
		{
			// Test that changing the URL path in the configuration file
			// results in an actual change in the URL that the webhook
			// authorizer calls out to.
			UrlPath:      "/authorize/again",
			ExpectedPath: "/authorize/again",
			Request: authorizer.AttributesRecord{
				User: &user.DefaultInfo{
					Name: "jane",
				},
				Verb:            "GET",
				Resource:        "somepod",
				Namespace:       "somenamespace",
				APIGroup:        "someapigroup",
				Path:            "somepath",
				ResourceRequest: true,
			},
			VerdictResponse: authorization.SubjectAccessReview{},
			Verdict:         false,
		},
		{
			// Test a positive verdict.
			UrlPath:      "/authorize",
			ExpectedPath: "/authorize",
			Request: authorizer.AttributesRecord{
				User: &user.DefaultInfo{
					Name: "jane",
				},
				Verb:            "GET",
				Resource:        "somepod",
				Namespace:       "somenamespace",
				APIGroup:        "someapigroup",
				Path:            "somepath",
				ResourceRequest: true,
			},
			VerdictResponse: authorization.SubjectAccessReview{
				Status: authorization.SubjectAccessReviewStatus{
					Allowed: true,
				},
			},
			Verdict: true,
		},
	}

	for i, ttt := range tt {
		// Ensures that defer is called after each loop iteration.
		func() {
			// Create a test harness.
			var url *url.URL
			ts := httptest.NewTLSServer(
				http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					resp := must(json.Marshal(ttt.VerdictResponse)).([]byte)
					w.Write(resp)
					url = r.URL
				}))
			defer ts.Close()

			// Save the certificate authority from the fake TLS server into a
			// cert file.  The client must have access to this data to validate
			// the request.  We make the CA file available on the filesystem,
			// and later point the kubeconfig file to this file path.
			certfile := must(ioutil.TempFile("", "ca-")).(*os.File)
			pem.Encode(certfile, &pem.Block{
				Type:  "CERTIFICATE",
				Bytes: ts.TLS.Certificates[0].Certificate[0]})
			certfile.Close()
			defer os.Remove(certfile.Name())

			// Create a kubeconfig file with configuration bits pointing to
			// correct files.
			kubeConfig := must(ioutil.TempFile("", "kubeconfig-")).(*os.File)
			defer os.Remove(kubeConfig.Name())

			// Format the kubeconfig data into the kubeconfig file.  Normally
			// all of this fixture setup will complete without error.
			tmpl := must(template.New("kubecofig").
				Parse(kubeConfigTmpl)).(*template.Template)
			data := kubeConfigData{
				Server:     fmt.Sprintf("%v%v", ts.URL, ttt.UrlPath),
				CAFilePath: certfile.Name(),
			}
			noerror(tmpl.Execute(kubeConfig, data))
			noerror(kubeConfig.Close())

			// And now, for the actual test case...
			transport := &http.Transport{}
			authz := must(NewWebhookAuthorizer(
				kubeConfig.Name(), 1*time.Second,
				2*time.Second, transport.Dial)).(*webhook.WebhookAuthorizer)

			verdict, reason, err := authz.Authorize(ttt.Request)
			if err != nil {
				if ttt.Error == nil {
					t.Fatalf("[%v] unexpected error: %v", i, err)
				} else if strings.Index(err.Error(), ttt.Error.Error()) == -1 {
					t.Errorf("[%v] unexpected error: actual: %v, expected: %v",
						i, err, ttt.Error)
				}
				return
			}
			if verdict != ttt.Verdict {
				t.Errorf("[%v] verdict mismatch: actual: %v, expected: %v",
					i, verdict, ttt.Verdict)
			}
			if reason != ttt.Reason {
				t.Errorf("[%v] reason mismatch: actual: %v, expected: %v",
					i, reason, ttt.Reason)
			}
			if !strings.HasSuffix(url.String(), ttt.ExpectedPath) {
				t.Errorf("[%v] url suffix mismatch: actual: %v, expected: %v",
					i, url, ttt.ExpectedPath)
			}
		}()

	}
}
