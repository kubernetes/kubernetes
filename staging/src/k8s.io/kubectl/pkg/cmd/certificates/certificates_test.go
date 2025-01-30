/*
Copyright 2020 The Kubernetes Authors.

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

package certificates

import (
	"bytes"
	"io"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/cobra"

	certificatesv1 "k8s.io/api/certificates/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
)

func TestCertificates(t *testing.T) {
	testcases := []struct {
		name            string
		nov1            bool
		nov1beta1       bool
		command         string
		force           bool
		args            []string
		expectFailure   bool
		expectActions   []string
		expectOutput    string
		expectErrOutput string
	}{
		{
			name:    "approve existing",
			command: "approve",
			args:    []string{"existing"},
			expectActions: []string{
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/existing`, // unstructured get
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/existing`, // typed get
				`PUT /apis/certificates.k8s.io/v1/certificatesigningrequests/existing/approval`,
			},
			expectOutput: `approved`,
		},
		{
			name:      "approve existing, no v1",
			nov1:      true,
			nov1beta1: true,
			command:   "approve",
			args:      []string{"existing"},
			expectActions: []string{
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/existing`, // unstructured get, 404
			},
			expectFailure:   true,
			expectErrOutput: `could not find the requested resource`,
		},
		{
			name:    "approve already approved",
			command: "approve",
			args:    []string{"approved"},
			expectActions: []string{
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/approved`, // unstructured get
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/approved`, // typed get
			},
			expectOutput: `approved`,
		},
		{
			name:    "approve already approved, force",
			command: "approve",
			args:    []string{"approved"},
			force:   true,
			expectActions: []string{
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/approved`, // unstructured get
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/approved`, // typed get
				`PUT /apis/certificates.k8s.io/v1/certificatesigningrequests/approved/approval`,
			},
			expectOutput: `approved`,
		},
		{
			name:    "approve already denied",
			command: "approve",
			args:    []string{"denied"},
			expectActions: []string{
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/denied`, // unstructured get
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/denied`, // typed get
			},
			expectFailure:   true,
			expectErrOutput: `is already Denied`,
		},

		{
			name:    "deny existing",
			command: "deny",
			args:    []string{"existing"},
			expectActions: []string{
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/existing`, // unstructured get
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/existing`, // typed get
				`PUT /apis/certificates.k8s.io/v1/certificatesigningrequests/existing/approval`,
			},
			expectOutput: `denied`,
		},
		{
			name:      "deny existing, no v1",
			nov1:      true,
			nov1beta1: true,
			command:   "deny",
			args:      []string{"existing"},
			expectActions: []string{
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/existing`, // unstructured get, 404
			},
			expectFailure:   true,
			expectErrOutput: `could not find the requested resource`,
		},
		{
			name:    "deny already denied",
			command: "deny",
			args:    []string{"denied"},
			expectActions: []string{
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/denied`, // unstructured get
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/denied`, // typed get
			},
			expectOutput: `denied`,
		},
		{
			name:    "deny already denied, force",
			command: "deny",
			args:    []string{"denied"},
			force:   true,
			expectActions: []string{
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/denied`, // unstructured get
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/denied`, // typed get
				`PUT /apis/certificates.k8s.io/v1/certificatesigningrequests/denied/approval`,
			},
			expectOutput: `denied`,
		},
		{
			name:    "deny already approved",
			command: "deny",
			args:    []string{"approved"},
			expectActions: []string{
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/approved`, // unstructured get
				`GET /apis/certificates.k8s.io/v1/certificatesigningrequests/approved`, // typed get
			},
			expectFailure:   true,
			expectErrOutput: `is already Approved`,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()
			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)

			existingV1 := &certificatesv1.CertificateSigningRequest{
				TypeMeta:   metav1.TypeMeta{APIVersion: "certificates.k8s.io/v1", Kind: "CertificateSigningRequest"},
				ObjectMeta: metav1.ObjectMeta{Name: "existing"},
			}

			approvedV1 := &certificatesv1.CertificateSigningRequest{
				TypeMeta:   metav1.TypeMeta{APIVersion: "certificates.k8s.io/v1", Kind: "CertificateSigningRequest"},
				ObjectMeta: metav1.ObjectMeta{Name: "approved"},
				Status:     certificatesv1.CertificateSigningRequestStatus{Conditions: []certificatesv1.CertificateSigningRequestCondition{{Type: certificatesv1.CertificateApproved}}},
			}

			deniedV1 := &certificatesv1.CertificateSigningRequest{
				TypeMeta:   metav1.TypeMeta{APIVersion: "certificates.k8s.io/v1", Kind: "CertificateSigningRequest"},
				ObjectMeta: metav1.ObjectMeta{Name: "denied"},
				Status:     certificatesv1.CertificateSigningRequestStatus{Conditions: []certificatesv1.CertificateSigningRequestCondition{{Type: certificatesv1.CertificateDenied}}},
			}

			actions := []string{}
			fakeClient := fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				actions = append(actions, req.Method+" "+req.URL.Path)
				switch p, m := req.URL.Path, req.Method; {
				case tc.nov1 && strings.HasPrefix(p, "/apis/certificates.k8s.io/v1/"):
					return &http.Response{StatusCode: http.StatusNotFound, Body: io.NopCloser(bytes.NewBuffer([]byte{}))}, nil

				case p == "/apis/certificates.k8s.io/v1/certificatesigningrequests/missing" && m == http.MethodGet:
					return &http.Response{StatusCode: http.StatusNotFound}, nil

				case p == "/apis/certificates.k8s.io/v1/certificatesigningrequests/existing" && m == http.MethodGet:
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, existingV1)}, nil
				case p == "/apis/certificates.k8s.io/v1/certificatesigningrequests/existing/approval" && m == http.MethodPut:
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, existingV1)}, nil

				case p == "/apis/certificates.k8s.io/v1/certificatesigningrequests/approved" && m == http.MethodGet:
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, approvedV1)}, nil
				case p == "/apis/certificates.k8s.io/v1/certificatesigningrequests/approved/approval" && m == http.MethodPut:
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, approvedV1)}, nil

				case p == "/apis/certificates.k8s.io/v1/certificatesigningrequests/denied" && m == http.MethodGet:
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, deniedV1)}, nil
				case p == "/apis/certificates.k8s.io/v1/certificatesigningrequests/denied/approval" && m == http.MethodPut:
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, deniedV1)}, nil

				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			})
			tf.UnstructuredClientForMappingFunc = func(gv schema.GroupVersion) (resource.RESTClient, error) {
				versionedAPIPath := ""
				if gv.Group == "" {
					versionedAPIPath = "/api/" + gv.Version
				} else {
					versionedAPIPath = "/apis/" + gv.Group + "/" + gv.Version
				}
				return &fake.RESTClient{
					VersionedAPIPath:     versionedAPIPath,
					NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
					Client:               fakeClient,
				}, nil
			}
			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
				Client:               fakeClient,
			}
			streams, _, buf, errbuf := genericiooptions.NewTestIOStreams()
			tf.ClientConfigVal.Transport = fakeClient.Transport

			defer func() {
				// Restore cmdutil behavior.
				cmdutil.DefaultBehaviorOnFatal()
			}()
			// Check exit code.
			cmdutil.BehaviorOnFatal(func(e string, code int) {
				if !tc.expectFailure {
					t.Log(e)
					t.Errorf("unexpected failure exit code %d", code)
				}
				errbuf.Write([]byte(e))
			})

			var cmd *cobra.Command
			switch tc.command {
			case "approve":
				cmd = NewCmdCertificateApprove(tf, streams)
			case "deny":
				cmd = NewCmdCertificateDeny(tf, streams)
			default:
				t.Errorf("unknown command: %s", tc.command)
			}

			if tc.force {
				cmd.Flags().Set("force", "true")
			}
			cmd.Run(cmd, tc.args)

			if !strings.Contains(buf.String(), tc.expectOutput) {
				t.Errorf("expected output to contain %q:\n%s", tc.expectOutput, buf.String())
			}
			if !strings.Contains(errbuf.String(), tc.expectErrOutput) {
				t.Errorf("expected error output to contain %q:\n%s", tc.expectErrOutput, errbuf.String())
			}
			if !reflect.DeepEqual(tc.expectActions, actions) {
				t.Logf("stdout:\n%s", buf.String())
				t.Logf("stderr:\n%s", errbuf.String())
				t.Errorf("expected\n%s\ngot\n%s", strings.Join(tc.expectActions, "\n"), strings.Join(actions, "\n"))
			}
		})
	}
}
