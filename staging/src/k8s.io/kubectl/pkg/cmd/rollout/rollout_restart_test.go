/*
Copyright 2021 The Kubernetes Authors.

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

package rollout

import (
	"bytes"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

var rolloutRestartGroupVersionEncoder = schema.GroupVersion{Group: "apps", Version: "v1"}

func TestRolloutRestartOne(t *testing.T) {
	deploymentName := "deployment/nginx-deployment"
	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutRestartGroupVersionEncoder)
	tf.Client = &RolloutRestartRESTClient{
		RESTClient: &fake.RESTClient{
			GroupVersion:         rolloutRestartGroupVersionEncoder,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == "/namespaces/test/deployments/nginx-deployment" && (m == "GET" || m == "PATCH"):
					responseDeployment := &appsv1.Deployment{}
					responseDeployment.Name = deploymentName
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, responseDeployment))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		},
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdRolloutRestart(tf, streams)

	cmd.Run(cmd, []string{deploymentName})
	expectedOutput := "deployment.apps/" + deploymentName + " restarted\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", expectedOutput, buf.String())
	}
}

func TestRolloutRestartError(t *testing.T) {
	deploymentName := "deployment/nginx-deployment"
	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutRestartGroupVersionEncoder)
	tf.Client = &RolloutRestartRESTClient{
		RESTClient: &fake.RESTClient{
			GroupVersion:         rolloutRestartGroupVersionEncoder,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == "/namespaces/test/deployments/nginx-deployment" && (m == "GET" || m == "PATCH"):
					responseDeployment := &appsv1.Deployment{}
					responseDeployment.Name = deploymentName
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, responseDeployment))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		},
	}

	streams, _, bufOut, _ := genericiooptions.NewTestIOStreams()
	opt := NewRolloutRestartOptions(streams)
	err := opt.Complete(tf, nil, []string{deploymentName})
	assert.NoError(t, err)
	err = opt.Validate()
	assert.NoError(t, err)
	opt.Restarter = func(obj runtime.Object) ([]byte, error) {
		return runtime.Encode(scheme.Codecs.LegacyCodec(appsv1.SchemeGroupVersion), obj)
	}

	expectedErr := "failed to create patch for nginx-deployment: if restart has already been triggered within the past second, please wait before attempting to trigger another"
	err = opt.RunRestart()
	if err == nil {
		t.Errorf("error expected but not fired")
	} else if err.Error() != expectedErr {
		t.Errorf("unexpected error fired %v", err)
	}

	if bufOut.String() != "" {
		t.Errorf("unexpected message")
	}
}

// Tests that giving selectors with no matching objects shows an error
func TestRolloutRestartSelectorNone(t *testing.T) {
	labelSelector := "app=test"

	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutRestartGroupVersionEncoder)
	tf.Client = &RolloutRestartRESTClient{
		RESTClient: &fake.RESTClient{
			GroupVersion:         rolloutRestartGroupVersionEncoder,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m, q := req.URL.Path, req.Method, req.URL.Query(); {
				case p == "/namespaces/test/deployments" && m == "GET" && q.Get("labelSelector") == labelSelector:
					// Return an empty list
					responseDeployments := &appsv1.DeploymentList{}
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, responseDeployments))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		},
	}

	streams, _, outBuf, errBuf := genericiooptions.NewTestIOStreams()
	cmd := NewCmdRolloutRestart(tf, streams)
	cmd.Flags().Set("selector", "app=test")

	cmd.Run(cmd, []string{"deployment"})
	if len(outBuf.String()) != 0 {
		t.Errorf("expected empty output, but got: %s", outBuf.String())
	}
	expectedError := "No resources found in test namespace.\n"
	if errBuf.String() != expectedError {
		t.Errorf("expected output: %s, but got: %s", expectedError, errBuf.String())
	}
}

// Tests that giving selectors with no matching objects shows an error
func TestRolloutRestartSelectorMany(t *testing.T) {
	firstDeployment := appsv1.Deployment{}
	firstDeployment.Name = "nginx-deployment-1"
	secondDeployment := appsv1.Deployment{}
	secondDeployment.Name = "nginx-deployment-2"
	labelSelector := "app=test"

	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace("test")

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutRestartGroupVersionEncoder)
	tf.Client = &RolloutRestartRESTClient{
		RESTClient: &fake.RESTClient{
			GroupVersion:         rolloutRestartGroupVersionEncoder,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m, q := req.URL.Path, req.Method, req.URL.Query(); {
				case p == "/namespaces/test/deployments" && m == "GET" && q.Get("labelSelector") == labelSelector:
					// Return the list of 2 deployments
					responseDeployments := &appsv1.DeploymentList{}
					responseDeployments.Items = []appsv1.Deployment{firstDeployment, secondDeployment}
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, responseDeployments))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
				case (p == "/namespaces/test/deployments/nginx-deployment-1" || p == "/namespaces/test/deployments/nginx-deployment-2") && m == "PATCH":
					// Pick deployment based on path
					responseDeployment := firstDeployment
					if strings.HasSuffix(p, "nginx-deployment-2") {
						responseDeployment = secondDeployment
					}
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, &responseDeployment))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		},
	}

	streams, _, buf, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdRolloutRestart(tf, streams)
	cmd.Flags().Set("selector", labelSelector)

	cmd.Run(cmd, []string{"deployment"})
	expectedOutput := "deployment.apps/" + firstDeployment.Name + " restarted\ndeployment.apps/" + secondDeployment.Name + " restarted\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", expectedOutput, buf.String())
	}
}

type RolloutRestartRESTClient struct {
	*fake.RESTClient
}
