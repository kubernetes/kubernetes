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
	"fmt"
	"io"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
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
	}
	if err.Error() != expectedErr {
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
	expectedError := "No resources found matching: [deployment], label selector: app=test in namespace \"test\".\n"
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

// TestRolloutRestartAllNamespaces tests that --all --all-namespaces restarts all rollout-capable resources across all namespaces.
func TestRolloutRestartAllNamespaces(t *testing.T) {
	defaultNS := "default"
	otherNS := "other"

	// Create resources in two different namespaces
	deployments := []appsv1.Deployment{
		{ObjectMeta: metav1.ObjectMeta{Name: "deploy-a", Namespace: defaultNS}},
		{ObjectMeta: metav1.ObjectMeta{Name: "deploy-b", Namespace: otherNS}},
	}

	daemonSets := []appsv1.DaemonSet{
		{ObjectMeta: metav1.ObjectMeta{Name: "ds-a", Namespace: defaultNS}},
		{ObjectMeta: metav1.ObjectMeta{Name: "ds-b", Namespace: otherNS}},
	}

	statefulSets := []appsv1.StatefulSet{
		{ObjectMeta: metav1.ObjectMeta{Name: "sts-a", Namespace: defaultNS}},
		{ObjectMeta: metav1.ObjectMeta{Name: "sts-b", Namespace: otherNS}},
	}

	ns := scheme.Codecs.WithoutConversion()
	tf := cmdtesting.NewTestFactory().WithNamespace(defaultNS)

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, rolloutRestartGroupVersionEncoder)

	tf.Client = &RolloutRestartRESTClient{
		RESTClient: &fake.RESTClient{
			GroupVersion:         rolloutRestartGroupVersionEncoder,
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				// List deployments across all namespaces
				case p == "/deployments" && m == "GET":
					resp := &appsv1.DeploymentList{Items: deployments}
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, resp))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil

				// List daemonsets across all namespaces
				case p == "/daemonsets" && m == "GET":
					resp := &appsv1.DaemonSetList{Items: daemonSets}
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, resp))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil

				// List statefulsets across all namespaces
				case p == "/statefulsets" && m == "GET":
					resp := &appsv1.StatefulSetList{Items: statefulSets}
					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, resp))))
					return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil

				// Patch requests to each resource
				case strings.HasPrefix(p, "/namespaces/") &&
					(strings.Contains(p, "/deployments/") || strings.Contains(p, "/daemonsets/") ||
						strings.Contains(p, "/statefulsets/")) && m == "PATCH":

					var obj runtime.Object
					if strings.Contains(p, "/deployments/") {
						name := extractNameFromPath(t, p)
						for _, d := range deployments {
							if d.Namespace == extractNamespaceFromPath(t, p) && d.Name == name {
								obj = &d
								break
							}
						}
					} else if strings.Contains(p, "/daemonsets/") {
						name := extractNameFromPath(t, p)
						for _, ds := range daemonSets {
							if ds.Namespace == extractNamespaceFromPath(t, p) && ds.Name == name {
								obj = &ds
								break
							}
						}
					} else if strings.Contains(p, "/statefulsets/") {
						name := extractNameFromPath(t, p)
						for _, sts := range statefulSets {
							if sts.Namespace == extractNamespaceFromPath(t, p) && sts.Name == name {
								obj = &sts
								break
							}
						}
					}

					if obj == nil {
						t.Fatalf("unexpected patch request: %v", p)
					}

					body := io.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, obj))))
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

	_ = cmd.Flags().Set("all-namespaces", "true")
	_ = cmd.Flags().Set("all", "true")

	cmd.Run(cmd, []string{})

	expectedOutput := ""
	for _, d := range deployments {
		expectedOutput += fmt.Sprintf("deployment.apps/%s restarted\n", d.Name)
	}
	for _, ds := range daemonSets {
		expectedOutput += fmt.Sprintf("daemonset.apps/%s restarted\n", ds.Name)
	}
	for _, sts := range statefulSets {
		expectedOutput += fmt.Sprintf("statefulset.apps/%s restarted\n", sts.Name)
	}

	if buf.String() != expectedOutput {
		t.Errorf("expected output:\n%s\nbut got:\n%s", expectedOutput, buf.String())
	}
}

func extractNameFromPath(t *testing.T, path string) string {
	parts := strings.Split(path, "/")
	if len(parts) < 5 {
		t.Fatalf("invalid path format: %s", path)
	}
	return parts[4]
}

func extractNamespaceFromPath(t *testing.T, path string) string {
	parts := strings.Split(path, "/")
	if len(parts) < 3 || parts[1] != "namespaces" {
		t.Fatalf("invalid path format: %s", path)
	}
	return parts[2]
}

type RolloutRestartRESTClient struct {
	*fake.RESTClient
}
