/*
Copyright 2016 The Kubernetes Authors.

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

package clusterinfo

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"os"
	"path"
	"testing"

	flag "github.com/spf13/pflag"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

func TestSetupOutputWriterNoOp(t *testing.T) {
	tests := []string{"", "-"}
	for _, test := range tests {
		_, _, buf, _ := genericclioptions.NewTestIOStreams()
		f := cmdtesting.NewTestFactory()
		defer f.Cleanup()

		writer := setupOutputWriter(test, buf, "/some/file/that/should/be/ignored")
		if writer != buf {
			t.Errorf("expected: %v, saw: %v", buf, writer)
		}
	}
}

func TestSetupOutputWriterFile(t *testing.T) {
	file := "output.json"
	dir, err := ioutil.TempDir(os.TempDir(), "out")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	fullPath := path.Join(dir, file)
	defer os.RemoveAll(dir)

	_, _, buf, _ := genericclioptions.NewTestIOStreams()
	f := cmdtesting.NewTestFactory()
	defer f.Cleanup()

	writer := setupOutputWriter(dir, buf, file)
	if writer == buf {
		t.Errorf("expected: %v, saw: %v", buf, writer)
	}
	output := "some data here"
	writer.Write([]byte(output))

	data, err := ioutil.ReadFile(fullPath)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if string(data) != output {
		t.Errorf("expected: %v, saw: %v", output, data)
	}
}

func TestCmdClusterInfoDump(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	ns := scheme.Codecs

	encodeResp := func(obj runtime.Object, ver runtime.GroupVersioner) io.ReadCloser {
		info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
		encoder := ns.EncoderForVersion(info.Serializer, ver)

		return ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, obj))))
	}

	encodeRespForCoreGV := func(obj runtime.Object) io.ReadCloser {
		return encodeResp(obj, corev1.SchemeGroupVersion)
	}

	encodeRespForAppsGV := func(obj runtime.Object) io.ReadCloser {
		return encodeResp(obj, appsv1.SchemeGroupVersion)
	}

	tests := map[string]struct {
		expNsListReq []string

		dumpAllNs        bool
		populateCmdFlags func(f *flag.FlagSet)
	}{
		"should dump default namespaces": {
			expNsListReq: []string{"kube-system", "default"},

			populateCmdFlags: func(f *flag.FlagSet) {
				// use default options
			},
		},

		"should dump requested namespaces": {
			expNsListReq: []string{"qa", "production"},

			populateCmdFlags: func(f *flag.FlagSet) {
				f.Set("namespaces", "qa,production")
			},
		},

		"should dump all available namespaces": {
			expNsListReq: []string{"qa", "production", "test", "kube-system"},

			dumpAllNs: true,
			populateCmdFlags: func(f *flag.FlagSet) {
				f.Set("all-namespaces", "true")
			},
		},
	}

	for tn, tc := range tests {
		t.Run(tn, func(t *testing.T) {
			expListReq := map[string]io.ReadCloser{
				"/api/v1/nodes": encodeRespForCoreGV(&corev1.NodeList{}),
			}

			for _, ns := range tc.expNsListReq {
				expListReq[fmt.Sprintf("/api/v1/namespaces/%s/events", ns)] = encodeRespForCoreGV(&corev1.EventList{})
				expListReq[fmt.Sprintf("/api/v1/namespaces/%s/replicationcontrollers", ns)] = encodeRespForCoreGV(&corev1.ReplicationControllerList{})
				expListReq[fmt.Sprintf("/api/v1/namespaces/%s/services", ns)] = encodeRespForCoreGV(&corev1.ServiceList{})
				expListReq[fmt.Sprintf("/api/v1/namespaces/%s/pods", ns)] = encodeRespForCoreGV(&corev1.PodList{})

				expListReq[fmt.Sprintf("/apis/apps/v1/namespaces/%s/daemonsets", ns)] = encodeRespForAppsGV(&appsv1.DaemonSetList{})
				expListReq[fmt.Sprintf("/apis/apps/v1/namespaces/%s/deployments", ns)] = encodeRespForAppsGV(&appsv1.DeploymentList{})
				expListReq[fmt.Sprintf("/apis/apps/v1/namespaces/%s/replicasets", ns)] = encodeRespForAppsGV(&appsv1.ReplicaSetList{})
			}

			if tc.dumpAllNs { // register expected namespaces
				var items []corev1.Namespace
				for _, nsName := range tc.expNsListReq {
					items = append(items, corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: nsName}})
				}
				expListReq["/api/v1/namespaces"] = encodeRespForCoreGV(&corev1.NamespaceList{
					Items: items,
				})
			}

			tf.Client = &fake.RESTClient{
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					respBody, found := expListReq[req.URL.Path]
					if !found {
						t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					}

					return &http.Response{
						StatusCode: http.StatusOK,
						Header:     cmdtesting.DefaultHeader(),
						Body:       respBody,
					}, nil
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

			ioStreams, _, _, _ := genericclioptions.NewTestIOStreams()
			cmd := NewCmdClusterInfoDump(tf, ioStreams)
			tc.populateCmdFlags(cmd.Flags())

			cmd.Run(cmd, []string{})
		})
	}
}
