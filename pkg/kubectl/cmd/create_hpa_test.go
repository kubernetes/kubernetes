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

package cmd

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/url"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	api "k8s.io/kubernetes/pkg/apis/core"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

var HPAGroupVersion = schema.GroupVersion{Group: "autoscaling", Version: "v1"}

func TestCreateHpa(t *testing.T) {
	rcObject := &api.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{Name: "fake-rc", Namespace: "test"},
		Spec: api.ReplicationControllerSpec{
			Replicas: 1,
		},
	}

	autoScalerObject := &autoscaling.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{Name: "fake-rc", Namespace: "test"},
	}

	f, tf, codec, ns := cmdtesting.NewAPIFactory()

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, HPAGroupVersion)

	hpaRestClient := &HPARESTClient{
		RESTClient: &fake.RESTClient{
			NegotiatedSerializer: ns,
			Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
				switch p, m := req.URL.Path, req.Method; {
				case p == "/namespaces/test/replicationcontrollers/fake-rc" && m == "GET":
					return &http.Response{StatusCode: 201, Header: defaultHeader(), Body: objBody(codec, rcObject)}, nil
				case p == "/namespaces/test/horizontalpodautoscalers" && m == "POST":
					return &http.Response{StatusCode: 201, Header: defaultHeader(), Body: ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(encoder, autoScalerObject))))}, nil
				default:
					t.Fatalf("unexpected request: %#v\n%#v", req.URL, req)
					return nil, nil
				}
			}),
		},
	}

	tf.Client = hpaRestClient
	tf.Namespace = "test"
	tf.Printer = &testPrinter{}

	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdCreateHorizontalPodAutoscaler(f, buf)
	cmd.Flags().Set("min", "2")
	cmd.Flags().Set("max", "10")
	cmd.Flags().Set("cpu-percent", "80")
	cmd.Run(cmd, []string{"replicationcontroller/fake-rc"})
	expectedOutput := "replicationcontroller \"fake-rc\" autoscaled\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", expectedOutput, buf.String())
	}
}

type HPARESTClient struct {
	*fake.RESTClient
}

func (c *HPARESTClient) Post() *restclient.Request {

	config := restclient.ContentConfig{
		ContentType:          runtime.ContentTypeJSON,
		GroupVersion:         &HPAGroupVersion,
		NegotiatedSerializer: c.NegotiatedSerializer,
	}

	ns := c.NegotiatedSerializer
	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	serializers := restclient.Serializers{
		Encoder: ns.EncoderForVersion(info.Serializer, HPAGroupVersion),
		Decoder: ns.DecoderToVersion(info.Serializer, HPAGroupVersion),
	}
	if info.StreamSerializer != nil {
		serializers.StreamingSerializer = info.StreamSerializer.Serializer
		serializers.Framer = info.StreamSerializer.Framer
	}
	return restclient.NewRequest(c, "POST", &url.URL{Host: "localhost"}, c.VersionedAPIPath, config, serializers, nil, nil)
}
