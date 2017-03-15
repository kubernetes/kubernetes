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

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestCreateHpa(t *testing.T) {
	rcObject := &api.ReplicationController{}
	rcObject.Name = "fake-rc"
	autoScalerObject := &autoscaling.HorizontalPodAutoscaler{}
	autoScalerObject.Name = "fake-rc"

	f, tf, codec, ns := cmdtesting.NewAPIFactory()

	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := ns.EncoderForVersion(info.Serializer, schema.GroupVersion{Group: "autoscaling", Version: "v1"})

	hpaRestClient := &HPARESTClient{
		RESTClient: &fake.RESTClient{
			APIRegistry:          api.Registry,
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

	cmd := NewCmdCreateHpa(f, buf)
	cmd.Flags().Set("max", "2")
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
	groupVersion := &c.APIRegistry.GroupOrDie("autoscaling").GroupVersion

	config := restclient.ContentConfig{
		ContentType:          runtime.ContentTypeJSON,
		GroupVersion:         groupVersion,
		NegotiatedSerializer: c.NegotiatedSerializer,
	}

	ns := c.NegotiatedSerializer
	info, _ := runtime.SerializerInfoForMediaType(ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	serializers := restclient.Serializers{
		Encoder: ns.EncoderForVersion(info.Serializer, groupVersion),
		Decoder: ns.DecoderToVersion(info.Serializer, groupVersion),
	}
	if info.StreamSerializer != nil {
		serializers.StreamingSerializer = info.StreamSerializer.Serializer
		serializers.Framer = info.StreamSerializer.Framer
	}
	return restclient.NewRequest(c, "POST", &url.URL{Host: "localhost"}, c.VersionedAPIPath, config, serializers, nil, nil)
}
