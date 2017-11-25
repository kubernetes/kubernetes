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
	"io"
	"net/http"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/core"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

// implement the Encoder interface to create codecs for working with batch objects
type customVersionCodec struct {
	ns           runtime.NegotiatedSerializer
	groupVersion *schema.GroupVersion
}

func (c customVersionCodec) Encode(obj runtime.Object, w io.Writer) error {
	info, _ := runtime.SerializerInfoForMediaType(c.ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	encoder := c.ns.EncoderForVersion(info.Serializer, c.groupVersion)
	return encoder.Encode(obj, w)
}

func (c customVersionCodec) Decode(data []byte, defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	info, _ := runtime.SerializerInfoForMediaType(c.ns.SupportedMediaTypes(), runtime.ContentTypeJSON)
	decoder := c.ns.DecoderToVersion(info.Serializer, c.groupVersion)
	return decoder.Decode(data, defaults, into)
}

var _ = runtime.Encoder(customVersionCodec{})

func TestCreateJobFromCronJob(t *testing.T) {
	jobSpec := batch.JobSpec{
		Template: core.PodTemplateSpec{
			Spec: core.PodSpec{
				Containers: []core.Container{
					{Image: "fake"},
				},
			},
		},
	}

	cjmiObject := &batch.CronJobManualRequest{
		CreatedJob: batch.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-cronjob-manual-1234567890",
			},
			Spec: jobSpec,
		},
	}

	cronJobObject := &batch.CronJob{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-cronjob",
		},
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1beta1",
			Kind:       "batch",
		},
		Spec: batch.CronJobSpec{
			Schedule: "* * * * *",
			JobTemplate: batch.JobTemplateSpec{
				Spec: jobSpec,
			},
		},
	}

	f, tf, _, ns := cmdtesting.NewAPIFactory()

	tf.Printer = &testPrinter{}
	tf.ClientConfig = &restclient.Config{
		ContentConfig: restclient.ContentConfig{
			GroupVersion: &schema.GroupVersion{Group: "batch", Version: "v1beta1"},
		},
	}

	alphaBatchCodec := &customVersionCodec{
		ns:           ns,
		groupVersion: &schema.GroupVersion{Group: "batch", Version: "v1beta1"},
	}

	tf.Namespace = "test"
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			switch p, m := req.URL.Path, req.Method; {
			case p == "/apis/batch/v1beta1/namespaces/test/cronjobs/test-cronjob" && m == "GET":
				return &http.Response{StatusCode: 200, Header: defaultHeader(), Body: objBody(alphaBatchCodec, cronJobObject)}, nil
			case p == "/apis/batch/v1beta1/namespaces/test/cronjobs/test-cronjob/instantiate" && m == "POST":
				return &http.Response{StatusCode: 201, Header: defaultHeader(), Body: objBody(alphaBatchCodec, cjmiObject)}, nil
			default:
				t.Fatalf("unexpected request for p=%s and m=%s: %#v\n%#v", p, m, req.URL, req)
				return nil, nil
			}
		}),
	}

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdCreateJob(f, buf)
	cmd.Flags().Set("from-cronjob", "test-cronjob")
	cmd.Run(cmd, []string{})
	expectedOutput := "job \"test-cronjob-manual-1234567890\" created\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", expectedOutput, buf.String())
	}
}
