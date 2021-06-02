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

package create

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

func TestCreatePriorityClass(t *testing.T) {
	pcName := "my-pc"
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	ns := scheme.Codecs.WithoutConversion()

	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Group: "scheduling.k8s.io", Version: "v1beta1"},
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(&bytes.Buffer{}),
			}, nil
		}),
	}
	tf.ClientConfigVal = &restclient.Config{}

	outputFormat := "name"

	ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdCreatePriorityClass(tf, ioStreams)
	cmd.Flags().Set("value", "1000")
	cmd.Flags().Set("global-default", "true")
	cmd.Flags().Set("description", "my priority")
	cmd.Flags().Set("dry-run", "client")
	cmd.Flags().Set("output", outputFormat)
	cmd.Flags().Set("preemption-policy", "Never")

	printFlags := genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme)
	printFlags.OutputFormat = &outputFormat

	options := &PriorityClassOptions{
		PrintFlags: printFlags,
		Name:       pcName,
		IOStreams:  ioStreams,
	}
	err := options.Complete(tf, cmd, []string{pcName})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = options.Run()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	expectedOutput := "priorityclass.scheduling.k8s.io/" + pcName + "\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", expectedOutput, buf.String())
	}
}
