/*
Copyright 2018 The Kubernetes Authors.

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
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/runtime/schema"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"

	"k8s.io/kubernetes/pkg/api/legacyscheme"
	cmdtesting "k8s.io/kubernetes/pkg/kubectl/cmd/testing"
)

func TestCreateDaemonSet(t *testing.T) {
	daeName := "jonny-dae"
	f, tf := cmdtesting.NewAPIFactory()
	ns := legacyscheme.Codecs
	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: "v1"},
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewBuffer([]byte("{}"))),
			}, nil
		}),
	}
	tf.ClientConfig = &restclient.Config{}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdCreateDaemonset(f, buf, buf)
	cmd.Flags().Set("dry-run", "true")
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("image", "hollywood/jonny.depp:v2")
	cmd.Run(cmd, []string{daeName})
	expectedOutput := daeName
	if !strings.Contains(buf.String(), expectedOutput) {
		t.Errorf("expected output: %s, but got: %s", expectedOutput, buf.String())
	}
}

func TestCreateDaemonsetNoImage(t *testing.T) {
	daeName := "jonny-dae"
	f, tf := cmdtesting.NewAPIFactory()
	ns := legacyscheme.Codecs
	tf.Client = &fake.RESTClient{
		GroupVersion:         schema.GroupVersion{Version: "v1"},
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(&bytes.Buffer{}),
			}, nil
		}),
	}
	tf.ClientConfig = &restclient.Config{}
	tf.Namespace = "test"

	buf := bytes.NewBuffer([]byte{})
	cmd := NewCmdCreateDaemonset(f, buf, buf)
	cmd.Flags().Set("dry-run", "true")
	cmd.Flags().Set("output", "name")
	err := CreateDaemonset(f, buf, buf, cmd, []string{daeName})
	assert.Error(t, err, "at least one image must be specified")
}
