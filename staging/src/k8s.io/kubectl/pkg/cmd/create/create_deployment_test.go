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

package create

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
)

func TestCreateDeployment(t *testing.T) {
	depName := "jonny-dep"
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	ns := scheme.Codecs.WithoutConversion()
	fakeDiscovery := "{\"kind\":\"APIResourceList\",\"apiVersion\":\"v1\",\"groupVersion\":\"apps/v1\",\"resources\":[{\"name\":\"deployments\",\"singularName\":\"\",\"namespaced\":true,\"kind\":\"Deployment\",\"verbs\":[\"create\",\"delete\",\"deletecollection\",\"get\",\"list\",\"patch\",\"update\",\"watch\"],\"shortNames\":[\"deploy\"],\"categories\":[\"all\"]}]}"
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewBuffer([]byte(fakeDiscovery))),
			}, nil
		}),
	}
	tf.ClientConfigVal = &restclient.Config{}

	ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdCreateDeployment(tf, ioStreams)
	cmd.Flags().Set("dry-run", "client")
	cmd.Flags().Set("output", "name")
	cmd.Flags().Set("image", "hollywood/jonny.depp:v2")
	cmd.Run(cmd, []string{depName})
	expectedOutput := "deployment.apps/" + depName + "\n"
	if buf.String() != expectedOutput {
		t.Errorf("expected output: %s, but got: %s", expectedOutput, buf.String())
	}
}

func TestCreateDeploymentWithPort(t *testing.T) {
	depName := "jonny-dep"
	port := "5701"
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	ns := scheme.Codecs.WithoutConversion()
	fakeDiscovery := "{\"kind\":\"APIResourceList\",\"apiVersion\":\"v1\",\"groupVersion\":\"apps/v1\",\"resources\":[{\"name\":\"deployments\",\"singularName\":\"\",\"namespaced\":true,\"kind\":\"Deployment\",\"verbs\":[\"create\",\"delete\",\"deletecollection\",\"get\",\"list\",\"patch\",\"update\",\"watch\"],\"shortNames\":[\"deploy\"],\"categories\":[\"all\"]}]}"
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewBuffer([]byte(fakeDiscovery))),
			}, nil
		}),
	}
	tf.ClientConfigVal = &restclient.Config{}

	ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdCreateDeployment(tf, ioStreams)
	cmd.Flags().Set("dry-run", "client")
	cmd.Flags().Set("output", "yaml")
	cmd.Flags().Set("port", port)
	cmd.Flags().Set("image", "hollywood/jonny.depp:v2")
	cmd.Run(cmd, []string{depName})
	if !strings.Contains(buf.String(), port) {
		t.Errorf("unexpected output: %s\nexpected to contain: %s", buf.String(), port)
	}
}

func TestCreateDeploymentWithReplicas(t *testing.T) {
	depName := "jonny-dep"
	replicas := "3"
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	ns := scheme.Codecs.WithoutConversion()
	fakeDiscovery := "{\"kind\":\"APIResourceList\",\"apiVersion\":\"v1\",\"groupVersion\":\"apps/v1\",\"resources\":[{\"name\":\"deployments\",\"singularName\":\"\",\"namespaced\":true,\"kind\":\"Deployment\",\"verbs\":[\"create\",\"delete\",\"deletecollection\",\"get\",\"list\",\"patch\",\"update\",\"watch\"],\"shortNames\":[\"deploy\"],\"categories\":[\"all\"]}]}"
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewBuffer([]byte(fakeDiscovery))),
			}, nil
		}),
	}
	tf.ClientConfigVal = &restclient.Config{}

	ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdCreateDeployment(tf, ioStreams)
	cmd.Flags().Set("dry-run", "client")
	cmd.Flags().Set("output", "jsonpath={.spec.replicas}")
	cmd.Flags().Set("replicas", replicas)
	cmd.Flags().Set("image", "hollywood/jonny.depp:v2")
	cmd.Run(cmd, []string{depName})
	if buf.String() != replicas {
		t.Errorf("expected output: %s, but got: %s", replicas, buf.String())
	}
}

func TestCreateDeploymentNoImage(t *testing.T) {
	depName := "jonny-dep"
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	ns := scheme.Codecs.WithoutConversion()
	fakeDiscovery := "{\"kind\":\"APIResourceList\",\"apiVersion\":\"v1\",\"groupVersion\":\"apps/v1\",\"resources\":[{\"name\":\"deployments\",\"singularName\":\"\",\"namespaced\":true,\"kind\":\"Deployment\",\"verbs\":[\"create\",\"delete\",\"deletecollection\",\"get\",\"list\",\"patch\",\"update\",\"watch\"],\"shortNames\":[\"deploy\"],\"categories\":[\"all\"]}]}"
	tf.Client = &fake.RESTClient{
		NegotiatedSerializer: ns,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
			return &http.Response{
				StatusCode: http.StatusOK,
				Body:       ioutil.NopCloser(bytes.NewBuffer([]byte(fakeDiscovery))),
			}, nil
		}),
	}
	tf.ClientConfigVal = &restclient.Config{}

	ioStreams := genericclioptions.NewTestIOStreamsDiscard()
	cmd := NewCmdCreateDeployment(tf, ioStreams)
	cmd.Flags().Set("output", "name")
	options := &CreateDeploymentOptions{
		PrintFlags:     genericclioptions.NewPrintFlags("created").WithTypeSetter(scheme.Scheme),
		DryRunStrategy: cmdutil.DryRunClient,
		IOStreams:      ioStreams,
	}

	err := options.Complete(tf, cmd, []string{depName})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	err = options.Run()
	assert.Error(t, err, "at least one image must be specified")
}
