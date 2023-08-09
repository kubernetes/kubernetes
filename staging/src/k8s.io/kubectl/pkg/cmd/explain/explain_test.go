/*
Copyright 2022 The Kubernetes Authors.

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

package explain

import (
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/meta"
	sptest "k8s.io/apimachinery/pkg/util/strategicpatch/testing"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/util/openapi"
)

var (
	fakeSchema        = sptest.Fake{Path: filepath.Join("..", "..", "..", "testdata", "openapi", "swagger.json")}
	FakeOpenAPISchema = testOpenAPISchema{
		OpenAPISchemaFn: func() (openapi.Resources, error) {
			s, err := fakeSchema.OpenAPISchema()
			if err != nil {
				return nil, err
			}
			return openapi.NewOpenAPIData(s)
		},
	}
)

type testOpenAPISchema struct {
	OpenAPISchemaFn func() (openapi.Resources, error)
}

func TestExplainInvalidArgs(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	opts := NewExplainOptions("kubectl", genericclioptions.NewTestIOStreamsDiscard())
	cmd := NewCmdExplain("kubectl", tf, genericclioptions.NewTestIOStreamsDiscard())
	err := opts.Complete(tf, cmd, []string{})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	err = opts.Validate()
	if err.Error() != "You must specify the type of resource to explain. Use \"kubectl api-resources\" for a complete list of supported resources.\n" {
		t.Error("unexpected non-error")
	}

	err = opts.Complete(tf, cmd, []string{"resource1", "resource2"})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	err = opts.Validate()
	if err.Error() != "We accept only this format: explain RESOURCE\n" {
		t.Error("unexpected non-error")
	}
}

func TestExplainNotExistResource(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	opts := NewExplainOptions("kubectl", genericclioptions.NewTestIOStreamsDiscard())
	cmd := NewCmdExplain("kubectl", tf, genericclioptions.NewTestIOStreamsDiscard())
	err := opts.Complete(tf, cmd, []string{"foo"})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	err = opts.Validate()
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	err = opts.Run()
	if _, ok := err.(*meta.NoResourceMatchError); !ok {
		t.Fatalf("unexpected error %v", err)
	}
}

func TestExplainNotExistVersion(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	opts := NewExplainOptions("kubectl", genericclioptions.NewTestIOStreamsDiscard())
	cmd := NewCmdExplain("kubectl", tf, genericclioptions.NewTestIOStreamsDiscard())
	err := opts.Complete(tf, cmd, []string{"pods"})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	opts.APIVersion = "v99"

	err = opts.Validate()
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	err = opts.Run()
	if err.Error() != "couldn't find resource for \"/v99, Kind=Pod\"" {
		t.Errorf("unexpected non-error")
	}
}

func TestExplain(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	tf.OpenAPISchemaFunc = FakeOpenAPISchema.OpenAPISchemaFn
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	ioStreams, _, buf, _ := genericclioptions.NewTestIOStreams()
	cmd := NewCmdExplain("kubectl", tf, ioStreams)
	cmd.Run(cmd, []string{"pods"})
	if !strings.Contains(buf.String(), "KIND:     Pod") {
		t.Fatalf("expected output should include pod kind")
	}

	cmd.Flags().Set("recursive", "true")
	cmd.Run(cmd, []string{"pods"})
	if !strings.Contains(buf.String(), "KIND:     Pod") ||
		!strings.Contains(buf.String(), "annotations\t<map[string]string>") {
		t.Fatalf("expected output should include pod kind")
	}

	cmd.Flags().Set("api-version", "batch/v1")
	cmd.Run(cmd, []string{"cronjobs"})
	if !strings.Contains(buf.String(), "VERSION:  batch/v1") {
		t.Fatalf("expected output should include pod batch/v1")
	}

	cmd.Flags().Set("api-version", "batch/v1beta1")
	cmd.Run(cmd, []string{"cronjobs"})
	if !strings.Contains(buf.String(), "VERSION:  batch/v1beta1") {
		t.Fatalf("expected output should include pod batch/v1beta1")
	}
}
