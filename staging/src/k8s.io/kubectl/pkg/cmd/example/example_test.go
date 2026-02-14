/*
Copyright 2025 The Kubernetes Authors.

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

package example

import (
	"regexp"
	"testing"

	"k8s.io/cli-runtime/pkg/genericiooptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestExampleInvalidArgs(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	o, err := NewFlags(genericiooptions.NewTestIOStreamsDiscard()).ToOptions(tf, []string{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err := o.Validate(); err == nil {
		t.Fatalf("expected validation error for missing args")
	}
}

func TestExampleList(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	streams, _, out, _ := genericiooptions.NewTestIOStreams()
	if err := func() error {
		cmd := NewCmdExample("kubectl", tf, streams)
		if err := cmd.Flags().Set("list", "true"); err != nil {
			return err
		}
		cmd.Run(cmd, []string{})
		return nil
	}(); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if out.Len() == 0 {
		t.Fatalf("expected list output, got empty")
	}
}

func TestExamplePodAndDeployment(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	streams, _, out, _ := genericiooptions.NewTestIOStreams()

	// Pod example
	cmd := NewCmdExample("kubectl", tf, streams)
	cmd.Run(cmd, []string{"pods"})
	matched, _ := regexp.MatchString(`(?m)^kind:\s*Pod$`, out.String())
	if !matched {
		t.Fatalf("expected kind: Pod in output, got:\n%s", out.String())
	}

	out.Reset()

	// Deployment example with overrides
	cmd = NewCmdExample("kubectl", tf, streams)
	if err := cmd.Flags().Set("replicas", "3"); err != nil {
		t.Fatal(err)
	}
	if err := cmd.Flags().Set("name", "web"); err != nil {
		t.Fatal(err)
	}
	cmd.Run(cmd, []string{"deployments"})
	matched, _ = regexp.MatchString(`(?m)^kind:\s*Deployment$`, out.String())
	if !matched {
		t.Fatalf("expected kind: Deployment in output, got:\n%s", out.String())
	}
	matched, _ = regexp.MatchString(`(?m)^  name:\s*web$`, out.String())
	if !matched {
		t.Fatalf("expected metadata.name: web in output, got:\n%s", out.String())
	}
	matched, _ = regexp.MatchString(`(?m)^  replicas:\s*3$`, out.String())
	if !matched {
		t.Fatalf("expected replicas: 3 in output, got:\n%s", out.String())
	}
}

func TestExampleFallbackPO(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	streams, _, out, _ := genericiooptions.NewTestIOStreams()
	cmd := NewCmdExample("kubectl", tf, streams)
	cmd.Run(cmd, []string{"po"})
	matched, _ := regexp.MatchString(`(?m)^kind:\s*Pod$`, out.String())
	if !matched {
		t.Fatalf("expected kind: Pod in output via fallback, got:\n%s", out.String())
	}
}
