/*
Copyright 2020 The Kubernetes Authors.

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

package version

import (
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestNewCmdVersionClientVersion(t *testing.T) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()
	streams, _, buf, _ := genericclioptions.NewTestIOStreams()
	o := NewOptions(streams)
	if err := o.Complete(tf, &cobra.Command{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if err := o.Validate(); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if err := o.Run(); err != nil {
		t.Errorf("Cannot execute version command: %v", err)
	}
	if !strings.Contains(buf.String(), "Client Version") {
		t.Errorf("unexpected output: %s", buf.String())
	}
}
