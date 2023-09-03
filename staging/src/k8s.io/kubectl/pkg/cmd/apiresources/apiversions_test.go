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

package apiresources

import (
	"testing"

	"github.com/spf13/cobra"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

func TestAPIVersionsComplete(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()
	cmd := NewCmdAPIVersions(tf, genericiooptions.NewTestIOStreamsDiscard())
	parentCmd := &cobra.Command{Use: "kubectl"}
	parentCmd.AddCommand(cmd)
	o := NewAPIVersionsOptions(genericiooptions.NewTestIOStreamsDiscard())

	err := o.Complete(tf, cmd, []string{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	err = o.Complete(tf, cmd, []string{"foo"})
	if err == nil {
		t.Fatalf("An error was expected but not returned")
	}
	expectedError := `unexpected arguments: [foo]
See 'kubectl api-versions -h' for help and examples`
	if err.Error() != expectedError {
		t.Fatalf("Unexpected error: %v\n expected: %v", err, expectedError)
	}
}

func TestAPIVersionsRun(t *testing.T) {
	dc := cmdtesting.NewFakeCachedDiscoveryClient()
	dc.Groups = []*v1.APIGroup{
		{
			Name: "",
			Versions: []v1.GroupVersionForDiscovery{
				{GroupVersion: "v1"},
			},
		},
		{
			Name: "foo",
			Versions: []v1.GroupVersionForDiscovery{
				{GroupVersion: "foo/v1beta1"},
				{GroupVersion: "foo/v1"},
				{GroupVersion: "foo/v2"},
			},
		},
		{
			Name: "bar",
			Versions: []v1.GroupVersionForDiscovery{
				{GroupVersion: "bar/v1"},
			},
		},
	}
	tf := cmdtesting.NewTestFactory().WithDiscoveryClient(dc)
	defer tf.Cleanup()

	ioStreams, _, out, errOut := genericiooptions.NewTestIOStreams()
	cmd := NewCmdAPIVersions(tf, ioStreams)
	cmd.Run(cmd, []string{})

	if errOut.Len() > 0 {
		t.Fatalf("unexpected error output: %s", errOut.String())
	}

	expectedOutput := `bar/v1
foo/v1
foo/v1beta1
foo/v2
v1
`
	if out.String() != expectedOutput {
		t.Fatalf("unexpected output: %s\nexpected: %s", out.String(), expectedOutput)
	}

	expectedInvalidations := 1
	if dc.Invalidations != expectedInvalidations {
		t.Fatalf("unexpected invalidations: %d, expected: %d", dc.Invalidations, expectedInvalidations)
	}
}
