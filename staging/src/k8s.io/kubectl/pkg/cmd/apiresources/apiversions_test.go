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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/discovery"
	fakediscovery "k8s.io/client-go/discovery/fake"
	clienttesting "k8s.io/client-go/testing"
)

// fakeCachedDiscoveryClient wraps a fake discovery client
type fakeCachedDiscoveryClient struct {
	discovery.DiscoveryInterface
	invalidations int
}

var _ discovery.CachedDiscoveryInterface = &fakeCachedDiscoveryClient{}

func (d *fakeCachedDiscoveryClient) Fresh() bool {
	return true
}

func (d *fakeCachedDiscoveryClient) Invalidate() {
	d.invalidations++
}

func newFakeCachedDiscoveryClient(resources []*metav1.APIResourceList) *fakeCachedDiscoveryClient {
	return &fakeCachedDiscoveryClient{
		DiscoveryInterface: &fakediscovery.FakeDiscovery{Fake: &clienttesting.Fake{Resources: resources}},
	}
}

func TestAPIVersionsToOptions(t *testing.T) {
	tf := genericclioptions.NewTestConfigFlags().WithDiscoveryClient(newFakeCachedDiscoveryClient(nil))
	flags := NewAPIVersionsFlags(tf, genericiooptions.NewTestIOStreamsDiscard())

	_, err := flags.ToOptions([]string{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	_, err = flags.ToOptions([]string{"foo"})
	if err == nil {
		t.Fatalf("An error was expected but not returned")
	}
	expectedError := `unexpected arguments: [foo]`
	if err.Error() != expectedError {
		t.Fatalf("Unexpected error: %v\n expected: %v", err, expectedError)
	}
}

func TestAPIVersionsRun(t *testing.T) {
	dc := newFakeCachedDiscoveryClient([]*metav1.APIResourceList{
		{GroupVersion: "v1"},
		{GroupVersion: "foo/v1beta1"},
		{GroupVersion: "foo/v1"},
		{GroupVersion: "foo/v2"},
		{GroupVersion: "bar/v1"},
	})
	tf := genericclioptions.NewTestConfigFlags().WithDiscoveryClient(dc)

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
	if dc.invalidations != expectedInvalidations {
		t.Fatalf("unexpected invalidations: %d, expected: %d", dc.invalidations, expectedInvalidations)
	}
}
