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

package store

import (
	"testing"

	"github.com/davecgh/go-spew/spew"

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/kubelet/kubeletconfig/checkpoint"
)

func TestReset(t *testing.T) {
	source, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
		Name:             "name",
		Namespace:        "namespace",
		UID:              "uid",
		KubeletConfigKey: "kubelet",
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	otherSource, _, err := checkpoint.NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
		Name:             "other-name",
		Namespace:        "namespace",
		UID:              "other-uid",
		KubeletConfigKey: "kubelet",
	}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	cases := []struct {
		s       *fakeStore
		updated bool
	}{
		{&fakeStore{assigned: nil, lastKnownGood: nil}, false},
		{&fakeStore{assigned: source, lastKnownGood: nil}, true},
		{&fakeStore{assigned: nil, lastKnownGood: source}, false},
		{&fakeStore{assigned: source, lastKnownGood: source}, true},
		{&fakeStore{assigned: source, lastKnownGood: otherSource}, true},
		{&fakeStore{assigned: otherSource, lastKnownGood: source}, true},
	}
	for _, c := range cases {
		updated, err := reset(c.s)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if c.s.assigned != nil || c.s.lastKnownGood != nil {
			t.Errorf("case %q, expect nil for assigned and last-known-good checkpoints, but still have %q and %q, respectively",
				spew.Sdump(c.s), c.s.assigned, c.s.lastKnownGood)
		}
		if c.updated != updated {
			t.Errorf("case %q, expect reset to return %t, but got %t", spew.Sdump(c.s), c.updated, updated)
		}
	}
}
