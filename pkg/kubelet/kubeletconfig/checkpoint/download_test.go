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

package checkpoint

import (
	"fmt"
	"testing"

	"github.com/davecgh/go-spew/spew"

	apiv1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	utiltest "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/test"
)

func TestNewRemoteConfigSource(t *testing.T) {
	cases := []struct {
		desc   string
		source *apiv1.NodeConfigSource
		expect RemoteConfigSource
		err    string
	}{
		{
			desc:   "all NodeConfigSource subfields nil",
			source: &apiv1.NodeConfigSource{},
			expect: nil,
			err:    "exactly one subfield must be non-nil",
		},
		{
			desc: "ConfigMap: valid reference",
			source: &apiv1.NodeConfigSource{
				ConfigMap: &apiv1.ConfigMapNodeConfigSource{
					Name:             "name",
					Namespace:        "namespace",
					UID:              "uid",
					KubeletConfigKey: "kubelet",
				}},
			expect: &remoteConfigMap{&apiv1.NodeConfigSource{
				ConfigMap: &apiv1.ConfigMapNodeConfigSource{
					Name:             "name",
					Namespace:        "namespace",
					UID:              "uid",
					KubeletConfigKey: "kubelet",
				}}},
			err: "",
		},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			source, _, err := NewRemoteConfigSource(c.source)
			utiltest.ExpectError(t, err, c.err)
			if err != nil {
				return
			}
			// underlying object should match the object passed in
			if !apiequality.Semantic.DeepEqual(c.expect.object(), source.object()) {
				t.Errorf("case %q, expect RemoteConfigSource %s but got %s", c.desc, spew.Sdump(c.expect), spew.Sdump(source))
			}
		})
	}
}

func TestRemoteConfigMapUID(t *testing.T) {
	const expect = "uid"
	source, _, err := NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
		Name:             "name",
		Namespace:        "namespace",
		UID:              expect,
		KubeletConfigKey: "kubelet",
	}})
	if err != nil {
		t.Fatalf("error constructing remote config source: %v", err)
	}
	uid := source.UID()
	if expect != uid {
		t.Errorf("expect %q, but got %q", expect, uid)
	}
}

func TestRemoteConfigMapAPIPath(t *testing.T) {
	const (
		name      = "name"
		namespace = "namespace"
	)
	source, _, err := NewRemoteConfigSource(&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
		Name:             name,
		Namespace:        namespace,
		UID:              "uid",
		KubeletConfigKey: "kubelet",
	}})
	if err != nil {
		t.Fatalf("error constructing remote config source: %v", err)
	}
	expect := fmt.Sprintf(configMapAPIPathFmt, namespace, name)
	path := source.APIPath()

	if expect != path {
		t.Errorf("expect %q, but got %q", expect, path)
	}
}

func TestRemoteConfigMapDownload(t *testing.T) {
	cm := &apiv1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "name",
			Namespace: "namespace",
			UID:       "uid",
		}}
	client := fakeclient.NewSimpleClientset(cm)
	payload, err := NewConfigMapPayload(cm)
	if err != nil {
		t.Fatalf("error constructing payload: %v", err)
	}

	makeSource := func(source *apiv1.NodeConfigSource) RemoteConfigSource {
		s, _, err := NewRemoteConfigSource(source)
		if err != nil {
			t.Fatalf("error constructing remote config source %v", err)
		}
		return s
	}

	cases := []struct {
		desc   string
		source RemoteConfigSource
		expect Payload
		err    string
	}{
		{
			desc: "object doesn't exist",
			source: makeSource(&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
				Name:             "bogus",
				Namespace:        "namespace",
				UID:              "bogus",
				KubeletConfigKey: "kubelet",
			}}),
			expect: nil,
			err:    "not found",
		},
		{
			desc: "UID is incorrect for namespace/name",
			source: makeSource(&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
				Name:             "name",
				Namespace:        "namespace",
				UID:              "bogus",
				KubeletConfigKey: "kubelet",
			}}),
			expect: nil,
			err:    "does not match",
		},
		{
			desc: "object exists and reference is correct",
			source: makeSource(&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{
				Name:             "name",
				Namespace:        "namespace",
				UID:              "uid",
				KubeletConfigKey: "kubelet",
			}}),
			expect: payload,
			err:    "",
		},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			payload, _, err := c.source.Download(client)
			utiltest.ExpectError(t, err, c.err)
			if err != nil {
				return
			}
			// downloaded object should match the expected
			if !apiequality.Semantic.DeepEqual(c.expect.object(), payload.object()) {
				t.Errorf("case %q, expect Checkpoint %s but got %s", c.desc, spew.Sdump(c.expect), spew.Sdump(payload))
			}
		})
	}
}

func TestEqualRemoteConfigSources(t *testing.T) {
	cases := []struct {
		desc   string
		a      RemoteConfigSource
		b      RemoteConfigSource
		expect bool
	}{
		{"both nil", nil, nil, true},
		{"a nil", nil, &remoteConfigMap{}, false},
		{"b nil", &remoteConfigMap{}, nil, false},
		{"neither nil, equal", &remoteConfigMap{}, &remoteConfigMap{}, true},
		{
			desc:   "neither nil, not equal",
			a:      &remoteConfigMap{&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{Name: "a"}}},
			b:      &remoteConfigMap{&apiv1.NodeConfigSource{ConfigMap: &apiv1.ConfigMapNodeConfigSource{KubeletConfigKey: "kubelet"}}},
			expect: false,
		},
	}

	for _, c := range cases {
		t.Run(c.desc, func(t *testing.T) {
			if EqualRemoteConfigSources(c.a, c.b) != c.expect {
				t.Errorf("expected EqualRemoteConfigSources to return %t, but got %t", c.expect, !c.expect)
			}
		})
	}
}
