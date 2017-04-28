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
	"testing"

	"github.com/davecgh/go-spew/spew"

	apiv1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
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
		// all NodeConfigSource subfields nil
		{"all NodeConfigSource subfields nil",
			&apiv1.NodeConfigSource{}, nil, "exactly one subfield must be non-nil"},
		{"ConfigMapRef: empty name, namespace, and UID",
			&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{}}, nil, "invalid ObjectReference"},
		// ConfigMapRef: empty name and namespace
		{"ConfigMapRef: empty name and namespace",
			&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{UID: "uid"}}, nil, "invalid ObjectReference"},
		// ConfigMapRef: empty name and UID
		{"ConfigMapRef: empty name and UID",
			&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Namespace: "namespace"}}, nil, "invalid ObjectReference"},
		// ConfigMapRef: empty namespace and UID
		{"ConfigMapRef: empty namespace and UID",
			&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Name: "name"}}, nil, "invalid ObjectReference"},
		// ConfigMapRef: empty UID
		{"ConfigMapRef: empty namespace and UID",
			&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Name: "name", Namespace: "namespace"}}, nil, "invalid ObjectReference"},
		// ConfigMapRef: empty namespace
		{"ConfigMapRef: empty namespace and UID",
			&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Name: "name", UID: "uid"}}, nil, "invalid ObjectReference"},
		// ConfigMapRef: empty name
		{"ConfigMapRef: empty namespace and UID",
			&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Namespace: "namespace", UID: "uid"}}, nil, "invalid ObjectReference"},
		// ConfigMapRef: valid reference
		{"ConfigMapRef: valid reference",
			&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Name: "name", Namespace: "namespace", UID: "uid"}},
			&remoteConfigMap{&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Name: "name", Namespace: "namespace", UID: "uid"}}}, ""},
	}

	for _, c := range cases {
		src, _, err := NewRemoteConfigSource(c.source)
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		// underlying object should match the object passed in
		if !apiequality.Semantic.DeepEqual(c.expect.object(), src.object()) {
			t.Errorf("case %q, expect RemoteConfigSource %s but got %s", c.desc, spew.Sdump(c.expect), spew.Sdump(src))
		}
	}
}

func TestRemoteConfigMapUID(t *testing.T) {
	cases := []string{"", "uid", "376dfb73-56db-11e7-a01e-42010a800002"}
	for _, uidIn := range cases {
		cpt := &remoteConfigMap{
			&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Name: "name", Namespace: "namespace", UID: types.UID(uidIn)}},
		}
		// UID method should return the correct value of the UID
		uidOut := cpt.UID()
		if uidIn != uidOut {
			t.Errorf("expect UID() to return %q, but got %q", uidIn, uidOut)
		}
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

	cases := []struct {
		desc   string
		source RemoteConfigSource
		expect Checkpoint
		err    string
	}{

		// object doesn't exist
		{"object doesn't exist",
			&remoteConfigMap{&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Name: "bogus", Namespace: "namespace", UID: "bogus"}}},
			nil, "could not download ConfigMap"},
		// UID of downloaded object doesn't match UID of referent found via namespace/name
		{"UID is incorrect for namespace/name",
			&remoteConfigMap{&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Name: "name", Namespace: "namespace", UID: "bogus"}}},
			nil, "does not match UID"},
		// successful download
		{"object exists and reference is correct",
			&remoteConfigMap{&apiv1.NodeConfigSource{ConfigMapRef: &apiv1.ObjectReference{Name: "name", Namespace: "namespace", UID: "uid"}}},
			&configMapCheckpoint{cm}, ""},
	}

	for _, c := range cases {
		cpt, _, err := c.source.Download(client)
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		// "downloaded" object should match the expected
		if !apiequality.Semantic.DeepEqual(c.expect.object(), cpt.object()) {
			t.Errorf("case %q, expect Checkpoint %s but got %s", c.desc, spew.Sdump(c.expect), spew.Sdump(cpt))
		}
	}
}
