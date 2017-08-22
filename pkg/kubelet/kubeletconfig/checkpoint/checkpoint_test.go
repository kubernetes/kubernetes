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
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilcodec "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/codec"
	utiltest "k8s.io/kubernetes/pkg/kubelet/kubeletconfig/util/test"
)

func TestDecodeCheckpoint(t *testing.T) {
	// generate correct Checkpoint for v1/ConfigMap test case
	cm, err := NewConfigMapCheckpoint(&apiv1.ConfigMap{ObjectMeta: metav1.ObjectMeta{UID: types.UID("uid")}})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// generate unsupported object encoding for unsupported type test case
	unsupported := newUnsupportedEncoded(t)

	// test cases
	cases := []struct {
		desc   string
		data   []byte
		expect Checkpoint // expect a deeply-equal Checkpoint to be returned from Decode
		err    string     // expect error to contain this substring
	}{
		// v1/ConfigMap
		{"v1/ConfigMap", []byte(`{"apiVersion": "v1","kind": "ConfigMap","metadata": {"uid": "uid"}}`), cm, ""},
		// malformed
		{"malformed", []byte("malformed"), nil, "failed to decode"},
		// no UID
		{"no UID", []byte(`{"apiVersion": "v1","kind": "ConfigMap"}`), nil, "ConfigMap must have a UID"},
		// well-formed, but unsupported type
		{"well-formed, but unsupported encoded type", unsupported, nil, "failed to convert"},
	}

	for _, c := range cases {
		cpt, err := DecodeCheckpoint(c.data)
		if utiltest.SkipRest(t, c.desc, err, c.err) {
			continue
		}
		// Unfortunately reflect.DeepEqual treats nil data structures as != empty data structures, so
		// we have to settle for semantic equality of the underlying checkpointed API objects.
		// If additional fields are added to the object that implements the Checkpoint interface,
		// they should be added to a named sub-object to facilitate a DeepEquals comparison
		// of the extra fields.
		// decoded checkpoint should match expected checkpoint
		if !apiequality.Semantic.DeepEqual(cpt.object(), c.expect.object()) {
			t.Errorf("case %q, expect checkpoint %s but got %s", c.desc, spew.Sdump(c.expect), spew.Sdump(cpt))
		}
	}
}

// newUnsupportedEncoded returns an encoding of an object that does not have a Checkpoint implementation
func newUnsupportedEncoded(t *testing.T) []byte {
	encoder, err := utilcodec.NewJSONEncoder(apiv1.GroupName)
	if err != nil {
		t.Fatalf("could not create an encoder, error: %v", err)
	}
	unsupported := &apiv1.Node{}
	data, err := runtime.Encode(encoder, unsupported)
	if err != nil {
		t.Fatalf("could not encode object, error: %v", err)
	}
	return data
}
