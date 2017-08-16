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

	apiv1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubelet/apis/kubeletconfig"
)

// Checkpoint represents a local copy of a config source (payload) object
type Checkpoint interface {
	// UID returns the UID of the config source object behind the Checkpoint
	UID() string
	// Parse parses the checkpoint into the internal KubeletConfiguration type
	Parse() (*kubeletconfig.KubeletConfiguration, error)
	// Encode returns a []byte representation of the config source object behind the Checkpoint
	Encode() ([]byte, error)

	// object returns the underlying checkpointed object. If you want to compare sources for equality, use EqualCheckpoints,
	// which compares the underlying checkpointed objects for semantic API equality.
	object() interface{}
}

// DecodeCheckpoint is a helper for using the apimachinery to decode serialized checkpoints
func DecodeCheckpoint(data []byte) (Checkpoint, error) {
	// decode the checkpoint
	obj, err := runtime.Decode(api.Codecs.UniversalDecoder(), data)
	if err != nil {
		return nil, fmt.Errorf("failed to decode, error: %v", err)
	}

	// TODO(mtaufen): for now we assume we are trying to load a ConfigMap checkpoint, may need to extend this if we allow other checkpoint types

	// convert it to the external ConfigMap type, so we're consistently working with the external type outside of the on-disk representation
	cm := &apiv1.ConfigMap{}
	err = api.Scheme.Convert(obj, cm, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to convert decoded object into a v1 ConfigMap, error: %v", err)
	}
	return NewConfigMapCheckpoint(cm)
}

// EqualCheckpoints compares two Checkpoints for equality, if their underlying objects are equal, so are the Checkpoints
func EqualCheckpoints(a, b Checkpoint) bool {
	if a != nil && b != nil {
		return apiequality.Semantic.DeepEqual(a.object(), b.object())
	}
	if a == nil && b == nil {
		return true
	}
	return false
}
