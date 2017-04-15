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

package util

import (
	"encoding/json"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	fed "k8s.io/kubernetes/federation/apis/federation"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

// UnmarshalPreferences reads the preferences from the annotations on the given object.
// It takes in an object and determines the supported types.
// Callers need to pass the string key used to store the annotations.
// Returns nil if the annotations with the given key are not found.
func UnmarshalPreferences(obj runtime.Object, key string) (*fed.ReplicaAllocationPreferences, error) {
	var pref fed.ReplicaAllocationPreferences
	var annotations map[string]string = nil
	if obj == nil {
		return nil, nil
	}

	switch obj := obj.(type) {
	default:
		return nil, fmt.Errorf("Unknnown object type while parsing annotations %v", obj)
	case *extensionsv1.Deployment:
		annotations = obj.Annotations
	case *extensionsv1.ReplicaSet:
		annotations = obj.Annotations
	}

	if annotations == nil {
		return nil, nil
	}
	prefString, found := annotations[key]
	if !found {
		return nil, nil
	}
	if err := json.Unmarshal([]byte(prefString), &pref); err != nil {
		return nil, err
	}
	return &pref, nil
}
