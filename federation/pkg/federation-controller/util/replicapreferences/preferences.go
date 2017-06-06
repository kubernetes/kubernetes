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

package replicapreferences

import (
	"encoding/json"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	fed "k8s.io/kubernetes/federation/apis/federation"
)

// GetAllocationPreferences reads the preferences from the annotations on the given object.
// It takes in an object and determines the supported types.
// Callers need to pass the string key used to store the annotations.
// Returns nil if the annotations with the given key are not found.
func GetAllocationPreferences(obj runtime.Object, key string) (*fed.ReplicaAllocationPreferences, error) {
	if obj == nil {
		return nil, nil
	}

	accessor, err := meta.Accessor(obj)
	if err != nil {
		return nil, err
	}
	annotations := accessor.GetAnnotations()
	if annotations == nil {
		return nil, nil
	}

	prefString, found := annotations[key]
	if !found {
		return nil, nil
	}

	var pref fed.ReplicaAllocationPreferences
	if err := json.Unmarshal([]byte(prefString), &pref); err != nil {
		return nil, err
	}
	return &pref, nil
}
