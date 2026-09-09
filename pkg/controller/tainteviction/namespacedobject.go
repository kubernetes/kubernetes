/*
Copyright 2025 The Kubernetes Authors.

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

package tainteviction

import (
	"k8s.io/apimachinery/pkg/types"
)

// NamespacedObject comprises a resource name with a mandatory namespace
// and optional UID. It gets rendered as "<namespace>/<name>[:<uid>]"
// (text output) or as an object (JSON output).
type NamespacedObject struct {
	types.NamespacedName
	UID types.UID
}

// String returns the general purpose string representation
func (n NamespacedObject) String() string {
	if n.UID != "" {
		return n.Namespace + string(types.Separator) + n.Name + ":" + string(n.UID)
	}
	return n.Namespace + string(types.Separator) + n.Name
}

// MarshalLog emits a struct containing required key/value pair
func (n NamespacedObject) MarshalLog() interface{} {
	return struct {
		Name      string    `json:"name"`
		Namespace string    `json:"namespace,omitempty"`
		UID       types.UID `json:"uid,omitempty"`
	}{
		Name:      n.Name,
		Namespace: n.Namespace,
		UID:       n.UID,
	}
}
