/*
Copyright 2020 The Kubernetes Authors.

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

package handlers

import (
	"context"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
)

// UpdatePersistedChange keeps track of the resourceVersion before we run the
// update, and looks again after the update, and checks for any
// difference. If a difference is found, then it will return true,
// otherwise it will return false.
type UpdatePersistedChange struct {
	resourceVersion string
}

func getResourceVersion(obj runtime.Object) (string, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return "", err
	}
	return accessor.GetResourceVersion(), nil
}

// TrackResourceVersion is a transformer that finds and records the
// resourceVersion from the object, before it's submitted to etcd.
func (u *UpdatePersistedChange) TrackResourceVersion(_ context.Context, obj, _ runtime.Object) (runtime.Object, error) {
	if rv, err := getResourceVersion(obj); err != nil {
		// Do nothing if we have an error
	} else {
		u.resourceVersion = rv
	}
	return obj, nil
}

// WasPersisted looked at the object, and returns true if its
// resourceVersion has changed, false otherwise.
func (u *UpdatePersistedChange) WasPersisted(obj runtime.Object) (bool, error) {
	rv, err := getResourceVersion(obj)
	if err != nil {
		// By default, assume that the object has changed.
		return true, err
	}
	return rv != u.resourceVersion, nil
}
