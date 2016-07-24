/*
Copyright 2016 The Kubernetes Authors.

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

package versioner

import (
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/runtime"
)

// UpdateObject sets storage metadata into an API object. Returns an error if the object
// cannot be updated correctly. May return nil if the requested object does not need metadata
// from database.
func UpdateObject(obj runtime.Object, resourceVersion uint64) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	versionString := ""
	if resourceVersion != 0 {
		versionString = strconv.FormatUint(resourceVersion, 10)
	}
	accessor.SetResourceVersion(versionString)
	return nil
}

// UpdateList sets the resource version into an API list object. Returns an error if the object
// cannot be updated correctly. May return nil if the requested object does not need metadata
// from database.
func UpdateList(obj runtime.Object, resourceVersion uint64) error {
	listMeta, err := api.ListMetaFor(obj)
	if err != nil || listMeta == nil {
		return err
	}
	versionString := ""
	if resourceVersion != 0 {
		versionString = strconv.FormatUint(resourceVersion, 10)
	}
	listMeta.ResourceVersion = versionString
	return nil
}

// ObjectResourceVersion returns the resource version (for persistence) of the specified object.
// Should return an error if the specified object does not have a persistable version.
func ObjectResourceVersion(obj runtime.Object) (uint64, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return 0, err
	}
	version := accessor.GetResourceVersion()
	if len(version) == 0 {
		return 0, nil
	}
	return strconv.ParseUint(version, 10, 64)
}

// CompareResourceVersion compares etcd resource versions.  Outside this API they are all strings,
// but etcd resource versions are special, they're actually ints, so we can easily compare them.
func CompareResourceVersion(lhs, rhs runtime.Object) int {
	lhsVersion, err := ObjectResourceVersion(lhs)
	if err != nil {
		// coder error
		panic(err)
	}
	rhsVersion, err := ObjectResourceVersion(rhs)
	if err != nil {
		// coder error
		panic(err)
	}

	if lhsVersion == rhsVersion {
		return 0
	}
	if lhsVersion < rhsVersion {
		return -1
	}

	return 1
}
