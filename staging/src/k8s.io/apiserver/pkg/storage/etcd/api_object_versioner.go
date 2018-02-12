/*
Copyright 2014 The Kubernetes Authors.

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

package etcd

import (
	"strconv"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage"
)

// APIObjectVersioner implements versioning and extracting etcd node information
// for objects that have an embedded ObjectMeta or ListMeta field.
type APIObjectVersioner struct{}

// UpdateObject implements Versioner
func (a APIObjectVersioner) UpdateObject(obj runtime.Object, resourceVersion string) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	accessor.SetResourceVersion(resourceVersion)
	return nil
}

// UpdateList implements Versioner
func (a APIObjectVersioner) UpdateList(obj runtime.Object, resourceVersion string, nextKey string) error {
	listAccessor, err := meta.ListAccessor(obj)
	if err != nil || listAccessor == nil {
		return err
	}
	listAccessor.SetResourceVersion(resourceVersion)
	listAccessor.SetContinue(nextKey)
	return nil
}

// UpdateObjectEtcdVersion calls first converts etcdResourceVersion to a string then calls UpdateObject
func (a APIObjectVersioner) UpdateObjectEtcdVersion(obj runtime.Object, etcdResourceVersion uint64) error {
	resourceVersion := a.EtcdRVToDisplayRV(etcdResourceVersion)
	return a.UpdateObject(obj, resourceVersion)
}

// UpdateListEtcdVersion calls first converts etcdResourceVersion to a string then calls UpdateList
func (a APIObjectVersioner) UpdateListEtcdVersion(obj runtime.Object, etcdResourceVersion uint64, nextKey string) error {
	resourceVersion := a.EtcdRVToDisplayRV(etcdResourceVersion)
	return a.UpdateList(obj, resourceVersion, nextKey)
}

// PrepareObjectForStorage clears resource version and self link prior to writing to etcd.
func (a APIObjectVersioner) PrepareObjectForStorage(obj runtime.Object) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	accessor.SetResourceVersion("")
	accessor.SetSelfLink("")
	return nil
}

// ObjectResourceVersion implements Versioner
func (a APIObjectVersioner) ObjectResourceVersion(obj runtime.Object) (uint64, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return 0, err
	}
	resourceVersion := accessor.GetResourceVersion()
	etcdResourceVersion, err := a.DisplayRVToEtcdRV(resourceVersion)
	return etcdResourceVersion, err
}

// ParseWatchResourceVersion takes a resource version argument and converts it to
// the etcd version we should pass to helper.Watch(). Because resourceVersion is
// an opaque value, the default watch behavior for non-zero watch is to watch
// the next value (if you pass "1", you will see updates from "2" onwards).
func (a APIObjectVersioner) ParseWatchResourceVersion(resourceVersion string) (uint64, error) {
	etcdResourceVersion, err := a.DisplayRVToEtcdRV(resourceVersion)
	if err != nil {
		return 0, storage.NewInvalidError(field.ErrorList{
			// Validation errors are supposed to return version-specific field
			// paths, but this is probably close enough.
			field.Invalid(field.NewPath("resourceVersion"), resourceVersion, err.Error()),
		})
	}
	return etcdResourceVersion, nil
}

// ParseListResourceVersion takes a resource version argument and converts it to
// the etcd version.
// TODO: reevaluate whether it is really clearer to have both this and the
// Watch version of this function, since they perform the same logic.
func (a APIObjectVersioner) ParseListResourceVersion(resourceVersion string) (uint64, error) {
	etcdResourceVersion, err := a.DisplayRVToEtcdRV(resourceVersion)
	if err != nil {
		return 0, storage.NewInvalidError(field.ErrorList{
			// Validation errors are supposed to return version-specific field
			// paths, but this is probably close enough.
			field.Invalid(field.NewPath("resourceVersion"), resourceVersion, err.Error()),
		})
	}
	return etcdResourceVersion, nil
}

// APIObjectVersioner implements Versioner
var Versioner storage.Versioner = APIObjectVersioner{}

// CompareObjectResourceVersion compares etcd resource versions of two objects.
func (a APIObjectVersioner) CompareObjectResourceVersion(lhs, rhs runtime.Object) int {
	lhsVersion, err := a.ObjectResourceVersion(lhs)
	if err != nil {
		// coder error
		panic(err)
	}
	rhsVersion, err := a.ObjectResourceVersion(rhs)
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

// CompareResourceVersion compares etcd resource versions.  Outside this API they are all strings,
// but etcd resource versions are special, they're actually ints, so we can easily compare them.
func (a APIObjectVersioner) CompareResourceVersion(lhs, rhs string) int {
	lhsVersion, err := a.DisplayRVToEtcdRV(lhs)
	if err != nil {
		// coder error
		panic(err)
	}
	rhsVersion, err := a.DisplayRVToEtcdRV(rhs)
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

// NextResourceVersion implements Versioner.
func (a APIObjectVersioner) NextResourceVersion(resourceVersion string) (string, error) {
	etcdResourceVersion, err := a.DisplayRVToEtcdRV(resourceVersion)
	if err != nil {
		return "", err
	}
	return a.EtcdRVToDisplayRV(etcdResourceVersion + 1), nil
}

// LastResourceVersion implements Versioner.
func (a APIObjectVersioner) LastResourceVersion(resourceVersion string) (string, error) {
	etcdResourceVersion, err := a.DisplayRVToEtcdRV(resourceVersion)
	if err != nil {
		return "", err
	}
	return a.EtcdRVToDisplayRV(etcdResourceVersion - 1), nil
}

// EtcdRVToDisplayRV takes a uint64 and formats it to a string, or returns "" if the value is 0
func (a APIObjectVersioner) EtcdRVToDisplayRV(versionUint uint64) string {
	if versionUint == 0 {
		return ""
	}
	return strconv.FormatUint(versionUint, 10)
}

// DisplayRVToEtcdRV takes a string and parses it's value as a uint64, or returns 0 if the string is ""
func (a APIObjectVersioner) DisplayRVToEtcdRV(versionString string) (uint64, error) {
	if versionString == "" {
		return 0, nil
	}
	return strconv.ParseUint(versionString, 10, 64)
}
