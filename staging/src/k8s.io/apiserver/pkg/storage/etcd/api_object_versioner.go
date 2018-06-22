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
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// APIObjectVersioner implements versioning and extracting etcd node information
// for objects that have an embedded ObjectMeta or ListMeta field.
type APIObjectVersioner struct {
	resourceVersionObfuscator Obfuscator
}

// NewAPIObjectVersioner instantiates a default APIObjectVersioner
func NewAPIObjectVersioner() APIObjectVersioner {
	obfuscationEnabled := utilfeature.DefaultFeatureGate.Enabled(features.ResourceVersionObfuscation)
	var obfuscator Obfuscator
	if obfuscationEnabled {
		obfuscator = NewFeistelObfuscator()
	} else {
		obfuscator = NewIdentityObfuscator()
	}
	return APIObjectVersioner{
		resourceVersionObfuscator: obfuscator,
	}
}

// Since the purpose of this obfuscator is not to securely encrypt the value, just to make it very
// clear what operations clients are allowed to reliably do to resource versions, it is okay to
// be transparent about what key is used.
// TODO: Change this to something else, probably the name of the collection the object belongs to.
const resourceVersionObfuscatorKey string = "version"

// UpdateObject implements Versioner
func (a APIObjectVersioner) UpdateObject(obj runtime.Object, storageBackendResourceVersion uint64) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return err
	}
	resourceVersion := a.etcdRVToDisplayRV(storageBackendResourceVersion)
	accessor.SetResourceVersion(resourceVersion)
	return nil
}

// UpdateList implements Versioner
func (a APIObjectVersioner) UpdateList(obj runtime.Object, storageBackendResourceVersion uint64, nextKey string) error {
	listAccessor, err := meta.ListAccessor(obj)
	if err != nil || listAccessor == nil {
		return err
	}
	resourceVersion := a.etcdRVToDisplayRV(storageBackendResourceVersion)
	listAccessor.SetResourceVersion(resourceVersion)
	listAccessor.SetContinue(nextKey)
	return nil
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
	storageBackendResourceVersion, err := a.displayRVToEtcdRV(resourceVersion)
	return storageBackendResourceVersion, err
}

// ParseWatchResourceVersion takes a resource version argument and converts it to
// the etcd version we should pass to helper.Watch(). Because resourceVersion is
// an opaque value, the default watch behavior for non-zero watch is to watch
// the next value (if you pass "1", you will see updates from "2" onwards).
func (a APIObjectVersioner) ParseWatchResourceVersion(resourceVersion string) (uint64, error) {
	storageBackendResourceVersion, err := a.displayRVToEtcdRV(resourceVersion)
	if err != nil {
		return 0, storage.NewInvalidError(field.ErrorList{
			// Validation errors are supposed to return version-specific field
			// paths, but this is probably close enough.
			field.Invalid(field.NewPath("resourceVersion"), resourceVersion, err.Error()),
		})
	}
	return storageBackendResourceVersion, nil
}

// ParseListResourceVersion takes a resource version argument and converts it to
// the etcd version.
// TODO: reevaluate whether it is really clearer to have both this and the
// Watch version of this function, since they perform the same logic.
func (a APIObjectVersioner) ParseListResourceVersion(resourceVersion string) (uint64, error) {
	storageBackendResourceVersion, err := a.displayRVToEtcdRV(resourceVersion)
	if err != nil {
		return 0, storage.NewInvalidError(field.ErrorList{
			// Validation errors are supposed to return version-specific field
			// paths, but this is probably close enough.
			field.Invalid(field.NewPath("resourceVersion"), resourceVersion, err.Error()),
		})
	}
	return storageBackendResourceVersion, nil
}

// APIObjectVersioner implements Versioner
var Versioner storage.Versioner = NewAPIObjectVersioner()

// CompareResourceVersion compares etcd resource versions.  Outside this API they are all strings,
// but etcd resource versions are special, they're actually ints, so we can easily compare them.
func (a APIObjectVersioner) CompareResourceVersion(lhs, rhs runtime.Object) int {
	lhsVersion, err := Versioner.ObjectResourceVersion(lhs)
	if err != nil {
		// coder error
		panic(err)
	}
	rhsVersion, err := Versioner.ObjectResourceVersion(rhs)
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

// etcdRVToDisplayRV takes a etcd resource version argument and converts it to
// a version safe for clients to use.
func (a APIObjectVersioner) etcdRVToDisplayRV(etcdResourceVersion uint64) string {
	clientResourceVersion := a.resourceVersionObfuscator.Encode(resourceVersionObfuscatorKey, etcdResourceVersion)
	resourceVersion := formatVersion(clientResourceVersion)
	return resourceVersion
}

// displayRVToEtcdRV takes a client resource version string and converts it to
// the etcd version.
func (a APIObjectVersioner) displayRVToEtcdRV(resourceVersion string) (uint64, error) {
	clientResourceVersion, err := parseVersion(resourceVersion)
	if err != nil {
		return 0, err
	}
	etcdResourceVersion := a.resourceVersionObfuscator.Decode(resourceVersionObfuscatorKey, clientResourceVersion)
	return etcdResourceVersion, nil
}

// parseVersion takes a string and parses it's value as a uint64, or returns 0 if the string is ""
func parseVersion(versionString string) (uint64, error) {
	if versionString == "" {
		return 0, nil
	}
	versionUint, err := strconv.ParseUint(versionString, 10, 64)
	return versionUint, err
}

// formatVersion takes a uint64 and formats it to a string, or returns "" if the value is 0
func formatVersion(versionUint uint64) string {
	versionString := ""
	if versionUint != 0 {
		versionString = strconv.FormatUint(versionUint, 10)
	}
	return versionString
}
