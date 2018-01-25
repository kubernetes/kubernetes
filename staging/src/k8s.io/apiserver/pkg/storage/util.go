/*
Copyright 2015 The Kubernetes Authors.

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

package storage

import (
	"fmt"
	"strconv"
	"strings"
	"sync/atomic"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/validation/path"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

type SimpleUpdateFunc func(runtime.Object) (runtime.Object, error)

// SimpleUpdateFunc converts SimpleUpdateFunc into UpdateFunc
func SimpleUpdate(fn SimpleUpdateFunc) UpdateFunc {
	return func(input runtime.Object, _ ResponseMeta) (runtime.Object, *uint64, error) {
		out, err := fn(input)
		return out, nil, err
	}
}

func EverythingFunc(runtime.Object) bool {
	return true
}

func NoTriggerFunc() []MatchValue {
	return nil
}

func NoTriggerPublisher(runtime.Object) []MatchValue {
	return nil
}

// ParseWatchResourceVersion takes a resource version argument and converts it to
// the etcd version we should pass to helper.Watch(). Because resourceVersion is
// an opaque value, the default watch behavior for non-zero watch is to watch
// the next value (if you pass "1", you will see updates from "2" onwards).
func ParseWatchResourceVersion(resourceVersion string) (uint64, error) {
	if resourceVersion == "" || resourceVersion == "0" {
		return 0, nil
	}
	version, err := strconv.ParseUint(resourceVersion, 10, 64)
	if err != nil {
		return 0, NewInvalidError(field.ErrorList{
			// Validation errors are supposed to return version-specific field
			// paths, but this is probably close enough.
			field.Invalid(field.NewPath("resourceVersion"), resourceVersion, err.Error()),
		})
	}
	return version, nil
}

// ParseListResourceVersion takes a resource version argument and converts it to
// the etcd version.
func ParseListResourceVersion(resourceVersion string) (uint64, error) {
	if resourceVersion == "" {
		return 0, nil
	}
	version, err := strconv.ParseUint(resourceVersion, 10, 64)
	return version, err
}

func NamespaceKeyFunc(prefix string, obj runtime.Object) (string, error) {
	meta, err := meta.Accessor(obj)
	if err != nil {
		return "", err
	}
	name := meta.GetName()
	if msgs := path.IsValidPathSegmentName(name); len(msgs) != 0 {
		return "", fmt.Errorf("invalid name: %v", msgs)
	}
	return prefix + "/" + meta.GetNamespace() + "/" + name, nil
}

func NoNamespaceKeyFunc(prefix string, obj runtime.Object) (string, error) {
	meta, err := meta.Accessor(obj)
	if err != nil {
		return "", err
	}
	name := meta.GetName()
	if msgs := path.IsValidPathSegmentName(name); len(msgs) != 0 {
		return "", fmt.Errorf("invalid name: %v", msgs)
	}
	return prefix + "/" + name, nil
}

// hasPathPrefix returns true if the string matches pathPrefix exactly, or if is prefixed with pathPrefix at a path segment boundary
func hasPathPrefix(s, pathPrefix string) bool {
	// Short circuit if s doesn't contain the prefix at all
	if !strings.HasPrefix(s, pathPrefix) {
		return false
	}

	pathPrefixLength := len(pathPrefix)

	if len(s) == pathPrefixLength {
		// Exact match
		return true
	}
	if strings.HasSuffix(pathPrefix, "/") {
		// pathPrefix already ensured a path segment boundary
		return true
	}
	if s[pathPrefixLength:pathPrefixLength+1] == "/" {
		// The next character in s is a path segment boundary
		// Check this instead of normalizing pathPrefix to avoid allocating on every call
		return true
	}
	return false
}

// HighWaterMark is a thread-safe object for tracking the maximum value seen
// for some quantity.
type HighWaterMark int64

// Update returns true if and only if 'current' is the highest value ever seen.
func (hwm *HighWaterMark) Update(current int64) bool {
	for {
		old := atomic.LoadInt64((*int64)(hwm))
		if current <= old {
			return false
		}
		if atomic.CompareAndSwapInt64((*int64)(hwm), old, current) {
			return true
		}
	}
}
