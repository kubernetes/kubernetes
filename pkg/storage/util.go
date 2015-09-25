/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"strconv"

	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/fielderrors"
)

type SimpleUpdateFunc func(runtime.Object) (runtime.Object, error)

// SimpleUpdateFunc converts SimpleUpdateFunc into UpdateFunc
func SimpleUpdate(fn SimpleUpdateFunc) UpdateFunc {
	return func(input runtime.Object, _ ResponseMeta) (runtime.Object, *uint64, error) {
		out, err := fn(input)
		return out, nil, err
	}
}

// ParseWatchResourceVersion takes a resource version argument and converts it to
// the etcd version we should pass to helper.Watch(). Because resourceVersion is
// an opaque value, the default watch behavior for non-zero watch is to watch
// the next value (if you pass "1", you will see updates from "2" onwards).
func ParseWatchResourceVersion(resourceVersion, kind string) (uint64, error) {
	if resourceVersion == "" || resourceVersion == "0" {
		return 0, nil
	}
	version, err := strconv.ParseUint(resourceVersion, 10, 64)
	if err != nil {
		// TODO: Does this need to be a ValidationErrorList?  I can't convince myself it does.
		return 0, errors.NewInvalid(kind, "", fielderrors.ValidationErrorList{fielderrors.NewFieldInvalid("resourceVersion", resourceVersion, err.Error())})
	}
	return version + 1, nil
}

func NamespaceKeyFunc(prefix string, obj runtime.Object) (string, error) {
	meta, err := meta.Accessor(obj)
	if err != nil {
		return "", err
	}
	return prefix + "/" + meta.Namespace() + "/" + meta.Name(), nil
}

func NoNamespaceKeyFunc(prefix string, obj runtime.Object) (string, error) {
	meta, err := meta.Accessor(obj)
	if err != nil {
		return "", err
	}
	return prefix + "/" + meta.Name(), nil
}
