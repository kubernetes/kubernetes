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

package storage

import (
	"fmt"
	"strconv"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
)

// objectSizeLabel is an internal label that is used to pass information about object size between etcd3.store and cacher.
// This label should not be exposed externally.
const objectSizeLabel = "k8s.io/object-size"

func SetObjectSizeLabel(accessor meta.MetadataAccessor, obj runtime.Object, size int64) error {
	labels, err := accessor.Labels(obj)
	if err != nil {
		return err
	}
	if labels == nil {
		labels = map[string]string{}
	}
	labels[objectSizeLabel] = strconv.FormatInt(size, 10)
	return accessor.SetLabels(obj, labels)
}

func ReadAndRemoveObjectSizeLabel(accessor meta.MetadataAccessor, obj runtime.Object) (int64, error) {
	labels, err := accessor.Labels(obj)
	if err != nil {
		return 0, err
	}
	if labels == nil {
		return 0, fmt.Errorf("expected object to have %q label", objectSizeLabel)
	}
	sizeStr, ok := labels[objectSizeLabel]
	if !ok {
		return 0, fmt.Errorf("expected object to have %q label", objectSizeLabel)
	}
	delete(labels, objectSizeLabel)
	if len(labels) == 0 {
		labels = nil
	}
	err = accessor.SetLabels(obj, labels)
	if err != nil {
		return 0, err
	}
	return strconv.ParseInt(sizeStr, 10, 64)
}
