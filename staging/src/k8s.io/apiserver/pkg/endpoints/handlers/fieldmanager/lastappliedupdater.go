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

package fieldmanager

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

const totalAnnotationSizeLimitB int64 = 256 * (1 << 10) // 256 kB

type lastAppliedUpdater struct {
	fieldManager Manager
}

var _ Manager = &lastAppliedUpdater{}

// NewLastAppliedUpdater sets the client-side apply annotation up to date with
// server-side apply managed fields
func NewLastAppliedUpdater(fieldManager Manager) Manager {
	return &lastAppliedUpdater{
		fieldManager: fieldManager,
	}
}

// Update implements Manager.
func (f *lastAppliedUpdater) Update(liveObj, newObj runtime.Object, managed Managed, manager string) (runtime.Object, Managed, error) {
	return f.fieldManager.Update(liveObj, newObj, managed, manager)
}

// server-side apply managed fields
func (f *lastAppliedUpdater) Apply(liveObj, newObj runtime.Object, managed Managed, manager string, force bool) (runtime.Object, Managed, error) {
	liveObj, managed, err := f.fieldManager.Apply(liveObj, newObj, managed, manager, force)
	if err != nil {
		return liveObj, managed, err
	}

	// Sync the client-side apply annotation only from kubectl server-side apply.
	// To opt-out of this behavior, users may specify a different field manager.
	//
	// If the client-side apply annotation doesn't exist,
	// then continue because we have no annotation to update
	if manager == "kubectl" && hasLastApplied(liveObj) {
		lastAppliedValue, err := buildLastApplied(newObj)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to build last-applied annotation: %v", err)
		}
		err = setLastApplied(liveObj, lastAppliedValue)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to set last-applied annotation: %v", err)
		}
	}
	return liveObj, managed, err
}

func hasLastApplied(obj runtime.Object) bool {
	var accessor, err = meta.Accessor(obj)
	if err != nil {
		panic(fmt.Sprintf("couldn't get accessor: %v", err))
	}
	var annotations = accessor.GetAnnotations()
	if annotations == nil {
		return false
	}
	lastApplied, ok := annotations[corev1.LastAppliedConfigAnnotation]
	return ok && len(lastApplied) > 0
}

func setLastApplied(obj runtime.Object, value string) error {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		panic(fmt.Sprintf("couldn't get accessor: %v", err))
	}
	var annotations = accessor.GetAnnotations()
	if annotations == nil {
		annotations = map[string]string{}
	}
	annotations[corev1.LastAppliedConfigAnnotation] = value
	if isAnnotationsValid(annotations) != nil {
		delete(annotations, corev1.LastAppliedConfigAnnotation)
	}
	accessor.SetAnnotations(annotations)
	return nil
}

func buildLastApplied(obj runtime.Object) (string, error) {
	obj = obj.DeepCopyObject()

	var accessor, err = meta.Accessor(obj)
	if err != nil {
		panic(fmt.Sprintf("couldn't get accessor: %v", err))
	}

	// Remove the annotation from the object before encoding the object
	var annotations = accessor.GetAnnotations()
	delete(annotations, corev1.LastAppliedConfigAnnotation)
	accessor.SetAnnotations(annotations)

	lastApplied, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj)
	if err != nil {
		return "", fmt.Errorf("couldn't encode object into last applied annotation: %v", err)
	}
	return string(lastApplied), nil
}

func isAnnotationsValid(annotations map[string]string) error {
	var totalSize int64
	for k, v := range annotations {
		totalSize += (int64)(len(k)) + (int64)(len(v))
	}
	if totalSize > (int64)(totalAnnotationSizeLimitB) {
		return fmt.Errorf("annotations size %d is larger than limit %d", totalSize, totalAnnotationSizeLimitB)
	}
	return nil
}
