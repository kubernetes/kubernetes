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

package handlers

import (
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/merge"
	"sigs.k8s.io/structured-merge-diff/typed"
)

type applyPatcher struct {
	*patcher

	parser  *gvkParser
	updater merge.Updater
}

const applyManager = "apply"

func (p *applyPatcher) applyPatchToCurrentObject(currentObject runtime.Object) (runtime.Object, error) {
	gvk := schema.GroupVersionKind{
		Group:   p.hubGroupVersion.Group,
		Version: p.hubGroupVersion.Version,
		Kind:    currentObject.GetObjectKind().GroupVersionKind().Kind,
	}
	pType := p.parser.Type(gvk)
	if pType == nil {
		return nil, fmt.Errorf("unable to find schema for type: %v", gvk)
	}
	vo, err := p.unsafeConvertor.ConvertToVersion(currentObject, p.hubGroupVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to convert object to version: %v", err)
	}
	current, err := runtime.DefaultUnstructuredConverter.ToUnstructured(vo)
	if err != nil {
		return nil, fmt.Errorf("failed to convert to unstructured: %v", err)
	}

	accessor, err := meta.Accessor(current)
	if err != nil {
		return nil, fmt.Errorf("couldn't get accessor: %v", err)
	}
	managedFields := accessor.GetManagedFields()
	accessor.SetManagedFields(nil)

	currentTyped, err := pType.FromUnstructured(current)
	if err != nil {
		return nil, fmt.Errorf("failed to create typed current object: %v", err)
	}

	newTyped, err := pType.FromYAML(typed.YAMLObject(p.patchBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to convert patch to typed object: %v", err)
	}

	// XXX: This needs to be converted from managedFields
	managed := fieldpath.ManagedFields{}

	// XXX: We don't have a force-flag yet, hence hard-coded to false.
	outputTyped, managed, err := p.updater.Apply(currentTyped, newTyped, fieldpath.APIVersion(p.hubGroupVersion.String()), managed, applyManager, false)
	if err != nil {
		return nil, err
	}

	output, ok := outputTyped.AsValue().ToUnstructured(true).(map[string]interface{})
	if !ok {
		return nil, errors.New("Unable to convert typed unstructured to object unstructured")
	}
	newObj := unstructured.Unstructured{Object: output}

	accessor, err = meta.Accessor(newObj)
	if err != nil {
		return nil, fmt.Errorf("couldn't get accessor on output: %v", err)
	}
	// XXX: This needs to be converted from managed
	accessor.SetManagedFields(managedFields)

	return &newObj, nil
}

func (p *applyPatcher) createNewObject() (runtime.Object, error) {
	return p.applyPatchToCurrentObject(p.restPatcher.New())
}

type failConverter struct{}

func (failConverter) Convert(object typed.TypedValue, version fieldpath.APIVersion) (typed.TypedValue, error) {
	if version != "" {
		return typed.TypedValue{}, fmt.Errorf("Converter can only handle empty version (received %v)", version)
	}
	return object, nil
}
