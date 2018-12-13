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
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/endpoints/handlers/apply"
	"sigs.k8s.io/structured-merge-diff/merge"
)

type applyPatcher struct {
	*patcher

	converter apply.TypeConverter
	updater   merge.Updater
}

const applyManager = "apply"

func (p *applyPatcher) applyPatchToCurrentObject(currentObject runtime.Object) (runtime.Object, error) {
	currentObject, err := p.unsafeConvertor.ConvertToVersion(currentObject, p.kind.GroupVersion())
	if err != nil {
		return nil, err
	}

	managed, err := apply.DecodeObjectManagedFields(currentObject)
	if err != nil {
		return nil, err
	}

	currentTyped, err := p.converter.ObjectToTyped(currentObject)
	if err != nil {
		return nil, fmt.Errorf("failed to create typed current object: %v", err)
	}

	newTyped, err := p.converter.YAMLToTyped(p.patchBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to convert patch to typed object: %v", err)
	}

	force := false
	if p.options.Force != nil {
		force = *p.options.Force
	}

	output, managed, err := p.updater.Apply(currentTyped, newTyped, managed, applyManager, force)
	if err != nil {
		return nil, err
	}

	newObject, err := p.converter.TypedToObject(output)
	if err != nil {
		return nil, fmt.Errorf("failed to convert output typed to object: %v", err)
	}

	if err := apply.EncodeObjectManagedFields(newObject, managed); err != nil {
		return nil, err
	}

	newObject, err = p.unsafeConvertor.ConvertToVersion(newObject, p.kind.GroupVersion())
	if err != nil {
		return nil, err
	}
	p.defaulter.Default(newObject)

	gvk := p.kind.GroupKind().WithVersion(runtime.APIVersionInternal)
	outputObj, err := p.unsafeConvertor.ConvertToVersion(newObject, gvk.GroupVersion())
	if err != nil {
		return nil, fmt.Errorf("failed to convert to unversioned: %v", err)
	}
	return outputObj, nil
}

func (p *applyPatcher) createNewObject() (runtime.Object, error) {
	obj, err := p.creater.New(p.kind)
	if err != nil {
		return nil, fmt.Errorf("failed to create new object (kind = %v): %v", p.kind, err)
	}
	return p.applyPatchToCurrentObject(obj)
}
