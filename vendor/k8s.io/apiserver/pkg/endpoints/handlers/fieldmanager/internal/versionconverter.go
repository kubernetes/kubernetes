/*
Copyright 2018 The Kubernetes Authors.

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

package internal

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/structured-merge-diff/fieldpath"
	"sigs.k8s.io/structured-merge-diff/merge"
	"sigs.k8s.io/structured-merge-diff/typed"
)

// versionConverter is an implementation of
// sigs.k8s.io/structured-merge-diff/merge.Converter
type versionConverter struct {
	typeConverter   TypeConverter
	objectConvertor runtime.ObjectConvertor
	hubGetter       func(from schema.GroupVersion) schema.GroupVersion
}

var _ merge.Converter = &versionConverter{}

// NewVersionConverter builds a VersionConverter from a TypeConverter and an ObjectConvertor.
func NewVersionConverter(t TypeConverter, o runtime.ObjectConvertor, h schema.GroupVersion) merge.Converter {
	return &versionConverter{
		typeConverter:   t,
		objectConvertor: o,
		hubGetter: func(from schema.GroupVersion) schema.GroupVersion {
			return schema.GroupVersion{
				Group:   from.Group,
				Version: h.Version,
			}
		},
	}
}

// NewCRDVersionConverter builds a VersionConverter for CRDs from a TypeConverter and an ObjectConvertor.
func NewCRDVersionConverter(t TypeConverter, o runtime.ObjectConvertor, h schema.GroupVersion) merge.Converter {
	return &versionConverter{
		typeConverter:   t,
		objectConvertor: o,
		hubGetter: func(from schema.GroupVersion) schema.GroupVersion {
			return h
		},
	}
}

// Convert implements sigs.k8s.io/structured-merge-diff/merge.Converter
func (v *versionConverter) Convert(object *typed.TypedValue, version fieldpath.APIVersion) (*typed.TypedValue, error) {
	// Convert the smd typed value to a kubernetes object.
	objectToConvert, err := v.typeConverter.TypedToObject(object)
	if err != nil {
		return object, err
	}

	// Parse the target groupVersion.
	groupVersion, err := schema.ParseGroupVersion(string(version))
	if err != nil {
		return object, err
	}

	// If attempting to convert to the same version as we already have, just return it.
	fromVersion := objectToConvert.GetObjectKind().GroupVersionKind().GroupVersion()
	if fromVersion == groupVersion {
		return object, nil
	}

	// Convert to internal
	internalObject, err := v.objectConvertor.ConvertToVersion(objectToConvert, v.hubGetter(fromVersion))
	if err != nil {
		return object, err
	}

	// Convert the object into the target version
	convertedObject, err := v.objectConvertor.ConvertToVersion(internalObject, groupVersion)
	if err != nil {
		return object, err
	}

	// Convert the object back to a smd typed value and return it.
	return v.typeConverter.ObjectToTyped(convertedObject)
}

// IsMissingVersionError
func (v *versionConverter) IsMissingVersionError(err error) bool {
	return runtime.IsNotRegisteredError(err) || isNoCorrespondingTypeError(err)
}
