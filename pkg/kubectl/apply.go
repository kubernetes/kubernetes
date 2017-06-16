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

package kubectl

import (
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

// GetOriginalConfiguration retrieves the original configuration of the object
// from the annotation, or nil if no annotation was found.
func GetOriginalConfiguration(mapping *meta.RESTMapping, obj runtime.Object) ([]byte, error) {
	annots, err := mapping.MetadataAccessor.Annotations(obj)
	if err != nil {
		return nil, err
	}

	if annots == nil {
		return nil, nil
	}

	original, ok := annots[api.LastAppliedConfigAnnotation]
	if !ok {
		return nil, nil
	}

	return []byte(original), nil
}

// SetOriginalConfiguration sets the original configuration of the object
// as the annotation on the object for later use in computing a three way patch.
func SetOriginalConfiguration(info *resource.Info, original []byte) error {
	if len(original) < 1 {
		return nil
	}

	accessor := info.Mapping.MetadataAccessor
	annots, err := accessor.Annotations(info.Object)
	if err != nil {
		return err
	}

	if annots == nil {
		annots = map[string]string{}
	}

	annots[api.LastAppliedConfigAnnotation] = string(original)
	if err := info.Mapping.MetadataAccessor.SetAnnotations(info.Object, annots); err != nil {
		return err
	}

	return nil
}

// GetModifiedConfiguration retrieves the modified configuration of the object.
// If annotate is true, it embeds the result as an annotation in the modified
// configuration. If an object was read from the command input, it will use that
// version of the object. Otherwise, it will use the version from the server.
func GetModifiedConfiguration(info *resource.Info, annotate bool, codec runtime.Encoder) ([]byte, error) {
	// First serialize the object without the annotation to prevent recursion,
	// then add that serialization to it as the annotation and serialize it again.
	var modified []byte
	if info.VersionedObject != nil {
		// If an object was read from input, use that version.
		accessor, err := meta.Accessor(info.VersionedObject)
		if err != nil {
			return nil, err
		}

		// Get the current annotations from the object.
		annots := accessor.GetAnnotations()
		if annots == nil {
			annots = map[string]string{}
		}

		original := annots[api.LastAppliedConfigAnnotation]
		delete(annots, api.LastAppliedConfigAnnotation)
		accessor.SetAnnotations(annots)
		// TODO: this needs to be abstracted - there should be no assumption that versioned object
		// can be marshalled to JSON.
		modified, err = runtime.Encode(codec, info.VersionedObject)
		if err != nil {
			return nil, err
		}

		if annotate {
			annots[api.LastAppliedConfigAnnotation] = string(modified)
			accessor.SetAnnotations(annots)
			// TODO: this needs to be abstracted - there should be no assumption that versioned object
			// can be marshalled to JSON.
			modified, err = runtime.Encode(codec, info.VersionedObject)
			if err != nil {
				return nil, err
			}
		}

		// Restore the object to its original condition.
		annots[api.LastAppliedConfigAnnotation] = original
		accessor.SetAnnotations(annots)
	} else {
		// Otherwise, use the server side version of the object.
		accessor := info.Mapping.MetadataAccessor
		// Get the current annotations from the object.
		annots, err := accessor.Annotations(info.Object)
		if err != nil {
			return nil, err
		}

		if annots == nil {
			annots = map[string]string{}
		}

		original := annots[api.LastAppliedConfigAnnotation]
		delete(annots, api.LastAppliedConfigAnnotation)
		if err := accessor.SetAnnotations(info.Object, annots); err != nil {
			return nil, err
		}

		modified, err = runtime.Encode(codec, info.Object)
		if err != nil {
			return nil, err
		}

		if annotate {
			annots[api.LastAppliedConfigAnnotation] = string(modified)
			if err := info.Mapping.MetadataAccessor.SetAnnotations(info.Object, annots); err != nil {
				return nil, err
			}

			modified, err = runtime.Encode(codec, info.Object)
			if err != nil {
				return nil, err
			}
		}

		// Restore the object to its original condition.
		annots[api.LastAppliedConfigAnnotation] = original
		if err := info.Mapping.MetadataAccessor.SetAnnotations(info.Object, annots); err != nil {
			return nil, err
		}
	}

	return modified, nil
}

// UpdateApplyAnnotation calls CreateApplyAnnotation if the last applied
// configuration annotation is already present. Otherwise, it does nothing.
func UpdateApplyAnnotation(info *resource.Info, codec runtime.Encoder) error {
	if original, err := GetOriginalConfiguration(info.Mapping, info.Object); err != nil || len(original) <= 0 {
		return err
	}
	return CreateApplyAnnotation(info, codec)
}

// CreateApplyAnnotation gets the modified configuration of the object,
// without embedding it again, and then sets it on the object as the annotation.
func CreateApplyAnnotation(info *resource.Info, codec runtime.Encoder) error {
	modified, err := GetModifiedConfiguration(info, false, codec)
	if err != nil {
		return err
	}
	return SetOriginalConfiguration(info, modified)
}

// Create the annotation used by kubectl apply only when createAnnotation is true
// Otherwise, only update the annotation when it already exists
func CreateOrUpdateAnnotation(createAnnotation bool, info *resource.Info, codec runtime.Encoder) error {
	if createAnnotation {
		return CreateApplyAnnotation(info, codec)
	}
	return UpdateApplyAnnotation(info, codec)
}
