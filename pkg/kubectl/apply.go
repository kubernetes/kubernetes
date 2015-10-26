/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"encoding/json"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

type debugError interface {
	DebugError() (msg string, args []interface{})
}

// LastAppliedConfigAnnotation is the annotation used to store the previous
// configuration of a resource for use in a three way diff by UpdateApplyAnnotation.
const LastAppliedConfigAnnotation = kubectlAnnotationPrefix + "last-applied-configuration"

// GetOriginalConfiguration retrieves the original configuration of the object
// from the annotation, or nil if no annotation was found.
func GetOriginalConfiguration(info *resource.Info) ([]byte, error) {
	annotations, err := info.Mapping.MetadataAccessor.Annotations(info.Object)
	if err != nil {
		return nil, err
	}

	if annotations == nil {
		return nil, nil
	}

	original, ok := annotations[LastAppliedConfigAnnotation]
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
	annotations, err := accessor.Annotations(info.Object)
	if err != nil {
		return err
	}

	if annotations == nil {
		annotations = map[string]string{}
	}

	annotations[LastAppliedConfigAnnotation] = string(original)
	if err := info.Mapping.MetadataAccessor.SetAnnotations(info.Object, annotations); err != nil {
		return err
	}

	return nil
}

// GetModifiedConfiguration retrieves the modified configuration of the object.
// If annotate is true, it embeds the result as an anotation in the modified
// configuration. If an object was read from the command input, it will use that
// version of the object. Otherwise, it will use the version from the server.
func GetModifiedConfiguration(info *resource.Info, annotate bool) ([]byte, error) {
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
		annotations := accessor.Annotations()
		if annotations == nil {
			annotations = map[string]string{}
		}

		original := annotations[LastAppliedConfigAnnotation]
		delete(annotations, LastAppliedConfigAnnotation)
		accessor.SetAnnotations(annotations)
		modified, err = json.Marshal(info.VersionedObject)
		if err != nil {
			return nil, err
		}

		if annotate {
			annotations[LastAppliedConfigAnnotation] = string(modified)
			accessor.SetAnnotations(annotations)
			modified, err = json.Marshal(info.VersionedObject)
			if err != nil {
				return nil, err
			}
		}

		// Restore the object to its original condition.
		annotations[LastAppliedConfigAnnotation] = original
		accessor.SetAnnotations(annotations)
	} else {
		// Otherwise, use the server side version of the object.
		accessor := info.Mapping.MetadataAccessor
		// Get the current annotations from the object.
		annotations, err := accessor.Annotations(info.Object)
		if err != nil {
			return nil, err
		}

		if annotations == nil {
			annotations = map[string]string{}
		}

		original := annotations[LastAppliedConfigAnnotation]
		delete(annotations, LastAppliedConfigAnnotation)
		if err := accessor.SetAnnotations(info.Object, annotations); err != nil {
			return nil, err
		}

		modified, err = info.Mapping.Codec.Encode(info.Object)
		if err != nil {
			return nil, err
		}

		if annotate {
			annotations[LastAppliedConfigAnnotation] = string(modified)
			if err := info.Mapping.MetadataAccessor.SetAnnotations(info.Object, annotations); err != nil {
				return nil, err
			}

			modified, err = info.Mapping.Codec.Encode(info.Object)
			if err != nil {
				return nil, err
			}
		}

		// Restore the object to its original condition.
		annotations[LastAppliedConfigAnnotation] = original
		if err := info.Mapping.MetadataAccessor.SetAnnotations(info.Object, annotations); err != nil {
			return nil, err
		}
	}

	return modified, nil
}

// UpdateApplyAnnotation gets the modified configuration of the object,
// without embedding it again, and then sets it on the object as the annotation.
func UpdateApplyAnnotation(info *resource.Info) error {
	modified, err := GetModifiedConfiguration(info, false)
	if err != nil {
		return err
	}

	if err := SetOriginalConfiguration(info, modified); err != nil {
		return err
	}

	return nil
}
