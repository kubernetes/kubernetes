/*
Copyright 2014 Google Inc. All rights reserved.

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

package api

import (
	"errors"
	"fmt"
	"regexp"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

var ErrNilObject = errors.New("Can't reference a nil object")

var versionFromSelfLink = regexp.MustCompile("/api/([^/]*)/")

// GetReference returns an ObjectReference which refers to the given
// object, or an error if the object doesn't follow the conventions
// that would allow this.
func GetReference(obj runtime.Object) (*ObjectReference, error) {
	if obj == nil {
		return nil, ErrNilObject
	}
	if ref, ok := obj.(*ObjectReference); ok {
		// Don't make a reference to a reference.
		return ref, nil
	}
	meta, err := meta.Accessor(obj)
	if err != nil {
		return nil, err
	}
	_, kind, err := Scheme.ObjectVersionAndKind(obj)
	if err != nil {
		return nil, err
	}
	version := versionFromSelfLink.FindStringSubmatch(meta.SelfLink())
	if len(version) < 2 {
		return nil, fmt.Errorf("unexpected self link format: '%v'; got version '%v'", meta.SelfLink(), version)
	}
	return &ObjectReference{
		Kind:            kind,
		APIVersion:      version[1],
		Name:            meta.Name(),
		Namespace:       meta.Namespace(),
		UID:             meta.UID(),
		ResourceVersion: meta.ResourceVersion(),
	}, nil
}

// GetPartialReference is exactly like GetReference, but allows you to set the FieldPath.
func GetPartialReference(obj runtime.Object, fieldPath string) (*ObjectReference, error) {
	ref, err := GetReference(obj)
	if err != nil {
		return nil, err
	}
	ref.FieldPath = fieldPath
	return ref, nil
}

// Allow clients to preemptively get a reference to an API object and pass it to places that
// intend only to get a reference to that object. This simplifies the event recording interface.
func (*ObjectReference) IsAnAPIObject() {}
