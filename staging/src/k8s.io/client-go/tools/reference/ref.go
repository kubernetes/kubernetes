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

package reference

import (
	"errors"
	"fmt"
	"net/url"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
)

var (
	// Errors that could be returned by GetReference.
	ErrNilObject  = errors.New("can't reference a nil object")
	ErrNoSelfLink = errors.New("selfLink was empty, can't make reference")
)

// GetReference returns an ObjectReference which refers to the given
// object, or an error if the object doesn't follow the conventions
// that would allow this.
// TODO: should take a meta.Interface see http://issue.k8s.io/7127
func GetReference(scheme *runtime.Scheme, obj runtime.Object) (*v1.ObjectReference, error) {
	if obj == nil {
		return nil, ErrNilObject
	}
	if ref, ok := obj.(*v1.ObjectReference); ok {
		// Don't make a reference to a reference.
		return ref, nil
	}

	gvk := obj.GetObjectKind().GroupVersionKind()

	// if the object referenced is actually persisted, we can just get kind from meta
	// if we are building an object reference to something not yet persisted, we should fallback to scheme
	kind := gvk.Kind
	if len(kind) == 0 {
		// TODO: this is wrong
		gvks, _, err := scheme.ObjectKinds(obj)
		if err != nil {
			return nil, err
		}
		kind = gvks[0].Kind
	}

	// An object that implements only List has enough metadata to build a reference
	var listMeta meta.List
	objectMeta, err := meta.Accessor(obj)
	if err != nil {
		listMeta, err = meta.ListAccessor(obj)
		if err != nil {
			return nil, err
		}
	} else {
		listMeta = objectMeta
	}

	// if the object referenced is actually persisted, we can also get version from meta
	version := gvk.GroupVersion().String()
	if len(version) == 0 {
		selfLink := listMeta.GetSelfLink()
		if len(selfLink) == 0 {
			return nil, ErrNoSelfLink
		}
		selfLinkUrl, err := url.Parse(selfLink)
		if err != nil {
			return nil, err
		}
		// example paths: /<prefix>/<version>/*
		parts := strings.Split(selfLinkUrl.Path, "/")
		if len(parts) < 3 {
			return nil, fmt.Errorf("unexpected self link format: '%v'; got version '%v'", selfLink, version)
		}
		version = parts[2]
	}

	// only has list metadata
	if objectMeta == nil {
		return &v1.ObjectReference{
			Kind:            kind,
			APIVersion:      version,
			ResourceVersion: listMeta.GetResourceVersion(),
		}, nil
	}

	return &v1.ObjectReference{
		Kind:            kind,
		APIVersion:      version,
		Name:            objectMeta.GetName(),
		Namespace:       objectMeta.GetNamespace(),
		UID:             objectMeta.GetUID(),
		ResourceVersion: objectMeta.GetResourceVersion(),
	}, nil
}

// GetPartialReference is exactly like GetReference, but allows you to set the FieldPath.
func GetPartialReference(scheme *runtime.Scheme, obj runtime.Object, fieldPath string) (*v1.ObjectReference, error) {
	ref, err := GetReference(scheme, obj)
	if err != nil {
		return nil, err
	}
	ref.FieldPath = fieldPath
	return ref, nil
}
