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

package conversion

import (
	"errors"
	"fmt"
	"net/url"

	"github.com/ugorji/go/codec"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

func (s *Scheme) DecodeToVersionedObject(data []byte) (interface{}, unversioned.GroupVersionKind, error) {
	kind, err := s.DataKind(data)
	if err != nil {
		return nil, unversioned.GroupVersionKind{}, err
	}

	internalGV, exists := s.InternalVersions[kind.Group]
	if !exists {
		return nil, unversioned.GroupVersionKind{}, fmt.Errorf("no internalVersion specified for %v", kind)
	}

	if len(kind.Group) == 0 && len(internalGV.Group) != 0 {
		return nil, unversioned.GroupVersionKind{}, fmt.Errorf("group not set in '%s'", string(data))
	}
	if len(kind.Version) == 0 && len(internalGV.Version) != 0 {
		return nil, unversioned.GroupVersionKind{}, fmt.Errorf("version not set in '%s'", string(data))
	}
	if kind.Kind == "" {
		return nil, unversioned.GroupVersionKind{}, fmt.Errorf("kind not set in '%s'", string(data))
	}

	obj, err := s.NewObject(kind)
	if err != nil {
		return nil, unversioned.GroupVersionKind{}, err
	}

	if err := codec.NewDecoderBytes(data, new(codec.JsonHandle)).Decode(obj); err != nil {
		return nil, unversioned.GroupVersionKind{}, err
	}
	return obj, kind, nil
}

// Decode converts a JSON string back into a pointer to an api object.
// Deduces the type based upon the fields added by the MetaInsertionFactory
// technique. The object will be converted, if necessary, into the
// s.InternalVersion type before being returned. Decode will not decode
// objects without version set unless InternalVersion is also "".
func (s *Scheme) Decode(data []byte) (interface{}, error) {
	return s.DecodeToVersion(data, unversioned.GroupVersion{})
}

// DecodeToVersion converts a JSON string back into a pointer to an api object.
// Deduces the type based upon the fields added by the MetaInsertionFactory
// technique. The object will be converted, if necessary, into the versioned
// type before being returned. Decode will not decode objects without version
// set unless version is also "".
// a GroupVersion with .IsEmpty() == true is means "use the internal version for
// the object's group"
func (s *Scheme) DecodeToVersion(data []byte, targetVersion unversioned.GroupVersion) (interface{}, error) {
	obj, sourceKind, err := s.DecodeToVersionedObject(data)
	if err != nil {
		return nil, err
	}
	// Version and Kind should be blank in memory.
	if err := s.SetVersionAndKind("", "", obj); err != nil {
		return nil, err
	}

	// if the targetVersion is empty, then we want the internal version, but the internal version varies by
	// group.  We can lookup the group now because we have knowledge of the group
	if targetVersion.IsEmpty() {
		exists := false
		targetVersion, exists = s.InternalVersions[sourceKind.Group]
		if !exists {
			return nil, fmt.Errorf("no internalVersion specified for %v", targetVersion)
		}
	}

	// Convert if needed.
	if targetVersion != sourceKind.GroupVersion() {
		objOut, err := s.NewObject(targetVersion.WithKind(sourceKind.Kind))
		if err != nil {
			return nil, err
		}
		flags, meta := s.generateConvertMeta(sourceKind.GroupVersion(), targetVersion, obj)
		if err := s.converter.Convert(obj, objOut, flags, meta); err != nil {
			return nil, err
		}
		obj = objOut
	}
	return obj, nil
}

// DecodeInto parses a JSON string and stores it in obj. Returns an error
// if data.Kind is set and doesn't match the type of obj. Obj should be a
// pointer to an api type.
// If obj's version doesn't match that in data, an attempt will be made to convert
// data into obj's version.
func (s *Scheme) DecodeInto(data []byte, obj interface{}) error {
	return s.DecodeIntoWithSpecifiedVersionKind(data, obj, unversioned.GroupVersionKind{})
}

// DecodeIntoWithSpecifiedVersionKind compares the passed in requestGroupVersionKind
// with data.Version and data.Kind, defaulting data.Version and
// data.Kind to the specified value if they are empty, or generating an error if
// data.Version and data.Kind are not empty and differ from the specified value.
// The function then implements the functionality of DecodeInto.
// If specifiedVersion and specifiedKind are empty, the function degenerates to
// DecodeInto.
func (s *Scheme) DecodeIntoWithSpecifiedVersionKind(data []byte, obj interface{}, requestedGVK unversioned.GroupVersionKind) error {
	if len(data) == 0 {
		return errors.New("empty input")
	}
	dataKind, err := s.DataKind(data)
	if err != nil {
		return err
	}
	if len(dataKind.Group) == 0 {
		dataKind.Group = requestedGVK.Group
	}
	if len(dataKind.Version) == 0 {
		dataKind.Version = requestedGVK.Version
	}
	if len(dataKind.Kind) == 0 {
		dataKind.Kind = requestedGVK.Kind
	}

	if len(requestedGVK.Group) > 0 && requestedGVK.Group != dataKind.Group {
		return errors.New(fmt.Sprintf("The fully qualified kind in the data (%v) does not match the specified apiVersion(%v)", dataKind, requestedGVK))
	}
	if len(requestedGVK.Version) > 0 && requestedGVK.Version != dataKind.Version {
		return errors.New(fmt.Sprintf("The fully qualified kind in the data (%v) does not match the specified apiVersion(%v)", dataKind, requestedGVK))
	}
	if len(requestedGVK.Kind) > 0 && requestedGVK.Kind != dataKind.Kind {
		return errors.New(fmt.Sprintf("The fully qualified kind in the data (%v) does not match the specified apiVersion(%v)", dataKind, requestedGVK))
	}

	objGVK, err := s.ObjectKind(obj)
	if err != nil {
		return err
	}
	// Assume objects with unset fields are being unmarshalled into the
	// correct type.
	if len(dataKind.Group) == 0 {
		dataKind.Group = objGVK.Group
	}
	if len(dataKind.Version) == 0 {
		dataKind.Version = objGVK.Version
	}
	if len(dataKind.Kind) == 0 {
		dataKind.Kind = objGVK.Kind
	}

	external, err := s.NewObject(dataKind)
	if err != nil {
		return err
	}
	if err := codec.NewDecoderBytes(data, new(codec.JsonHandle)).Decode(external); err != nil {
		return err
	}
	flags, meta := s.generateConvertMeta(dataKind.GroupVersion(), objGVK.GroupVersion(), external)
	if err := s.converter.Convert(external, obj, flags, meta); err != nil {
		return err
	}

	// Version and Kind should be blank in memory.
	return s.SetVersionAndKind("", "", obj)
}

func (s *Scheme) DecodeParametersInto(parameters url.Values, obj interface{}) error {
	if err := s.Convert(&parameters, obj); err != nil {
		return err
	}
	// TODO: Should we do any convertion here?
	return nil
}
