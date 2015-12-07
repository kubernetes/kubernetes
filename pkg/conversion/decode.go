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

func (s *Scheme) DecodeToVersionedObject(data []byte) (interface{}, string, string, error) {
	gvk, err := s.DataKind(data)
	if err != nil {
		return nil, "", "", err
	}

	internalGV, exists := s.InternalVersions[gvk.Group]
	if !exists {
		return nil, "", "", fmt.Errorf("no internalVersion specified for %v", gvk)
	}

	if len(gvk.Group) == 0 && len(internalGV.Group) != 0 {
		return nil, "", "", fmt.Errorf("group not set in '%s'", string(data))
	}
	if len(gvk.Version) == 0 && len(internalGV.Version) != 0 {
		return nil, "", "", fmt.Errorf("version not set in '%s'", string(data))
	}
	if gvk.Kind == "" {
		return nil, "", "", fmt.Errorf("kind not set in '%s'", string(data))
	}

	obj, err := s.NewObject(gvk.GroupVersion().String(), gvk.Kind)
	if err != nil {
		return nil, "", "", err
	}

	if err := codec.NewDecoderBytes(data, new(codec.JsonHandle)).Decode(obj); err != nil {
		return nil, "", "", err
	}
	return obj, gvk.GroupVersion().String(), gvk.Kind, nil
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
func (s *Scheme) DecodeToVersion(data []byte, gv unversioned.GroupVersion) (interface{}, error) {
	obj, sourceVersion, kind, err := s.DecodeToVersionedObject(data)
	if err != nil {
		return nil, err
	}
	// Version and Kind should be blank in memory.
	if err := s.SetVersionAndKind("", "", obj); err != nil {
		return nil, err
	}

	sourceGV, err := unversioned.ParseGroupVersion(sourceVersion)
	if err != nil {
		return nil, err
	}

	// if the gv is empty, then we want the internal version, but the internal version varies by
	// group.  We can lookup the group now because we have knowledge of the group
	if gv.IsEmpty() {
		exists := false
		gv, exists = s.InternalVersions[sourceGV.Group]
		if !exists {
			return nil, fmt.Errorf("no internalVersion specified for %v", gv)
		}
	}

	// Convert if needed.
	if gv != sourceGV {
		objOut, err := s.NewObject(gv.String(), kind)
		if err != nil {
			return nil, err
		}
		flags, meta := s.generateConvertMeta(sourceGV, gv, obj)
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
	dataGVK, err := s.DataKind(data)
	if err != nil {
		return err
	}
	if len(dataGVK.Group) == 0 {
		dataGVK.Group = requestedGVK.Group
	}
	if len(dataGVK.Version) == 0 {
		dataGVK.Version = requestedGVK.Version
	}
	if len(dataGVK.Kind) == 0 {
		dataGVK.Kind = requestedGVK.Kind
	}

	if len(requestedGVK.Group) > 0 && requestedGVK.Group != dataGVK.Group {
		return errors.New(fmt.Sprintf("The fully qualified kind in the data (%v) does not match the specified apiVersion(%v)", dataGVK, requestedGVK))
	}
	if len(requestedGVK.Version) > 0 && requestedGVK.Version != dataGVK.Version {
		return errors.New(fmt.Sprintf("The fully qualified kind in the data (%v) does not match the specified apiVersion(%v)", dataGVK, requestedGVK))
	}
	if len(requestedGVK.Kind) > 0 && requestedGVK.Kind != dataGVK.Kind {
		return errors.New(fmt.Sprintf("The fully qualified kind in the data (%v) does not match the specified apiVersion(%v)", dataGVK, requestedGVK))
	}

	objGVK, err := s.ObjectKind(obj)
	if err != nil {
		return err
	}
	// Assume objects with unset fields are being unmarshalled into the
	// correct type.
	if len(dataGVK.Group) == 0 {
		dataGVK.Group = objGVK.Group
	}
	if len(dataGVK.Version) == 0 {
		dataGVK.Version = objGVK.Version
	}
	if len(dataGVK.Kind) == 0 {
		dataGVK.Kind = objGVK.Kind
	}

	external, err := s.NewObject(dataGVK.GroupVersion().String(), dataGVK.Kind)
	if err != nil {
		return err
	}
	if err := codec.NewDecoderBytes(data, new(codec.JsonHandle)).Decode(external); err != nil {
		return err
	}
	flags, meta := s.generateConvertMeta(dataGVK.GroupVersion(), objGVK.GroupVersion(), external)
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
