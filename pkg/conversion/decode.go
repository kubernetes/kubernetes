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
	version, kind, err := s.DataVersionAndKind(data)
	if err != nil {
		return nil, "", "", err
	}

	gv, err := unversioned.ParseGroupVersion(version)
	if err != nil {
		return nil, "", "", err
	}

	internalGV, exists := s.InternalVersions[gv.Group]
	if !exists {
		return nil, "", "", fmt.Errorf("no internalVersion specified for %v", gv)
	}

	if len(gv.Version) == 0 && len(internalGV.Version) != 0 {
		return nil, "", "", fmt.Errorf("version not set in '%s'", string(data))
	}
	if kind == "" {
		return nil, "", "", fmt.Errorf("kind not set in '%s'", string(data))
	}
	obj, err := s.NewObject(version, kind)
	if err != nil {
		return nil, "", "", err
	}

	if err := codec.NewDecoderBytes(data, new(codec.JsonHandle)).Decode(obj); err != nil {
		return nil, "", "", err
	}
	return obj, version, kind, nil
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
		flags, meta := s.generateConvertMeta(sourceVersion, gv.String(), obj)
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
	dataVersion, dataKind, err := s.DataVersionAndKind(data)
	if err != nil {
		return err
	}
	if dataVersion == "" {
		dataVersion = requestedGVK.GroupVersion().String()
	}
	if dataKind == "" {
		dataKind = requestedGVK.Kind
	}
	if (len(requestedGVK.Group) > 0 || len(requestedGVK.Version) > 0) && (dataVersion != requestedGVK.GroupVersion().String()) {
		return errors.New(fmt.Sprintf("The apiVersion in the data (%s) does not match the specified apiVersion(%v)", dataVersion, requestedGVK.GroupVersion()))
	}
	if len(requestedGVK.Kind) > 0 && (dataKind != requestedGVK.Kind) {
		return errors.New(fmt.Sprintf("The kind in the data (%s) does not match the specified kind(%v)", dataKind, requestedGVK))
	}

	objVersion, objKind, err := s.ObjectVersionAndKind(obj)
	if err != nil {
		return err
	}
	if dataKind == "" {
		// Assume objects with unset Kind fields are being unmarshalled into the
		// correct type.
		dataKind = objKind
	}
	if dataVersion == "" {
		// Assume objects with unset Version fields are being unmarshalled into the
		// correct type.
		dataVersion = objVersion
	}

	external, err := s.NewObject(dataVersion, dataKind)
	if err != nil {
		return err
	}
	if err := codec.NewDecoderBytes(data, new(codec.JsonHandle)).Decode(external); err != nil {
		return err
	}
	flags, meta := s.generateConvertMeta(dataVersion, objVersion, external)
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
