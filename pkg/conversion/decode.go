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

package conversion

import (
	"errors"
	"fmt"

	"gopkg.in/v1/yaml"
)

// Decode converts a YAML or JSON string back into a pointer to an api object.
// Deduces the type based upon the fields added by the MetaInsertionFactory
// technique. The object will be converted, if necessary, into the
// s.InternalVersion type before being returned. Decode will not decode
// objects without version set unless InternalVersion is also "".
func (s *Scheme) Decode(data []byte) (interface{}, error) {
	version, kind, err := s.DataVersionAndKind(data)
	if err != nil {
		return nil, err
	}
	if version == "" && s.InternalVersion != "" {
		return nil, fmt.Errorf("version not set in '%s'", string(data))
	}
	obj, err := s.NewObject(version, kind)
	if err != nil {
		return nil, err
	}

	// yaml is a superset of json, so we use it to decode here. That way,
	// we understand both.
	err = yaml.Unmarshal(data, obj)
	if err != nil {
		return nil, err
	}

	// Version and Kind should be blank in memory.
	err = s.SetVersionAndKind("", "", obj)
	if err != nil {
		return nil, err
	}

	// Convert if needed.
	if s.InternalVersion != version {
		objOut, err := s.NewObject(s.InternalVersion, kind)
		if err != nil {
			return nil, err
		}
		err = s.converter.Convert(obj, objOut, 0, s.generateConvertMeta(version, s.InternalVersion))
		if err != nil {
			return nil, err
		}
		obj = objOut
	}
	return obj, nil
}

// DecodeInto parses a YAML or JSON string and stores it in obj. Returns an error
// if data.Kind is set and doesn't match the type of obj. Obj should be a
// pointer to an api type.
// If obj's version doesn't match that in data, an attempt will be made to convert
// data into obj's version.
func (s *Scheme) DecodeInto(data []byte, obj interface{}) error {
	if len(data) == 0 {
		// This is valid YAML, but it's a bad idea not to return an error
		// for an empty string-- that's almost certainly not what the caller
		// was expecting.
		return errors.New("empty input")
	}
	dataVersion, dataKind, err := s.DataVersionAndKind(data)
	if err != nil {
		return err
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
	if dataKind != objKind {
		return fmt.Errorf("data of kind '%v', obj of type '%v'", dataKind, objKind)
	}
	if dataVersion == "" {
		// Assume objects with unset Version fields are being unmarshalled into the
		// correct type.
		dataVersion = objVersion
	}

	if objVersion == dataVersion {
		// Easy case!
		err = yaml.Unmarshal(data, obj)
		if err != nil {
			return err
		}
	} else {
		external, err := s.NewObject(dataVersion, dataKind)
		if err != nil {
			return fmt.Errorf("Unable to create new object of type ('%s', '%s')", dataVersion, dataKind)
		}
		// yaml is a superset of json, so we use it to decode here. That way,
		// we understand both.
		err = yaml.Unmarshal(data, external)
		if err != nil {
			return err
		}
		err = s.converter.Convert(external, obj, 0, s.generateConvertMeta(dataVersion, objVersion))
		if err != nil {
			return err
		}
	}

	// Version and Kind should be blank in memory.
	return s.SetVersionAndKind("", "", obj)
}
