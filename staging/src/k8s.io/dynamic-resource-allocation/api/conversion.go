/*
Copyright 2024 The Kubernetes Authors.

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
	"strings"

	"github.com/blang/semver/v4"

	resourceapi "k8s.io/api/resource/v1"
	conversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
)

var (
	localSchemeBuilder runtime.SchemeBuilder
	AddToScheme        = localSchemeBuilder.AddToScheme
)

// resourceSliceScope is set up by Convert_v1_ResourceSlice_To_api_ResourceSlice and
// required by the other conversion functions for that direction. Converting
// the other structs separately is not supported, they have to be embedded
// inside a resourceapi.ResourceSlice.
type resourceSliceScope struct {
	conversion.Scope
	driverName UniqueString
	slice      *ResourceSlice
}

func Convert_v1_ResourceSlice_To_api_ResourceSlice(in *resourceapi.ResourceSlice, out *ResourceSlice, s conversion.Scope) error {
	out.uniqueStringMap = make(map[string]UniqueString)
	rs := &resourceSliceScope{
		Scope:      s,
		driverName: out.MakeUniqueString(in.Spec.Driver),
		slice:      out,
	}
	return autoConvert_v1_ResourceSlice_To_api_ResourceSlice(in, out, rs)
}

// Convert_api_ResourceSlice_To_v1_ResourceSlice converts api.ResourceSlice to v1.ResourceSlice.
// The internal uniqueStringMap cache does not need to be converted.
func Convert_api_ResourceSlice_To_v1_ResourceSlice(in *ResourceSlice, out *resourceapi.ResourceSlice, s conversion.Scope) error {
	return autoConvert_api_ResourceSlice_To_v1_ResourceSlice(in, out, s)
}

func Convert_api_UniqueString_To_string(in *UniqueString, out *string, s conversion.Scope) error {
	if *in == NullUniqueString {
		*out = ""
		return nil
	}
	*out = in.String()
	return nil
}

func Convert_string_To_api_UniqueString(in *string, out *UniqueString, s conversion.Scope) error {
	rs, ok := s.(*resourceSliceScope)
	if !ok {
		return errors.New("Convert_string_To_api_UniqueString may only be called when converting a ResourceSlice")
	}
	if *in == "" {
		*out = NullUniqueString
		return nil
	}
	*out = rs.slice.MakeUniqueString(*in)
	return nil
}
func Convert_Map_v1_QualifiedName_To_v1_DeviceAttribute_To_api_DeviceAttributes(in *map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, out *DeviceAttributes, s conversion.Scope) error {
	rs, ok := s.(*resourceSliceScope)
	if !ok {
		return errors.New("Convert_Map_v1_QualifiedName_To_v1_DeviceAttribute_To_api_DeviceAttributes may only be called when converting a ResourceSlice")
	}

	return Convert_v1_To_api_DeviceAttributes(in, out, rs)
}

func Convert_v1_To_api_DeviceAttributes(in *map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, out *DeviceAttributes, rs *resourceSliceScope) error {
	if *in == nil {
		*out = DeviceAttributes{}
		return nil
	}

	// Let's assume that each driver uses at most its own domain and the standard resource.k8s.io.
	m := DeviceAttributes{
		Nested:     make(map[UniqueString]map[UniqueString]any, 2),
		DriverName: rs.driverName,
	}

	for k, v := range *in {
		var domain, name UniqueString
		sep := strings.Index(string(k), "/")
		if sep < 0 {
			// Plain name, uses driver name as domain.
			domain = rs.driverName
			name = rs.slice.MakeUniqueString(string(k))
			// A fully-qualified entry for the same name, if already recorded,
			// wins (shouldn't happen unless a DRA driver made a mistake).
			if _, exists := m.Nested[domain][name]; exists {
				continue
			}
		} else {
			// Fully qualified string, contains both domain and name.
			domain = rs.slice.MakeUniqueString(string(k[:sep]))
			name = rs.slice.MakeUniqueString(string(k[sep+1:]))
			if domain == rs.driverName {
				if m.DriverNameQualifiedIDs == nil {
					m.DriverNameQualifiedIDs = sets.New[UniqueString]()
				}
				m.DriverNameQualifiedIDs.Insert(name)
			}
		}

		inner := m.Nested[domain]
		if inner == nil {
			inner = make(map[UniqueString]any)
		}
		// Resolve one-of to the right value instead of storing the full struct.
		var attrValue any
		switch {
		case v.IntValue != nil:
			attrValue = *v.IntValue
		case v.BoolValue != nil:
			attrValue = *v.BoolValue
		case v.StringValue != nil:
			attrValue = *v.StringValue
		case v.VersionValue != nil:
			ver, err := semver.New(*v.VersionValue)
			if err != nil {
				// Should not happen, input must be valid.
				return err
			}
			attrValue = *ver
		case len(v.IntValues) > 0:
			attrValue = v.IntValues
		case len(v.BoolValues) > 0:
			attrValue = v.BoolValues
		case len(v.StringValues) > 0:
			attrValue = v.StringValues
		case len(v.VersionValues) > 0:
			versions := make([]semver.Version, len(v.VersionValues))
			for i, verStr := range v.VersionValues {
				ver, err := semver.New(verStr)
				if err != nil {
					// Should not happen, input must be valid.
					return err
				}
				versions[i] = *ver
			}
			attrValue = versions
		}
		inner[name] = attrValue
		m.Nested[domain] = inner
	}

	*out = m
	return nil
}

func Convert_api_DeviceAttributes_To_Map_v1_QualifiedName_To_v1_DeviceAttribute(in *DeviceAttributes, out *map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, s conversion.Scope) error {
	if in.Nested == nil {
		*out = nil
		return nil
	}
	m := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
	for domain, inner := range in.Nested {
		for id, attrValue := range inner {
			var attr resourceapi.DeviceAttribute
			switch attrValue := attrValue.(type) {
			case int64:
				attr.IntValue = &attrValue
			case bool:
				attr.BoolValue = &attrValue
			case string:
				attr.StringValue = &attrValue
			case semver.Version:
				attr.VersionValue = new(attrValue.String())
			case []int64:
				attr.IntValues = attrValue
			case []bool:
				attr.BoolValues = attrValue
			case []string:
				attr.StringValues = attrValue
			case []semver.Version:
				versions := make([]string, len(attrValue))
				for i, ver := range attrValue {
					versions[i] = ver.String()
				}
				attr.VersionValues = versions

			}
			if domain == in.DriverName && !in.DriverNameQualifiedIDs.Has(id) {
				m[resourceapi.QualifiedName(id.String())] = attr
			} else {
				m[resourceapi.QualifiedName(domain.String()+"/"+id.String())] = attr
			}
		}
	}
	*out = m
	return nil
}

func Convert_Map_v1_QualifiedName_To_v1_DeviceCapacity_To_api_DeviceCapacities(in *map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, out *DeviceCapacities, s conversion.Scope) error {
	rs, ok := s.(*resourceSliceScope)
	if !ok {
		return errors.New("Convert_Map_v1_QualifiedName_To_v1_DeviceCapacity_To_api_DeviceCapacities may only be called when converting a ResourceSlice")
	}
	return Convert_v1_To_api_DeviceCapacities(in, out, rs)
}

func Convert_v1_To_api_DeviceCapacities(in *map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, out *DeviceCapacities, rs *resourceSliceScope) error {
	if *in == nil {
		*out = DeviceCapacities{}
		return nil
	}

	// Same logic as in Convert_v1_To_api_DeviceAttributes above...
	m := DeviceCapacities{
		Nested:     make(map[UniqueString]map[UniqueString]*DeviceCapacity, 2),
		DriverName: rs.driverName,
	}
	for k, v := range *in {
		var domain, name UniqueString
		sep := strings.Index(string(k), "/")
		if sep < 0 {
			// Plain name, uses driver name as domain.
			domain = rs.driverName
			name = rs.slice.MakeUniqueString(string(k))
			// A fully-qualified entry for the same name, if already recorded,
			// wins (shouldn't happen unless a DRA driver made a mistake).
			if _, exists := m.Nested[domain][name]; exists {
				continue
			}
		} else {
			domain = rs.slice.MakeUniqueString(string(k[:sep]))
			name = rs.slice.MakeUniqueString(string(k[sep+1:]))
			if domain == rs.driverName {
				if m.DriverNameQualifiedIDs == nil {
					m.DriverNameQualifiedIDs = sets.New[UniqueString]()
				}
				m.DriverNameQualifiedIDs.Insert(name)
			}
		}

		inner := m.Nested[domain]
		if inner == nil {
			inner = make(map[UniqueString]*DeviceCapacity)
		}
		valueCopy := v.Value.DeepCopy()
		inner[name] = &DeviceCapacity{
			Value:         valueCopy,
			RequestPolicy: v.RequestPolicy.DeepCopy(),
		}
		m.Nested[domain] = inner
	}
	*out = m
	return nil
}

func Convert_api_DeviceCapacities_To_Map_v1_QualifiedName_To_v1_DeviceCapacity(in *DeviceCapacities, out *map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, s conversion.Scope) error {
	if in.Nested == nil {
		*out = nil
		return nil
	}
	m := make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity)
	for domain, inner := range in.Nested {
		for id, cap := range inner {
			v := resourceapi.DeviceCapacity{
				Value:         cap.Value.DeepCopy(),
				RequestPolicy: cap.RequestPolicy.DeepCopy(),
			}
			if domain == in.DriverName && !in.DriverNameQualifiedIDs.Has(id) {
				m[resourceapi.QualifiedName(id.String())] = v
			} else {
				m[resourceapi.QualifiedName(domain.String()+"/"+id.String())] = v
			}
		}
	}
	*out = m
	return nil
}
