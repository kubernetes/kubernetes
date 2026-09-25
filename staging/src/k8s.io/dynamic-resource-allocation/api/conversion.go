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
	"k8s.io/apimachinery/pkg/api/resource"
	conversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	apiservercel "k8s.io/apiserver/pkg/cel"
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

	// First convert fully-qualified entries ("fully-qualified wins" in case of conflicting entries with and without the driver name).
	for k, v := range *in {
		sep := strings.Index(string(k), "/")
		if sep < 0 {
			// Handled below.
			continue
		}
		// Fully qualified string, contains both domain and name.
		domain := rs.slice.MakeUniqueString(string(k[:sep]))
		name := rs.slice.MakeUniqueString(string(k[sep+1:]))

		if domain == rs.driverName {
			if m.DriverNameQualifiedIDs == nil {
				m.DriverNameQualifiedIDs = sets.New[UniqueString]()
			}
			m.DriverNameQualifiedIDs.Insert(name)
		}
		if err := addAttribute(&m, domain, name, v); err != nil {
			return err
		}
	}

	// Now unqualified entries.
	for k, v := range *in {
		sep := strings.Index(string(k), "/")
		if sep >= 0 {
			// Handled above.
			continue
		}
		// Plain name, uses driver name as domain.
		domain := rs.driverName
		name := rs.slice.MakeUniqueString(string(k))
		if m.DriverNameQualifiedIDs.Has(name) {
			// A conflicting entry! Ignore it. Shouldn't happen unless a DRA driver made a mistake.
			continue
		}
		if err := addAttribute(&m, domain, name, v); err != nil {
			return err
		}
	}

	*out = m
	return nil
}

func addAttribute(m *DeviceAttributes, domain, id UniqueString, v resourceapi.DeviceAttribute) error {
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
		attrValue = apiservercel.Semver{Version: *ver}
	case len(v.IntValues) > 0:
		attrValue = v.IntValues
	case len(v.BoolValues) > 0:
		attrValue = v.BoolValues
	case len(v.StringValues) > 0:
		attrValue = v.StringValues
	case len(v.VersionValues) > 0:
		versions := make([]apiservercel.Semver, len(v.VersionValues))
		for i, verStr := range v.VersionValues {
			ver, err := semver.New(verStr)
			if err != nil {
				// Should not happen, input must be valid.
				return err
			}
			versions[i].Version = *ver
		}
		attrValue = versions
	}
	inner[id] = attrValue
	m.Nested[domain] = inner
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
			case apiservercel.Semver:
				attr.VersionValue = new(attrValue.Version.String())
			case []int64:
				attr.IntValues = attrValue
			case []bool:
				attr.BoolValues = attrValue
			case []string:
				attr.StringValues = attrValue
			case []apiservercel.Semver:
				versions := make([]string, len(attrValue))
				for i, ver := range attrValue {
					versions[i] = ver.Version.String()
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
		Nested:     make(map[UniqueString]map[UniqueString]DeviceCapacity, 2),
		DriverName: rs.driverName,
	}
	for k, v := range *in {
		sep := strings.Index(string(k), "/")
		if sep < 0 {
			continue
		}
		domain := rs.slice.MakeUniqueString(string(k[:sep]))
		name := rs.slice.MakeUniqueString(string(k[sep+1:]))

		if domain == rs.driverName {
			if m.DriverNameQualifiedIDs == nil {
				m.DriverNameQualifiedIDs = sets.New[UniqueString]()
			}
			m.DriverNameQualifiedIDs.Insert(name)
		}
		addCapacity(&m, domain, name, v)
	}
	for k, v := range *in {
		sep := strings.Index(string(k), "/")
		if sep >= 0 {
			continue
		}
		domain := rs.driverName
		name := rs.slice.MakeUniqueString(string(k))
		if m.DriverNameQualifiedIDs.Has(name) {
			continue
		}
		addCapacity(&m, domain, name, v)
	}
	*out = m
	return nil
}

func addCapacity(m *DeviceCapacities, domain, id UniqueString, v resourceapi.DeviceCapacity) {
	inner := m.Nested[domain]
	if inner == nil {
		inner = make(map[UniqueString]DeviceCapacity)
	}
	valueCopy := v.Value.DeepCopy()
	inner[id] = DeviceCapacity{
		Value:         apiservercel.Quantity{Quantity: &valueCopy},
		RequestPolicy: v.RequestPolicy.DeepCopy(),
	}
	m.Nested[domain] = inner
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

// Convert_cel_Quantity_To_resource_Quantity converts from an apiservercel.Quantity to resource.Quantity.
func Convert_cel_Quantity_To_resource_Quantity(in *apiservercel.Quantity, out *resource.Quantity, s conversion.Scope) error {
	if in.Quantity == nil {
		*out = resource.Quantity{}
		return nil
	}
	*out = in.DeepCopy()
	return nil
}

// Convert_resource_Quantity_To_cel_Quantity converts from a resource.Quantity to apiservercel.Quantity.
func Convert_resource_Quantity_To_cel_Quantity(in *resource.Quantity, out *apiservercel.Quantity, s conversion.Scope) error {
	q := in.DeepCopy()
	out.Quantity = &q
	return nil
}
