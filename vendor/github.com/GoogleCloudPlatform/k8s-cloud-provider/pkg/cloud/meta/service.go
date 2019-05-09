/*
Copyright 2018 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package meta

import (
	"errors"
	"fmt"
	"reflect"
	"sort"
)

// ServiceInfo defines the entry for a Service that code will be generated for.
type ServiceInfo struct {
	// Object is the Go name of the object type that the service deals
	// with. Example: "ForwardingRule".
	Object string
	// Service is the Go name of the service struct i.e. where the methods
	// are defined. Examples: "GlobalForwardingRules".
	Service string
	// Resource is the plural noun of the resource in the compute API URL (e.g.
	// "forwardingRules").
	Resource string
	// version if unspecified will be assumed to be VersionGA.
	version     Version
	keyType     KeyType
	serviceType reflect.Type

	additionalMethods   []string
	options             int
	aggregatedListField string
}

// Version returns the version of the Service, defaulting to GA if APIVersion
// is empty.
func (i *ServiceInfo) Version() Version {
	if i.version == "" {
		return VersionGA
	}
	return i.version
}

// VersionTitle returns the capitalized golang CamelCase name for the version.
func (i *ServiceInfo) VersionTitle() string {
	switch i.Version() {
	case VersionGA:
		return "GA"
	case VersionAlpha:
		return "Alpha"
	case VersionBeta:
		return "Beta"
	}
	panic(fmt.Errorf("invalid version %q", i.Version()))
}

// WrapType is the name of the wrapper service type.
func (i *ServiceInfo) WrapType() string {
	switch i.Version() {
	case VersionGA:
		return i.Service
	case VersionAlpha:
		return "Alpha" + i.Service
	case VersionBeta:
		return "Beta" + i.Service
	}
	return "Invalid"
}

// WrapTypeOps is the name of the additional operations type.
func (i *ServiceInfo) WrapTypeOps() string {
	return i.WrapType() + "Ops"
}

// FQObjectType is fully qualified name of the object (e.g. compute.Instance).
func (i *ServiceInfo) FQObjectType() string {
	return fmt.Sprintf("%v.%v", i.Version(), i.Object)
}

// ObjectListType is the compute List type for the object (contains Items field).
func (i *ServiceInfo) ObjectListType() string {
	return fmt.Sprintf("%v.%vList", i.Version(), i.Object)
}

// ObjectAggregatedListType is the compute List type for the object (contains Items field).
func (i *ServiceInfo) ObjectAggregatedListType() string {
	return fmt.Sprintf("%v.%vAggregatedList", i.Version(), i.Object)
}

// MockWrapType is the name of the concrete mock for this type.
func (i *ServiceInfo) MockWrapType() string {
	return "Mock" + i.WrapType()
}

// MockField is the name of the field in the mock struct.
func (i *ServiceInfo) MockField() string {
	return "Mock" + i.WrapType()
}

// GCEWrapType is the name of the GCE wrapper type.
func (i *ServiceInfo) GCEWrapType() string {
	return "GCE" + i.WrapType()
}

// Field is the name of the GCE struct.
func (i *ServiceInfo) Field() string {
	return "gce" + i.WrapType()
}

// Methods returns a list of additional methods to generate code for.
func (i *ServiceInfo) Methods() []*Method {
	methods := map[string]bool{}
	for _, m := range i.additionalMethods {
		methods[m] = true
	}

	var ret []*Method
	for j := 0; j < i.serviceType.NumMethod(); j++ {
		m := i.serviceType.Method(j)
		if _, ok := methods[m.Name]; !ok {
			continue
		}
		ret = append(ret, newMethod(i, m))
		methods[m.Name] = false
	}

	for k, b := range methods {
		if b {
			panic(fmt.Errorf("method %q was not found in service %q", k, i.Service))
		}
	}

	return ret
}

// KeyIsGlobal is true if the key is global.
func (i *ServiceInfo) KeyIsGlobal() bool {
	return i.keyType == Global
}

// KeyIsRegional is true if the key is regional.
func (i *ServiceInfo) KeyIsRegional() bool {
	return i.keyType == Regional
}

// KeyIsZonal is true if the key is zonal.
func (i *ServiceInfo) KeyIsZonal() bool {
	return i.keyType == Zonal
}

// KeyIsProject is true if the key represents the project resource.
func (i *ServiceInfo) KeyIsProject() bool {
	// Projects are a special resource for ResourceId because there is no 'key' value. This func
	// is used by the generator to not accept a key parameter.
	return i.Service == "Projects"
}

// MakeKey returns the call used to create the appropriate key type.
func (i *ServiceInfo) MakeKey(name, location string) string {
	switch i.keyType {
	case Global:
		return fmt.Sprintf("GlobalKey(%q)", name)
	case Regional:
		return fmt.Sprintf("RegionalKey(%q, %q)", name, location)
	case Zonal:
		return fmt.Sprintf("ZonalKey(%q, %q)", name, location)
	}
	return "Invalid"
}

// GenerateGet is true if the method is to be generated.
func (i *ServiceInfo) GenerateGet() bool {
	return i.options&NoGet == 0
}

// GenerateList is true if the method is to be generated.
func (i *ServiceInfo) GenerateList() bool {
	return i.options&NoList == 0
}

// GenerateDelete is true if the method is to be generated.
func (i *ServiceInfo) GenerateDelete() bool {
	return i.options&NoDelete == 0
}

// GenerateInsert is true if the method is to be generated.
func (i *ServiceInfo) GenerateInsert() bool {
	return i.options&NoInsert == 0
}

// GenerateCustomOps is true if we should generated a xxxOps interface for
// adding additional methods to the generated interface.
func (i *ServiceInfo) GenerateCustomOps() bool {
	return i.options&CustomOps != 0
}

// AggregatedList is true if the method is to be generated.
func (i *ServiceInfo) AggregatedList() bool {
	return i.options&AggregatedList != 0
}

// AggregatedListField is the name of the field used for the aggregated list
// call. This is typically the same as the name of the service, but can be
// customized by setting the aggregatedListField field.
func (i *ServiceInfo) AggregatedListField() string {
	if i.aggregatedListField == "" {
		return i.Service
	}
	return i.aggregatedListField
}

// ServiceGroup is a grouping of the same service but at different API versions.
type ServiceGroup struct {
	Alpha *ServiceInfo
	Beta  *ServiceInfo
	GA    *ServiceInfo
}

// Service returns any ServiceInfo string belonging to the ServiceGroup.
func (sg *ServiceGroup) Service() string {
	return sg.ServiceInfo().Service
}

// ServiceInfo returns any ServiceInfo object belonging to the ServiceGroup.
func (sg *ServiceGroup) ServiceInfo() *ServiceInfo {
	switch {
	case sg.GA != nil:
		return sg.GA
	case sg.Alpha != nil:
		return sg.Alpha
	case sg.Beta != nil:
		return sg.Beta
	default:
		panic(errors.New("service group is empty"))
	}
}

// HasGA returns true if this object has a GA representation.
func (sg *ServiceGroup) HasGA() bool {
	return sg.GA != nil
}

// HasAlpha returns true if this object has a Alpha representation.
func (sg *ServiceGroup) HasAlpha() bool {
	return sg.Alpha != nil
}

// HasBeta returns true if this object has a Beta representation.
func (sg *ServiceGroup) HasBeta() bool {
	return sg.Beta != nil
}

// groupServices together by version.
func groupServices(services []*ServiceInfo) map[string]*ServiceGroup {
	ret := map[string]*ServiceGroup{}
	for _, si := range services {
		if _, ok := ret[si.Service]; !ok {
			ret[si.Service] = &ServiceGroup{}
		}
		group := ret[si.Service]
		switch si.Version() {
		case VersionAlpha:
			group.Alpha = si
		case VersionBeta:
			group.Beta = si
		case VersionGA:
			group.GA = si
		}
	}
	return ret
}

// AllServicesByGroup is a map of service name to ServicesGroup.
var AllServicesByGroup map[string]*ServiceGroup

// SortedServicesGroups is a slice of Servicegroup sorted by Service name.
var SortedServicesGroups []*ServiceGroup

func init() {
	AllServicesByGroup = groupServices(AllServices)

	for _, sg := range AllServicesByGroup {
		SortedServicesGroups = append(SortedServicesGroups, sg)
	}
	sort.Slice(SortedServicesGroups, func(i, j int) bool {
		return SortedServicesGroups[i].Service() < SortedServicesGroups[j].Service()
	})
}
