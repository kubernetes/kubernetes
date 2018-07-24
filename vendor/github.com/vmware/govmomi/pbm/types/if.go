/*
Copyright (c) 2014-2018 VMware, Inc. All Rights Reserved.

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

package types

import (
	"reflect"

	"github.com/vmware/govmomi/vim25/types"
)

func (b *PbmCapabilityConstraints) GetPbmCapabilityConstraints() *PbmCapabilityConstraints { return b }

type BasePbmCapabilityConstraints interface {
	GetPbmCapabilityConstraints() *PbmCapabilityConstraints
}

func init() {
	types.Add("BasePbmCapabilityConstraints", reflect.TypeOf((*PbmCapabilityConstraints)(nil)).Elem())
}

func (b *PbmCapabilityProfile) GetPbmCapabilityProfile() *PbmCapabilityProfile { return b }

type BasePbmCapabilityProfile interface {
	GetPbmCapabilityProfile() *PbmCapabilityProfile
}

func init() {
	types.Add("BasePbmCapabilityProfile", reflect.TypeOf((*PbmCapabilityProfile)(nil)).Elem())
}

func (b *PbmCapabilityProfilePropertyMismatchFault) GetPbmCapabilityProfilePropertyMismatchFault() *PbmCapabilityProfilePropertyMismatchFault {
	return b
}

type BasePbmCapabilityProfilePropertyMismatchFault interface {
	GetPbmCapabilityProfilePropertyMismatchFault() *PbmCapabilityProfilePropertyMismatchFault
}

func init() {
	types.Add("BasePbmCapabilityProfilePropertyMismatchFault", reflect.TypeOf((*PbmCapabilityProfilePropertyMismatchFault)(nil)).Elem())
}

func (b *PbmCapabilityTypeInfo) GetPbmCapabilityTypeInfo() *PbmCapabilityTypeInfo { return b }

type BasePbmCapabilityTypeInfo interface {
	GetPbmCapabilityTypeInfo() *PbmCapabilityTypeInfo
}

func init() {
	types.Add("BasePbmCapabilityTypeInfo", reflect.TypeOf((*PbmCapabilityTypeInfo)(nil)).Elem())
}

func (b *PbmCompatibilityCheckFault) GetPbmCompatibilityCheckFault() *PbmCompatibilityCheckFault {
	return b
}

type BasePbmCompatibilityCheckFault interface {
	GetPbmCompatibilityCheckFault() *PbmCompatibilityCheckFault
}

func init() {
	types.Add("BasePbmCompatibilityCheckFault", reflect.TypeOf((*PbmCompatibilityCheckFault)(nil)).Elem())
}

func (b *PbmFault) GetPbmFault() *PbmFault { return b }

type BasePbmFault interface {
	GetPbmFault() *PbmFault
}

func init() {
	types.Add("BasePbmFault", reflect.TypeOf((*PbmFault)(nil)).Elem())
}

func (b *PbmLineOfServiceInfo) GetPbmLineOfServiceInfo() *PbmLineOfServiceInfo { return b }

type BasePbmLineOfServiceInfo interface {
	GetPbmLineOfServiceInfo() *PbmLineOfServiceInfo
}

func init() {
	types.Add("BasePbmLineOfServiceInfo", reflect.TypeOf((*PbmLineOfServiceInfo)(nil)).Elem())
}

func (b *PbmPlacementMatchingResources) GetPbmPlacementMatchingResources() *PbmPlacementMatchingResources {
	return b
}

type BasePbmPlacementMatchingResources interface {
	GetPbmPlacementMatchingResources() *PbmPlacementMatchingResources
}

func init() {
	types.Add("BasePbmPlacementMatchingResources", reflect.TypeOf((*PbmPlacementMatchingResources)(nil)).Elem())
}

func (b *PbmPlacementRequirement) GetPbmPlacementRequirement() *PbmPlacementRequirement { return b }

type BasePbmPlacementRequirement interface {
	GetPbmPlacementRequirement() *PbmPlacementRequirement
}

func init() {
	types.Add("BasePbmPlacementRequirement", reflect.TypeOf((*PbmPlacementRequirement)(nil)).Elem())
}

func (b *PbmProfile) GetPbmProfile() *PbmProfile { return b }

type BasePbmProfile interface {
	GetPbmProfile() *PbmProfile
}

func init() {
	types.Add("BasePbmProfile", reflect.TypeOf((*PbmProfile)(nil)).Elem())
}

func (b *PbmPropertyMismatchFault) GetPbmPropertyMismatchFault() *PbmPropertyMismatchFault { return b }

type BasePbmPropertyMismatchFault interface {
	GetPbmPropertyMismatchFault() *PbmPropertyMismatchFault
}

func init() {
	types.Add("BasePbmPropertyMismatchFault", reflect.TypeOf((*PbmPropertyMismatchFault)(nil)).Elem())
}
