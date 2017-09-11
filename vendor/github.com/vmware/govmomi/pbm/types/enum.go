/*
Copyright (c) 2014-2017 VMware, Inc. All Rights Reserved.

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

type PbmBuiltinGenericType string

const (
	PbmBuiltinGenericTypeVMW_RANGE = PbmBuiltinGenericType("VMW_RANGE")
	PbmBuiltinGenericTypeVMW_SET   = PbmBuiltinGenericType("VMW_SET")
)

func init() {
	types.Add("pbm:PbmBuiltinGenericType", reflect.TypeOf((*PbmBuiltinGenericType)(nil)).Elem())
}

type PbmBuiltinType string

const (
	PbmBuiltinTypeXSD_LONG     = PbmBuiltinType("XSD_LONG")
	PbmBuiltinTypeXSD_SHORT    = PbmBuiltinType("XSD_SHORT")
	PbmBuiltinTypeXSD_INTEGER  = PbmBuiltinType("XSD_INTEGER")
	PbmBuiltinTypeXSD_INT      = PbmBuiltinType("XSD_INT")
	PbmBuiltinTypeXSD_STRING   = PbmBuiltinType("XSD_STRING")
	PbmBuiltinTypeXSD_BOOLEAN  = PbmBuiltinType("XSD_BOOLEAN")
	PbmBuiltinTypeXSD_DOUBLE   = PbmBuiltinType("XSD_DOUBLE")
	PbmBuiltinTypeXSD_DATETIME = PbmBuiltinType("XSD_DATETIME")
	PbmBuiltinTypeVMW_TIMESPAN = PbmBuiltinType("VMW_TIMESPAN")
	PbmBuiltinTypeVMW_POLICY   = PbmBuiltinType("VMW_POLICY")
)

func init() {
	types.Add("pbm:PbmBuiltinType", reflect.TypeOf((*PbmBuiltinType)(nil)).Elem())
}

type PbmCapabilityOperator string

const (
	PbmCapabilityOperatorNOT = PbmCapabilityOperator("NOT")
)

func init() {
	types.Add("pbm:PbmCapabilityOperator", reflect.TypeOf((*PbmCapabilityOperator)(nil)).Elem())
}

type PbmCapabilityTimeUnitType string

const (
	PbmCapabilityTimeUnitTypeSECONDS = PbmCapabilityTimeUnitType("SECONDS")
	PbmCapabilityTimeUnitTypeMINUTES = PbmCapabilityTimeUnitType("MINUTES")
	PbmCapabilityTimeUnitTypeHOURS   = PbmCapabilityTimeUnitType("HOURS")
	PbmCapabilityTimeUnitTypeDAYS    = PbmCapabilityTimeUnitType("DAYS")
	PbmCapabilityTimeUnitTypeWEEKS   = PbmCapabilityTimeUnitType("WEEKS")
	PbmCapabilityTimeUnitTypeMONTHS  = PbmCapabilityTimeUnitType("MONTHS")
	PbmCapabilityTimeUnitTypeYEARS   = PbmCapabilityTimeUnitType("YEARS")
)

func init() {
	types.Add("pbm:PbmCapabilityTimeUnitType", reflect.TypeOf((*PbmCapabilityTimeUnitType)(nil)).Elem())
}

type PbmComplianceResultComplianceTaskStatus string

const (
	PbmComplianceResultComplianceTaskStatusInProgress = PbmComplianceResultComplianceTaskStatus("inProgress")
	PbmComplianceResultComplianceTaskStatusSuccess    = PbmComplianceResultComplianceTaskStatus("success")
	PbmComplianceResultComplianceTaskStatusFailed     = PbmComplianceResultComplianceTaskStatus("failed")
)

func init() {
	types.Add("pbm:PbmComplianceResultComplianceTaskStatus", reflect.TypeOf((*PbmComplianceResultComplianceTaskStatus)(nil)).Elem())
}

type PbmComplianceStatus string

const (
	PbmComplianceStatusCompliant     = PbmComplianceStatus("compliant")
	PbmComplianceStatusNonCompliant  = PbmComplianceStatus("nonCompliant")
	PbmComplianceStatusUnknown       = PbmComplianceStatus("unknown")
	PbmComplianceStatusNotApplicable = PbmComplianceStatus("notApplicable")
	PbmComplianceStatusOutOfDate     = PbmComplianceStatus("outOfDate")
)

func init() {
	types.Add("pbm:PbmComplianceStatus", reflect.TypeOf((*PbmComplianceStatus)(nil)).Elem())
}

type PbmIofilterInfoFilterType string

const (
	PbmIofilterInfoFilterTypeINSPECTION         = PbmIofilterInfoFilterType("INSPECTION")
	PbmIofilterInfoFilterTypeCOMPRESSION        = PbmIofilterInfoFilterType("COMPRESSION")
	PbmIofilterInfoFilterTypeENCRYPTION         = PbmIofilterInfoFilterType("ENCRYPTION")
	PbmIofilterInfoFilterTypeREPLICATION        = PbmIofilterInfoFilterType("REPLICATION")
	PbmIofilterInfoFilterTypeCACHE              = PbmIofilterInfoFilterType("CACHE")
	PbmIofilterInfoFilterTypeDATAPROVIDER       = PbmIofilterInfoFilterType("DATAPROVIDER")
	PbmIofilterInfoFilterTypeDATASTOREIOCONTROL = PbmIofilterInfoFilterType("DATASTOREIOCONTROL")
)

func init() {
	types.Add("pbm:PbmIofilterInfoFilterType", reflect.TypeOf((*PbmIofilterInfoFilterType)(nil)).Elem())
}

type PbmLineOfServiceInfoLineOfServiceEnum string

const (
	PbmLineOfServiceInfoLineOfServiceEnumINSPECTION           = PbmLineOfServiceInfoLineOfServiceEnum("INSPECTION")
	PbmLineOfServiceInfoLineOfServiceEnumCOMPRESSION          = PbmLineOfServiceInfoLineOfServiceEnum("COMPRESSION")
	PbmLineOfServiceInfoLineOfServiceEnumENCRYPTION           = PbmLineOfServiceInfoLineOfServiceEnum("ENCRYPTION")
	PbmLineOfServiceInfoLineOfServiceEnumREPLICATION          = PbmLineOfServiceInfoLineOfServiceEnum("REPLICATION")
	PbmLineOfServiceInfoLineOfServiceEnumCACHING              = PbmLineOfServiceInfoLineOfServiceEnum("CACHING")
	PbmLineOfServiceInfoLineOfServiceEnumPERSISTENCE          = PbmLineOfServiceInfoLineOfServiceEnum("PERSISTENCE")
	PbmLineOfServiceInfoLineOfServiceEnumDATA_PROVIDER        = PbmLineOfServiceInfoLineOfServiceEnum("DATA_PROVIDER")
	PbmLineOfServiceInfoLineOfServiceEnumDATASTORE_IO_CONTROL = PbmLineOfServiceInfoLineOfServiceEnum("DATASTORE_IO_CONTROL")
)

func init() {
	types.Add("pbm:PbmLineOfServiceInfoLineOfServiceEnum", reflect.TypeOf((*PbmLineOfServiceInfoLineOfServiceEnum)(nil)).Elem())
}

type PbmObjectType string

const (
	PbmObjectTypeVirtualMachine         = PbmObjectType("virtualMachine")
	PbmObjectTypeVirtualMachineAndDisks = PbmObjectType("virtualMachineAndDisks")
	PbmObjectTypeVirtualDiskId          = PbmObjectType("virtualDiskId")
	PbmObjectTypeVirtualDiskUUID        = PbmObjectType("virtualDiskUUID")
	PbmObjectTypeDatastore              = PbmObjectType("datastore")
	PbmObjectTypeUnknown                = PbmObjectType("unknown")
)

func init() {
	types.Add("pbm:PbmObjectType", reflect.TypeOf((*PbmObjectType)(nil)).Elem())
}

type PbmProfileCategoryEnum string

const (
	PbmProfileCategoryEnumREQUIREMENT         = PbmProfileCategoryEnum("REQUIREMENT")
	PbmProfileCategoryEnumRESOURCE            = PbmProfileCategoryEnum("RESOURCE")
	PbmProfileCategoryEnumDATA_SERVICE_POLICY = PbmProfileCategoryEnum("DATA_SERVICE_POLICY")
)

func init() {
	types.Add("pbm:PbmProfileCategoryEnum", reflect.TypeOf((*PbmProfileCategoryEnum)(nil)).Elem())
}

type PbmProfileResourceTypeEnum string

const (
	PbmProfileResourceTypeEnumSTORAGE = PbmProfileResourceTypeEnum("STORAGE")
)

func init() {
	types.Add("pbm:PbmProfileResourceTypeEnum", reflect.TypeOf((*PbmProfileResourceTypeEnum)(nil)).Elem())
}

type PbmSystemCreatedProfileType string

const (
	PbmSystemCreatedProfileTypeVsanDefaultProfile = PbmSystemCreatedProfileType("VsanDefaultProfile")
	PbmSystemCreatedProfileTypeVVolDefaultProfile = PbmSystemCreatedProfileType("VVolDefaultProfile")
)

func init() {
	types.Add("pbm:PbmSystemCreatedProfileType", reflect.TypeOf((*PbmSystemCreatedProfileType)(nil)).Elem())
}

type PbmVmOperation string

const (
	PbmVmOperationCREATE      = PbmVmOperation("CREATE")
	PbmVmOperationRECONFIGURE = PbmVmOperation("RECONFIGURE")
	PbmVmOperationMIGRATE     = PbmVmOperation("MIGRATE")
	PbmVmOperationCLONE       = PbmVmOperation("CLONE")
)

func init() {
	types.Add("pbm:PbmVmOperation", reflect.TypeOf((*PbmVmOperation)(nil)).Elem())
}

type PbmVvolType string

const (
	PbmVvolTypeConfig = PbmVvolType("Config")
	PbmVvolTypeData   = PbmVvolType("Data")
	PbmVvolTypeSwap   = PbmVvolType("Swap")
)

func init() {
	types.Add("pbm:PbmVvolType", reflect.TypeOf((*PbmVvolType)(nil)).Elem())
}
