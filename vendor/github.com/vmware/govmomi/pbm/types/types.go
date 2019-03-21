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
	"time"

	"github.com/vmware/govmomi/vim25/types"
)

type ArrayOfPbmCapabilityConstraintInstance struct {
	PbmCapabilityConstraintInstance []PbmCapabilityConstraintInstance `xml:"PbmCapabilityConstraintInstance,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmCapabilityConstraintInstance", reflect.TypeOf((*ArrayOfPbmCapabilityConstraintInstance)(nil)).Elem())
}

type ArrayOfPbmCapabilityInstance struct {
	PbmCapabilityInstance []PbmCapabilityInstance `xml:"PbmCapabilityInstance,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmCapabilityInstance", reflect.TypeOf((*ArrayOfPbmCapabilityInstance)(nil)).Elem())
}

type ArrayOfPbmCapabilityMetadata struct {
	PbmCapabilityMetadata []PbmCapabilityMetadata `xml:"PbmCapabilityMetadata,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmCapabilityMetadata", reflect.TypeOf((*ArrayOfPbmCapabilityMetadata)(nil)).Elem())
}

type ArrayOfPbmCapabilityMetadataPerCategory struct {
	PbmCapabilityMetadataPerCategory []PbmCapabilityMetadataPerCategory `xml:"PbmCapabilityMetadataPerCategory,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmCapabilityMetadataPerCategory", reflect.TypeOf((*ArrayOfPbmCapabilityMetadataPerCategory)(nil)).Elem())
}

type ArrayOfPbmCapabilityPropertyInstance struct {
	PbmCapabilityPropertyInstance []PbmCapabilityPropertyInstance `xml:"PbmCapabilityPropertyInstance,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmCapabilityPropertyInstance", reflect.TypeOf((*ArrayOfPbmCapabilityPropertyInstance)(nil)).Elem())
}

type ArrayOfPbmCapabilityPropertyMetadata struct {
	PbmCapabilityPropertyMetadata []PbmCapabilityPropertyMetadata `xml:"PbmCapabilityPropertyMetadata,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmCapabilityPropertyMetadata", reflect.TypeOf((*ArrayOfPbmCapabilityPropertyMetadata)(nil)).Elem())
}

type ArrayOfPbmCapabilitySchema struct {
	PbmCapabilitySchema []PbmCapabilitySchema `xml:"PbmCapabilitySchema,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmCapabilitySchema", reflect.TypeOf((*ArrayOfPbmCapabilitySchema)(nil)).Elem())
}

type ArrayOfPbmCapabilitySubProfile struct {
	PbmCapabilitySubProfile []PbmCapabilitySubProfile `xml:"PbmCapabilitySubProfile,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmCapabilitySubProfile", reflect.TypeOf((*ArrayOfPbmCapabilitySubProfile)(nil)).Elem())
}

type ArrayOfPbmCapabilityVendorNamespaceInfo struct {
	PbmCapabilityVendorNamespaceInfo []PbmCapabilityVendorNamespaceInfo `xml:"PbmCapabilityVendorNamespaceInfo,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmCapabilityVendorNamespaceInfo", reflect.TypeOf((*ArrayOfPbmCapabilityVendorNamespaceInfo)(nil)).Elem())
}

type ArrayOfPbmCapabilityVendorResourceTypeInfo struct {
	PbmCapabilityVendorResourceTypeInfo []PbmCapabilityVendorResourceTypeInfo `xml:"PbmCapabilityVendorResourceTypeInfo,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmCapabilityVendorResourceTypeInfo", reflect.TypeOf((*ArrayOfPbmCapabilityVendorResourceTypeInfo)(nil)).Elem())
}

type ArrayOfPbmCompliancePolicyStatus struct {
	PbmCompliancePolicyStatus []PbmCompliancePolicyStatus `xml:"PbmCompliancePolicyStatus,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmCompliancePolicyStatus", reflect.TypeOf((*ArrayOfPbmCompliancePolicyStatus)(nil)).Elem())
}

type ArrayOfPbmComplianceResult struct {
	PbmComplianceResult []PbmComplianceResult `xml:"PbmComplianceResult,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmComplianceResult", reflect.TypeOf((*ArrayOfPbmComplianceResult)(nil)).Elem())
}

type ArrayOfPbmDatastoreSpaceStatistics struct {
	PbmDatastoreSpaceStatistics []PbmDatastoreSpaceStatistics `xml:"PbmDatastoreSpaceStatistics,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmDatastoreSpaceStatistics", reflect.TypeOf((*ArrayOfPbmDatastoreSpaceStatistics)(nil)).Elem())
}

type ArrayOfPbmDefaultProfileInfo struct {
	PbmDefaultProfileInfo []PbmDefaultProfileInfo `xml:"PbmDefaultProfileInfo,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmDefaultProfileInfo", reflect.TypeOf((*ArrayOfPbmDefaultProfileInfo)(nil)).Elem())
}

type ArrayOfPbmPlacementCompatibilityResult struct {
	PbmPlacementCompatibilityResult []PbmPlacementCompatibilityResult `xml:"PbmPlacementCompatibilityResult,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmPlacementCompatibilityResult", reflect.TypeOf((*ArrayOfPbmPlacementCompatibilityResult)(nil)).Elem())
}

type ArrayOfPbmPlacementHub struct {
	PbmPlacementHub []PbmPlacementHub `xml:"PbmPlacementHub,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmPlacementHub", reflect.TypeOf((*ArrayOfPbmPlacementHub)(nil)).Elem())
}

type ArrayOfPbmPlacementMatchingResources struct {
	PbmPlacementMatchingResources []BasePbmPlacementMatchingResources `xml:"PbmPlacementMatchingResources,omitempty,typeattr"`
}

func init() {
	types.Add("pbm:ArrayOfPbmPlacementMatchingResources", reflect.TypeOf((*ArrayOfPbmPlacementMatchingResources)(nil)).Elem())
}

type ArrayOfPbmPlacementRequirement struct {
	PbmPlacementRequirement []BasePbmPlacementRequirement `xml:"PbmPlacementRequirement,omitempty,typeattr"`
}

func init() {
	types.Add("pbm:ArrayOfPbmPlacementRequirement", reflect.TypeOf((*ArrayOfPbmPlacementRequirement)(nil)).Elem())
}

type ArrayOfPbmPlacementResourceUtilization struct {
	PbmPlacementResourceUtilization []PbmPlacementResourceUtilization `xml:"PbmPlacementResourceUtilization,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmPlacementResourceUtilization", reflect.TypeOf((*ArrayOfPbmPlacementResourceUtilization)(nil)).Elem())
}

type ArrayOfPbmProfile struct {
	PbmProfile []BasePbmProfile `xml:"PbmProfile,omitempty,typeattr"`
}

func init() {
	types.Add("pbm:ArrayOfPbmProfile", reflect.TypeOf((*ArrayOfPbmProfile)(nil)).Elem())
}

type ArrayOfPbmProfileId struct {
	PbmProfileId []PbmProfileId `xml:"PbmProfileId,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmProfileId", reflect.TypeOf((*ArrayOfPbmProfileId)(nil)).Elem())
}

type ArrayOfPbmProfileOperationOutcome struct {
	PbmProfileOperationOutcome []PbmProfileOperationOutcome `xml:"PbmProfileOperationOutcome,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmProfileOperationOutcome", reflect.TypeOf((*ArrayOfPbmProfileOperationOutcome)(nil)).Elem())
}

type ArrayOfPbmProfileResourceType struct {
	PbmProfileResourceType []PbmProfileResourceType `xml:"PbmProfileResourceType,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmProfileResourceType", reflect.TypeOf((*ArrayOfPbmProfileResourceType)(nil)).Elem())
}

type ArrayOfPbmProfileType struct {
	PbmProfileType []PbmProfileType `xml:"PbmProfileType,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmProfileType", reflect.TypeOf((*ArrayOfPbmProfileType)(nil)).Elem())
}

type ArrayOfPbmQueryProfileResult struct {
	PbmQueryProfileResult []PbmQueryProfileResult `xml:"PbmQueryProfileResult,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmQueryProfileResult", reflect.TypeOf((*ArrayOfPbmQueryProfileResult)(nil)).Elem())
}

type ArrayOfPbmQueryReplicationGroupResult struct {
	PbmQueryReplicationGroupResult []PbmQueryReplicationGroupResult `xml:"PbmQueryReplicationGroupResult,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmQueryReplicationGroupResult", reflect.TypeOf((*ArrayOfPbmQueryReplicationGroupResult)(nil)).Elem())
}

type ArrayOfPbmRollupComplianceResult struct {
	PbmRollupComplianceResult []PbmRollupComplianceResult `xml:"PbmRollupComplianceResult,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmRollupComplianceResult", reflect.TypeOf((*ArrayOfPbmRollupComplianceResult)(nil)).Elem())
}

type ArrayOfPbmServerObjectRef struct {
	PbmServerObjectRef []PbmServerObjectRef `xml:"PbmServerObjectRef,omitempty"`
}

func init() {
	types.Add("pbm:ArrayOfPbmServerObjectRef", reflect.TypeOf((*ArrayOfPbmServerObjectRef)(nil)).Elem())
}

type PbmAboutInfo struct {
	types.DynamicData

	Name         string `xml:"name"`
	Version      string `xml:"version"`
	InstanceUuid string `xml:"instanceUuid"`
}

func init() {
	types.Add("pbm:PbmAboutInfo", reflect.TypeOf((*PbmAboutInfo)(nil)).Elem())
}

type PbmAlreadyExists struct {
	PbmFault

	Name string `xml:"name,omitempty"`
}

func init() {
	types.Add("pbm:PbmAlreadyExists", reflect.TypeOf((*PbmAlreadyExists)(nil)).Elem())
}

type PbmAlreadyExistsFault PbmAlreadyExists

func init() {
	types.Add("pbm:PbmAlreadyExistsFault", reflect.TypeOf((*PbmAlreadyExistsFault)(nil)).Elem())
}

type PbmAssignDefaultRequirementProfile PbmAssignDefaultRequirementProfileRequestType

func init() {
	types.Add("pbm:PbmAssignDefaultRequirementProfile", reflect.TypeOf((*PbmAssignDefaultRequirementProfile)(nil)).Elem())
}

type PbmAssignDefaultRequirementProfileRequestType struct {
	This       types.ManagedObjectReference `xml:"_this"`
	Profile    PbmProfileId                 `xml:"profile"`
	Datastores []PbmPlacementHub            `xml:"datastores"`
}

func init() {
	types.Add("pbm:PbmAssignDefaultRequirementProfileRequestType", reflect.TypeOf((*PbmAssignDefaultRequirementProfileRequestType)(nil)).Elem())
}

type PbmAssignDefaultRequirementProfileResponse struct {
}

type PbmCapabilityConstraintInstance struct {
	types.DynamicData

	PropertyInstance []PbmCapabilityPropertyInstance `xml:"propertyInstance"`
}

func init() {
	types.Add("pbm:PbmCapabilityConstraintInstance", reflect.TypeOf((*PbmCapabilityConstraintInstance)(nil)).Elem())
}

type PbmCapabilityConstraints struct {
	types.DynamicData
}

func init() {
	types.Add("pbm:PbmCapabilityConstraints", reflect.TypeOf((*PbmCapabilityConstraints)(nil)).Elem())
}

type PbmCapabilityDescription struct {
	types.DynamicData

	Description PbmExtendedElementDescription `xml:"description"`
	Value       types.AnyType                 `xml:"value,typeattr"`
}

func init() {
	types.Add("pbm:PbmCapabilityDescription", reflect.TypeOf((*PbmCapabilityDescription)(nil)).Elem())
}

type PbmCapabilityDiscreteSet struct {
	types.DynamicData

	Values []types.AnyType `xml:"values,typeattr"`
}

func init() {
	types.Add("pbm:PbmCapabilityDiscreteSet", reflect.TypeOf((*PbmCapabilityDiscreteSet)(nil)).Elem())
}

type PbmCapabilityGenericTypeInfo struct {
	PbmCapabilityTypeInfo

	GenericTypeName string `xml:"genericTypeName"`
}

func init() {
	types.Add("pbm:PbmCapabilityGenericTypeInfo", reflect.TypeOf((*PbmCapabilityGenericTypeInfo)(nil)).Elem())
}

type PbmCapabilityInstance struct {
	types.DynamicData

	Id         PbmCapabilityMetadataUniqueId     `xml:"id"`
	Constraint []PbmCapabilityConstraintInstance `xml:"constraint"`
}

func init() {
	types.Add("pbm:PbmCapabilityInstance", reflect.TypeOf((*PbmCapabilityInstance)(nil)).Elem())
}

type PbmCapabilityMetadata struct {
	types.DynamicData

	Id                       PbmCapabilityMetadataUniqueId   `xml:"id"`
	Summary                  PbmExtendedElementDescription   `xml:"summary"`
	Mandatory                *bool                           `xml:"mandatory"`
	Hint                     *bool                           `xml:"hint"`
	KeyId                    string                          `xml:"keyId,omitempty"`
	AllowMultipleConstraints *bool                           `xml:"allowMultipleConstraints"`
	PropertyMetadata         []PbmCapabilityPropertyMetadata `xml:"propertyMetadata"`
}

func init() {
	types.Add("pbm:PbmCapabilityMetadata", reflect.TypeOf((*PbmCapabilityMetadata)(nil)).Elem())
}

type PbmCapabilityMetadataPerCategory struct {
	types.DynamicData

	SubCategory        string                  `xml:"subCategory"`
	CapabilityMetadata []PbmCapabilityMetadata `xml:"capabilityMetadata"`
}

func init() {
	types.Add("pbm:PbmCapabilityMetadataPerCategory", reflect.TypeOf((*PbmCapabilityMetadataPerCategory)(nil)).Elem())
}

type PbmCapabilityMetadataUniqueId struct {
	types.DynamicData

	Namespace string `xml:"namespace"`
	Id        string `xml:"id"`
}

func init() {
	types.Add("pbm:PbmCapabilityMetadataUniqueId", reflect.TypeOf((*PbmCapabilityMetadataUniqueId)(nil)).Elem())
}

type PbmCapabilityNamespaceInfo struct {
	types.DynamicData

	Version   string                         `xml:"version"`
	Namespace string                         `xml:"namespace"`
	Info      *PbmExtendedElementDescription `xml:"info,omitempty"`
}

func init() {
	types.Add("pbm:PbmCapabilityNamespaceInfo", reflect.TypeOf((*PbmCapabilityNamespaceInfo)(nil)).Elem())
}

type PbmCapabilityProfile struct {
	PbmProfile

	ProfileCategory          string                       `xml:"profileCategory"`
	ResourceType             PbmProfileResourceType       `xml:"resourceType"`
	Constraints              BasePbmCapabilityConstraints `xml:"constraints,typeattr"`
	GenerationId             int64                        `xml:"generationId,omitempty"`
	IsDefault                bool                         `xml:"isDefault"`
	SystemCreatedProfileType string                       `xml:"systemCreatedProfileType,omitempty"`
	LineOfService            string                       `xml:"lineOfService,omitempty"`
}

func init() {
	types.Add("pbm:PbmCapabilityProfile", reflect.TypeOf((*PbmCapabilityProfile)(nil)).Elem())
}

type PbmCapabilityProfileCreateSpec struct {
	types.DynamicData

	Name         string                       `xml:"name"`
	Description  string                       `xml:"description,omitempty"`
	Category     string                       `xml:"category,omitempty"`
	ResourceType PbmProfileResourceType       `xml:"resourceType"`
	Constraints  BasePbmCapabilityConstraints `xml:"constraints,typeattr"`
}

func init() {
	types.Add("pbm:PbmCapabilityProfileCreateSpec", reflect.TypeOf((*PbmCapabilityProfileCreateSpec)(nil)).Elem())
}

type PbmCapabilityProfilePropertyMismatchFault struct {
	PbmPropertyMismatchFault

	ResourcePropertyInstance PbmCapabilityPropertyInstance `xml:"resourcePropertyInstance"`
}

func init() {
	types.Add("pbm:PbmCapabilityProfilePropertyMismatchFault", reflect.TypeOf((*PbmCapabilityProfilePropertyMismatchFault)(nil)).Elem())
}

type PbmCapabilityProfilePropertyMismatchFaultFault BasePbmCapabilityProfilePropertyMismatchFault

func init() {
	types.Add("pbm:PbmCapabilityProfilePropertyMismatchFaultFault", reflect.TypeOf((*PbmCapabilityProfilePropertyMismatchFaultFault)(nil)).Elem())
}

type PbmCapabilityProfileUpdateSpec struct {
	types.DynamicData

	Name        string                       `xml:"name,omitempty"`
	Description string                       `xml:"description,omitempty"`
	Constraints BasePbmCapabilityConstraints `xml:"constraints,omitempty,typeattr"`
}

func init() {
	types.Add("pbm:PbmCapabilityProfileUpdateSpec", reflect.TypeOf((*PbmCapabilityProfileUpdateSpec)(nil)).Elem())
}

type PbmCapabilityPropertyInstance struct {
	types.DynamicData

	Id       string        `xml:"id"`
	Operator string        `xml:"operator,omitempty"`
	Value    types.AnyType `xml:"value,typeattr"`
}

func init() {
	types.Add("pbm:PbmCapabilityPropertyInstance", reflect.TypeOf((*PbmCapabilityPropertyInstance)(nil)).Elem())
}

type PbmCapabilityPropertyMetadata struct {
	types.DynamicData

	Id                   string                        `xml:"id"`
	Summary              PbmExtendedElementDescription `xml:"summary"`
	Mandatory            bool                          `xml:"mandatory"`
	Type                 BasePbmCapabilityTypeInfo     `xml:"type,omitempty,typeattr"`
	DefaultValue         types.AnyType                 `xml:"defaultValue,omitempty,typeattr"`
	AllowedValue         types.AnyType                 `xml:"allowedValue,omitempty,typeattr"`
	RequirementsTypeHint string                        `xml:"requirementsTypeHint,omitempty"`
}

func init() {
	types.Add("pbm:PbmCapabilityPropertyMetadata", reflect.TypeOf((*PbmCapabilityPropertyMetadata)(nil)).Elem())
}

type PbmCapabilityRange struct {
	types.DynamicData

	Min types.AnyType `xml:"min,typeattr"`
	Max types.AnyType `xml:"max,typeattr"`
}

func init() {
	types.Add("pbm:PbmCapabilityRange", reflect.TypeOf((*PbmCapabilityRange)(nil)).Elem())
}

type PbmCapabilitySchema struct {
	types.DynamicData

	VendorInfo                    PbmCapabilitySchemaVendorInfo      `xml:"vendorInfo"`
	NamespaceInfo                 PbmCapabilityNamespaceInfo         `xml:"namespaceInfo"`
	LineOfService                 BasePbmLineOfServiceInfo           `xml:"lineOfService,omitempty,typeattr"`
	CapabilityMetadataPerCategory []PbmCapabilityMetadataPerCategory `xml:"capabilityMetadataPerCategory"`
}

func init() {
	types.Add("pbm:PbmCapabilitySchema", reflect.TypeOf((*PbmCapabilitySchema)(nil)).Elem())
}

type PbmCapabilitySchemaVendorInfo struct {
	types.DynamicData

	VendorUuid string                        `xml:"vendorUuid"`
	Info       PbmExtendedElementDescription `xml:"info"`
}

func init() {
	types.Add("pbm:PbmCapabilitySchemaVendorInfo", reflect.TypeOf((*PbmCapabilitySchemaVendorInfo)(nil)).Elem())
}

type PbmCapabilitySubProfile struct {
	types.DynamicData

	Name           string                  `xml:"name"`
	Capability     []PbmCapabilityInstance `xml:"capability"`
	ForceProvision *bool                   `xml:"forceProvision"`
}

func init() {
	types.Add("pbm:PbmCapabilitySubProfile", reflect.TypeOf((*PbmCapabilitySubProfile)(nil)).Elem())
}

type PbmCapabilitySubProfileConstraints struct {
	PbmCapabilityConstraints

	SubProfiles []PbmCapabilitySubProfile `xml:"subProfiles"`
}

func init() {
	types.Add("pbm:PbmCapabilitySubProfileConstraints", reflect.TypeOf((*PbmCapabilitySubProfileConstraints)(nil)).Elem())
}

type PbmCapabilityTimeSpan struct {
	types.DynamicData

	Value int32  `xml:"value"`
	Unit  string `xml:"unit"`
}

func init() {
	types.Add("pbm:PbmCapabilityTimeSpan", reflect.TypeOf((*PbmCapabilityTimeSpan)(nil)).Elem())
}

type PbmCapabilityTypeInfo struct {
	types.DynamicData

	TypeName string `xml:"typeName"`
}

func init() {
	types.Add("pbm:PbmCapabilityTypeInfo", reflect.TypeOf((*PbmCapabilityTypeInfo)(nil)).Elem())
}

type PbmCapabilityVendorNamespaceInfo struct {
	types.DynamicData

	VendorInfo    PbmCapabilitySchemaVendorInfo `xml:"vendorInfo"`
	NamespaceInfo PbmCapabilityNamespaceInfo    `xml:"namespaceInfo"`
}

func init() {
	types.Add("pbm:PbmCapabilityVendorNamespaceInfo", reflect.TypeOf((*PbmCapabilityVendorNamespaceInfo)(nil)).Elem())
}

type PbmCapabilityVendorResourceTypeInfo struct {
	types.DynamicData

	ResourceType        string                             `xml:"resourceType"`
	VendorNamespaceInfo []PbmCapabilityVendorNamespaceInfo `xml:"vendorNamespaceInfo"`
}

func init() {
	types.Add("pbm:PbmCapabilityVendorResourceTypeInfo", reflect.TypeOf((*PbmCapabilityVendorResourceTypeInfo)(nil)).Elem())
}

type PbmCheckCompatibility PbmCheckCompatibilityRequestType

func init() {
	types.Add("pbm:PbmCheckCompatibility", reflect.TypeOf((*PbmCheckCompatibility)(nil)).Elem())
}

type PbmCheckCompatibilityRequestType struct {
	This         types.ManagedObjectReference `xml:"_this"`
	HubsToSearch []PbmPlacementHub            `xml:"hubsToSearch,omitempty"`
	Profile      PbmProfileId                 `xml:"profile"`
}

func init() {
	types.Add("pbm:PbmCheckCompatibilityRequestType", reflect.TypeOf((*PbmCheckCompatibilityRequestType)(nil)).Elem())
}

type PbmCheckCompatibilityResponse struct {
	Returnval []PbmPlacementCompatibilityResult `xml:"returnval,omitempty"`
}

type PbmCheckCompatibilityWithSpec PbmCheckCompatibilityWithSpecRequestType

func init() {
	types.Add("pbm:PbmCheckCompatibilityWithSpec", reflect.TypeOf((*PbmCheckCompatibilityWithSpec)(nil)).Elem())
}

type PbmCheckCompatibilityWithSpecRequestType struct {
	This         types.ManagedObjectReference   `xml:"_this"`
	HubsToSearch []PbmPlacementHub              `xml:"hubsToSearch,omitempty"`
	ProfileSpec  PbmCapabilityProfileCreateSpec `xml:"profileSpec"`
}

func init() {
	types.Add("pbm:PbmCheckCompatibilityWithSpecRequestType", reflect.TypeOf((*PbmCheckCompatibilityWithSpecRequestType)(nil)).Elem())
}

type PbmCheckCompatibilityWithSpecResponse struct {
	Returnval []PbmPlacementCompatibilityResult `xml:"returnval,omitempty"`
}

type PbmCheckCompliance PbmCheckComplianceRequestType

func init() {
	types.Add("pbm:PbmCheckCompliance", reflect.TypeOf((*PbmCheckCompliance)(nil)).Elem())
}

type PbmCheckComplianceRequestType struct {
	This     types.ManagedObjectReference `xml:"_this"`
	Entities []PbmServerObjectRef         `xml:"entities"`
	Profile  *PbmProfileId                `xml:"profile,omitempty"`
}

func init() {
	types.Add("pbm:PbmCheckComplianceRequestType", reflect.TypeOf((*PbmCheckComplianceRequestType)(nil)).Elem())
}

type PbmCheckComplianceResponse struct {
	Returnval []PbmComplianceResult `xml:"returnval,omitempty"`
}

type PbmCheckRequirements PbmCheckRequirementsRequestType

func init() {
	types.Add("pbm:PbmCheckRequirements", reflect.TypeOf((*PbmCheckRequirements)(nil)).Elem())
}

type PbmCheckRequirementsRequestType struct {
	This                        types.ManagedObjectReference  `xml:"_this"`
	HubsToSearch                []PbmPlacementHub             `xml:"hubsToSearch,omitempty"`
	PlacementSubjectRef         *PbmServerObjectRef           `xml:"placementSubjectRef,omitempty"`
	PlacementSubjectRequirement []BasePbmPlacementRequirement `xml:"placementSubjectRequirement,omitempty,typeattr"`
}

func init() {
	types.Add("pbm:PbmCheckRequirementsRequestType", reflect.TypeOf((*PbmCheckRequirementsRequestType)(nil)).Elem())
}

type PbmCheckRequirementsResponse struct {
	Returnval []PbmPlacementCompatibilityResult `xml:"returnval,omitempty"`
}

type PbmCheckRollupCompliance PbmCheckRollupComplianceRequestType

func init() {
	types.Add("pbm:PbmCheckRollupCompliance", reflect.TypeOf((*PbmCheckRollupCompliance)(nil)).Elem())
}

type PbmCheckRollupComplianceRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	Entity []PbmServerObjectRef         `xml:"entity"`
}

func init() {
	types.Add("pbm:PbmCheckRollupComplianceRequestType", reflect.TypeOf((*PbmCheckRollupComplianceRequestType)(nil)).Elem())
}

type PbmCheckRollupComplianceResponse struct {
	Returnval []PbmRollupComplianceResult `xml:"returnval,omitempty"`
}

type PbmCompatibilityCheckFault struct {
	PbmFault

	Hub PbmPlacementHub `xml:"hub"`
}

func init() {
	types.Add("pbm:PbmCompatibilityCheckFault", reflect.TypeOf((*PbmCompatibilityCheckFault)(nil)).Elem())
}

type PbmCompatibilityCheckFaultFault BasePbmCompatibilityCheckFault

func init() {
	types.Add("pbm:PbmCompatibilityCheckFaultFault", reflect.TypeOf((*PbmCompatibilityCheckFaultFault)(nil)).Elem())
}

type PbmComplianceOperationalStatus struct {
	types.DynamicData

	Healthy           *bool      `xml:"healthy"`
	OperationETA      *time.Time `xml:"operationETA"`
	OperationProgress int64      `xml:"operationProgress,omitempty"`
	Transitional      *bool      `xml:"transitional"`
}

func init() {
	types.Add("pbm:PbmComplianceOperationalStatus", reflect.TypeOf((*PbmComplianceOperationalStatus)(nil)).Elem())
}

type PbmCompliancePolicyStatus struct {
	types.DynamicData

	ExpectedValue PbmCapabilityInstance  `xml:"expectedValue"`
	CurrentValue  *PbmCapabilityInstance `xml:"currentValue,omitempty"`
}

func init() {
	types.Add("pbm:PbmCompliancePolicyStatus", reflect.TypeOf((*PbmCompliancePolicyStatus)(nil)).Elem())
}

type PbmComplianceResult struct {
	types.DynamicData

	CheckTime            time.Time                       `xml:"checkTime"`
	Entity               PbmServerObjectRef              `xml:"entity"`
	Profile              *PbmProfileId                   `xml:"profile,omitempty"`
	ComplianceTaskStatus string                          `xml:"complianceTaskStatus,omitempty"`
	ComplianceStatus     string                          `xml:"complianceStatus"`
	Mismatch             bool                            `xml:"mismatch"`
	ViolatedPolicies     []PbmCompliancePolicyStatus     `xml:"violatedPolicies,omitempty"`
	ErrorCause           []types.LocalizedMethodFault    `xml:"errorCause,omitempty"`
	OperationalStatus    *PbmComplianceOperationalStatus `xml:"operationalStatus,omitempty"`
	Info                 *PbmExtendedElementDescription  `xml:"info,omitempty"`
}

func init() {
	types.Add("pbm:PbmComplianceResult", reflect.TypeOf((*PbmComplianceResult)(nil)).Elem())
}

type PbmCreate PbmCreateRequestType

func init() {
	types.Add("pbm:PbmCreate", reflect.TypeOf((*PbmCreate)(nil)).Elem())
}

type PbmCreateRequestType struct {
	This       types.ManagedObjectReference   `xml:"_this"`
	CreateSpec PbmCapabilityProfileCreateSpec `xml:"createSpec"`
}

func init() {
	types.Add("pbm:PbmCreateRequestType", reflect.TypeOf((*PbmCreateRequestType)(nil)).Elem())
}

type PbmCreateResponse struct {
	Returnval PbmProfileId `xml:"returnval"`
}

type PbmDataServiceToPoliciesMap struct {
	types.DynamicData

	DataServicePolicy     PbmProfileId                `xml:"dataServicePolicy"`
	ParentStoragePolicies []PbmProfileId              `xml:"parentStoragePolicies,omitempty"`
	Fault                 *types.LocalizedMethodFault `xml:"fault,omitempty"`
}

func init() {
	types.Add("pbm:PbmDataServiceToPoliciesMap", reflect.TypeOf((*PbmDataServiceToPoliciesMap)(nil)).Elem())
}

type PbmDatastoreSpaceStatistics struct {
	types.DynamicData

	ProfileId         string `xml:"profileId,omitempty"`
	PhysicalTotalInMB int64  `xml:"physicalTotalInMB"`
	PhysicalFreeInMB  int64  `xml:"physicalFreeInMB"`
	PhysicalUsedInMB  int64  `xml:"physicalUsedInMB"`
	LogicalLimitInMB  int64  `xml:"logicalLimitInMB,omitempty"`
	LogicalFreeInMB   int64  `xml:"logicalFreeInMB"`
	LogicalUsedInMB   int64  `xml:"logicalUsedInMB"`
}

func init() {
	types.Add("pbm:PbmDatastoreSpaceStatistics", reflect.TypeOf((*PbmDatastoreSpaceStatistics)(nil)).Elem())
}

type PbmDefaultCapabilityProfile struct {
	PbmCapabilityProfile

	VvolType    []string `xml:"vvolType"`
	ContainerId string   `xml:"containerId"`
}

func init() {
	types.Add("pbm:PbmDefaultCapabilityProfile", reflect.TypeOf((*PbmDefaultCapabilityProfile)(nil)).Elem())
}

type PbmDefaultProfileAppliesFault struct {
	PbmCompatibilityCheckFault
}

func init() {
	types.Add("pbm:PbmDefaultProfileAppliesFault", reflect.TypeOf((*PbmDefaultProfileAppliesFault)(nil)).Elem())
}

type PbmDefaultProfileAppliesFaultFault PbmDefaultProfileAppliesFault

func init() {
	types.Add("pbm:PbmDefaultProfileAppliesFaultFault", reflect.TypeOf((*PbmDefaultProfileAppliesFaultFault)(nil)).Elem())
}

type PbmDefaultProfileInfo struct {
	types.DynamicData

	Datastores     []PbmPlacementHub `xml:"datastores"`
	DefaultProfile BasePbmProfile    `xml:"defaultProfile,omitempty,typeattr"`
}

func init() {
	types.Add("pbm:PbmDefaultProfileInfo", reflect.TypeOf((*PbmDefaultProfileInfo)(nil)).Elem())
}

type PbmDelete PbmDeleteRequestType

func init() {
	types.Add("pbm:PbmDelete", reflect.TypeOf((*PbmDelete)(nil)).Elem())
}

type PbmDeleteRequestType struct {
	This      types.ManagedObjectReference `xml:"_this"`
	ProfileId []PbmProfileId               `xml:"profileId"`
}

func init() {
	types.Add("pbm:PbmDeleteRequestType", reflect.TypeOf((*PbmDeleteRequestType)(nil)).Elem())
}

type PbmDeleteResponse struct {
	Returnval []PbmProfileOperationOutcome `xml:"returnval,omitempty"`
}

type PbmDuplicateName struct {
	PbmFault

	Name string `xml:"name"`
}

func init() {
	types.Add("pbm:PbmDuplicateName", reflect.TypeOf((*PbmDuplicateName)(nil)).Elem())
}

type PbmDuplicateNameFault PbmDuplicateName

func init() {
	types.Add("pbm:PbmDuplicateNameFault", reflect.TypeOf((*PbmDuplicateNameFault)(nil)).Elem())
}

type PbmExtendedElementDescription struct {
	types.DynamicData

	Label                   string              `xml:"label"`
	Summary                 string              `xml:"summary"`
	Key                     string              `xml:"key"`
	MessageCatalogKeyPrefix string              `xml:"messageCatalogKeyPrefix"`
	MessageArg              []types.KeyAnyValue `xml:"messageArg,omitempty"`
}

func init() {
	types.Add("pbm:PbmExtendedElementDescription", reflect.TypeOf((*PbmExtendedElementDescription)(nil)).Elem())
}

type PbmFault struct {
	types.MethodFault
}

func init() {
	types.Add("pbm:PbmFault", reflect.TypeOf((*PbmFault)(nil)).Elem())
}

type PbmFaultFault BasePbmFault

func init() {
	types.Add("pbm:PbmFaultFault", reflect.TypeOf((*PbmFaultFault)(nil)).Elem())
}

type PbmFaultInvalidLogin struct {
	PbmFault
}

func init() {
	types.Add("pbm:PbmFaultInvalidLogin", reflect.TypeOf((*PbmFaultInvalidLogin)(nil)).Elem())
}

type PbmFaultInvalidLoginFault PbmFaultInvalidLogin

func init() {
	types.Add("pbm:PbmFaultInvalidLoginFault", reflect.TypeOf((*PbmFaultInvalidLoginFault)(nil)).Elem())
}

type PbmFaultNotFound struct {
	PbmFault
}

func init() {
	types.Add("pbm:PbmFaultNotFound", reflect.TypeOf((*PbmFaultNotFound)(nil)).Elem())
}

type PbmFaultNotFoundFault PbmFaultNotFound

func init() {
	types.Add("pbm:PbmFaultNotFoundFault", reflect.TypeOf((*PbmFaultNotFoundFault)(nil)).Elem())
}

type PbmFaultProfileStorageFault struct {
	PbmFault
}

func init() {
	types.Add("pbm:PbmFaultProfileStorageFault", reflect.TypeOf((*PbmFaultProfileStorageFault)(nil)).Elem())
}

type PbmFaultProfileStorageFaultFault PbmFaultProfileStorageFault

func init() {
	types.Add("pbm:PbmFaultProfileStorageFaultFault", reflect.TypeOf((*PbmFaultProfileStorageFaultFault)(nil)).Elem())
}

type PbmFetchCapabilityMetadata PbmFetchCapabilityMetadataRequestType

func init() {
	types.Add("pbm:PbmFetchCapabilityMetadata", reflect.TypeOf((*PbmFetchCapabilityMetadata)(nil)).Elem())
}

type PbmFetchCapabilityMetadataRequestType struct {
	This         types.ManagedObjectReference `xml:"_this"`
	ResourceType *PbmProfileResourceType      `xml:"resourceType,omitempty"`
	VendorUuid   string                       `xml:"vendorUuid,omitempty"`
}

func init() {
	types.Add("pbm:PbmFetchCapabilityMetadataRequestType", reflect.TypeOf((*PbmFetchCapabilityMetadataRequestType)(nil)).Elem())
}

type PbmFetchCapabilityMetadataResponse struct {
	Returnval []PbmCapabilityMetadataPerCategory `xml:"returnval,omitempty"`
}

type PbmFetchCapabilitySchema PbmFetchCapabilitySchemaRequestType

func init() {
	types.Add("pbm:PbmFetchCapabilitySchema", reflect.TypeOf((*PbmFetchCapabilitySchema)(nil)).Elem())
}

type PbmFetchCapabilitySchemaRequestType struct {
	This          types.ManagedObjectReference `xml:"_this"`
	VendorUuid    string                       `xml:"vendorUuid,omitempty"`
	LineOfService []string                     `xml:"lineOfService,omitempty"`
}

func init() {
	types.Add("pbm:PbmFetchCapabilitySchemaRequestType", reflect.TypeOf((*PbmFetchCapabilitySchemaRequestType)(nil)).Elem())
}

type PbmFetchCapabilitySchemaResponse struct {
	Returnval []PbmCapabilitySchema `xml:"returnval,omitempty"`
}

type PbmFetchComplianceResult PbmFetchComplianceResultRequestType

func init() {
	types.Add("pbm:PbmFetchComplianceResult", reflect.TypeOf((*PbmFetchComplianceResult)(nil)).Elem())
}

type PbmFetchComplianceResultRequestType struct {
	This     types.ManagedObjectReference `xml:"_this"`
	Entities []PbmServerObjectRef         `xml:"entities"`
	Profile  *PbmProfileId                `xml:"profile,omitempty"`
}

func init() {
	types.Add("pbm:PbmFetchComplianceResultRequestType", reflect.TypeOf((*PbmFetchComplianceResultRequestType)(nil)).Elem())
}

type PbmFetchComplianceResultResponse struct {
	Returnval []PbmComplianceResult `xml:"returnval,omitempty"`
}

type PbmFetchResourceType PbmFetchResourceTypeRequestType

func init() {
	types.Add("pbm:PbmFetchResourceType", reflect.TypeOf((*PbmFetchResourceType)(nil)).Elem())
}

type PbmFetchResourceTypeRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("pbm:PbmFetchResourceTypeRequestType", reflect.TypeOf((*PbmFetchResourceTypeRequestType)(nil)).Elem())
}

type PbmFetchResourceTypeResponse struct {
	Returnval []PbmProfileResourceType `xml:"returnval,omitempty"`
}

type PbmFetchRollupComplianceResult PbmFetchRollupComplianceResultRequestType

func init() {
	types.Add("pbm:PbmFetchRollupComplianceResult", reflect.TypeOf((*PbmFetchRollupComplianceResult)(nil)).Elem())
}

type PbmFetchRollupComplianceResultRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	Entity []PbmServerObjectRef         `xml:"entity"`
}

func init() {
	types.Add("pbm:PbmFetchRollupComplianceResultRequestType", reflect.TypeOf((*PbmFetchRollupComplianceResultRequestType)(nil)).Elem())
}

type PbmFetchRollupComplianceResultResponse struct {
	Returnval []PbmRollupComplianceResult `xml:"returnval,omitempty"`
}

type PbmFetchVendorInfo PbmFetchVendorInfoRequestType

func init() {
	types.Add("pbm:PbmFetchVendorInfo", reflect.TypeOf((*PbmFetchVendorInfo)(nil)).Elem())
}

type PbmFetchVendorInfoRequestType struct {
	This         types.ManagedObjectReference `xml:"_this"`
	ResourceType *PbmProfileResourceType      `xml:"resourceType,omitempty"`
}

func init() {
	types.Add("pbm:PbmFetchVendorInfoRequestType", reflect.TypeOf((*PbmFetchVendorInfoRequestType)(nil)).Elem())
}

type PbmFetchVendorInfoResponse struct {
	Returnval []PbmCapabilityVendorResourceTypeInfo `xml:"returnval,omitempty"`
}

type PbmFindApplicableDefaultProfile PbmFindApplicableDefaultProfileRequestType

func init() {
	types.Add("pbm:PbmFindApplicableDefaultProfile", reflect.TypeOf((*PbmFindApplicableDefaultProfile)(nil)).Elem())
}

type PbmFindApplicableDefaultProfileRequestType struct {
	This       types.ManagedObjectReference `xml:"_this"`
	Datastores []PbmPlacementHub            `xml:"datastores"`
}

func init() {
	types.Add("pbm:PbmFindApplicableDefaultProfileRequestType", reflect.TypeOf((*PbmFindApplicableDefaultProfileRequestType)(nil)).Elem())
}

type PbmFindApplicableDefaultProfileResponse struct {
	Returnval []BasePbmProfile `xml:"returnval,omitempty,typeattr"`
}

type PbmIncompatibleVendorSpecificRuleSet struct {
	PbmCapabilityProfilePropertyMismatchFault
}

func init() {
	types.Add("pbm:PbmIncompatibleVendorSpecificRuleSet", reflect.TypeOf((*PbmIncompatibleVendorSpecificRuleSet)(nil)).Elem())
}

type PbmIncompatibleVendorSpecificRuleSetFault PbmIncompatibleVendorSpecificRuleSet

func init() {
	types.Add("pbm:PbmIncompatibleVendorSpecificRuleSetFault", reflect.TypeOf((*PbmIncompatibleVendorSpecificRuleSetFault)(nil)).Elem())
}

type PbmLegacyHubsNotSupported struct {
	PbmFault

	Hubs []PbmPlacementHub `xml:"hubs"`
}

func init() {
	types.Add("pbm:PbmLegacyHubsNotSupported", reflect.TypeOf((*PbmLegacyHubsNotSupported)(nil)).Elem())
}

type PbmLegacyHubsNotSupportedFault PbmLegacyHubsNotSupported

func init() {
	types.Add("pbm:PbmLegacyHubsNotSupportedFault", reflect.TypeOf((*PbmLegacyHubsNotSupportedFault)(nil)).Elem())
}

type PbmLineOfServiceInfo struct {
	types.DynamicData

	LineOfService string                         `xml:"lineOfService"`
	Name          PbmExtendedElementDescription  `xml:"name"`
	Description   *PbmExtendedElementDescription `xml:"description,omitempty"`
}

func init() {
	types.Add("pbm:PbmLineOfServiceInfo", reflect.TypeOf((*PbmLineOfServiceInfo)(nil)).Elem())
}

type PbmNonExistentHubs struct {
	PbmFault

	Hubs []PbmPlacementHub `xml:"hubs"`
}

func init() {
	types.Add("pbm:PbmNonExistentHubs", reflect.TypeOf((*PbmNonExistentHubs)(nil)).Elem())
}

type PbmNonExistentHubsFault PbmNonExistentHubs

func init() {
	types.Add("pbm:PbmNonExistentHubsFault", reflect.TypeOf((*PbmNonExistentHubsFault)(nil)).Elem())
}

type PbmPersistenceBasedDataServiceInfo struct {
	PbmLineOfServiceInfo

	CompatiblePersistenceSchemaNamespace []string `xml:"compatiblePersistenceSchemaNamespace,omitempty"`
}

func init() {
	types.Add("pbm:PbmPersistenceBasedDataServiceInfo", reflect.TypeOf((*PbmPersistenceBasedDataServiceInfo)(nil)).Elem())
}

type PbmPlacementCapabilityConstraintsRequirement struct {
	PbmPlacementRequirement

	Constraints BasePbmCapabilityConstraints `xml:"constraints,typeattr"`
}

func init() {
	types.Add("pbm:PbmPlacementCapabilityConstraintsRequirement", reflect.TypeOf((*PbmPlacementCapabilityConstraintsRequirement)(nil)).Elem())
}

type PbmPlacementCapabilityProfileRequirement struct {
	PbmPlacementRequirement

	ProfileId PbmProfileId `xml:"profileId"`
}

func init() {
	types.Add("pbm:PbmPlacementCapabilityProfileRequirement", reflect.TypeOf((*PbmPlacementCapabilityProfileRequirement)(nil)).Elem())
}

type PbmPlacementCompatibilityResult struct {
	types.DynamicData

	Hub               PbmPlacementHub                     `xml:"hub"`
	MatchingResources []BasePbmPlacementMatchingResources `xml:"matchingResources,omitempty,typeattr"`
	HowMany           int64                               `xml:"howMany,omitempty"`
	Utilization       []PbmPlacementResourceUtilization   `xml:"utilization,omitempty"`
	Warning           []types.LocalizedMethodFault        `xml:"warning,omitempty"`
	Error             []types.LocalizedMethodFault        `xml:"error,omitempty"`
}

func init() {
	types.Add("pbm:PbmPlacementCompatibilityResult", reflect.TypeOf((*PbmPlacementCompatibilityResult)(nil)).Elem())
}

type PbmPlacementHub struct {
	types.DynamicData

	HubType string `xml:"hubType"`
	HubId   string `xml:"hubId"`
}

func init() {
	types.Add("pbm:PbmPlacementHub", reflect.TypeOf((*PbmPlacementHub)(nil)).Elem())
}

type PbmPlacementMatchingReplicationResources struct {
	PbmPlacementMatchingResources

	ReplicationGroup []types.ReplicationGroupId `xml:"replicationGroup,omitempty"`
}

func init() {
	types.Add("pbm:PbmPlacementMatchingReplicationResources", reflect.TypeOf((*PbmPlacementMatchingReplicationResources)(nil)).Elem())
}

type PbmPlacementMatchingResources struct {
	types.DynamicData
}

func init() {
	types.Add("pbm:PbmPlacementMatchingResources", reflect.TypeOf((*PbmPlacementMatchingResources)(nil)).Elem())
}

type PbmPlacementRequirement struct {
	types.DynamicData
}

func init() {
	types.Add("pbm:PbmPlacementRequirement", reflect.TypeOf((*PbmPlacementRequirement)(nil)).Elem())
}

type PbmPlacementResourceUtilization struct {
	types.DynamicData

	Name            PbmExtendedElementDescription `xml:"name"`
	Description     PbmExtendedElementDescription `xml:"description"`
	AvailableBefore int64                         `xml:"availableBefore,omitempty"`
	AvailableAfter  int64                         `xml:"availableAfter,omitempty"`
	Total           int64                         `xml:"total,omitempty"`
}

func init() {
	types.Add("pbm:PbmPlacementResourceUtilization", reflect.TypeOf((*PbmPlacementResourceUtilization)(nil)).Elem())
}

type PbmProfile struct {
	types.DynamicData

	ProfileId       PbmProfileId `xml:"profileId"`
	Name            string       `xml:"name"`
	Description     string       `xml:"description,omitempty"`
	CreationTime    time.Time    `xml:"creationTime"`
	CreatedBy       string       `xml:"createdBy"`
	LastUpdatedTime time.Time    `xml:"lastUpdatedTime"`
	LastUpdatedBy   string       `xml:"lastUpdatedBy"`
}

func init() {
	types.Add("pbm:PbmProfile", reflect.TypeOf((*PbmProfile)(nil)).Elem())
}

type PbmProfileId struct {
	types.DynamicData

	UniqueId string `xml:"uniqueId"`
}

func init() {
	types.Add("pbm:PbmProfileId", reflect.TypeOf((*PbmProfileId)(nil)).Elem())
}

type PbmProfileOperationOutcome struct {
	types.DynamicData

	ProfileId PbmProfileId                `xml:"profileId"`
	Fault     *types.LocalizedMethodFault `xml:"fault,omitempty"`
}

func init() {
	types.Add("pbm:PbmProfileOperationOutcome", reflect.TypeOf((*PbmProfileOperationOutcome)(nil)).Elem())
}

type PbmProfileResourceType struct {
	types.DynamicData

	ResourceType string `xml:"resourceType"`
}

func init() {
	types.Add("pbm:PbmProfileResourceType", reflect.TypeOf((*PbmProfileResourceType)(nil)).Elem())
}

type PbmProfileType struct {
	types.DynamicData

	UniqueId string `xml:"uniqueId"`
}

func init() {
	types.Add("pbm:PbmProfileType", reflect.TypeOf((*PbmProfileType)(nil)).Elem())
}

type PbmPropertyMismatchFault struct {
	PbmCompatibilityCheckFault

	CapabilityInstanceId        PbmCapabilityMetadataUniqueId `xml:"capabilityInstanceId"`
	RequirementPropertyInstance PbmCapabilityPropertyInstance `xml:"requirementPropertyInstance"`
}

func init() {
	types.Add("pbm:PbmPropertyMismatchFault", reflect.TypeOf((*PbmPropertyMismatchFault)(nil)).Elem())
}

type PbmPropertyMismatchFaultFault BasePbmPropertyMismatchFault

func init() {
	types.Add("pbm:PbmPropertyMismatchFaultFault", reflect.TypeOf((*PbmPropertyMismatchFaultFault)(nil)).Elem())
}

type PbmQueryAssociatedEntities PbmQueryAssociatedEntitiesRequestType

func init() {
	types.Add("pbm:PbmQueryAssociatedEntities", reflect.TypeOf((*PbmQueryAssociatedEntities)(nil)).Elem())
}

type PbmQueryAssociatedEntitiesRequestType struct {
	This     types.ManagedObjectReference `xml:"_this"`
	Profiles []PbmProfileId               `xml:"profiles,omitempty"`
}

func init() {
	types.Add("pbm:PbmQueryAssociatedEntitiesRequestType", reflect.TypeOf((*PbmQueryAssociatedEntitiesRequestType)(nil)).Elem())
}

type PbmQueryAssociatedEntitiesResponse struct {
	Returnval []PbmQueryProfileResult `xml:"returnval,omitempty"`
}

type PbmQueryAssociatedEntity PbmQueryAssociatedEntityRequestType

func init() {
	types.Add("pbm:PbmQueryAssociatedEntity", reflect.TypeOf((*PbmQueryAssociatedEntity)(nil)).Elem())
}

type PbmQueryAssociatedEntityRequestType struct {
	This       types.ManagedObjectReference `xml:"_this"`
	Profile    PbmProfileId                 `xml:"profile"`
	EntityType string                       `xml:"entityType,omitempty"`
}

func init() {
	types.Add("pbm:PbmQueryAssociatedEntityRequestType", reflect.TypeOf((*PbmQueryAssociatedEntityRequestType)(nil)).Elem())
}

type PbmQueryAssociatedEntityResponse struct {
	Returnval []PbmServerObjectRef `xml:"returnval,omitempty"`
}

type PbmQueryAssociatedProfile PbmQueryAssociatedProfileRequestType

func init() {
	types.Add("pbm:PbmQueryAssociatedProfile", reflect.TypeOf((*PbmQueryAssociatedProfile)(nil)).Elem())
}

type PbmQueryAssociatedProfileRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	Entity PbmServerObjectRef           `xml:"entity"`
}

func init() {
	types.Add("pbm:PbmQueryAssociatedProfileRequestType", reflect.TypeOf((*PbmQueryAssociatedProfileRequestType)(nil)).Elem())
}

type PbmQueryAssociatedProfileResponse struct {
	Returnval []PbmProfileId `xml:"returnval,omitempty"`
}

type PbmQueryAssociatedProfiles PbmQueryAssociatedProfilesRequestType

func init() {
	types.Add("pbm:PbmQueryAssociatedProfiles", reflect.TypeOf((*PbmQueryAssociatedProfiles)(nil)).Elem())
}

type PbmQueryAssociatedProfilesRequestType struct {
	This     types.ManagedObjectReference `xml:"_this"`
	Entities []PbmServerObjectRef         `xml:"entities"`
}

func init() {
	types.Add("pbm:PbmQueryAssociatedProfilesRequestType", reflect.TypeOf((*PbmQueryAssociatedProfilesRequestType)(nil)).Elem())
}

type PbmQueryAssociatedProfilesResponse struct {
	Returnval []PbmQueryProfileResult `xml:"returnval,omitempty"`
}

type PbmQueryByRollupComplianceStatus PbmQueryByRollupComplianceStatusRequestType

func init() {
	types.Add("pbm:PbmQueryByRollupComplianceStatus", reflect.TypeOf((*PbmQueryByRollupComplianceStatus)(nil)).Elem())
}

type PbmQueryByRollupComplianceStatusRequestType struct {
	This   types.ManagedObjectReference `xml:"_this"`
	Status string                       `xml:"status"`
}

func init() {
	types.Add("pbm:PbmQueryByRollupComplianceStatusRequestType", reflect.TypeOf((*PbmQueryByRollupComplianceStatusRequestType)(nil)).Elem())
}

type PbmQueryByRollupComplianceStatusResponse struct {
	Returnval []PbmServerObjectRef `xml:"returnval,omitempty"`
}

type PbmQueryDefaultRequirementProfile PbmQueryDefaultRequirementProfileRequestType

func init() {
	types.Add("pbm:PbmQueryDefaultRequirementProfile", reflect.TypeOf((*PbmQueryDefaultRequirementProfile)(nil)).Elem())
}

type PbmQueryDefaultRequirementProfileRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
	Hub  PbmPlacementHub              `xml:"hub"`
}

func init() {
	types.Add("pbm:PbmQueryDefaultRequirementProfileRequestType", reflect.TypeOf((*PbmQueryDefaultRequirementProfileRequestType)(nil)).Elem())
}

type PbmQueryDefaultRequirementProfileResponse struct {
	Returnval *PbmProfileId `xml:"returnval,omitempty"`
}

type PbmQueryDefaultRequirementProfiles PbmQueryDefaultRequirementProfilesRequestType

func init() {
	types.Add("pbm:PbmQueryDefaultRequirementProfiles", reflect.TypeOf((*PbmQueryDefaultRequirementProfiles)(nil)).Elem())
}

type PbmQueryDefaultRequirementProfilesRequestType struct {
	This       types.ManagedObjectReference `xml:"_this"`
	Datastores []PbmPlacementHub            `xml:"datastores"`
}

func init() {
	types.Add("pbm:PbmQueryDefaultRequirementProfilesRequestType", reflect.TypeOf((*PbmQueryDefaultRequirementProfilesRequestType)(nil)).Elem())
}

type PbmQueryDefaultRequirementProfilesResponse struct {
	Returnval []PbmDefaultProfileInfo `xml:"returnval"`
}

type PbmQueryMatchingHub PbmQueryMatchingHubRequestType

func init() {
	types.Add("pbm:PbmQueryMatchingHub", reflect.TypeOf((*PbmQueryMatchingHub)(nil)).Elem())
}

type PbmQueryMatchingHubRequestType struct {
	This         types.ManagedObjectReference `xml:"_this"`
	HubsToSearch []PbmPlacementHub            `xml:"hubsToSearch,omitempty"`
	Profile      PbmProfileId                 `xml:"profile"`
}

func init() {
	types.Add("pbm:PbmQueryMatchingHubRequestType", reflect.TypeOf((*PbmQueryMatchingHubRequestType)(nil)).Elem())
}

type PbmQueryMatchingHubResponse struct {
	Returnval []PbmPlacementHub `xml:"returnval,omitempty"`
}

type PbmQueryMatchingHubWithSpec PbmQueryMatchingHubWithSpecRequestType

func init() {
	types.Add("pbm:PbmQueryMatchingHubWithSpec", reflect.TypeOf((*PbmQueryMatchingHubWithSpec)(nil)).Elem())
}

type PbmQueryMatchingHubWithSpecRequestType struct {
	This         types.ManagedObjectReference   `xml:"_this"`
	HubsToSearch []PbmPlacementHub              `xml:"hubsToSearch,omitempty"`
	CreateSpec   PbmCapabilityProfileCreateSpec `xml:"createSpec"`
}

func init() {
	types.Add("pbm:PbmQueryMatchingHubWithSpecRequestType", reflect.TypeOf((*PbmQueryMatchingHubWithSpecRequestType)(nil)).Elem())
}

type PbmQueryMatchingHubWithSpecResponse struct {
	Returnval []PbmPlacementHub `xml:"returnval,omitempty"`
}

type PbmQueryProfile PbmQueryProfileRequestType

func init() {
	types.Add("pbm:PbmQueryProfile", reflect.TypeOf((*PbmQueryProfile)(nil)).Elem())
}

type PbmQueryProfileRequestType struct {
	This            types.ManagedObjectReference `xml:"_this"`
	ResourceType    PbmProfileResourceType       `xml:"resourceType"`
	ProfileCategory string                       `xml:"profileCategory,omitempty"`
}

func init() {
	types.Add("pbm:PbmQueryProfileRequestType", reflect.TypeOf((*PbmQueryProfileRequestType)(nil)).Elem())
}

type PbmQueryProfileResponse struct {
	Returnval []PbmProfileId `xml:"returnval,omitempty"`
}

type PbmQueryProfileResult struct {
	types.DynamicData

	Object    PbmServerObjectRef          `xml:"object"`
	ProfileId []PbmProfileId              `xml:"profileId,omitempty"`
	Fault     *types.LocalizedMethodFault `xml:"fault,omitempty"`
}

func init() {
	types.Add("pbm:PbmQueryProfileResult", reflect.TypeOf((*PbmQueryProfileResult)(nil)).Elem())
}

type PbmQueryReplicationGroupResult struct {
	types.DynamicData

	Object             PbmServerObjectRef          `xml:"object"`
	ReplicationGroupId *types.ReplicationGroupId   `xml:"replicationGroupId,omitempty"`
	Fault              *types.LocalizedMethodFault `xml:"fault,omitempty"`
}

func init() {
	types.Add("pbm:PbmQueryReplicationGroupResult", reflect.TypeOf((*PbmQueryReplicationGroupResult)(nil)).Elem())
}

type PbmQueryReplicationGroups PbmQueryReplicationGroupsRequestType

func init() {
	types.Add("pbm:PbmQueryReplicationGroups", reflect.TypeOf((*PbmQueryReplicationGroups)(nil)).Elem())
}

type PbmQueryReplicationGroupsRequestType struct {
	This     types.ManagedObjectReference `xml:"_this"`
	Entities []PbmServerObjectRef         `xml:"entities,omitempty"`
}

func init() {
	types.Add("pbm:PbmQueryReplicationGroupsRequestType", reflect.TypeOf((*PbmQueryReplicationGroupsRequestType)(nil)).Elem())
}

type PbmQueryReplicationGroupsResponse struct {
	Returnval []PbmQueryReplicationGroupResult `xml:"returnval,omitempty"`
}

type PbmQuerySpaceStatsForStorageContainer PbmQuerySpaceStatsForStorageContainerRequestType

func init() {
	types.Add("pbm:PbmQuerySpaceStatsForStorageContainer", reflect.TypeOf((*PbmQuerySpaceStatsForStorageContainer)(nil)).Elem())
}

type PbmQuerySpaceStatsForStorageContainerRequestType struct {
	This                types.ManagedObjectReference `xml:"_this"`
	Datastore           PbmServerObjectRef           `xml:"datastore"`
	CapabilityProfileId []PbmProfileId               `xml:"capabilityProfileId,omitempty"`
}

func init() {
	types.Add("pbm:PbmQuerySpaceStatsForStorageContainerRequestType", reflect.TypeOf((*PbmQuerySpaceStatsForStorageContainerRequestType)(nil)).Elem())
}

type PbmQuerySpaceStatsForStorageContainerResponse struct {
	Returnval []PbmDatastoreSpaceStatistics `xml:"returnval,omitempty"`
}

type PbmResetDefaultRequirementProfile PbmResetDefaultRequirementProfileRequestType

func init() {
	types.Add("pbm:PbmResetDefaultRequirementProfile", reflect.TypeOf((*PbmResetDefaultRequirementProfile)(nil)).Elem())
}

type PbmResetDefaultRequirementProfileRequestType struct {
	This    types.ManagedObjectReference `xml:"_this"`
	Profile *PbmProfileId                `xml:"profile,omitempty"`
}

func init() {
	types.Add("pbm:PbmResetDefaultRequirementProfileRequestType", reflect.TypeOf((*PbmResetDefaultRequirementProfileRequestType)(nil)).Elem())
}

type PbmResetDefaultRequirementProfileResponse struct {
}

type PbmResetVSanDefaultProfile PbmResetVSanDefaultProfileRequestType

func init() {
	types.Add("pbm:PbmResetVSanDefaultProfile", reflect.TypeOf((*PbmResetVSanDefaultProfile)(nil)).Elem())
}

type PbmResetVSanDefaultProfileRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("pbm:PbmResetVSanDefaultProfileRequestType", reflect.TypeOf((*PbmResetVSanDefaultProfileRequestType)(nil)).Elem())
}

type PbmResetVSanDefaultProfileResponse struct {
}

type PbmResourceInUse struct {
	PbmFault

	Type string `xml:"type,omitempty"`
	Name string `xml:"name,omitempty"`
}

func init() {
	types.Add("pbm:PbmResourceInUse", reflect.TypeOf((*PbmResourceInUse)(nil)).Elem())
}

type PbmResourceInUseFault PbmResourceInUse

func init() {
	types.Add("pbm:PbmResourceInUseFault", reflect.TypeOf((*PbmResourceInUseFault)(nil)).Elem())
}

type PbmRetrieveContent PbmRetrieveContentRequestType

func init() {
	types.Add("pbm:PbmRetrieveContent", reflect.TypeOf((*PbmRetrieveContent)(nil)).Elem())
}

type PbmRetrieveContentRequestType struct {
	This       types.ManagedObjectReference `xml:"_this"`
	ProfileIds []PbmProfileId               `xml:"profileIds"`
}

func init() {
	types.Add("pbm:PbmRetrieveContentRequestType", reflect.TypeOf((*PbmRetrieveContentRequestType)(nil)).Elem())
}

type PbmRetrieveContentResponse struct {
	Returnval []BasePbmProfile `xml:"returnval,typeattr"`
}

type PbmRetrieveServiceContent PbmRetrieveServiceContentRequestType

func init() {
	types.Add("pbm:PbmRetrieveServiceContent", reflect.TypeOf((*PbmRetrieveServiceContent)(nil)).Elem())
}

type PbmRetrieveServiceContentRequestType struct {
	This types.ManagedObjectReference `xml:"_this"`
}

func init() {
	types.Add("pbm:PbmRetrieveServiceContentRequestType", reflect.TypeOf((*PbmRetrieveServiceContentRequestType)(nil)).Elem())
}

type PbmRetrieveServiceContentResponse struct {
	Returnval PbmServiceInstanceContent `xml:"returnval"`
}

type PbmRollupComplianceResult struct {
	types.DynamicData

	OldestCheckTime             time.Time                    `xml:"oldestCheckTime"`
	Entity                      PbmServerObjectRef           `xml:"entity"`
	OverallComplianceStatus     string                       `xml:"overallComplianceStatus"`
	OverallComplianceTaskStatus string                       `xml:"overallComplianceTaskStatus,omitempty"`
	Result                      []PbmComplianceResult        `xml:"result,omitempty"`
	ErrorCause                  []types.LocalizedMethodFault `xml:"errorCause,omitempty"`
	ProfileMismatch             bool                         `xml:"profileMismatch"`
}

func init() {
	types.Add("pbm:PbmRollupComplianceResult", reflect.TypeOf((*PbmRollupComplianceResult)(nil)).Elem())
}

type PbmServerObjectRef struct {
	types.DynamicData

	ObjectType string `xml:"objectType"`
	Key        string `xml:"key"`
	ServerUuid string `xml:"serverUuid,omitempty"`
}

func init() {
	types.Add("pbm:PbmServerObjectRef", reflect.TypeOf((*PbmServerObjectRef)(nil)).Elem())
}

type PbmServiceInstanceContent struct {
	types.DynamicData

	AboutInfo                 PbmAboutInfo                  `xml:"aboutInfo"`
	SessionManager            types.ManagedObjectReference  `xml:"sessionManager"`
	CapabilityMetadataManager types.ManagedObjectReference  `xml:"capabilityMetadataManager"`
	ProfileManager            types.ManagedObjectReference  `xml:"profileManager"`
	ComplianceManager         types.ManagedObjectReference  `xml:"complianceManager"`
	PlacementSolver           types.ManagedObjectReference  `xml:"placementSolver"`
	ReplicationManager        *types.ManagedObjectReference `xml:"replicationManager,omitempty"`
}

func init() {
	types.Add("pbm:PbmServiceInstanceContent", reflect.TypeOf((*PbmServiceInstanceContent)(nil)).Elem())
}

type PbmUpdate PbmUpdateRequestType

func init() {
	types.Add("pbm:PbmUpdate", reflect.TypeOf((*PbmUpdate)(nil)).Elem())
}

type PbmUpdateRequestType struct {
	This       types.ManagedObjectReference   `xml:"_this"`
	ProfileId  PbmProfileId                   `xml:"profileId"`
	UpdateSpec PbmCapabilityProfileUpdateSpec `xml:"updateSpec"`
}

func init() {
	types.Add("pbm:PbmUpdateRequestType", reflect.TypeOf((*PbmUpdateRequestType)(nil)).Elem())
}

type PbmUpdateResponse struct {
}

type PbmVaioDataServiceInfo struct {
	PbmLineOfServiceInfo
}

func init() {
	types.Add("pbm:PbmVaioDataServiceInfo", reflect.TypeOf((*PbmVaioDataServiceInfo)(nil)).Elem())
}

type VersionURI string

func init() {
	types.Add("pbm:versionURI", reflect.TypeOf((*VersionURI)(nil)).Elem())
}
