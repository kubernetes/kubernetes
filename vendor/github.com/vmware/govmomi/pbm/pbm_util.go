/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package pbm

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/vmware/govmomi/pbm/types"
)

// A struct to capture pbm create spec details.
type CapabilityProfileCreateSpec struct {
	Name           string
	SubProfileName string
	Description    string
	Category       string
	CapabilityList []Capability
}

// A struct to capture pbm capability instance details.
type Capability struct {
	ID           string
	Namespace    string
	PropertyList []Property
}

// A struct to capture pbm property instance details.
type Property struct {
	ID       string
	Operator string
	Value    string
	DataType string
}

func CreateCapabilityProfileSpec(pbmCreateSpec CapabilityProfileCreateSpec) (*types.PbmCapabilityProfileCreateSpec, error) {
	capabilities, err := createCapabilityInstances(pbmCreateSpec.CapabilityList)
	if err != nil {
		return nil, err
	}

	pbmCapabilityProfileSpec := types.PbmCapabilityProfileCreateSpec{
		Name:        pbmCreateSpec.Name,
		Description: pbmCreateSpec.Description,
		Category:    pbmCreateSpec.Category,
		ResourceType: types.PbmProfileResourceType{
			ResourceType: string(types.PbmProfileResourceTypeEnumSTORAGE),
		},
		Constraints: &types.PbmCapabilitySubProfileConstraints{
			SubProfiles: []types.PbmCapabilitySubProfile{
				types.PbmCapabilitySubProfile{
					Capability: capabilities,
					Name:       pbmCreateSpec.SubProfileName,
				},
			},
		},
	}
	return &pbmCapabilityProfileSpec, nil
}

func createCapabilityInstances(rules []Capability) ([]types.PbmCapabilityInstance, error) {
	var capabilityInstances []types.PbmCapabilityInstance
	for _, capabilityRule := range rules {
		capability := types.PbmCapabilityInstance{
			Id: types.PbmCapabilityMetadataUniqueId{
				Namespace: capabilityRule.Namespace,
				Id:        capabilityRule.ID,
			},
		}

		var propertyInstances []types.PbmCapabilityPropertyInstance
		for _, propertyRule := range capabilityRule.PropertyList {
			property := types.PbmCapabilityPropertyInstance{
				Id: propertyRule.ID,
			}
			if propertyRule.Operator != "" {
				property.Operator = propertyRule.Operator
			}
			var err error
			switch strings.ToLower(propertyRule.DataType) {
			case "int":
				// Go int32 is marshalled to xsi:int whereas Go int is marshalled to xsi:long when sending down the wire.
				var val int32
				val, err = verifyPropertyValueIsInt(propertyRule.Value, propertyRule.DataType)
				property.Value = val
			case "bool":
				var val bool
				val, err = verifyPropertyValueIsBoolean(propertyRule.Value, propertyRule.DataType)
				property.Value = val
			case "string":
				property.Value = propertyRule.Value
			case "set":
				set := types.PbmCapabilityDiscreteSet{}
				for _, val := range strings.Split(propertyRule.Value, ",") {
					set.Values = append(set.Values, val)
				}
				property.Value = set
			default:
				return nil, fmt.Errorf("invalid value: %q with datatype: %q", propertyRule.Value, propertyRule.Value)
			}
			if err != nil {
				return nil, fmt.Errorf("invalid value: %q with datatype: %q", propertyRule.Value, propertyRule.Value)
			}
			propertyInstances = append(propertyInstances, property)
		}
		constraintInstances := []types.PbmCapabilityConstraintInstance{
			types.PbmCapabilityConstraintInstance{
				PropertyInstance: propertyInstances,
			},
		}
		capability.Constraint = constraintInstances
		capabilityInstances = append(capabilityInstances, capability)
	}
	return capabilityInstances, nil
}

// Verify if the capability value is of type integer.
func verifyPropertyValueIsInt(propertyValue string, dataType string) (int32, error) {
	val, err := strconv.ParseInt(propertyValue, 10, 32)
	if err != nil {
		return -1, err
	}
	return int32(val), nil
}

// Verify if the capability value is of type integer.
func verifyPropertyValueIsBoolean(propertyValue string, dataType string) (bool, error) {
	val, err := strconv.ParseBool(propertyValue)
	if err != nil {
		return false, err
	}
	return val, nil
}
