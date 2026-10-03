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

package experimental

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	apiservercel "k8s.io/apiserver/pkg/cel"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/klog/v2"
)

// attributeProvider defines how constraints retrieve device attributes.
type attributeProvider interface {
	lookupAttribute(request *requestData, device *draapi.Device, deviceID DeviceID, attribute draapi.FullyQualifiedName) (any, error)
}

// distinctAttributeConstraint compares an attribute value across devices.
// All devices must share the same value. When the set of devices is
// empty, any device that has the attribute can be added. After that,
// only matching devices can be added.
//
// We don't need to track *which* devices are part of the set, only
// how many.
type distinctAttributeConstraint struct {
	logger            klog.Logger // Includes name and attribute name, so no need to repeat in log messages.
	requestNames      sets.Set[string]
	attributeName     draapi.FullyQualifiedName
	features          Features
	attributeProvider attributeProvider

	attributes []any
}

func (m *distinctAttributeConstraint) add(request *requestData, device *draapi.Device, deviceID DeviceID) (bool, error) {
	if m.requestNames.Len() > 0 && !m.matches(request) {
		// Device not affected by constraint.
		return true, nil
	}

	attribute, err := m.attributeProvider.lookupAttribute(request, device, deviceID, m.attributeName)
	if err != nil {
		return false, err
	}
	if attribute == nil {
		// Doesn't have the attribute.
		m.logger.V(7).Info("Constraint not satisfied, attribute not set")
		return false, nil
	}

	if !m.matchesAttribute(attribute) {
		m.logger.V(7).Info("Constraint not satisfied, has some duplicated attributes")
		return false, nil
	}
	m.attributes = append(m.attributes, attribute)
	m.logger.V(7).Info("Constraint satisfied by device", "device", deviceID, "numDevices", len(m.attributes))
	return true, nil
}

func (m *distinctAttributeConstraint) remove(request *requestData, device *draapi.Device, deviceID DeviceID) {
	if m.requestNames.Len() > 0 && !m.matches(request) {
		// Device not affected by constraint.
		return
	}

	m.attributes = m.attributes[:len(m.attributes)-1]
	m.logger.V(7).Info("Device removed from constraint set", "device", deviceID, "numDevices", len(m.attributes))
}

func (m *distinctAttributeConstraint) matches(request *requestData) bool {
	if request.parentRequest != nil {
		requestName := request.parentRequest.name()
		subRequestName := request.request.name()
		fullSubRequestName := fmt.Sprintf("%s/%s", requestName, subRequestName)
		return m.requestNames.Has(requestName) || m.requestNames.Has(fullSubRequestName)
	} else {
		return m.requestNames.Has(request.request.name())
	}
}

func (m *distinctAttributeConstraint) matchesAttribute(attribute any) bool {
	if m.features.ListTypeAttributes {
		// Set-based comparison for ListAttributes feature:
		// Check that the new device's attribute set is disjoint from all existing devices.
		// This implements "Pairwise Disjoint" semantics for distinct attributes.
		newSet := attributeAsSet(attribute)
		if newSet == nil {
			m.logger.V(7).Info("Unknown attribute type")
			return false
		}

		// Check that the new device is disjoint from each existing device
		for _, attr := range m.attributes {
			existingSet := attributeAsSet(attr)
			if existingSet == nil {
				continue
			}
			if newSet.intersection(existingSet) != nil {
				// New device has common elements with an existing device.
				// This violates the distinct constraint.
				m.logger.V(7).Info("Constraint not satisfied, devices have common elements")
				return false
			}
		}

		// New device is disjoint from all existing devices.
		// The constraint is satisfied.
		m.logger.V(7).Info("Constraint satisfied, new device is disjoint from all existing devices")
		return true
	}

	// Scalar comparison (existing behavior)
	for _, attr := range m.attributes {
		switch existing := attr.(type) {
		case string:
			candidate, ok := attribute.(string)
			if !ok {
				m.logger.V(7).Info("Attribute types don't match", "existing", attr, "candidate", attribute)
				return true
			}
			if existing == candidate {
				m.logger.V(7).Info("String values duplicated")
				return false
			}
			m.logger.V(7).Info("Attribute values don't match", "existing", attr, "candidate", attribute)
		case int64:
			candidate, ok := attribute.(int64)
			if !ok {
				m.logger.V(7).Info("Attribute types don't match", "existing", attr, "candidate", attribute)
				return true
			}
			if existing == candidate {
				m.logger.V(7).Info("Int values duplicated")
				return false
			}
			m.logger.V(7).Info("Attribute values don't match", "existing", attr, "candidate", attribute)
		case bool:
			candidate, ok := attribute.(bool)
			if !ok {
				m.logger.V(7).Info("Attribute types don't match", "existing", attr, "candidate", attribute)
				return true
			}
			if existing == candidate {
				m.logger.V(7).Info("Bool values duplicated")
				return false
			}
			m.logger.V(7).Info("Attribute values don't match", "existing", attr, "candidate", attribute)
		case apiservercel.Semver:
			candidate, ok := attribute.(apiservercel.Semver)
			if !ok {
				m.logger.V(7).Info("Attribute types don't match", "existing", attr, "candidate", attribute)
				return true
			}
			if existing.Version.Equals(candidate.Version) {
				m.logger.V(7).Info("Version values duplicated")
				return false
			}
			m.logger.V(7).Info("Attribute values don't match", "existing", attr, "candidate", attribute)
		default:
			// Unknown value type, cannot match.
			m.logger.V(7).Info("Distinct attribute type unknown", "existing", attr)
			return false
		}
	}
	// All distinct
	return true
}
