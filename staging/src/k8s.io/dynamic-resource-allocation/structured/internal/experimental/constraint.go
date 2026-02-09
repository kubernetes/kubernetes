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

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	draapi "k8s.io/dynamic-resource-allocation/api"
	"k8s.io/klog/v2"
)

// distinctAttributeConstraint compares an attribute value across devices.
// All devices must share the same value. When the set of devices is
// empty, any device that has the attribute can be added. After that,
// only matching devices can be added.
//
// We don't need to track *which* devices are part of the set, only
// how many.
type distinctAttributeConstraint struct {
	logger        klog.Logger // Includes name and attribute name, so no need to repeat in log messages.
	requestNames  sets.Set[string]
	attributeName resourceapi.FullyQualifiedName

	attributes map[string]resourceapi.DeviceAttribute
	numDevices int
}

func (m *distinctAttributeConstraint) add(requestName, subRequestName string, device *draapi.Device, deviceID DeviceID) bool {
	if m.requestNames.Len() > 0 && !m.matches(requestName, subRequestName) {
		// Device not affected by constraint.
		return true
	}

	attribute := lookupAttribute(device, deviceID, m.attributeName)
	if attribute == nil {
		// Doesn't have the attribute.
		m.logger.V(7).Info("Constraint not satisfied, attribute not set")
		return false
	}

	if m.numDevices == 0 {
		// The first device can always get picked.
		m.attributes[requestName] = *attribute
		m.numDevices = 1
		m.logger.V(7).Info("First attribute added")
		return true
	}

	if !m.matchesAttribute(*attribute) {
		m.logger.V(7).Info("Constraint not satisfied, has some duplicated attributes")
		return false
	}
	m.attributes[requestName] = *attribute
	m.numDevices++
	m.logger.V(7).Info("Constraint satisfied by device", "device", deviceID, "numDevices", m.numDevices)
	return true

}

func (m *distinctAttributeConstraint) remove(requestName, subRequestName string, device *draapi.Device, deviceID DeviceID) {
	if m.requestNames.Len() > 0 && !m.matches(requestName, subRequestName) {
		// Device not affected by constraint.
		return
	}
	delete(m.attributes, requestName)
	m.numDevices--
	m.logger.V(7).Info("Device removed from constraint set", "device", deviceID, "numDevices", m.numDevices)
}

func (m *distinctAttributeConstraint) matches(requestName, subRequestName string) bool {
	if subRequestName == "" {
		return m.requestNames.Has(requestName)
	} else {
		fullSubRequestName := fmt.Sprintf("%s/%s", requestName, subRequestName)
		return m.requestNames.Has(requestName) || m.requestNames.Has(fullSubRequestName)
	}
}

func (m *distinctAttributeConstraint) matchesAttribute(attribute resourceapi.DeviceAttribute) bool {
	for _, attr := range m.attributes {
		switch {
		case attribute.StringValue != nil:
			if attr.StringValue != nil && *attribute.StringValue == *attr.StringValue {
				m.logger.V(7).Info("String values duplicated")
				return false
			}
		case attribute.IntValue != nil:
			if attr.IntValue != nil && *attribute.IntValue == *attr.IntValue {
				m.logger.V(7).Info("Int values duplicated")
				return false
			}
		case attribute.BoolValue != nil:
			if attr.BoolValue != nil && *attribute.BoolValue == *attr.BoolValue {
				m.logger.V(7).Info("Bool values duplicated")
				return false
			}
		case attribute.VersionValue != nil:
			// semver 2.0.0 requires that version strings are in their
			// minimal form (in particular, no leading zeros). Therefore a
			// strict "exact equal" check can do a string comparison.
			if attr.VersionValue != nil && *attribute.VersionValue == *attr.VersionValue {
				m.logger.V(7).Info("Version values duplicated")
				return false
			}
		default:
			// Unknown value type, cannot match.
			// This condition should not be reached
			// as the unknown value type should be failed on CEL compile (getAttributeValue).
			m.logger.V(7).Info("Distinct attribute type unknown")
			return false
		}
	}
	// All distinct
	return true
}
