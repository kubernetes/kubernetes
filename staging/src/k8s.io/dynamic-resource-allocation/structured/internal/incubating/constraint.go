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

package incubating

import (
	"fmt"

	"github.com/blang/semver/v4"

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
	attributeName draapi.FullyQualifiedName

	attributes []any
}

func (m *distinctAttributeConstraint) add(requestName, subRequestName string, device *draapi.Device, deviceID DeviceID) (bool, error) {
	if m.requestNames.Len() > 0 && !m.matches(requestName, subRequestName) {
		// Device not affected by constraint.
		return true, nil
	}

	attribute, ok := device.Attributes.Lookup(m.attributeName)
	if !ok {
		// Doesn't have the attribute.
		m.logger.V(7).Info("Constraint not satisfied, attribute not set")
		return false, nil
	}
	if attribute == nil {
		// Has the attribute, but with a value that this client cannot represent
		// (for example, a value type added in a newer API version).
		return false, fmt.Errorf("device %s: unsupported attribute value for %s", deviceID, m.attributeName)
	}

	switch attribute.(type) {
	case string:
	case int64:
	case bool:
	case semver.Version:
	default:
		// Unknown value type, cannot compare. Better stop instead of
		// silently ignoring the device.
		return false, fmt.Errorf("device %s: unsupported attribute value for %s", deviceID, m.attributeName)
	}

	matches := m.matchesAttribute(attribute)
	if !matches {
		m.logger.V(7).Info("Constraint not satisfied, has some duplicated attributes")
		return false, nil
	}
	m.attributes = append(m.attributes, attribute)
	m.logger.V(7).Info("Constraint satisfied by device", "device", deviceID, "numDevices", len(m.attributes))
	return true, nil
}

func (m *distinctAttributeConstraint) remove(requestName, subRequestName string, device *draapi.Device, deviceID DeviceID) {
	if m.requestNames.Len() > 0 && !m.matches(requestName, subRequestName) {
		// Device not affected by constraint.
		return
	}

	m.attributes = m.attributes[:len(m.attributes)-1]
	m.logger.V(7).Info("Device removed from constraint set", "device", deviceID, "numDevices", len(m.attributes))
}

func (m *distinctAttributeConstraint) matches(requestName, subRequestName string) bool {
	if subRequestName == "" {
		return m.requestNames.Has(requestName)
	} else {
		fullSubRequestName := fmt.Sprintf("%s/%s", requestName, subRequestName)
		return m.requestNames.Has(requestName) || m.requestNames.Has(fullSubRequestName)
	}
}

func (m *distinctAttributeConstraint) matchesAttribute(attribute any) bool {
	for _, attr := range m.attributes {
		switch existing := attr.(type) {
		case string:
			candidate, ok := attribute.(string)
			if !ok {
				m.logger.V(7).Info("Attribute types don't match", "existing", attr, "candidate", attribute)
				continue
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
				continue
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
				continue
			}
			if existing == candidate {
				m.logger.V(7).Info("Bool values duplicated")
				return false
			}
			m.logger.V(7).Info("Attribute values don't match", "existing", attr, "candidate", attribute)
		case semver.Version:
			candidate, ok := attribute.(semver.Version)
			if !ok {
				m.logger.V(7).Info("Attribute types don't match", "existing", attr, "candidate", attribute)
				continue
			}
			if existing.Equals(candidate) {
				m.logger.V(7).Info("Version values duplicated")
				return false
			}
			m.logger.V(7).Info("Attribute values don't match", "existing", attr, "candidate", attribute)
		default:
			// Dead code, existing values were checked already.
			return false
		}
	}
	// All distinct
	return true
}
