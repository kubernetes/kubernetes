/*
Copyright The Kubernetes Authors.

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

package deviceattribute

import (
	"fmt"
	"regexp"
	"strings"

	resourceapi "k8s.io/api/resource/v1"
)

// GetMdevUUIDAttribute returns a DeviceAttribute with the mediated device (mdev) UUID
// as a string value.
//
// It returns an error if the mdev UUID is empty or is not in the canonical UUID format
// (8-4-4-4-12 hex digits), e.g., "123e4567-e89b-12d3-a456-426614174000".
func GetMdevUUIDAttribute(mdevUUID string) (DeviceAttribute, error) {
	if err := verifyMdevUUIDFormat(mdevUUID); err != nil {
		return DeviceAttribute{}, err
	}

	normalizedUUID := strings.ToLower(mdevUUID)

	attr := DeviceAttribute{
		Name:  StandardDeviceAttributeMdevUUID,
		Value: resourceapi.DeviceAttribute{StringValue: &normalizedUUID},
	}

	return attr, nil
}

// verifyMdevUUIDFormat verifies that the mdev UUID is in the canonical UUID format (8-4-4-4-12 hex digits).
func verifyMdevUUIDFormat(mdevUUID string) error {
	if mdevUUID == "" {
		return fmt.Errorf("mdev UUID cannot be empty")
	}

	uuidRegexp := regexp.MustCompile(`^(?i)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$`)
	if !uuidRegexp.MatchString(mdevUUID) {
		return fmt.Errorf("invalid mdev UUID format: %s", mdevUUID)
	}

	return nil
}
