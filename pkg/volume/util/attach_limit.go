/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"crypto/sha1"
	"encoding/hex"
)

// This file is a common place holder for volume limit utility constants
// shared between volume package and scheduler

const (
	// CSIAttachLimitPrefix defines prefix used for CSI volumes
	CSIAttachLimitPrefix = "attachable-volumes-csi-"

	// ResourceNameLengthLimit stores maximum allowed Length for a ResourceName
	ResourceNameLengthLimit = 63
)

// GetCSIAttachLimitKey returns limit key used for CSI volumes
func GetCSIAttachLimitKey(driverName string) string {
	csiPrefixLength := len(CSIAttachLimitPrefix)
	totalkeyLength := csiPrefixLength + len(driverName)
	if totalkeyLength >= ResourceNameLengthLimit {
		charsFromDriverName := driverName[:23]
		hash := sha1.New()
		hash.Write([]byte(driverName))
		hashed := hex.EncodeToString(hash.Sum(nil))
		hashed = hashed[:16]
		return CSIAttachLimitPrefix + charsFromDriverName + hashed
	}
	return CSIAttachLimitPrefix + driverName
}
