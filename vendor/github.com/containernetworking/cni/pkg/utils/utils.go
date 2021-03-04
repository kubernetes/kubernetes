// Copyright 2019 CNI authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package utils

import (
	"bytes"
	"fmt"
	"regexp"
	"unicode"

	"github.com/containernetworking/cni/pkg/types"
)

const (
	// cniValidNameChars is the regexp used to validate valid characters in
	// containerID and networkName
	cniValidNameChars = `[a-zA-Z0-9][a-zA-Z0-9_.\-]`

	// maxInterfaceNameLength is the length max of a valid interface name
	maxInterfaceNameLength = 15
)

var cniReg = regexp.MustCompile(`^` + cniValidNameChars + `*$`)

// ValidateContainerID will validate that the supplied containerID is not empty does not contain invalid characters
func ValidateContainerID(containerID string) *types.Error {

	if containerID == "" {
		return types.NewError(types.ErrUnknownContainer, "missing containerID", "")
	}
	if !cniReg.MatchString(containerID) {
		return types.NewError(types.ErrInvalidEnvironmentVariables, "invalid characters in containerID", containerID)
	}
	return nil
}

// ValidateNetworkName will validate that the supplied networkName does not contain invalid characters
func ValidateNetworkName(networkName string) *types.Error {

	if networkName == "" {
		return types.NewError(types.ErrInvalidNetworkConfig, "missing network name:", "")
	}
	if !cniReg.MatchString(networkName) {
		return types.NewError(types.ErrInvalidNetworkConfig, "invalid characters found in network name", networkName)
	}
	return nil
}

// ValidateInterfaceName will validate the interface name based on the three rules below
// 1. The name must not be empty
// 2. The name must be less than 16 characters
// 3. The name must not be "." or ".."
// 3. The name must not contain / or : or any whitespace characters
// ref to https://github.com/torvalds/linux/blob/master/net/core/dev.c#L1024
func ValidateInterfaceName(ifName string) *types.Error {
	if len(ifName) == 0 {
		return types.NewError(types.ErrInvalidEnvironmentVariables, "interface name is empty", "")
	}
	if len(ifName) > maxInterfaceNameLength {
		return types.NewError(types.ErrInvalidEnvironmentVariables, "interface name is too long", fmt.Sprintf("interface name should be less than %d characters", maxInterfaceNameLength+1))
	}
	if ifName == "." || ifName == ".." {
		return types.NewError(types.ErrInvalidEnvironmentVariables, "interface name is . or ..", "")
	}
	for _, r := range bytes.Runes([]byte(ifName)) {
		if r == '/' || r == ':' || unicode.IsSpace(r) {
			return types.NewError(types.ErrInvalidEnvironmentVariables, "interface name contains / or : or whitespace characters", "")
		}
	}

	return nil
}
