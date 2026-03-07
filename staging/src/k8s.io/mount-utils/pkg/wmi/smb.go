//go:build windows
// +build windows

/*
Copyright 2025 The Kubernetes Authors.

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

package wmi

import (
	"fmt"
	"strings"
)

const (
	MSFTSmbGlobalMappingClass = "MSFT_SmbGlobalMapping"
)

// Refer to https://learn.microsoft.com/en-us/previous-versions/windows/desktop/smb/msft-smbmapping
const (
	SmbMappingStatusOK uint32 = iota
	SmbMappingStatusPaused
	SmbMappingStatusDisconnected
	SmbMappingStatusNetworkError
	SmbMappingStatusConnecting
	SmbMappingStatusReconnecting
	SmbMappingStatusUnavailable
)

const (
	credentialDelimiter = ":"
)

func escapeUserName(userName string) string {
	// refer to https://github.com/PowerShell/PowerShell/blob/9303de597da55963a6e26a8fe164d0b256ca3d4d/src/Microsoft.PowerShell.Commands.Management/cimSupport/cmdletization/cim/cimConverter.cs#L169-L170
	userName = strings.ReplaceAll(userName, "\\", "\\\\")
	userName = strings.ReplaceAll(userName, credentialDelimiter, "\\"+credentialDelimiter)
	return userName
}

// QuerySmbGlobalMappingByRemotePath retrieves the SMB global mapping from its remote path.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM MSFT_SmbGlobalMapping
//
// Refer to https://pkg.go.dev/github.com/microsoft/wmi/server2019/root/microsoft/windows/smb#MSFT_SmbGlobalMapping
// for the WMI class definition.
func QuerySmbGlobalMappingByRemotePath(scope *Scope, remotePath string) (*COMDispatchObject, error) {
	q := NewQuery(MSFTSmbGlobalMappingClass).
		WithNamespace(WMINamespaceSmb).
		WithCondition("RemotePath", "=", remotePath)

	smb, err := QueryFirstObjectWithBuilder(scope, q)
	if err != nil {
		return nil, fmt.Errorf("failed to query SMB global mapping by remote path %s: %w", remotePath, err)
	}

	return smb, nil
}

// GetSmbGlobalMappingStatus returns the status of an SMB global mapping.
func GetSmbGlobalMappingStatus(smb *COMDispatchObject) (uint32, error) {
	return smb.GetUint32Property("Status")
}

// RemoveSmbGlobalMappingByRemotePath removes an SMB global mapping matching to the remote path.
//
// Refer to https://pkg.go.dev/github.com/microsoft/wmi/server2019/root/microsoft/windows/smb#MSFT_SmbGlobalMapping
// for the WMI class definition.
func RemoveSmbGlobalMappingByRemotePath(scope *Scope, remotePath string) error {
	q := NewQuery(MSFTSmbGlobalMappingClass).
		WithNamespace(WMINamespaceSmb).
		WithCondition("RemotePath", "=", remotePath)

	smb, err := QueryFirstObjectWithBuilder(scope, q)
	if err != nil {
		return fmt.Errorf("failed to query SMB global mapping by remote path %s: %w", remotePath, err)
	}

	result, err := smb.CallUint32("Remove", true)
	if err != nil {
		return fmt.Errorf("failed to remove SMB global mapping by remote path %s: %w", remotePath, err)
	}
	if result != 0 {
		return NewWMIError(MSFTSmbGlobalMappingClass, "Remove", smb.Dispatch(), result)
	}
	return nil
}

// NewSmbGlobalMapping creates a new SMB global mapping to the remote path.
//
// Refer to https://pkg.go.dev/github.com/microsoft/wmi/server2019/root/microsoft/windows/smb#MSFT_SmbGlobalMapping
// for the WMI class definition.
func NewSmbGlobalMapping(remotePath, username, password string, requirePrivacy bool) error {
	params := map[string]interface{}{
		"RemotePath":     remotePath,
		"RequirePrivacy": requirePrivacy,
	}
	if username != "" {
		// refer to https://github.com/PowerShell/PowerShell/blob/9303de597da55963a6e26a8fe164d0b256ca3d4d/src/Microsoft.PowerShell.Commands.Management/cimSupport/cmdletization/cim/cimConverter.cs#L166-L178
		// on how SMB credential is handled in PowerShell
		params["Credential"] = escapeUserName(username) + credentialDelimiter + password
	}

	result, _, err := CallMethodOnWMIClass(WMINamespaceSmb, MSFTSmbGlobalMappingClass, "Create", params, DiscardOutputParameter)
	if err != nil {
		return fmt.Errorf("failed to create SMB global mapping: %w", err)
	}
	if result != 0 {
		return NewWMIError(MSFTSmbGlobalMappingClass, "Create", nil, result)
	}
	return nil
}
