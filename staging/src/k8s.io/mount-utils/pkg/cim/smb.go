//go:build windows
// +build windows

package cim

import (
	"strings"

	"github.com/microsoft/wmi/pkg/base/query"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
)

// Refer to https://learn.microsoft.com/en-us/previous-versions/windows/desktop/smb/msft-smbmapping
const (
	SmbMappingStatusOK int32 = iota
	SmbMappingStatusPaused
	SmbMappingStatusDisconnected
	SmbMappingStatusNetworkError
	SmbMappingStatusConnecting
	SmbMappingStatusReconnecting
	SmbMappingStatusUnavailable

	credentialDelimiter = ":"
)

// escapeQueryParameter escapes a parameter for WMI Queries
func escapeQueryParameter(s string) string {
	s = strings.ReplaceAll(s, "'", "''")
	s = strings.ReplaceAll(s, "\\", "\\\\")
	return s
}

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
func QuerySmbGlobalMappingByRemotePath(remotePath string) (*cim.WmiInstance, error) {
	smbQuery := query.NewWmiQuery("MSFT_SmbGlobalMapping", "RemotePath", escapeQueryParameter(remotePath))
	instances, err := QueryInstances(WMINamespaceSmb, smbQuery)
	if err != nil {
		return nil, err
	}

	return instances[0], err
}

// GetSmbGlobalMappingStatus returns the status of an SMB global mapping.
func GetSmbGlobalMappingStatus(inst *cim.WmiInstance) (int32, error) {
	statusProp, err := inst.GetProperty("Status")
	if err != nil {
		return SmbMappingStatusUnavailable, err
	}

	return statusProp.(int32), nil
}

// RemoveSmbGlobalMappingByRemotePath removes an SMB global mapping matching to the remote path.
//
// Refer to https://pkg.go.dev/github.com/microsoft/wmi/server2019/root/microsoft/windows/smb#MSFT_SmbGlobalMapping
// for the WMI class definition.
func RemoveSmbGlobalMappingByRemotePath(remotePath string) error {
	smbQuery := query.NewWmiQuery("MSFT_SmbGlobalMapping", "RemotePath", escapeQueryParameter(remotePath))
	instances, err := QueryInstances(WMINamespaceSmb, smbQuery)
	if err != nil {
		return err
	}

	_, err = instances[0].InvokeMethod("Remove", true)
	return err
}

// NewSmbGlobalMapping creates a new SMB global mapping to the remote path.
//
// Refer to https://pkg.go.dev/github.com/microsoft/wmi/server2019/root/microsoft/windows/smb#MSFT_SmbGlobalMapping
// for the WMI class definition.
func NewSmbGlobalMapping(remotePath, username, password string, requirePrivacy bool) (int, error) {
	params := map[string]interface{}{
		"RemotePath":     remotePath,
		"RequirePrivacy": requirePrivacy,
	}
	if username != "" {
		// refer to https://github.com/PowerShell/PowerShell/blob/9303de597da55963a6e26a8fe164d0b256ca3d4d/src/Microsoft.PowerShell.Commands.Management/cimSupport/cmdletization/cim/cimConverter.cs#L166-L178
		// on how SMB credential is handled in PowerShell
		params["Credential"] = escapeUserName(username) + credentialDelimiter + password
	}

	result, _, err := InvokeCimMethod(WMINamespaceSmb, "MSFT_SmbGlobalMapping", "Create", params)
	return result, err
}
