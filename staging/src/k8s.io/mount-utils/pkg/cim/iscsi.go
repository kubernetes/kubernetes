//go:build windows
// +build windows

package cim

import (
	"fmt"
	"strconv"

	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/server2019/root/microsoft/windows/storage"
)

var (
	ISCSITargetPortalDefaultSelectorList = []string{"TargetPortalAddress", "TargetPortalPortNumber"}
)

// ListISCSITargetPortals retrieves a list of iSCSI target portals.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM MSFT_IscsiTargetPortal
//
// Refer to https://learn.microsoft.com/en-us/previous-versions/windows/desktop/iscsidisc/msft-iscsitargetportal
// for the WMI class definition.
func ListISCSITargetPortals(selectorList []string) ([]*storage.MSFT_iSCSITargetPortal, error) {
	q := query.NewWmiQueryWithSelectList("MSFT_IscsiTargetPortal", selectorList)
	instances, err := QueryInstances(WMINamespaceStorage, q)
	if IgnoreNotFound(err) != nil {
		return nil, err
	}

	var targetPortals []*storage.MSFT_iSCSITargetPortal
	for _, instance := range instances {
		portal, err := storage.NewMSFT_iSCSITargetPortalEx1(instance)
		if err != nil {
			return nil, fmt.Errorf("failed to query iSCSI target portal %v. error: %v", instance, err)
		}

		targetPortals = append(targetPortals, portal)
	}

	return targetPortals, nil
}

// QueryISCSITargetPortal retrieves information about a specific iSCSI target portal
// identified by its network address and port number.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM MSFT_IscsiTargetPortal
//	  WHERE TargetPortalAddress = '<address>'
//	    AND TargetPortalPortNumber = '<port>'
//
// Refer to https://learn.microsoft.com/en-us/previous-versions/windows/desktop/iscsidisc/msft-iscsitargetportal
// for the WMI class definition.
func QueryISCSITargetPortal(address string, port uint32, selectorList []string) (*storage.MSFT_iSCSITargetPortal, error) {
	portalQuery := query.NewWmiQueryWithSelectList(
		"MSFT_iSCSITargetPortal", selectorList,
		"TargetPortalAddress", address,
		"TargetPortalPortNumber", strconv.Itoa(int(port)))
	instances, err := QueryInstances(WMINamespaceStorage, portalQuery)
	if err != nil {
		return nil, err
	}

	targetPortal, err := storage.NewMSFT_iSCSITargetPortalEx1(instances[0])
	if err != nil {
		return nil, fmt.Errorf("failed to query iSCSI target portal at (%s:%d). error: %v", address, port, err)
	}

	return targetPortal, nil
}

// ListISCSITargetsByTargetPortalAddressAndPort retrieves ISCSI targets by address and port of an iSCSI target portal.
func ListISCSITargetsByTargetPortalAddressAndPort(address string, port uint32, selectorList []string) ([]*storage.MSFT_iSCSITarget, error) {
	instance, err := QueryISCSITargetPortal(address, port, selectorList)
	if err != nil {
		return nil, err
	}

	targets, err := ListISCSITargetsByTargetPortal([]*storage.MSFT_iSCSITargetPortal{instance})
	return targets, err
}

// NewISCSITargetPortal creates a new iSCSI target portal.
//
// Refer to https://learn.microsoft.com/en-us/previous-versions/windows/desktop/iscsidisc/msft-iscsitargetportal-new
// for the WMI method definition.
func NewISCSITargetPortal(targetPortalAddress string,
	targetPortalPortNumber uint32,
	initiatorInstanceName *string,
	initiatorPortalAddress *string,
	isHeaderDigest *bool,
	isDataDigest *bool) (*storage.MSFT_iSCSITargetPortal, error) {
	params := map[string]interface{}{
		"TargetPortalAddress":    targetPortalAddress,
		"TargetPortalPortNumber": targetPortalPortNumber,
	}
	if initiatorInstanceName != nil {
		params["InitiatorInstanceName"] = *initiatorInstanceName
	}
	if initiatorPortalAddress != nil {
		params["InitiatorPortalAddress"] = *initiatorPortalAddress
	}
	if isHeaderDigest != nil {
		params["IsHeaderDigest"] = *isHeaderDigest
	}
	if isDataDigest != nil {
		params["IsDataDigest"] = *isDataDigest
	}
	result, _, err := InvokeCimMethod(WMINamespaceStorage, "MSFT_iSCSITargetPortal", "New", params)
	if err != nil {
		return nil, fmt.Errorf("failed to create iSCSI target portal with %v. result: %d, error: %v", params, result, err)
	}

	return QueryISCSITargetPortal(targetPortalAddress, targetPortalPortNumber, nil)
}

// ParseISCSITargetPortal retrieves the portal address and port number of an iSCSI target portal.
func ParseISCSITargetPortal(instance *storage.MSFT_iSCSITargetPortal) (string, uint32, error) {
	portalAddress, err := instance.GetPropertyTargetPortalAddress()
	if err != nil {
		return "", 0, fmt.Errorf("failed parsing target portal address %v. err: %w", instance, err)
	}

	portalPort, err := instance.GetProperty("TargetPortalPortNumber")
	if err != nil {
		return "", 0, fmt.Errorf("failed parsing target portal port number %v. err: %w", instance, err)
	}

	return portalAddress, uint32(portalPort.(int32)), nil
}

// RemoveISCSITargetPortal removes an iSCSI target portal.
//
// Refer to https://learn.microsoft.com/en-us/previous-versions/windows/desktop/iscsidisc/msft-iscsitargetportal-remove
// for the WMI method definition.
func RemoveISCSITargetPortal(instance *storage.MSFT_iSCSITargetPortal) (int, error) {
	address, port, err := ParseISCSITargetPortal(instance)
	if err != nil {
		return 0, fmt.Errorf("failed to parse target portal %v. error: %v", instance, err)
	}

	result, err := instance.InvokeMethodWithReturn("Remove",
		nil,
		nil,
		int(port),
		address,
	)
	return int(result), err
}

// ListISCSITargetsByTargetPortal retrieves all iSCSI targets from the specified iSCSI target portal
// using MSFT_iSCSITargetToiSCSITargetPortal association.
//
// WMI association MSFT_iSCSITargetToiSCSITargetPortal:
//
//	iSCSITarget                                                                  | iSCSITargetPortal
//	-----------                                                                  | -----------------
//	MSFT_iSCSITarget (NodeAddress = "iqn.1991-05.com.microsoft:win-8e2evaq9q...) | MSFT_iSCSITargetPortal (TargetPortalAdd...
//
// Refer to https://learn.microsoft.com/en-us/previous-versions/windows/desktop/iscsidisc/msft-iscsitarget
// for the WMI class definition.
func ListISCSITargetsByTargetPortal(portals []*storage.MSFT_iSCSITargetPortal) ([]*storage.MSFT_iSCSITarget, error) {
	var targets []*storage.MSFT_iSCSITarget
	for _, portal := range portals {
		collection, err := portal.GetAssociated("MSFT_iSCSITargetToiSCSITargetPortal", "MSFT_iSCSITarget", "iSCSITarget", "iSCSITargetPortal")
		if err != nil {
			return nil, fmt.Errorf("failed to query associated iSCSITarget for %v. error: %v", portal, err)
		}

		for _, instance := range collection {
			target, err := storage.NewMSFT_iSCSITargetEx1(instance)
			if err != nil {
				return nil, fmt.Errorf("failed to query iSCSI target %v. error: %v", instance, err)
			}

			targets = append(targets, target)
		}
	}

	return targets, nil
}

// QueryISCSITarget retrieves the iSCSI target from the specified portal address, portal and node address.
func QueryISCSITarget(address string, port uint32, nodeAddress string) (*storage.MSFT_iSCSITarget, error) {
	portal, err := QueryISCSITargetPortal(address, port, nil)
	if err != nil {
		return nil, err
	}

	targets, err := ListISCSITargetsByTargetPortal([]*storage.MSFT_iSCSITargetPortal{portal})
	if err != nil {
		return nil, err
	}

	for _, target := range targets {
		targetNodeAddress, err := GetISCSITargetNodeAddress(target)
		if err != nil {
			return nil, fmt.Errorf("failed to query iSCSI target %v. error: %v", target, err)
		}

		if targetNodeAddress == nodeAddress {
			return target, nil
		}
	}

	return nil, nil
}

// GetISCSITargetNodeAddress returns the node address of an iSCSI target.
func GetISCSITargetNodeAddress(target *storage.MSFT_iSCSITarget) (string, error) {
	nodeAddress, err := target.GetProperty("NodeAddress")
	if err != nil {
		return "", err
	}

	return nodeAddress.(string), err
}

// IsISCSITargetConnected returns whether the iSCSI target is connected.
func IsISCSITargetConnected(target *storage.MSFT_iSCSITarget) (bool, error) {
	return target.GetPropertyIsConnected()
}

// QueryISCSISessionByTarget retrieves the iSCSI session from the specified iSCSI target
// using MSFT_iSCSITargetToiSCSISession association.
//
// WMI association MSFT_iSCSITargetToiSCSISession:
//
//	iSCSISession                                                                | iSCSITarget
//	------------                                                                | -----------
//	MSFT_iSCSISession (SessionIdentifier = "ffffac0cacbff010-4000013700000016") | MSFT_iSCSITarget (NodeAddress = "iqn.199...
//
// Refer to https://learn.microsoft.com/en-us/previous-versions/windows/desktop/iscsidisc/msft-iscsisession
// for the WMI class definition.
func QueryISCSISessionByTarget(target *storage.MSFT_iSCSITarget) (*storage.MSFT_iSCSISession, error) {
	collection, err := target.GetAssociated("MSFT_iSCSITargetToiSCSISession", "MSFT_iSCSISession", "iSCSISession", "iSCSITarget")
	if err != nil {
		return nil, fmt.Errorf("failed to query associated iSCSISession for %v. error: %v", target, err)
	}

	if len(collection) == 0 {
		return nil, nil
	}

	session, err := storage.NewMSFT_iSCSISessionEx1(collection[0])
	return session, err
}

// UnregisterISCSISession unregisters the iSCSI session so that it is no longer persistent.
//
// Refer https://learn.microsoft.com/en-us/previous-versions/windows/desktop/iscsidisc/msft-iscsisession-unregister
// for the WMI method definition.
func UnregisterISCSISession(session *storage.MSFT_iSCSISession) (int, error) {
	result, err := session.InvokeMethodWithReturn("Unregister")
	return int(result), err
}

// SetISCSISessionChapSecret sets a CHAP secret key for use with iSCSI initiator connections.
//
// Refer https://learn.microsoft.com/en-us/previous-versions/windows/desktop/iscsidisc/msft-iscsitarget-disconnect
// for the WMI method definition.
func SetISCSISessionChapSecret(mutualChapSecret string) (int, error) {
	result, _, err := InvokeCimMethod(WMINamespaceStorage, "MSFT_iSCSISession", "SetCHAPSecret", map[string]interface{}{"ChapSecret": mutualChapSecret})
	return result, err
}

// GetISCSISessionIdentifier returns the identifier of an iSCSI session.
func GetISCSISessionIdentifier(session *storage.MSFT_iSCSISession) (string, error) {
	return session.GetPropertySessionIdentifier()
}

// IsISCSISessionPersistent returns whether an iSCSI session is persistent.
func IsISCSISessionPersistent(session *storage.MSFT_iSCSISession) (bool, error) {
	return session.GetPropertyIsPersistent()
}

// ListDisksByTarget find all disks associated with an iSCSITarget.
// It finds out the iSCSIConnections from MSFT_iSCSITargetToiSCSIConnection association,
// then locate MSFT_Disk objects from MSFT_iSCSIConnectionToDisk association.
//
// WMI association MSFT_iSCSITargetToiSCSIConnection:
//
//	iSCSIConnection                                                     | iSCSITarget
//	---------------                                                     | -----------
//	MSFT_iSCSIConnection (ConnectionIdentifier = "ffffac0cacbff010-15") | MSFT_iSCSITarget (NodeAddress = "iqn.1991-05.com...
//
// WMI association MSFT_iSCSIConnectionToDisk:
//
//	Disk                                                               | iSCSIConnection
//	----                                                               | ---------------
//	MSFT_Disk (ObjectId = "{1}\\WIN-8E2EVAQ9QSB\root/Microsoft/Win...) | MSFT_iSCSIConnection (ConnectionIdentifier = "fff...
//
// Refer to https://learn.microsoft.com/en-us/previous-versions/windows/desktop/iscsidisc/msft-iscsiconnection
// for the WMI class definition.
func ListDisksByTarget(target *storage.MSFT_iSCSITarget) ([]*storage.MSFT_Disk, error) {
	// list connections to the given iSCSI target
	collection, err := target.GetAssociated("MSFT_iSCSITargetToiSCSIConnection", "MSFT_iSCSIConnection", "iSCSIConnection", "iSCSITarget")
	if err != nil {
		return nil, fmt.Errorf("failed to query associated iSCSISession for %v. error: %v", target, err)
	}

	if len(collection) == 0 {
		return nil, nil
	}

	var result []*storage.MSFT_Disk
	for _, conn := range collection {
		instances, err := conn.GetAssociated("MSFT_iSCSIConnectionToDisk", "MSFT_Disk", "Disk", "iSCSIConnection")
		if err != nil {
			return nil, fmt.Errorf("failed to query associated disk for %v. error: %v", target, err)
		}

		for _, instance := range instances {
			disk, err := storage.NewMSFT_DiskEx1(instance)
			if err != nil {
				return nil, fmt.Errorf("failed to query associated disk %v. error: %v", instance, err)
			}

			result = append(result, disk)
		}
	}

	return result, err
}

// ConnectISCSITarget establishes a connection to an iSCSI target with optional CHAP authentication credential.
//
// Refer https://learn.microsoft.com/en-us/previous-versions/windows/desktop/iscsidisc/msft-iscsitarget-connect
// for the WMI method definition.
func ConnectISCSITarget(portalAddress string, portalPortNumber uint32, nodeAddress string, authType string, chapUsername *string, chapSecret *string) (int, error) {
	inParams := map[string]interface{}{
		"NodeAddress":            nodeAddress,
		"TargetPortalAddress":    portalAddress,
		"TargetPortalPortNumber": int(portalPortNumber),
		"AuthenticationType":     authType,
	}
	// InitiatorPortalAddress
	// IsDataDigest
	// IsHeaderDigest
	// ReportToPnP
	if chapUsername != nil {
		inParams["ChapUsername"] = *chapUsername
	}
	if chapSecret != nil {
		inParams["ChapSecret"] = *chapSecret
	}

	result, _, err := InvokeCimMethod(WMINamespaceStorage, "MSFT_iSCSITarget", "Connect", inParams)
	return result, err
}

// DisconnectISCSITarget disconnects the specified session between an iSCSI initiator and an iSCSI target.
//
// Refer https://learn.microsoft.com/en-us/previous-versions/windows/desktop/iscsidisc/msft-iscsitarget-disconnect
// for the WMI method definition.
func DisconnectISCSITarget(target *storage.MSFT_iSCSITarget, sessionIdentifier string) (int, error) {
	result, err := target.InvokeMethodWithReturn("Disconnect", sessionIdentifier)
	return int(result), err
}
