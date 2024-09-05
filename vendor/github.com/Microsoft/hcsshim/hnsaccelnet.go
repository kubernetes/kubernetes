//go:build windows

package hcsshim

import (
	"errors"

	"github.com/Microsoft/hcsshim/internal/hns"
)

// HNSNnvManagementMacAddress represents management mac address
// which needs to be excluded from VF reassignment
type HNSNnvManagementMacAddress = hns.HNSNnvManagementMacAddress

// HNSNnvManagementMacList represents a list of management
// mac addresses for exclusion from VF reassignment
type HNSNnvManagementMacList = hns.HNSNnvManagementMacList

var (
	ErrorEmptyMacAddressList = errors.New("management mac_address list is empty")
)

// SetNnvManagementMacAddresses sets a list of
// management mac addresses in hns for exclusion from VF reassignment.
func SetNnvManagementMacAddresses(managementMacAddresses []string) (*HNSNnvManagementMacList, error) {
	if len(managementMacAddresses) == 0 {
		return nil, ErrorEmptyMacAddressList
	}
	nnvManagementMacList := &HNSNnvManagementMacList{}
	for _, mac := range managementMacAddresses {
		nnvManagementMacList.MacAddressList = append(nnvManagementMacList.MacAddressList, HNSNnvManagementMacAddress{MacAddress: mac})
	}
	return nnvManagementMacList.Set()
}

// GetNnvManagementMacAddresses retrieves a list of
// management mac addresses in hns for exclusion from VF reassignment.
func GetNnvManagementMacAddresses() (*HNSNnvManagementMacList, error) {
	return hns.GetNnvManagementMacAddressList()
}

// DeleteNnvManagementMacAddresses delete list of
// management mac addresses in hns which are excluded from VF reassignment.
func DeleteNnvManagementMacAddresses() (*HNSNnvManagementMacList, error) {
	return hns.DeleteNnvManagementMacAddressList()
}
