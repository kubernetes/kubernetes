//go:build windows

package hns

import (
	"encoding/json"

	"github.com/sirupsen/logrus"
)

// HNSNnvManagementMacAddress represents management mac address
// which needs to be excluded from VF reassignment
type HNSNnvManagementMacAddress struct {
	MacAddress string `json:",omitempty"`
}

// HNSNnvManagementMacList represents a list of management
// mac addresses for exclusion from VF reassignment
type HNSNnvManagementMacList struct {
	MacAddressList []HNSNnvManagementMacAddress `json:",omitempty"`
}

// HNSNnvManagementMacRequest makes a HNS call to modify/query NnvManagementMacList
func HNSNnvManagementMacRequest(method, path, request string) (*HNSNnvManagementMacList, error) {
	nnvManagementMacList := &HNSNnvManagementMacList{}
	err := hnsCall(method, "/accelnet/"+path, request, &nnvManagementMacList)
	if err != nil {
		return nil, err
	}
	return nnvManagementMacList, nil
}

// Set ManagementMacAddressList by sending "POST" NnvManagementMacRequest to HNS.
func (nnvManagementMacList *HNSNnvManagementMacList) Set() (*HNSNnvManagementMacList, error) {
	operation := "Set"
	title := "hcnshim::nnvManagementMacList::" + operation
	logrus.Debugf(title+" id=%s", nnvManagementMacList.MacAddressList)

	jsonString, err := json.Marshal(nnvManagementMacList)
	if err != nil {
		return nil, err
	}
	return HNSNnvManagementMacRequest("POST", "", string(jsonString))
}

// Get ManagementMacAddressList by sending "GET" NnvManagementMacRequest to HNS.
func GetNnvManagementMacAddressList() (*HNSNnvManagementMacList, error) {
	operation := "Get"
	title := "hcnshim::nnvManagementMacList::" + operation
	logrus.Debugf(title)
	return HNSNnvManagementMacRequest("GET", "", "")
}

// Delete ManagementMacAddressList by sending "DELETE" NnvManagementMacRequest to HNS.
func DeleteNnvManagementMacAddressList() (*HNSNnvManagementMacList, error) {
	operation := "Delete"
	title := "hcnshim::nnvManagementMacList::" + operation
	logrus.Debugf(title)
	return HNSNnvManagementMacRequest("DELETE", "", "")
}
