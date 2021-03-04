package hns

import (
	"encoding/json"
	"net"
	"strings"

	"github.com/sirupsen/logrus"
)

// HNSEndpoint represents a network endpoint in HNS
type HNSEndpoint struct {
	Id                 string            `json:"ID,omitempty"`
	Name               string            `json:",omitempty"`
	VirtualNetwork     string            `json:",omitempty"`
	VirtualNetworkName string            `json:",omitempty"`
	Policies           []json.RawMessage `json:",omitempty"`
	MacAddress         string            `json:",omitempty"`
	IPAddress          net.IP            `json:",omitempty"`
	IPv6Address        net.IP            `json:",omitempty"`
	DNSSuffix          string            `json:",omitempty"`
	DNSServerList      string            `json:",omitempty"`
	GatewayAddress     string            `json:",omitempty"`
	GatewayAddressV6   string            `json:",omitempty"`
	EnableInternalDNS  bool              `json:",omitempty"`
	DisableICC         bool              `json:",omitempty"`
	PrefixLength       uint8             `json:",omitempty"`
	IPv6PrefixLength   uint8             `json:",omitempty"`
	IsRemoteEndpoint   bool              `json:",omitempty"`
	EnableLowMetric    bool              `json:",omitempty"`
	Namespace          *Namespace        `json:",omitempty"`
	EncapOverhead      uint16            `json:",omitempty"`
}

//SystemType represents the type of the system on which actions are done
type SystemType string

// SystemType const
const (
	ContainerType      SystemType = "Container"
	VirtualMachineType SystemType = "VirtualMachine"
	HostType           SystemType = "Host"
)

// EndpointAttachDetachRequest is the structure used to send request to the container to modify the system
// Supported resource types are Network and Request Types are Add/Remove
type EndpointAttachDetachRequest struct {
	ContainerID    string     `json:"ContainerId,omitempty"`
	SystemType     SystemType `json:"SystemType"`
	CompartmentID  uint16     `json:"CompartmentId,omitempty"`
	VirtualNICName string     `json:"VirtualNicName,omitempty"`
}

// EndpointResquestResponse is object to get the endpoint request response
type EndpointResquestResponse struct {
	Success bool
	Error   string
}

// HNSEndpointRequest makes a HNS call to modify/query a network endpoint
func HNSEndpointRequest(method, path, request string) (*HNSEndpoint, error) {
	endpoint := &HNSEndpoint{}
	err := hnsCall(method, "/endpoints/"+path, request, &endpoint)
	if err != nil {
		return nil, err
	}

	return endpoint, nil
}

// HNSListEndpointRequest makes a HNS call to query the list of available endpoints
func HNSListEndpointRequest() ([]HNSEndpoint, error) {
	var endpoint []HNSEndpoint
	err := hnsCall("GET", "/endpoints/", "", &endpoint)
	if err != nil {
		return nil, err
	}

	return endpoint, nil
}

// GetHNSEndpointByID get the Endpoint by ID
func GetHNSEndpointByID(endpointID string) (*HNSEndpoint, error) {
	return HNSEndpointRequest("GET", endpointID, "")
}

// GetHNSEndpointByName gets the endpoint filtered by Name
func GetHNSEndpointByName(endpointName string) (*HNSEndpoint, error) {
	hnsResponse, err := HNSListEndpointRequest()
	if err != nil {
		return nil, err
	}
	for _, hnsEndpoint := range hnsResponse {
		if hnsEndpoint.Name == endpointName {
			return &hnsEndpoint, nil
		}
	}
	return nil, EndpointNotFoundError{EndpointName: endpointName}
}

type endpointAttachInfo struct {
	SharedContainers json.RawMessage `json:",omitempty"`
}

func (endpoint *HNSEndpoint) IsAttached(vID string) (bool, error) {
	attachInfo := endpointAttachInfo{}
	err := hnsCall("GET", "/endpoints/"+endpoint.Id, "", &attachInfo)

	// Return false allows us to just return the err
	if err != nil {
		return false, err
	}

	if strings.Contains(strings.ToLower(string(attachInfo.SharedContainers)), strings.ToLower(vID)) {
		return true, nil
	}

	return false, nil

}

// Create Endpoint by sending EndpointRequest to HNS. TODO: Create a separate HNS interface to place all these methods
func (endpoint *HNSEndpoint) Create() (*HNSEndpoint, error) {
	operation := "Create"
	title := "hcsshim::HNSEndpoint::" + operation
	logrus.Debugf(title+" id=%s", endpoint.Id)

	jsonString, err := json.Marshal(endpoint)
	if err != nil {
		return nil, err
	}
	return HNSEndpointRequest("POST", "", string(jsonString))
}

// Delete Endpoint by sending EndpointRequest to HNS
func (endpoint *HNSEndpoint) Delete() (*HNSEndpoint, error) {
	operation := "Delete"
	title := "hcsshim::HNSEndpoint::" + operation
	logrus.Debugf(title+" id=%s", endpoint.Id)

	return HNSEndpointRequest("DELETE", endpoint.Id, "")
}

// Update Endpoint
func (endpoint *HNSEndpoint) Update() (*HNSEndpoint, error) {
	operation := "Update"
	title := "hcsshim::HNSEndpoint::" + operation
	logrus.Debugf(title+" id=%s", endpoint.Id)
	jsonString, err := json.Marshal(endpoint)
	if err != nil {
		return nil, err
	}
	err = hnsCall("POST", "/endpoints/"+endpoint.Id, string(jsonString), &endpoint)

	return endpoint, err
}

// ApplyACLPolicy applies a set of ACL Policies on the Endpoint
func (endpoint *HNSEndpoint) ApplyACLPolicy(policies ...*ACLPolicy) error {
	operation := "ApplyACLPolicy"
	title := "hcsshim::HNSEndpoint::" + operation
	logrus.Debugf(title+" id=%s", endpoint.Id)

	for _, policy := range policies {
		if policy == nil {
			continue
		}
		jsonString, err := json.Marshal(policy)
		if err != nil {
			return err
		}
		endpoint.Policies = append(endpoint.Policies, jsonString)
	}

	_, err := endpoint.Update()
	return err
}

// ApplyProxyPolicy applies a set of Proxy Policies on the Endpoint
func (endpoint *HNSEndpoint) ApplyProxyPolicy(policies ...*ProxyPolicy) error {
	operation := "ApplyProxyPolicy"
	title := "hcsshim::HNSEndpoint::" + operation
	logrus.Debugf(title+" id=%s", endpoint.Id)

	for _, policy := range policies {
		if policy == nil {
			continue
		}
		jsonString, err := json.Marshal(policy)
		if err != nil {
			return err
		}
		endpoint.Policies = append(endpoint.Policies, jsonString)
	}

	_, err := endpoint.Update()
	return err
}

// ContainerAttach attaches an endpoint to container
func (endpoint *HNSEndpoint) ContainerAttach(containerID string, compartmentID uint16) error {
	operation := "ContainerAttach"
	title := "hcsshim::HNSEndpoint::" + operation
	logrus.Debugf(title+" id=%s", endpoint.Id)

	requestMessage := &EndpointAttachDetachRequest{
		ContainerID:   containerID,
		CompartmentID: compartmentID,
		SystemType:    ContainerType,
	}
	response := &EndpointResquestResponse{}
	jsonString, err := json.Marshal(requestMessage)
	if err != nil {
		return err
	}
	return hnsCall("POST", "/endpoints/"+endpoint.Id+"/attach", string(jsonString), &response)
}

// ContainerDetach detaches an endpoint from container
func (endpoint *HNSEndpoint) ContainerDetach(containerID string) error {
	operation := "ContainerDetach"
	title := "hcsshim::HNSEndpoint::" + operation
	logrus.Debugf(title+" id=%s", endpoint.Id)

	requestMessage := &EndpointAttachDetachRequest{
		ContainerID: containerID,
		SystemType:  ContainerType,
	}
	response := &EndpointResquestResponse{}

	jsonString, err := json.Marshal(requestMessage)
	if err != nil {
		return err
	}
	return hnsCall("POST", "/endpoints/"+endpoint.Id+"/detach", string(jsonString), &response)
}

// HostAttach attaches a nic on the host
func (endpoint *HNSEndpoint) HostAttach(compartmentID uint16) error {
	operation := "HostAttach"
	title := "hcsshim::HNSEndpoint::" + operation
	logrus.Debugf(title+" id=%s", endpoint.Id)
	requestMessage := &EndpointAttachDetachRequest{
		CompartmentID: compartmentID,
		SystemType:    HostType,
	}
	response := &EndpointResquestResponse{}

	jsonString, err := json.Marshal(requestMessage)
	if err != nil {
		return err
	}
	return hnsCall("POST", "/endpoints/"+endpoint.Id+"/attach", string(jsonString), &response)

}

// HostDetach detaches a nic on the host
func (endpoint *HNSEndpoint) HostDetach() error {
	operation := "HostDetach"
	title := "hcsshim::HNSEndpoint::" + operation
	logrus.Debugf(title+" id=%s", endpoint.Id)
	requestMessage := &EndpointAttachDetachRequest{
		SystemType: HostType,
	}
	response := &EndpointResquestResponse{}

	jsonString, err := json.Marshal(requestMessage)
	if err != nil {
		return err
	}
	return hnsCall("POST", "/endpoints/"+endpoint.Id+"/detach", string(jsonString), &response)
}

// VirtualMachineNICAttach attaches a endpoint to a virtual machine
func (endpoint *HNSEndpoint) VirtualMachineNICAttach(virtualMachineNICName string) error {
	operation := "VirtualMachineNicAttach"
	title := "hcsshim::HNSEndpoint::" + operation
	logrus.Debugf(title+" id=%s", endpoint.Id)
	requestMessage := &EndpointAttachDetachRequest{
		VirtualNICName: virtualMachineNICName,
		SystemType:     VirtualMachineType,
	}
	response := &EndpointResquestResponse{}

	jsonString, err := json.Marshal(requestMessage)
	if err != nil {
		return err
	}
	return hnsCall("POST", "/endpoints/"+endpoint.Id+"/attach", string(jsonString), &response)
}

// VirtualMachineNICDetach detaches a endpoint  from a virtual machine
func (endpoint *HNSEndpoint) VirtualMachineNICDetach() error {
	operation := "VirtualMachineNicDetach"
	title := "hcsshim::HNSEndpoint::" + operation
	logrus.Debugf(title+" id=%s", endpoint.Id)

	requestMessage := &EndpointAttachDetachRequest{
		SystemType: VirtualMachineType,
	}
	response := &EndpointResquestResponse{}

	jsonString, err := json.Marshal(requestMessage)
	if err != nil {
		return err
	}
	return hnsCall("POST", "/endpoints/"+endpoint.Id+"/detach", string(jsonString), &response)
}
