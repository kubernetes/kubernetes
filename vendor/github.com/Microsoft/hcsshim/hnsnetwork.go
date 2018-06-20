package hcsshim

import (
	"encoding/json"
	"net"

	"github.com/sirupsen/logrus"
)

// Subnet is assoicated with a network and represents a list
// of subnets available to the network
type Subnet struct {
	AddressPrefix  string            `json:",omitempty"`
	GatewayAddress string            `json:",omitempty"`
	Policies       []json.RawMessage `json:",omitempty"`
}

// MacPool is assoicated with a network and represents a list
// of macaddresses available to the network
type MacPool struct {
	StartMacAddress string `json:",omitempty"`
	EndMacAddress   string `json:",omitempty"`
}

// HNSNetwork represents a network in HNS
type HNSNetwork struct {
	Id                   string            `json:"ID,omitempty"`
	Name                 string            `json:",omitempty"`
	Type                 string            `json:",omitempty"`
	NetworkAdapterName   string            `json:",omitempty"`
	SourceMac            string            `json:",omitempty"`
	Policies             []json.RawMessage `json:",omitempty"`
	MacPools             []MacPool         `json:",omitempty"`
	Subnets              []Subnet          `json:",omitempty"`
	DNSSuffix            string            `json:",omitempty"`
	DNSServerList        string            `json:",omitempty"`
	DNSServerCompartment uint32            `json:",omitempty"`
	ManagementIP         string            `json:",omitempty"`
	AutomaticDNS         bool              `json:",omitempty"`
}

type hnsNetworkResponse struct {
	Success bool
	Error   string
	Output  HNSNetwork
}

type hnsResponse struct {
	Success bool
	Error   string
	Output  json.RawMessage
}

// HNSNetworkRequest makes a call into HNS to update/query a single network
func HNSNetworkRequest(method, path, request string) (*HNSNetwork, error) {
	var network HNSNetwork
	err := hnsCall(method, "/networks/"+path, request, &network)
	if err != nil {
		return nil, err
	}

	return &network, nil
}

// HNSListNetworkRequest makes a HNS call to query the list of available networks
func HNSListNetworkRequest(method, path, request string) ([]HNSNetwork, error) {
	var network []HNSNetwork
	err := hnsCall(method, "/networks/"+path, request, &network)
	if err != nil {
		return nil, err
	}

	return network, nil
}

// GetHNSNetworkByID
func GetHNSNetworkByID(networkID string) (*HNSNetwork, error) {
	return HNSNetworkRequest("GET", networkID, "")
}

// GetHNSNetworkName filtered by Name
func GetHNSNetworkByName(networkName string) (*HNSNetwork, error) {
	hsnnetworks, err := HNSListNetworkRequest("GET", "", "")
	if err != nil {
		return nil, err
	}
	for _, hnsnetwork := range hsnnetworks {
		if hnsnetwork.Name == networkName {
			return &hnsnetwork, nil
		}
	}
	return nil, NetworkNotFoundError{NetworkName: networkName}
}

// Create Network by sending NetworkRequest to HNS.
func (network *HNSNetwork) Create() (*HNSNetwork, error) {
	operation := "Create"
	title := "HCSShim::HNSNetwork::" + operation
	logrus.Debugf(title+" id=%s", network.Id)

	jsonString, err := json.Marshal(network)
	if err != nil {
		return nil, err
	}
	return HNSNetworkRequest("POST", "", string(jsonString))
}

// Delete Network by sending NetworkRequest to HNS
func (network *HNSNetwork) Delete() (*HNSNetwork, error) {
	operation := "Delete"
	title := "HCSShim::HNSNetwork::" + operation
	logrus.Debugf(title+" id=%s", network.Id)

	return HNSNetworkRequest("DELETE", network.Id, "")
}

// Creates an endpoint on the Network.
func (network *HNSNetwork) NewEndpoint(ipAddress net.IP, macAddress net.HardwareAddr) *HNSEndpoint {
	return &HNSEndpoint{
		VirtualNetwork: network.Id,
		IPAddress:      ipAddress,
		MacAddress:     string(macAddress),
	}
}

func (network *HNSNetwork) CreateEndpoint(endpoint *HNSEndpoint) (*HNSEndpoint, error) {
	operation := "CreateEndpoint"
	title := "HCSShim::HNSNetwork::" + operation
	logrus.Debugf(title+" id=%s, endpointId=%s", network.Id, endpoint.Id)

	endpoint.VirtualNetwork = network.Id
	return endpoint.Create()
}

func (network *HNSNetwork) CreateRemoteEndpoint(endpoint *HNSEndpoint) (*HNSEndpoint, error) {
	operation := "CreateRemoteEndpoint"
	title := "HCSShim::HNSNetwork::" + operation
	logrus.Debugf(title+" id=%s", network.Id)
	endpoint.IsRemoteEndpoint = true
	return network.CreateEndpoint(endpoint)
}
