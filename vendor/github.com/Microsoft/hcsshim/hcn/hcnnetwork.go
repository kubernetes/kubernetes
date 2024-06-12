//go:build windows

package hcn

import (
	"encoding/json"
	"errors"

	"github.com/Microsoft/go-winio/pkg/guid"
	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/sirupsen/logrus"
)

// Route is associated with a subnet.
type Route struct {
	NextHop           string `json:",omitempty"`
	DestinationPrefix string `json:",omitempty"`
	Metric            uint16 `json:",omitempty"`
}

// Subnet is associated with a Ipam.
type Subnet struct {
	IpAddressPrefix string            `json:",omitempty"`
	Policies        []json.RawMessage `json:",omitempty"`
	Routes          []Route           `json:",omitempty"`
}

// Ipam (Internet Protocol Address Management) is associated with a network
// and represents the address space(s) of a network.
type Ipam struct {
	Type    string   `json:",omitempty"` // Ex: Static, DHCP
	Subnets []Subnet `json:",omitempty"`
}

// MacRange is associated with MacPool and respresents the start and end addresses.
type MacRange struct {
	StartMacAddress string `json:",omitempty"`
	EndMacAddress   string `json:",omitempty"`
}

// MacPool is associated with a network and represents pool of MacRanges.
type MacPool struct {
	Ranges []MacRange `json:",omitempty"`
}

// Dns (Domain Name System is associated with a network).
type Dns struct {
	Domain     string   `json:",omitempty"`
	Search     []string `json:",omitempty"`
	ServerList []string `json:",omitempty"`
	Options    []string `json:",omitempty"`
}

// NetworkType are various networks.
type NetworkType string

// NetworkType const
const (
	NAT         NetworkType = "NAT"
	Transparent NetworkType = "Transparent"
	L2Bridge    NetworkType = "L2Bridge"
	L2Tunnel    NetworkType = "L2Tunnel"
	ICS         NetworkType = "ICS"
	Private     NetworkType = "Private"
	Overlay     NetworkType = "Overlay"
)

// NetworkFlags are various network flags.
type NetworkFlags uint32

// NetworkFlags const
const (
	None                NetworkFlags = 0
	EnableNonPersistent NetworkFlags = 8
	DisableHostPort     NetworkFlags = 1024
)

// HostComputeNetwork represents a network
type HostComputeNetwork struct {
	Id            string          `json:"ID,omitempty"`
	Name          string          `json:",omitempty"`
	Type          NetworkType     `json:",omitempty"`
	Policies      []NetworkPolicy `json:",omitempty"`
	MacPool       MacPool         `json:",omitempty"`
	Dns           Dns             `json:",omitempty"`
	Ipams         []Ipam          `json:",omitempty"`
	Flags         NetworkFlags    `json:",omitempty"` // 0: None
	Health        Health          `json:",omitempty"`
	SchemaVersion SchemaVersion   `json:",omitempty"`
}

// NetworkResourceType are the 3 different Network settings resources.
type NetworkResourceType string

var (
	// NetworkResourceTypePolicy is for Network's policies. Ex: RemoteSubnet
	NetworkResourceTypePolicy NetworkResourceType = "Policy"
	// NetworkResourceTypeDNS is for Network's DNS settings.
	NetworkResourceTypeDNS NetworkResourceType = "DNS"
	// NetworkResourceTypeExtension is for Network's extension settings.
	NetworkResourceTypeExtension NetworkResourceType = "Extension"
)

// ModifyNetworkSettingRequest is the structure used to send request to modify an network.
// Used to update DNS/extension/policy on an network.
type ModifyNetworkSettingRequest struct {
	ResourceType NetworkResourceType `json:",omitempty"` // Policy, DNS, Extension
	RequestType  RequestType         `json:",omitempty"` // Add, Remove, Update, Refresh
	Settings     json.RawMessage     `json:",omitempty"`
}

type PolicyNetworkRequest struct {
	Policies []NetworkPolicy `json:",omitempty"`
}

func getNetwork(networkGUID guid.GUID, query string) (*HostComputeNetwork, error) {
	// Open network.
	var (
		networkHandle    hcnNetwork
		resultBuffer     *uint16
		propertiesBuffer *uint16
	)
	hr := hcnOpenNetwork(&networkGUID, &networkHandle, &resultBuffer)
	if err := checkForErrors("hcnOpenNetwork", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query network.
	hr = hcnQueryNetworkProperties(networkHandle, query, &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryNetworkProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close network.
	hr = hcnCloseNetwork(networkHandle)
	if err := checkForErrors("hcnCloseNetwork", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to HostComputeNetwork
	var outputNetwork HostComputeNetwork

	// If HNS sets the network type to NAT (i.e. '0' in HNS.Schema.Network.NetworkMode),
	// the value will be omitted from the JSON blob. We therefore need to initialize NAT here before
	// unmarshaling the JSON blob.
	outputNetwork.Type = NAT

	if err := json.Unmarshal([]byte(properties), &outputNetwork); err != nil {
		return nil, err
	}
	return &outputNetwork, nil
}

func enumerateNetworks(query string) ([]HostComputeNetwork, error) {
	// Enumerate all Network Guids
	var (
		resultBuffer  *uint16
		networkBuffer *uint16
	)
	hr := hcnEnumerateNetworks(query, &networkBuffer, &resultBuffer)
	if err := checkForErrors("hcnEnumerateNetworks", hr, resultBuffer); err != nil {
		return nil, err
	}

	networks := interop.ConvertAndFreeCoTaskMemString(networkBuffer)
	var networkIds []guid.GUID
	if err := json.Unmarshal([]byte(networks), &networkIds); err != nil {
		return nil, err
	}

	var outputNetworks []HostComputeNetwork
	for _, networkGUID := range networkIds {
		network, err := getNetwork(networkGUID, query)
		if err != nil {
			return nil, err
		}
		outputNetworks = append(outputNetworks, *network)
	}
	return outputNetworks, nil
}

func createNetwork(settings string) (*HostComputeNetwork, error) {
	// Create new network.
	var (
		networkHandle    hcnNetwork
		resultBuffer     *uint16
		propertiesBuffer *uint16
	)
	networkGUID := guid.GUID{}
	hr := hcnCreateNetwork(&networkGUID, settings, &networkHandle, &resultBuffer)
	if err := checkForErrors("hcnCreateNetwork", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query network.
	hcnQuery := defaultQuery()
	query, err := json.Marshal(hcnQuery)
	if err != nil {
		return nil, err
	}
	hr = hcnQueryNetworkProperties(networkHandle, string(query), &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryNetworkProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close network.
	hr = hcnCloseNetwork(networkHandle)
	if err := checkForErrors("hcnCloseNetwork", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to HostComputeNetwork
	var outputNetwork HostComputeNetwork

	// If HNS sets the network type to NAT (i.e. '0' in HNS.Schema.Network.NetworkMode),
	// the value will be omitted from the JSON blob. We therefore need to initialize NAT here before
	// unmarshaling the JSON blob.
	outputNetwork.Type = NAT

	if err := json.Unmarshal([]byte(properties), &outputNetwork); err != nil {
		return nil, err
	}
	return &outputNetwork, nil
}

func modifyNetwork(networkID string, settings string) (*HostComputeNetwork, error) {
	networkGUID, err := guid.FromString(networkID)
	if err != nil {
		return nil, errInvalidNetworkID
	}
	// Open Network
	var (
		networkHandle    hcnNetwork
		resultBuffer     *uint16
		propertiesBuffer *uint16
	)
	hr := hcnOpenNetwork(&networkGUID, &networkHandle, &resultBuffer)
	if err := checkForErrors("hcnOpenNetwork", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Modify Network
	hr = hcnModifyNetwork(networkHandle, settings, &resultBuffer)
	if err := checkForErrors("hcnModifyNetwork", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query network.
	hcnQuery := defaultQuery()
	query, err := json.Marshal(hcnQuery)
	if err != nil {
		return nil, err
	}
	hr = hcnQueryNetworkProperties(networkHandle, string(query), &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryNetworkProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close network.
	hr = hcnCloseNetwork(networkHandle)
	if err := checkForErrors("hcnCloseNetwork", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to HostComputeNetwork
	var outputNetwork HostComputeNetwork

	// If HNS sets the network type to NAT (i.e. '0' in HNS.Schema.Network.NetworkMode),
	// the value will be omitted from the JSON blob. We therefore need to initialize NAT here before
	// unmarshaling the JSON blob.
	outputNetwork.Type = NAT

	if err := json.Unmarshal([]byte(properties), &outputNetwork); err != nil {
		return nil, err
	}
	return &outputNetwork, nil
}

func deleteNetwork(networkID string) error {
	networkGUID, err := guid.FromString(networkID)
	if err != nil {
		return errInvalidNetworkID
	}
	var resultBuffer *uint16
	hr := hcnDeleteNetwork(&networkGUID, &resultBuffer)
	if err := checkForErrors("hcnDeleteNetwork", hr, resultBuffer); err != nil {
		return err
	}
	return nil
}

// ListNetworks makes a call to list all available networks.
func ListNetworks() ([]HostComputeNetwork, error) {
	hcnQuery := defaultQuery()
	networks, err := ListNetworksQuery(hcnQuery)
	if err != nil {
		return nil, err
	}
	return networks, nil
}

// ListNetworksQuery makes a call to query the list of available networks.
func ListNetworksQuery(query HostComputeQuery) ([]HostComputeNetwork, error) {
	queryJSON, err := json.Marshal(query)
	if err != nil {
		return nil, err
	}

	networks, err := enumerateNetworks(string(queryJSON))
	if err != nil {
		return nil, err
	}
	return networks, nil
}

// GetNetworkByID returns the network specified by Id.
func GetNetworkByID(networkID string) (*HostComputeNetwork, error) {
	hcnQuery := defaultQuery()
	mapA := map[string]string{"ID": networkID}
	filter, err := json.Marshal(mapA)
	if err != nil {
		return nil, err
	}
	hcnQuery.Filter = string(filter)

	networks, err := ListNetworksQuery(hcnQuery)
	if err != nil {
		return nil, err
	}
	if len(networks) == 0 {
		return nil, NetworkNotFoundError{NetworkID: networkID}
	}
	return &networks[0], err
}

// GetNetworkByName returns the network specified by Name.
func GetNetworkByName(networkName string) (*HostComputeNetwork, error) {
	hcnQuery := defaultQuery()
	mapA := map[string]string{"Name": networkName}
	filter, err := json.Marshal(mapA)
	if err != nil {
		return nil, err
	}
	hcnQuery.Filter = string(filter)

	networks, err := ListNetworksQuery(hcnQuery)
	if err != nil {
		return nil, err
	}
	if len(networks) == 0 {
		return nil, NetworkNotFoundError{NetworkName: networkName}
	}
	return &networks[0], err
}

// Create Network.
func (network *HostComputeNetwork) Create() (*HostComputeNetwork, error) {
	logrus.Debugf("hcn::HostComputeNetwork::Create id=%s", network.Id)
	for _, ipam := range network.Ipams {
		for _, subnet := range ipam.Subnets {
			if subnet.IpAddressPrefix != "" {
				hasDefault := false
				for _, route := range subnet.Routes {
					if route.NextHop == "" {
						return nil, errors.New("network create error, subnet has address prefix but no gateway specified")
					}
					if route.DestinationPrefix == "0.0.0.0/0" || route.DestinationPrefix == "::/0" {
						hasDefault = true
					}
				}
				if !hasDefault {
					return nil, errors.New("network create error, no default gateway")
				}
			}
		}
	}

	jsonString, err := json.Marshal(network)
	if err != nil {
		return nil, err
	}

	logrus.Debugf("hcn::HostComputeNetwork::Create JSON: %s", jsonString)
	network, hcnErr := createNetwork(string(jsonString))
	if hcnErr != nil {
		return nil, hcnErr
	}
	return network, nil
}

// Delete Network.
func (network *HostComputeNetwork) Delete() error {
	logrus.Debugf("hcn::HostComputeNetwork::Delete id=%s", network.Id)

	if err := deleteNetwork(network.Id); err != nil {
		return err
	}
	return nil
}

// ModifyNetworkSettings updates the Policy for a network.
func (network *HostComputeNetwork) ModifyNetworkSettings(request *ModifyNetworkSettingRequest) error {
	logrus.Debugf("hcn::HostComputeNetwork::ModifyNetworkSettings id=%s", network.Id)

	networkSettingsRequest, err := json.Marshal(request)
	if err != nil {
		return err
	}

	_, err = modifyNetwork(network.Id, string(networkSettingsRequest))
	if err != nil {
		return err
	}
	return nil
}

// AddPolicy applies a Policy (ex: RemoteSubnet) on the Network.
func (network *HostComputeNetwork) AddPolicy(networkPolicy PolicyNetworkRequest) error {
	logrus.Debugf("hcn::HostComputeNetwork::AddPolicy id=%s", network.Id)

	settingsJSON, err := json.Marshal(networkPolicy)
	if err != nil {
		return err
	}
	requestMessage := &ModifyNetworkSettingRequest{
		ResourceType: NetworkResourceTypePolicy,
		RequestType:  RequestTypeAdd,
		Settings:     settingsJSON,
	}

	return network.ModifyNetworkSettings(requestMessage)
}

// RemovePolicy removes a Policy (ex: RemoteSubnet) from the Network.
func (network *HostComputeNetwork) RemovePolicy(networkPolicy PolicyNetworkRequest) error {
	logrus.Debugf("hcn::HostComputeNetwork::RemovePolicy id=%s", network.Id)

	settingsJSON, err := json.Marshal(networkPolicy)
	if err != nil {
		return err
	}
	requestMessage := &ModifyNetworkSettingRequest{
		ResourceType: NetworkResourceTypePolicy,
		RequestType:  RequestTypeRemove,
		Settings:     settingsJSON,
	}

	return network.ModifyNetworkSettings(requestMessage)
}

// CreateEndpoint creates an endpoint on the Network.
func (network *HostComputeNetwork) CreateEndpoint(endpoint *HostComputeEndpoint) (*HostComputeEndpoint, error) {
	isRemote := endpoint.Flags&EndpointFlagsRemoteEndpoint != 0
	logrus.Debugf("hcn::HostComputeNetwork::CreatEndpoint, networkId=%s remote=%t", network.Id, isRemote)

	endpoint.HostComputeNetwork = network.Id
	endpointSettings, err := json.Marshal(endpoint)
	if err != nil {
		return nil, err
	}
	newEndpoint, err := createEndpoint(network.Id, string(endpointSettings))
	if err != nil {
		return nil, err
	}
	return newEndpoint, nil
}

// CreateRemoteEndpoint creates a remote endpoint on the Network.
func (network *HostComputeNetwork) CreateRemoteEndpoint(endpoint *HostComputeEndpoint) (*HostComputeEndpoint, error) {
	endpoint.Flags = EndpointFlagsRemoteEndpoint | endpoint.Flags
	return network.CreateEndpoint(endpoint)
}
