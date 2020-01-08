package hcn

import (
	"encoding/json"
	"errors"

	"github.com/Microsoft/go-winio/pkg/guid"
	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/sirupsen/logrus"
)

// IpConfig is assoicated with an endpoint
type IpConfig struct {
	IpAddress    string `json:",omitempty"`
	PrefixLength uint8  `json:",omitempty"`
}

// EndpointFlags are special settings on an endpoint.
type EndpointFlags uint32

var (
	// EndpointFlagsNone is the default.
	EndpointFlagsNone EndpointFlags
	// EndpointFlagsRemoteEndpoint means that an endpoint is on another host.
	EndpointFlagsRemoteEndpoint EndpointFlags = 1
)

// HostComputeEndpoint represents a network endpoint
type HostComputeEndpoint struct {
	Id                   string           `json:"ID,omitempty"`
	Name                 string           `json:",omitempty"`
	HostComputeNetwork   string           `json:",omitempty"` // GUID
	HostComputeNamespace string           `json:",omitempty"` // GUID
	Policies             []EndpointPolicy `json:",omitempty"`
	IpConfigurations     []IpConfig       `json:",omitempty"`
	Dns                  Dns              `json:",omitempty"`
	Routes               []Route          `json:",omitempty"`
	MacAddress           string           `json:",omitempty"`
	Flags                EndpointFlags    `json:",omitempty"`
	SchemaVersion        SchemaVersion    `json:",omitempty"`
}

// EndpointResourceType are the two different Endpoint settings resources.
type EndpointResourceType string

var (
	// EndpointResourceTypePolicy is for Endpoint Policies. Ex: ACL, NAT
	EndpointResourceTypePolicy EndpointResourceType = "Policy"
	// EndpointResourceTypePort is for Endpoint Port settings.
	EndpointResourceTypePort EndpointResourceType = "Port"
)

// ModifyEndpointSettingRequest is the structure used to send request to modify an endpoint.
// Used to update policy/port on an endpoint.
type ModifyEndpointSettingRequest struct {
	ResourceType EndpointResourceType `json:",omitempty"` // Policy, Port
	RequestType  RequestType          `json:",omitempty"` // Add, Remove, Update, Refresh
	Settings     json.RawMessage      `json:",omitempty"`
}

type PolicyEndpointRequest struct {
	Policies []EndpointPolicy `json:",omitempty"`
}

func getEndpoint(endpointGuid guid.GUID, query string) (*HostComputeEndpoint, error) {
	// Open endpoint.
	var (
		endpointHandle   hcnEndpoint
		resultBuffer     *uint16
		propertiesBuffer *uint16
	)
	hr := hcnOpenEndpoint(&endpointGuid, &endpointHandle, &resultBuffer)
	if err := checkForErrors("hcnOpenEndpoint", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query endpoint.
	hr = hcnQueryEndpointProperties(endpointHandle, query, &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryEndpointProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close endpoint.
	hr = hcnCloseEndpoint(endpointHandle)
	if err := checkForErrors("hcnCloseEndpoint", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to HostComputeEndpoint
	var outputEndpoint HostComputeEndpoint
	if err := json.Unmarshal([]byte(properties), &outputEndpoint); err != nil {
		return nil, err
	}
	return &outputEndpoint, nil
}

func enumerateEndpoints(query string) ([]HostComputeEndpoint, error) {
	// Enumerate all Endpoint Guids
	var (
		resultBuffer   *uint16
		endpointBuffer *uint16
	)
	hr := hcnEnumerateEndpoints(query, &endpointBuffer, &resultBuffer)
	if err := checkForErrors("hcnEnumerateEndpoints", hr, resultBuffer); err != nil {
		return nil, err
	}

	endpoints := interop.ConvertAndFreeCoTaskMemString(endpointBuffer)
	var endpointIds []guid.GUID
	err := json.Unmarshal([]byte(endpoints), &endpointIds)
	if err != nil {
		return nil, err
	}

	var outputEndpoints []HostComputeEndpoint
	for _, endpointGuid := range endpointIds {
		endpoint, err := getEndpoint(endpointGuid, query)
		if err != nil {
			return nil, err
		}
		outputEndpoints = append(outputEndpoints, *endpoint)
	}
	return outputEndpoints, nil
}

func createEndpoint(networkId string, endpointSettings string) (*HostComputeEndpoint, error) {
	networkGuid, err := guid.FromString(networkId)
	if err != nil {
		return nil, errInvalidNetworkID
	}
	// Open network.
	var networkHandle hcnNetwork
	var resultBuffer *uint16
	hr := hcnOpenNetwork(&networkGuid, &networkHandle, &resultBuffer)
	if err := checkForErrors("hcnOpenNetwork", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Create endpoint.
	endpointId := guid.GUID{}
	var endpointHandle hcnEndpoint
	hr = hcnCreateEndpoint(networkHandle, &endpointId, endpointSettings, &endpointHandle, &resultBuffer)
	if err := checkForErrors("hcnCreateEndpoint", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query endpoint.
	hcnQuery := defaultQuery()
	query, err := json.Marshal(hcnQuery)
	if err != nil {
		return nil, err
	}
	var propertiesBuffer *uint16
	hr = hcnQueryEndpointProperties(endpointHandle, string(query), &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryEndpointProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close endpoint.
	hr = hcnCloseEndpoint(endpointHandle)
	if err := checkForErrors("hcnCloseEndpoint", hr, nil); err != nil {
		return nil, err
	}
	// Close network.
	hr = hcnCloseNetwork(networkHandle)
	if err := checkForErrors("hcnCloseNetwork", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to HostComputeEndpoint
	var outputEndpoint HostComputeEndpoint
	if err := json.Unmarshal([]byte(properties), &outputEndpoint); err != nil {
		return nil, err
	}
	return &outputEndpoint, nil
}

func modifyEndpoint(endpointId string, settings string) (*HostComputeEndpoint, error) {
	endpointGuid, err := guid.FromString(endpointId)
	if err != nil {
		return nil, errInvalidEndpointID
	}
	// Open endpoint
	var (
		endpointHandle   hcnEndpoint
		resultBuffer     *uint16
		propertiesBuffer *uint16
	)
	hr := hcnOpenEndpoint(&endpointGuid, &endpointHandle, &resultBuffer)
	if err := checkForErrors("hcnOpenEndpoint", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Modify endpoint
	hr = hcnModifyEndpoint(endpointHandle, settings, &resultBuffer)
	if err := checkForErrors("hcnModifyEndpoint", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query endpoint.
	hcnQuery := defaultQuery()
	query, err := json.Marshal(hcnQuery)
	if err != nil {
		return nil, err
	}
	hr = hcnQueryEndpointProperties(endpointHandle, string(query), &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryEndpointProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close endpoint.
	hr = hcnCloseEndpoint(endpointHandle)
	if err := checkForErrors("hcnCloseEndpoint", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to HostComputeEndpoint
	var outputEndpoint HostComputeEndpoint
	if err := json.Unmarshal([]byte(properties), &outputEndpoint); err != nil {
		return nil, err
	}
	return &outputEndpoint, nil
}

func deleteEndpoint(endpointId string) error {
	endpointGuid, err := guid.FromString(endpointId)
	if err != nil {
		return errInvalidEndpointID
	}
	var resultBuffer *uint16
	hr := hcnDeleteEndpoint(&endpointGuid, &resultBuffer)
	if err := checkForErrors("hcnDeleteEndpoint", hr, resultBuffer); err != nil {
		return err
	}
	return nil
}

// ListEndpoints makes a call to list all available endpoints.
func ListEndpoints() ([]HostComputeEndpoint, error) {
	hcnQuery := defaultQuery()
	endpoints, err := ListEndpointsQuery(hcnQuery)
	if err != nil {
		return nil, err
	}
	return endpoints, nil
}

// ListEndpointsQuery makes a call to query the list of available endpoints.
func ListEndpointsQuery(query HostComputeQuery) ([]HostComputeEndpoint, error) {
	queryJson, err := json.Marshal(query)
	if err != nil {
		return nil, err
	}

	endpoints, err := enumerateEndpoints(string(queryJson))
	if err != nil {
		return nil, err
	}
	return endpoints, nil
}

// ListEndpointsOfNetwork queries the list of endpoints on a network.
func ListEndpointsOfNetwork(networkId string) ([]HostComputeEndpoint, error) {
	hcnQuery := defaultQuery()
	// TODO: Once query can convert schema, change to {HostComputeNetwork:networkId}
	mapA := map[string]string{"VirtualNetwork": networkId}
	filter, err := json.Marshal(mapA)
	if err != nil {
		return nil, err
	}
	hcnQuery.Filter = string(filter)

	return ListEndpointsQuery(hcnQuery)
}

// GetEndpointByID returns an endpoint specified by Id
func GetEndpointByID(endpointId string) (*HostComputeEndpoint, error) {
	hcnQuery := defaultQuery()
	mapA := map[string]string{"ID": endpointId}
	filter, err := json.Marshal(mapA)
	if err != nil {
		return nil, err
	}
	hcnQuery.Filter = string(filter)

	endpoints, err := ListEndpointsQuery(hcnQuery)
	if err != nil {
		return nil, err
	}
	if len(endpoints) == 0 {
		return nil, EndpointNotFoundError{EndpointID: endpointId}
	}
	return &endpoints[0], err
}

// GetEndpointByName returns an endpoint specified by Name
func GetEndpointByName(endpointName string) (*HostComputeEndpoint, error) {
	hcnQuery := defaultQuery()
	mapA := map[string]string{"Name": endpointName}
	filter, err := json.Marshal(mapA)
	if err != nil {
		return nil, err
	}
	hcnQuery.Filter = string(filter)

	endpoints, err := ListEndpointsQuery(hcnQuery)
	if err != nil {
		return nil, err
	}
	if len(endpoints) == 0 {
		return nil, EndpointNotFoundError{EndpointName: endpointName}
	}
	return &endpoints[0], err
}

// Create Endpoint.
func (endpoint *HostComputeEndpoint) Create() (*HostComputeEndpoint, error) {
	logrus.Debugf("hcn::HostComputeEndpoint::Create id=%s", endpoint.Id)

	if endpoint.HostComputeNamespace != "" {
		return nil, errors.New("endpoint create error, endpoint json HostComputeNamespace is read only and should not be set")
	}

	jsonString, err := json.Marshal(endpoint)
	if err != nil {
		return nil, err
	}

	logrus.Debugf("hcn::HostComputeEndpoint::Create JSON: %s", jsonString)
	endpoint, hcnErr := createEndpoint(endpoint.HostComputeNetwork, string(jsonString))
	if hcnErr != nil {
		return nil, hcnErr
	}
	return endpoint, nil
}

// Delete Endpoint.
func (endpoint *HostComputeEndpoint) Delete() error {
	logrus.Debugf("hcn::HostComputeEndpoint::Delete id=%s", endpoint.Id)

	if err := deleteEndpoint(endpoint.Id); err != nil {
		return err
	}
	return nil
}

// ModifyEndpointSettings updates the Port/Policy of an Endpoint.
func ModifyEndpointSettings(endpointId string, request *ModifyEndpointSettingRequest) error {
	logrus.Debugf("hcn::HostComputeEndpoint::ModifyEndpointSettings id=%s", endpointId)

	endpointSettingsRequest, err := json.Marshal(request)
	if err != nil {
		return err
	}

	_, err = modifyEndpoint(endpointId, string(endpointSettingsRequest))
	if err != nil {
		return err
	}
	return nil
}

// ApplyPolicy applies a Policy (ex: ACL) on the Endpoint.
func (endpoint *HostComputeEndpoint) ApplyPolicy(requestType RequestType, endpointPolicy PolicyEndpointRequest) error {
	logrus.Debugf("hcn::HostComputeEndpoint::ApplyPolicy id=%s", endpoint.Id)

	settingsJson, err := json.Marshal(endpointPolicy)
	if err != nil {
		return err
	}
	requestMessage := &ModifyEndpointSettingRequest{
		ResourceType: EndpointResourceTypePolicy,
		RequestType:  requestType,
		Settings:     settingsJson,
	}

	return ModifyEndpointSettings(endpoint.Id, requestMessage)
}

// NamespaceAttach modifies a Namespace to add an endpoint.
func (endpoint *HostComputeEndpoint) NamespaceAttach(namespaceId string) error {
	return AddNamespaceEndpoint(namespaceId, endpoint.Id)
}

// NamespaceDetach modifies a Namespace to remove an endpoint.
func (endpoint *HostComputeEndpoint) NamespaceDetach(namespaceId string) error {
	return RemoveNamespaceEndpoint(namespaceId, endpoint.Id)
}
