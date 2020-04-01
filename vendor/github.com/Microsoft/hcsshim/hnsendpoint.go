package hcsshim

import (
	"github.com/Microsoft/hcsshim/internal/hns"
)

// HNSEndpoint represents a network endpoint in HNS
type HNSEndpoint = hns.HNSEndpoint

// Namespace represents a Compartment.
type Namespace = hns.Namespace

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
type EndpointAttachDetachRequest = hns.EndpointAttachDetachRequest

// EndpointResquestResponse is object to get the endpoint request response
type EndpointResquestResponse = hns.EndpointResquestResponse

// HNSEndpointRequest makes a HNS call to modify/query a network endpoint
func HNSEndpointRequest(method, path, request string) (*HNSEndpoint, error) {
	return hns.HNSEndpointRequest(method, path, request)
}

// HNSListEndpointRequest makes a HNS call to query the list of available endpoints
func HNSListEndpointRequest() ([]HNSEndpoint, error) {
	return hns.HNSListEndpointRequest()
}

// HotAttachEndpoint makes a HCS Call to attach the endpoint to the container
func HotAttachEndpoint(containerID string, endpointID string) error {
	endpoint, err := GetHNSEndpointByID(endpointID)
	isAttached, err := endpoint.IsAttached(containerID)
	if isAttached {
		return err
	}
	return modifyNetworkEndpoint(containerID, endpointID, Add)
}

// HotDetachEndpoint makes a HCS Call to detach the endpoint from the container
func HotDetachEndpoint(containerID string, endpointID string) error {
	endpoint, err := GetHNSEndpointByID(endpointID)
	isAttached, err := endpoint.IsAttached(containerID)
	if !isAttached {
		return err
	}
	return modifyNetworkEndpoint(containerID, endpointID, Remove)
}

// ModifyContainer corresponding to the container id, by sending a request
func modifyContainer(id string, request *ResourceModificationRequestResponse) error {
	container, err := OpenContainer(id)
	if err != nil {
		if IsNotExist(err) {
			return ErrComputeSystemDoesNotExist
		}
		return getInnerError(err)
	}
	defer container.Close()
	err = container.Modify(request)
	if err != nil {
		if IsNotSupported(err) {
			return ErrPlatformNotSupported
		}
		return getInnerError(err)
	}

	return nil
}

func modifyNetworkEndpoint(containerID string, endpointID string, request RequestType) error {
	requestMessage := &ResourceModificationRequestResponse{
		Resource: Network,
		Request:  request,
		Data:     endpointID,
	}
	err := modifyContainer(containerID, requestMessage)

	if err != nil {
		return err
	}

	return nil
}

// GetHNSEndpointByID get the Endpoint by ID
func GetHNSEndpointByID(endpointID string) (*HNSEndpoint, error) {
	return hns.GetHNSEndpointByID(endpointID)
}

// GetHNSEndpointByName gets the endpoint filtered by Name
func GetHNSEndpointByName(endpointName string) (*HNSEndpoint, error) {
	return hns.GetHNSEndpointByName(endpointName)
}
