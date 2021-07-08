package hcn

import (
	"encoding/json"

	"github.com/Microsoft/go-winio/pkg/guid"
	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/sirupsen/logrus"
)

// LoadBalancerPortMapping is associated with HostComputeLoadBalancer
type LoadBalancerPortMapping struct {
	Protocol         uint32                       `json:",omitempty"` // EX: TCP = 6, UDP = 17
	InternalPort     uint16                       `json:",omitempty"`
	ExternalPort     uint16                       `json:",omitempty"`
	DistributionType LoadBalancerDistribution     `json:",omitempty"` // EX: Distribute per connection = 0, distribute traffic of the same protocol per client IP = 1, distribute per client IP = 2
	Flags            LoadBalancerPortMappingFlags `json:",omitempty"`
}

// HostComputeLoadBalancer represents software load balancer.
type HostComputeLoadBalancer struct {
	Id                   string                    `json:"ID,omitempty"`
	HostComputeEndpoints []string                  `json:",omitempty"`
	SourceVIP            string                    `json:",omitempty"`
	FrontendVIPs         []string                  `json:",omitempty"`
	PortMappings         []LoadBalancerPortMapping `json:",omitempty"`
	SchemaVersion        SchemaVersion             `json:",omitempty"`
	Flags                LoadBalancerFlags         `json:",omitempty"` // 0: None, 1: EnableDirectServerReturn
}

//LoadBalancerFlags modify settings for a loadbalancer.
type LoadBalancerFlags uint32

var (
	// LoadBalancerFlagsNone is the default.
	LoadBalancerFlagsNone LoadBalancerFlags = 0
	// LoadBalancerFlagsDSR enables Direct Server Return (DSR)
	LoadBalancerFlagsDSR LoadBalancerFlags = 1
)

// LoadBalancerPortMappingFlags are special settings on a loadbalancer.
type LoadBalancerPortMappingFlags uint32

var (
	// LoadBalancerPortMappingFlagsNone is the default.
	LoadBalancerPortMappingFlagsNone LoadBalancerPortMappingFlags
	// LoadBalancerPortMappingFlagsILB enables internal loadbalancing.
	LoadBalancerPortMappingFlagsILB LoadBalancerPortMappingFlags = 1
	// LoadBalancerPortMappingFlagsLocalRoutedVIP enables VIP access from the host.
	LoadBalancerPortMappingFlagsLocalRoutedVIP LoadBalancerPortMappingFlags = 2
	// LoadBalancerPortMappingFlagsUseMux enables DSR for NodePort access of VIP.
	LoadBalancerPortMappingFlagsUseMux LoadBalancerPortMappingFlags = 4
	// LoadBalancerPortMappingFlagsPreserveDIP delivers packets with destination IP as the VIP.
	LoadBalancerPortMappingFlagsPreserveDIP LoadBalancerPortMappingFlags = 8
)

// LoadBalancerDistribution specifies how the loadbalancer distributes traffic.
type LoadBalancerDistribution uint32

var (
	// LoadBalancerDistributionNone is the default and loadbalances each connection to the same pod.
	LoadBalancerDistributionNone LoadBalancerDistribution
	// LoadBalancerDistributionSourceIPProtocol loadbalances all traffic of the same protocol from a client IP to the same pod.
	LoadBalancerDistributionSourceIPProtocol LoadBalancerDistribution = 1
	// LoadBalancerDistributionSourceIP loadbalances all traffic from a client IP to the same pod.
	LoadBalancerDistributionSourceIP LoadBalancerDistribution = 2
)

func getLoadBalancer(loadBalancerGuid guid.GUID, query string) (*HostComputeLoadBalancer, error) {
	// Open loadBalancer.
	var (
		loadBalancerHandle hcnLoadBalancer
		resultBuffer       *uint16
		propertiesBuffer   *uint16
	)
	hr := hcnOpenLoadBalancer(&loadBalancerGuid, &loadBalancerHandle, &resultBuffer)
	if err := checkForErrors("hcnOpenLoadBalancer", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query loadBalancer.
	hr = hcnQueryLoadBalancerProperties(loadBalancerHandle, query, &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryLoadBalancerProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close loadBalancer.
	hr = hcnCloseLoadBalancer(loadBalancerHandle)
	if err := checkForErrors("hcnCloseLoadBalancer", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to HostComputeLoadBalancer
	var outputLoadBalancer HostComputeLoadBalancer
	if err := json.Unmarshal([]byte(properties), &outputLoadBalancer); err != nil {
		return nil, err
	}
	return &outputLoadBalancer, nil
}

func enumerateLoadBalancers(query string) ([]HostComputeLoadBalancer, error) {
	// Enumerate all LoadBalancer Guids
	var (
		resultBuffer       *uint16
		loadBalancerBuffer *uint16
	)
	hr := hcnEnumerateLoadBalancers(query, &loadBalancerBuffer, &resultBuffer)
	if err := checkForErrors("hcnEnumerateLoadBalancers", hr, resultBuffer); err != nil {
		return nil, err
	}

	loadBalancers := interop.ConvertAndFreeCoTaskMemString(loadBalancerBuffer)
	var loadBalancerIds []guid.GUID
	if err := json.Unmarshal([]byte(loadBalancers), &loadBalancerIds); err != nil {
		return nil, err
	}

	var outputLoadBalancers []HostComputeLoadBalancer
	for _, loadBalancerGuid := range loadBalancerIds {
		loadBalancer, err := getLoadBalancer(loadBalancerGuid, query)
		if err != nil {
			return nil, err
		}
		outputLoadBalancers = append(outputLoadBalancers, *loadBalancer)
	}
	return outputLoadBalancers, nil
}

func createLoadBalancer(settings string) (*HostComputeLoadBalancer, error) {
	// Create new loadBalancer.
	var (
		loadBalancerHandle hcnLoadBalancer
		resultBuffer       *uint16
		propertiesBuffer   *uint16
	)
	loadBalancerGuid := guid.GUID{}
	hr := hcnCreateLoadBalancer(&loadBalancerGuid, settings, &loadBalancerHandle, &resultBuffer)
	if err := checkForErrors("hcnCreateLoadBalancer", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query loadBalancer.
	hcnQuery := defaultQuery()
	query, err := json.Marshal(hcnQuery)
	if err != nil {
		return nil, err
	}
	hr = hcnQueryLoadBalancerProperties(loadBalancerHandle, string(query), &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryLoadBalancerProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close loadBalancer.
	hr = hcnCloseLoadBalancer(loadBalancerHandle)
	if err := checkForErrors("hcnCloseLoadBalancer", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to HostComputeLoadBalancer
	var outputLoadBalancer HostComputeLoadBalancer
	if err := json.Unmarshal([]byte(properties), &outputLoadBalancer); err != nil {
		return nil, err
	}
	return &outputLoadBalancer, nil
}

func deleteLoadBalancer(loadBalancerId string) error {
	loadBalancerGuid, err := guid.FromString(loadBalancerId)
	if err != nil {
		return errInvalidLoadBalancerID
	}
	var resultBuffer *uint16
	hr := hcnDeleteLoadBalancer(&loadBalancerGuid, &resultBuffer)
	if err := checkForErrors("hcnDeleteLoadBalancer", hr, resultBuffer); err != nil {
		return err
	}
	return nil
}

// ListLoadBalancers makes a call to list all available loadBalancers.
func ListLoadBalancers() ([]HostComputeLoadBalancer, error) {
	hcnQuery := defaultQuery()
	loadBalancers, err := ListLoadBalancersQuery(hcnQuery)
	if err != nil {
		return nil, err
	}
	return loadBalancers, nil
}

// ListLoadBalancersQuery makes a call to query the list of available loadBalancers.
func ListLoadBalancersQuery(query HostComputeQuery) ([]HostComputeLoadBalancer, error) {
	queryJson, err := json.Marshal(query)
	if err != nil {
		return nil, err
	}

	loadBalancers, err := enumerateLoadBalancers(string(queryJson))
	if err != nil {
		return nil, err
	}
	return loadBalancers, nil
}

// GetLoadBalancerByID returns the LoadBalancer specified by Id.
func GetLoadBalancerByID(loadBalancerId string) (*HostComputeLoadBalancer, error) {
	hcnQuery := defaultQuery()
	mapA := map[string]string{"ID": loadBalancerId}
	filter, err := json.Marshal(mapA)
	if err != nil {
		return nil, err
	}
	hcnQuery.Filter = string(filter)

	loadBalancers, err := ListLoadBalancersQuery(hcnQuery)
	if err != nil {
		return nil, err
	}
	if len(loadBalancers) == 0 {
		return nil, LoadBalancerNotFoundError{LoadBalancerId: loadBalancerId}
	}
	return &loadBalancers[0], err
}

// Create LoadBalancer.
func (loadBalancer *HostComputeLoadBalancer) Create() (*HostComputeLoadBalancer, error) {
	logrus.Debugf("hcn::HostComputeLoadBalancer::Create id=%s", loadBalancer.Id)

	jsonString, err := json.Marshal(loadBalancer)
	if err != nil {
		return nil, err
	}

	logrus.Debugf("hcn::HostComputeLoadBalancer::Create JSON: %s", jsonString)
	loadBalancer, hcnErr := createLoadBalancer(string(jsonString))
	if hcnErr != nil {
		return nil, hcnErr
	}
	return loadBalancer, nil
}

// Delete LoadBalancer.
func (loadBalancer *HostComputeLoadBalancer) Delete() error {
	logrus.Debugf("hcn::HostComputeLoadBalancer::Delete id=%s", loadBalancer.Id)

	if err := deleteLoadBalancer(loadBalancer.Id); err != nil {
		return err
	}
	return nil
}

// AddEndpoint add an endpoint to a LoadBalancer
func (loadBalancer *HostComputeLoadBalancer) AddEndpoint(endpoint *HostComputeEndpoint) (*HostComputeLoadBalancer, error) {
	logrus.Debugf("hcn::HostComputeLoadBalancer::AddEndpoint loadBalancer=%s endpoint=%s", loadBalancer.Id, endpoint.Id)

	err := loadBalancer.Delete()
	if err != nil {
		return nil, err
	}

	// Add Endpoint to the Existing List
	loadBalancer.HostComputeEndpoints = append(loadBalancer.HostComputeEndpoints, endpoint.Id)

	return loadBalancer.Create()
}

// RemoveEndpoint removes an endpoint from a LoadBalancer
func (loadBalancer *HostComputeLoadBalancer) RemoveEndpoint(endpoint *HostComputeEndpoint) (*HostComputeLoadBalancer, error) {
	logrus.Debugf("hcn::HostComputeLoadBalancer::RemoveEndpoint loadBalancer=%s endpoint=%s", loadBalancer.Id, endpoint.Id)

	err := loadBalancer.Delete()
	if err != nil {
		return nil, err
	}

	// Create a list of all the endpoints besides the one being removed
	var endpoints []string
	for _, endpointReference := range loadBalancer.HostComputeEndpoints {
		if endpointReference == endpoint.Id {
			continue
		}
		endpoints = append(endpoints, endpointReference)
	}
	loadBalancer.HostComputeEndpoints = endpoints
	return loadBalancer.Create()
}

// AddLoadBalancer for the specified endpoints
func AddLoadBalancer(endpoints []HostComputeEndpoint, flags LoadBalancerFlags, portMappingFlags LoadBalancerPortMappingFlags, sourceVIP string, frontendVIPs []string, protocol uint16, internalPort uint16, externalPort uint16) (*HostComputeLoadBalancer, error) {
	logrus.Debugf("hcn::HostComputeLoadBalancer::AddLoadBalancer endpointId=%v, LoadBalancerFlags=%v, LoadBalancerPortMappingFlags=%v, sourceVIP=%s, frontendVIPs=%v, protocol=%v, internalPort=%v, externalPort=%v", endpoints, flags, portMappingFlags, sourceVIP, frontendVIPs, protocol, internalPort, externalPort)

	loadBalancer := &HostComputeLoadBalancer{
		SourceVIP: sourceVIP,
		PortMappings: []LoadBalancerPortMapping{
			{
				Protocol:     uint32(protocol),
				InternalPort: internalPort,
				ExternalPort: externalPort,
				Flags:        portMappingFlags,
			},
		},
		FrontendVIPs: frontendVIPs,
		SchemaVersion: SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		Flags: flags,
	}

	for _, endpoint := range endpoints {
		loadBalancer.HostComputeEndpoints = append(loadBalancer.HostComputeEndpoints, endpoint.Id)
	}

	return loadBalancer.Create()
}
