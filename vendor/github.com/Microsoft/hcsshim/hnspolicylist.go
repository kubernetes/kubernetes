package hcsshim

import (
	"encoding/json"

	"github.com/sirupsen/logrus"
)

// RoutePolicy is a structure defining schema for Route based Policy
type RoutePolicy struct {
	Policy
	DestinationPrefix string `json:"DestinationPrefix,omitempty"`
	NextHop           string `json:"NextHop,omitempty"`
	EncapEnabled      bool   `json:"NeedEncap,omitempty"`
}

// ELBPolicy is a structure defining schema for ELB LoadBalancing based Policy
type ELBPolicy struct {
	LBPolicy
	SourceVIP string   `json:"SourceVIP,omitempty"`
	VIPs      []string `json:"VIPs,omitempty"`
	ILB       bool     `json:"ILB,omitempty"`
}

// LBPolicy is a structure defining schema for LoadBalancing based Policy
type LBPolicy struct {
	Policy
	Protocol     uint16 `json:"Protocol,omitempty"`
	InternalPort uint16
	ExternalPort uint16
}

// PolicyList is a structure defining schema for Policy list request
type PolicyList struct {
	ID                 string            `json:"ID,omitempty"`
	EndpointReferences []string          `json:"References,omitempty"`
	Policies           []json.RawMessage `json:"Policies,omitempty"`
}

// HNSPolicyListRequest makes a call into HNS to update/query a single network
func HNSPolicyListRequest(method, path, request string) (*PolicyList, error) {
	var policy PolicyList
	err := hnsCall(method, "/policylists/"+path, request, &policy)
	if err != nil {
		return nil, err
	}

	return &policy, nil
}

// HNSListPolicyListRequest gets all the policy list
func HNSListPolicyListRequest() ([]PolicyList, error) {
	var plist []PolicyList
	err := hnsCall("GET", "/policylists/", "", &plist)
	if err != nil {
		return nil, err
	}

	return plist, nil
}

// PolicyListRequest makes a HNS call to modify/query a network policy list
func PolicyListRequest(method, path, request string) (*PolicyList, error) {
	policylist := &PolicyList{}
	err := hnsCall(method, "/policylists/"+path, request, &policylist)
	if err != nil {
		return nil, err
	}

	return policylist, nil
}

// GetPolicyListByID get the policy list by ID
func GetPolicyListByID(policyListID string) (*PolicyList, error) {
	return PolicyListRequest("GET", policyListID, "")
}

// Create PolicyList by sending PolicyListRequest to HNS.
func (policylist *PolicyList) Create() (*PolicyList, error) {
	operation := "Create"
	title := "HCSShim::PolicyList::" + operation
	logrus.Debugf(title+" id=%s", policylist.ID)
	jsonString, err := json.Marshal(policylist)
	if err != nil {
		return nil, err
	}
	return PolicyListRequest("POST", "", string(jsonString))
}

// Delete deletes PolicyList
func (policylist *PolicyList) Delete() (*PolicyList, error) {
	operation := "Delete"
	title := "HCSShim::PolicyList::" + operation
	logrus.Debugf(title+" id=%s", policylist.ID)

	return PolicyListRequest("DELETE", policylist.ID, "")
}

// AddEndpoint add an endpoint to a Policy List
func (policylist *PolicyList) AddEndpoint(endpoint *HNSEndpoint) (*PolicyList, error) {
	operation := "AddEndpoint"
	title := "HCSShim::PolicyList::" + operation
	logrus.Debugf(title+" id=%s, endpointId:%s", policylist.ID, endpoint.Id)

	_, err := policylist.Delete()
	if err != nil {
		return nil, err
	}

	// Add Endpoint to the Existing List
	policylist.EndpointReferences = append(policylist.EndpointReferences, "/endpoints/"+endpoint.Id)

	return policylist.Create()
}

// RemoveEndpoint removes an endpoint from the Policy List
func (policylist *PolicyList) RemoveEndpoint(endpoint *HNSEndpoint) (*PolicyList, error) {
	operation := "RemoveEndpoint"
	title := "HCSShim::PolicyList::" + operation
	logrus.Debugf(title+" id=%s, endpointId:%s", policylist.ID, endpoint.Id)

	_, err := policylist.Delete()
	if err != nil {
		return nil, err
	}

	elementToRemove := "/endpoints/" + endpoint.Id

	var references []string

	for _, endpointReference := range policylist.EndpointReferences {
		if endpointReference == elementToRemove {
			continue
		}
		references = append(references, endpointReference)
	}
	policylist.EndpointReferences = references
	return policylist.Create()
}

// AddLoadBalancer policy list for the specified endpoints
func AddLoadBalancer(endpoints []HNSEndpoint, isILB bool, sourceVIP, vip string, protocol uint16, internalPort uint16, externalPort uint16) (*PolicyList, error) {
	operation := "AddLoadBalancer"
	title := "HCSShim::PolicyList::" + operation
	logrus.Debugf(title+" endpointId=%v, isILB=%v, sourceVIP=%s, vip=%s, protocol=%v, internalPort=%v, externalPort=%v", endpoints, isILB, sourceVIP, vip, protocol, internalPort, externalPort)

	policylist := &PolicyList{}

	elbPolicy := &ELBPolicy{
		SourceVIP: sourceVIP,
		ILB:       isILB,
	}

	if len(vip) > 0 {
		elbPolicy.VIPs = []string{vip}
	}
	elbPolicy.Type = ExternalLoadBalancer
	elbPolicy.Protocol = protocol
	elbPolicy.InternalPort = internalPort
	elbPolicy.ExternalPort = externalPort

	for _, endpoint := range endpoints {
		policylist.EndpointReferences = append(policylist.EndpointReferences, "/endpoints/"+endpoint.Id)
	}

	jsonString, err := json.Marshal(elbPolicy)
	if err != nil {
		return nil, err
	}
	policylist.Policies = append(policylist.Policies, jsonString)
	return policylist.Create()
}

// AddRoute adds route policy list for the specified endpoints
func AddRoute(endpoints []HNSEndpoint, destinationPrefix string, nextHop string, encapEnabled bool) (*PolicyList, error) {
	operation := "AddRoute"
	title := "HCSShim::PolicyList::" + operation
	logrus.Debugf(title+" destinationPrefix:%s", destinationPrefix)

	policylist := &PolicyList{}

	rPolicy := &RoutePolicy{
		DestinationPrefix: destinationPrefix,
		NextHop:           nextHop,
		EncapEnabled:      encapEnabled,
	}
	rPolicy.Type = Route

	for _, endpoint := range endpoints {
		policylist.EndpointReferences = append(policylist.EndpointReferences, "/endpoints/"+endpoint.Id)
	}

	jsonString, err := json.Marshal(rPolicy)
	if err != nil {
		return nil, err
	}

	policylist.Policies = append(policylist.Policies, jsonString)
	return policylist.Create()
}
