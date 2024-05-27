//go:build windows

package hcsshim

import (
	"github.com/Microsoft/hcsshim/internal/hns"
)

// RoutePolicy is a structure defining schema for Route based Policy
type RoutePolicy = hns.RoutePolicy

// ELBPolicy is a structure defining schema for ELB LoadBalancing based Policy
type ELBPolicy = hns.ELBPolicy

// LBPolicy is a structure defining schema for LoadBalancing based Policy
type LBPolicy = hns.LBPolicy

// PolicyList is a structure defining schema for Policy list request
type PolicyList = hns.PolicyList

// HNSPolicyListRequest makes a call into HNS to update/query a single network
func HNSPolicyListRequest(method, path, request string) (*PolicyList, error) {
	return hns.HNSPolicyListRequest(method, path, request)
}

// HNSListPolicyListRequest gets all the policy list
func HNSListPolicyListRequest() ([]PolicyList, error) {
	return hns.HNSListPolicyListRequest()
}

// PolicyListRequest makes a HNS call to modify/query a network policy list
func PolicyListRequest(method, path, request string) (*PolicyList, error) {
	return hns.PolicyListRequest(method, path, request)
}

// GetPolicyListByID get the policy list by ID
func GetPolicyListByID(policyListID string) (*PolicyList, error) {
	return hns.GetPolicyListByID(policyListID)
}

// AddLoadBalancer policy list for the specified endpoints
func AddLoadBalancer(endpoints []HNSEndpoint, isILB bool, sourceVIP, vip string, protocol uint16, internalPort uint16, externalPort uint16) (*PolicyList, error) {
	return hns.AddLoadBalancer(endpoints, isILB, sourceVIP, vip, protocol, internalPort, externalPort)
}

// AddRoute adds route policy list for the specified endpoints
func AddRoute(endpoints []HNSEndpoint, destinationPrefix string, nextHop string, encapEnabled bool) (*PolicyList, error) {
	return hns.AddRoute(endpoints, destinationPrefix, nextHop, encapEnabled)
}
