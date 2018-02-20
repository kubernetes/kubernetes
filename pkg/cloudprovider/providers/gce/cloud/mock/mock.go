/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package mock encapsulates mocks for testing GCE provider functionality.
// These methods are used to override the mock objects' methods in order to
// intercept the standard processing and to add custom logic for test purposes.
//
//  // Example usage:
// cloud := cloud.NewMockGCE()
// cloud.MockTargetPools.AddInstanceHook = mock.AddInstanceHook
package mock

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	alpha "google.golang.org/api/compute/v0.alpha"
	beta "google.golang.org/api/compute/v0.beta"
	ga "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce/cloud/meta"
)

// gceObject is an abstraction of all GCE API object in go client
type gceObject interface {
	MarshalJSON() ([]byte, error)
}

// AddInstanceHook mocks adding a Instance to MockTargetPools
func AddInstanceHook(ctx context.Context, key *meta.Key, req *ga.TargetPoolsAddInstanceRequest, m *cloud.MockTargetPools) error {
	pool, err := m.Get(ctx, key)
	if err != nil {
		return &googleapi.Error{
			Code:    http.StatusNotFound,
			Message: fmt.Sprintf("Key: %s was not found in TargetPools", key.String()),
		}
	}

	for _, instance := range req.Instances {
		pool.Instances = append(pool.Instances, instance.Instance)
	}

	return nil
}

// RemoveInstanceHook mocks removing a Instance from MockTargetPools
func RemoveInstanceHook(ctx context.Context, key *meta.Key, req *ga.TargetPoolsRemoveInstanceRequest, m *cloud.MockTargetPools) error {
	pool, err := m.Get(ctx, key)
	if err != nil {
		return &googleapi.Error{
			Code:    http.StatusNotFound,
			Message: fmt.Sprintf("Key: %s was not found in TargetPools", key.String()),
		}
	}

	for _, instanceToRemove := range req.Instances {
		for i, instance := range pool.Instances {
			if instanceToRemove.Instance == instance {
				// Delete instance from pool.Instances without preserving order
				pool.Instances[i] = pool.Instances[len(pool.Instances)-1]
				pool.Instances = pool.Instances[:len(pool.Instances)-1]
				break
			}
		}
	}

	return nil
}

func convertAndInsertAlphaForwardingRule(key *meta.Key, obj gceObject, mRules map[meta.Key]*cloud.MockForwardingRulesObj, version meta.Version, projectID string) (bool, error) {
	if !key.Valid() {
		return false, fmt.Errorf("invalid GCE key (%+v)", key)
	}

	if _, ok := mRules[*key]; ok {
		err := &googleapi.Error{
			Code:    http.StatusConflict,
			Message: fmt.Sprintf("MockForwardingRule %v exists", key),
		}
		return false, err
	}

	enc, err := obj.MarshalJSON()
	if err != nil {
		return false, err
	}
	var fwdRule alpha.ForwardingRule
	if err := json.Unmarshal(enc, &fwdRule); err != nil {
		return false, err
	}
	// Set the default values for the Alpha fields.
	if fwdRule.NetworkTier == "" {
		fwdRule.NetworkTier = cloud.NetworkTierDefault.ToGCEValue()
	}

	fwdRule.Name = key.Name
	if fwdRule.SelfLink == "" {
		fwdRule.SelfLink = cloud.SelfLink(version, projectID, "forwardingRules", key)
	}

	mRules[*key] = &cloud.MockForwardingRulesObj{Obj: fwdRule}
	return true, nil
}

// InsertFwdRuleHook mocks inserting a ForwardingRule. ForwardingRules are
// expected to default to Premium tier if no NetworkTier is specified.
func InsertFwdRuleHook(ctx context.Context, key *meta.Key, obj *ga.ForwardingRule, m *cloud.MockForwardingRules) (bool, error) {
	projectID := m.ProjectRouter.ProjectID(ctx, meta.VersionGA, "forwardingRules")
	return convertAndInsertAlphaForwardingRule(key, obj, m.Objects, meta.VersionGA, projectID)
}

// InsertBetaFwdRuleHook mocks inserting a BetaForwardingRule.
func InsertBetaFwdRuleHook(ctx context.Context, key *meta.Key, obj *beta.ForwardingRule, m *cloud.MockForwardingRules) (bool, error) {
	projectID := m.ProjectRouter.ProjectID(ctx, meta.VersionBeta, "forwardingRules")
	return convertAndInsertAlphaForwardingRule(key, obj, m.Objects, meta.VersionBeta, projectID)
}

// InsertAlphaFwdRuleHook mocks inserting an AlphaForwardingRule.
func InsertAlphaFwdRuleHook(ctx context.Context, key *meta.Key, obj *alpha.ForwardingRule, m *cloud.MockForwardingRules) (bool, error) {
	projectID := m.ProjectRouter.ProjectID(ctx, meta.VersionAlpha, "forwardingRules")
	return convertAndInsertAlphaForwardingRule(key, obj, m.Objects, meta.VersionAlpha, projectID)
}

// Used to assign Addresses with no IP a unique IP address
var ipCounter = 1

func convertAndInsertAlphaAddress(key *meta.Key, obj gceObject, mAddrs map[meta.Key]*cloud.MockAddressesObj, version meta.Version, projectID string) (bool, error) {
	if !key.Valid() {
		return false, fmt.Errorf("invalid GCE key (%+v)", key)
	}

	if _, ok := mAddrs[*key]; ok {
		err := &googleapi.Error{
			Code:    http.StatusConflict,
			Message: fmt.Sprintf("MockAddresses %v exists", key),
		}
		return false, err
	}

	enc, err := obj.MarshalJSON()
	if err != nil {
		return false, err
	}
	var addr alpha.Address
	if err := json.Unmarshal(enc, &addr); err != nil {
		return false, err
	}

	// Set default address type if not present.
	if addr.AddressType == "" {
		addr.AddressType = string(cloud.SchemeExternal)
	}

	var existingAddresses []*ga.Address
	for _, obj := range mAddrs {
		existingAddresses = append(existingAddresses, obj.ToGA())
	}

	for _, existingAddr := range existingAddresses {
		if addr.Address == existingAddr.Address {
			msg := fmt.Sprintf("MockAddresses IP %v in use", addr.Address)

			// When the IP is already in use, this call returns a StatusBadRequest
			// if the address is an external address, and StatusConflict if an
			// internal address. This is to be consistent with actual GCE API.
			errorCode := http.StatusConflict
			if addr.AddressType == string(cloud.SchemeExternal) {
				errorCode = http.StatusBadRequest
			}

			return false, &googleapi.Error{Code: errorCode, Message: msg}
		}
	}

	// Set default values used in tests
	addr.Name = key.Name
	if addr.SelfLink == "" {
		addr.SelfLink = cloud.SelfLink(version, projectID, "addresses", key)
	}

	if addr.Address == "" {
		addr.Address = fmt.Sprintf("1.2.3.%d", ipCounter)
		ipCounter++
	}

	// Set the default values for the Alpha fields.
	if addr.NetworkTier == "" {
		addr.NetworkTier = cloud.NetworkTierDefault.ToGCEValue()
	}

	mAddrs[*key] = &cloud.MockAddressesObj{Obj: addr}
	return true, nil
}

// InsertAddressHook mocks inserting an Address.
func InsertAddressHook(ctx context.Context, key *meta.Key, obj *ga.Address, m *cloud.MockAddresses) (bool, error) {
	projectID := m.ProjectRouter.ProjectID(ctx, meta.VersionGA, "addresses")
	return convertAndInsertAlphaAddress(key, obj, m.Objects, meta.VersionGA, projectID)
}

// InsertBetaAddressHook mocks inserting a BetaAddress.
func InsertBetaAddressHook(ctx context.Context, key *meta.Key, obj *beta.Address, m *cloud.MockAddresses) (bool, error) {
	projectID := m.ProjectRouter.ProjectID(ctx, meta.VersionBeta, "addresses")
	return convertAndInsertAlphaAddress(key, obj, m.Objects, meta.VersionBeta, projectID)
}

// InsertAlphaAddressHook mocks inserting an Address. Addresses are expected to
// default to Premium tier if no NetworkTier is specified.
func InsertAlphaAddressHook(ctx context.Context, key *meta.Key, obj *alpha.Address, m *cloud.MockAlphaAddresses) (bool, error) {
	projectID := m.ProjectRouter.ProjectID(ctx, meta.VersionBeta, "addresses")
	return convertAndInsertAlphaAddress(key, obj, m.Objects, meta.VersionAlpha, projectID)
}
