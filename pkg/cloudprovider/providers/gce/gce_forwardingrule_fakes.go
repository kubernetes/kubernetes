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

package gce

import (
	"encoding/json"
	"fmt"
	"net/http"

	computealpha "google.golang.org/api/compute/v0.alpha"
	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
)

type FakeCloudForwardingRuleService struct {
	// fwdRulesByRegionAndName
	// Outer key is for region string; inner key is for fwdRuleess name.
	fwdRulesByRegionAndName map[string]map[string]*computealpha.ForwardingRule
}

// FakeCloudForwardingRuleService Implements CloudForwardingRuleService
var _ CloudForwardingRuleService = &FakeCloudForwardingRuleService{}

func NewFakeCloudForwardingRuleService() *FakeCloudForwardingRuleService {
	return &FakeCloudForwardingRuleService{
		fwdRulesByRegionAndName: make(map[string]map[string]*computealpha.ForwardingRule),
	}
}

// SetRegionalForwardingRulees sets the fwdRuleesses of ther region. This is used for
// setting the test environment.
func (f *FakeCloudForwardingRuleService) SetRegionalForwardingRulees(region string, fwdRules []*computealpha.ForwardingRule) {
	// Reset fwdRuleesses in the region.
	f.fwdRulesByRegionAndName[region] = make(map[string]*computealpha.ForwardingRule)

	for _, fwdRule := range fwdRules {
		f.fwdRulesByRegionAndName[region][fwdRule.Name] = fwdRule
	}
}

func (f *FakeCloudForwardingRuleService) CreateAlphaRegionForwardingRule(fwdRule *computealpha.ForwardingRule, region string) error {
	if _, exists := f.fwdRulesByRegionAndName[region]; !exists {
		f.fwdRulesByRegionAndName[region] = make(map[string]*computealpha.ForwardingRule)
	}

	if _, exists := f.fwdRulesByRegionAndName[region][fwdRule.Name]; exists {
		return &googleapi.Error{Code: http.StatusConflict}
	}

	f.fwdRulesByRegionAndName[region][fwdRule.Name] = fwdRule
	return nil
}

func (f *FakeCloudForwardingRuleService) CreateRegionForwardingRule(fwdRule *compute.ForwardingRule, region string) error {
	alphafwdRule := convertToAlphaForwardingRule(fwdRule)
	return f.CreateAlphaRegionForwardingRule(alphafwdRule, region)
}

func (f *FakeCloudForwardingRuleService) DeleteRegionForwardingRule(name, region string) error {
	if _, exists := f.fwdRulesByRegionAndName[region]; !exists {
		return makeGoogleAPINotFoundError("")
	}

	if _, exists := f.fwdRulesByRegionAndName[region][name]; !exists {
		return makeGoogleAPINotFoundError("")
	}
	delete(f.fwdRulesByRegionAndName[region], name)
	return nil
}

func (f *FakeCloudForwardingRuleService) GetAlphaRegionForwardingRule(name, region string) (*computealpha.ForwardingRule, error) {
	if _, exists := f.fwdRulesByRegionAndName[region]; !exists {
		return nil, makeGoogleAPINotFoundError("")
	}

	if fwdRule, exists := f.fwdRulesByRegionAndName[region][name]; !exists {
		return nil, makeGoogleAPINotFoundError("")
	} else {
		return fwdRule, nil
	}
}

func (f *FakeCloudForwardingRuleService) GetRegionForwardingRule(name, region string) (*compute.ForwardingRule, error) {
	fwdRule, err := f.GetAlphaRegionForwardingRule(name, region)
	if fwdRule != nil {
		return convertToV1ForwardingRule(fwdRule), err
	}
	return nil, err
}

func (f *FakeCloudForwardingRuleService) getNetworkTierFromForwardingRule(name, region string) (string, error) {
	fwdRule, err := f.GetAlphaRegionForwardingRule(name, region)
	if err != nil {
		return "", err
	}
	return fwdRule.NetworkTier, nil
}

func convertToV1ForwardingRule(object gceObject) *compute.ForwardingRule {
	enc, err := object.MarshalJSON()
	if err != nil {
		panic(fmt.Sprintf("Failed to encode to json: %v", err))
	}
	var fwdRule compute.ForwardingRule
	if err := json.Unmarshal(enc, &fwdRule); err != nil {
		panic(fmt.Sprintf("Failed to convert GCE apiObject %v to v1 fwdRuleess: %v", object, err))
	}
	return &fwdRule
}

func convertToAlphaForwardingRule(object gceObject) *computealpha.ForwardingRule {
	enc, err := object.MarshalJSON()
	if err != nil {
		panic(fmt.Sprintf("Failed to encode to json: %v", err))
	}
	var fwdRule computealpha.ForwardingRule
	if err := json.Unmarshal(enc, &fwdRule); err != nil {
		panic(fmt.Sprintf("Failed to convert GCE apiObject %v to alpha fwdRuleess: %v", object, err))
	}
	// Set the default values for the Alpha fields.
	fwdRule.NetworkTier = NetworkTierDefault.ToGCEValue()

	return &fwdRule
}
