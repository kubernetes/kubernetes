/*
Copyright 2018 The Kubernetes Authors.

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

package azure

import (
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIsMasterNode(t *testing.T) {
	if isMasterNode(&v1.Node{}) {
		t.Errorf("Empty node should not be master!")
	}
	if isMasterNode(&v1.Node{
		ObjectMeta: meta.ObjectMeta{
			Labels: map[string]string{
				nodeLabelRole: "worker",
			},
		},
	}) {
		t.Errorf("Node labelled 'worker' should not be master!")
	}
	if !isMasterNode(&v1.Node{
		ObjectMeta: meta.ObjectMeta{
			Labels: map[string]string{
				nodeLabelRole: "master",
			},
		},
	}) {
		t.Errorf("Node should be master!")
	}
}

func TestGetLastSegment(t *testing.T) {
	tests := []struct {
		ID        string
		expected  string
		expectErr bool
	}{
		{
			ID:        "",
			expected:  "",
			expectErr: true,
		},
		{
			ID:        "foo/",
			expected:  "",
			expectErr: true,
		},
		{
			ID:        "foo/bar",
			expected:  "bar",
			expectErr: false,
		},
		{
			ID:        "foo/bar/baz",
			expected:  "baz",
			expectErr: false,
		},
	}

	for _, test := range tests {
		s, e := getLastSegment(test.ID)
		if test.expectErr && e == nil {
			t.Errorf("Expected err, but it was nil")
			continue
		}
		if !test.expectErr && e != nil {
			t.Errorf("Unexpected error: %v", e)
			continue
		}
		if s != test.expected {
			t.Errorf("expected: %s, got %s", test.expected, s)
		}
	}
}

func TestGenerateStorageAccountName(t *testing.T) {
	tests := []struct {
		prefix string
	}{
		{
			prefix: "",
		},
		{
			prefix: "pvc",
		},
		{
			prefix: "1234512345123451234512345",
		},
	}

	for _, test := range tests {
		accountName := generateStorageAccountName(test.prefix)
		if len(accountName) > storageAccountNameMaxLength || len(accountName) < 3 {
			t.Errorf("input prefix: %s, output account name: %s, length not in [3,%d]", test.prefix, accountName, storageAccountNameMaxLength)
		}

		for _, char := range accountName {
			if (char < 'a' || char > 'z') && (char < '0' || char > '9') {
				t.Errorf("input prefix: %s, output account name: %s, there is non-digit or non-letter(%q)", test.prefix, accountName, char)
				break
			}
		}
	}
}

func TestMapLoadBalancerNameToVMSet(t *testing.T) {
	az := getTestCloud()
	az.PrimaryAvailabilitySetName = "primary"

	cases := []struct {
		description   string
		lbName        string
		useStandardLB bool
		clusterName   string
		expectedVMSet string
	}{
		{
			description:   "default external LB should map to primary vmset",
			lbName:        "azure",
			clusterName:   "azure",
			expectedVMSet: "primary",
		},
		{
			description:   "default internal LB should map to primary vmset",
			lbName:        "azure-internal",
			clusterName:   "azure",
			expectedVMSet: "primary",
		},
		{
			description:   "non-default external LB should map to its own vmset",
			lbName:        "azuretest-internal",
			clusterName:   "azure",
			expectedVMSet: "azuretest",
		},
		{
			description:   "non-default internal LB should map to its own vmset",
			lbName:        "azuretest-internal",
			clusterName:   "azure",
			expectedVMSet: "azuretest",
		},
	}

	for _, c := range cases {
		if c.useStandardLB {
			az.Config.LoadBalancerSku = loadBalancerSkuStandard
		} else {
			az.Config.LoadBalancerSku = loadBalancerSkuBasic
		}
		vmset := az.mapLoadBalancerNameToVMSet(c.lbName, c.clusterName)
		assert.Equal(t, c.expectedVMSet, vmset, c.description)
	}
}

func TestGetAzureLoadBalancerName(t *testing.T) {
	az := getTestCloud()
	az.PrimaryAvailabilitySetName = "primary"

	cases := []struct {
		description   string
		vmSet         string
		isInternal    bool
		useStandardLB bool
		clusterName   string
		expected      string
	}{
		{
			description: "default external LB should get primary vmset",
			vmSet:       "primary",
			clusterName: "azure",
			expected:    "azure",
		},
		{
			description: "default internal LB should get primary vmset",
			vmSet:       "primary",
			clusterName: "azure",
			isInternal:  true,
			expected:    "azure-internal",
		},
		{
			description: "non-default external LB should get its own vmset",
			vmSet:       "as",
			clusterName: "azure",
			expected:    "as",
		},
		{
			description: "non-default internal LB should get its own vmset",
			vmSet:       "as",
			clusterName: "azure",
			isInternal:  true,
			expected:    "as-internal",
		},
		{
			description:   "default standard external LB should get cluster name",
			vmSet:         "primary",
			useStandardLB: true,
			clusterName:   "azure",
			expected:      "azure",
		},
		{
			description:   "default standard internal LB should get cluster name",
			vmSet:         "primary",
			useStandardLB: true,
			isInternal:    true,
			clusterName:   "azure",
			expected:      "azure-internal",
		},
		{
			description:   "non-default standard external LB should get cluster-name",
			vmSet:         "as",
			useStandardLB: true,
			clusterName:   "azure",
			expected:      "azure",
		},
		{
			description:   "non-default standard internal LB should get cluster-name",
			vmSet:         "as",
			useStandardLB: true,
			isInternal:    true,
			clusterName:   "azure",
			expected:      "azure-internal",
		},
	}

	for _, c := range cases {
		if c.useStandardLB {
			az.Config.LoadBalancerSku = loadBalancerSkuStandard
		} else {
			az.Config.LoadBalancerSku = loadBalancerSkuBasic
		}
		loadbalancerName := az.getAzureLoadBalancerName(c.clusterName, c.vmSet, c.isInternal)
		assert.Equal(t, c.expected, loadbalancerName, c.description)
	}
}
