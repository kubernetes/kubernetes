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

package azure

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetVmssInstanceID(t *testing.T) {
	tests := []struct {
		msg                string
		machineName        string
		expectError        bool
		expectedInstanceID string
	}{{
		msg:         "invalid vmss instance name",
		machineName: "vmvm",
		expectError: true,
	},
		{
			msg:                "valid vmss instance name",
			machineName:        "vm00000Z",
			expectError:        false,
			expectedInstanceID: "35",
		},
	}

	for i, test := range tests {
		instanceID, err := getVmssInstanceID(test.machineName)
		if test.expectError {
			assert.Error(t, err, fmt.Sprintf("TestCase[%d]: %s", i, test.msg))
		} else {
			assert.Equal(t, test.expectedInstanceID, instanceID, fmt.Sprintf("TestCase[%d]: %s", i, test.msg))
		}
	}
}

func TestMapLoadBalancerNameToAvailabilitySet(t *testing.T) {
	az := getTestCloud()
	az.PrimaryAvailabilitySetName = "primary"

	cases := []struct {
		description   string
		lbName        string
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
			lbName:        "azuretest",
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
		vmset := az.mapLoadBalancerNameToAvailabilitySet(c.lbName, c.clusterName)
		assert.Equal(t, c.expectedVMSet, vmset, c.description)
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
