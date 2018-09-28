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
	"context"
	"fmt"
	"net"
	"net/http"
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2018-04-01/compute"
	"k8s.io/apimachinery/pkg/types"
)

func setTestVirtualMachines(c *Cloud, vmList []string) {
	virtualMachineClient := c.VirtualMachinesClient.(*fakeAzureVirtualMachinesClient)
	store := map[string]map[string]compute.VirtualMachine{
		"rg": make(map[string]compute.VirtualMachine),
	}

	for i := range vmList {
		nodeName := vmList[i]
		instanceID := fmt.Sprintf("/subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/%s", nodeName)
		store["rg"][nodeName] = compute.VirtualMachine{
			Name:     &nodeName,
			ID:       &instanceID,
			Location: &c.Location,
		}
	}

	virtualMachineClient.setFakeStore(store)
}

func TestInstanceID(t *testing.T) {
	cloud := getTestCloud()
	cloud.metadata = &InstanceMetadata{}

	testcases := []struct {
		name         string
		vmList       []string
		nodeName     string
		metadataName string
		expected     string
		expectError  bool
	}{
		{
			name:         "InstanceID should get instanceID if node's name are equal to metadataName",
			vmList:       []string{"vm1"},
			nodeName:     "vm1",
			metadataName: "vm1",
			expected:     "/subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/vm1",
		},
		{
			name:         "InstanceID should get instanceID from Azure API if node is not local instance",
			vmList:       []string{"vm2"},
			nodeName:     "vm2",
			metadataName: "vm1",
			expected:     "/subscriptions/script/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/vm2",
		},
		{
			name:        "InstanceID should report error if VM doesn't exist",
			vmList:      []string{"vm1"},
			nodeName:    "vm3",
			expectError: true,
		},
	}

	for _, test := range testcases {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Errorf("Test [%s] unexpected error: %v", test.name, err)
		}

		mux := http.NewServeMux()
		mux.Handle("/instance/compute", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprintf(w, fmt.Sprintf("{\"name\":\"%s\"}", test.metadataName))
		}))
		go func() {
			http.Serve(listener, mux)
		}()
		defer listener.Close()

		cloud.metadata.baseURL = "http://" + listener.Addr().String() + "/"
		setTestVirtualMachines(cloud, test.vmList)
		instanceID, err := cloud.InstanceID(context.Background(), types.NodeName(test.nodeName))
		if test.expectError {
			if err == nil {
				t.Errorf("Test [%s] unexpected nil err", test.name)
			}
		} else {
			if err != nil {
				t.Errorf("Test [%s] unexpected error: %v", test.name, err)
			}
		}

		if instanceID != test.expected {
			t.Errorf("Test [%s] unexpected instanceID: %s, expected %q", test.name, instanceID, test.expected)
		}
	}
}
