// +build go1.7

package vmutils

// Copyright 2017 Microsoft Corporation
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

import (
	"encoding/xml"
	"testing"

	vm "github.com/Azure/azure-sdk-for-go/services/classic/management/virtualmachine"
)

func Test_AddAzureVMExtensionConfiguration(t *testing.T) {

	role := vm.Role{}
	AddAzureVMExtensionConfiguration(&role,
		"nameOfExtension", "nameOfPublisher", "versionOfExtension", "nameOfReference", "state", []byte{1, 2, 3}, []byte{})

	data, err := xml.MarshalIndent(role, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if expected := `<Role>
  <ConfigurationSets></ConfigurationSets>
  <ResourceExtensionReferences>
    <ResourceExtensionReference>
      <ReferenceName>nameOfReference</ReferenceName>
      <Publisher>nameOfPublisher</Publisher>
      <Name>nameOfExtension</Name>
      <Version>versionOfExtension</Version>
      <ResourceExtensionParameterValues>
        <ResourceExtensionParameterValue>
          <Key>ignored</Key>
          <Value>AQID</Value>
          <Type>Public</Type>
        </ResourceExtensionParameterValue>
      </ResourceExtensionParameterValues>
      <State>state</State>
    </ResourceExtensionReference>
  </ResourceExtensionReferences>
  <DataVirtualHardDisks></DataVirtualHardDisks>
</Role>`; string(data) != expected {
		t.Fatalf("Expected %q, but got %q", expected, string(data))
	}
}
