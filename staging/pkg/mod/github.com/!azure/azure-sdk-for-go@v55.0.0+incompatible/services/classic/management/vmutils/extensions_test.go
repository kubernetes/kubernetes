// +build go1.7

package vmutils

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

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
