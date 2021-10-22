// +build go1.7

package storageservice

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See License.txt in the project root for license information.

import (
	"encoding/xml"
	"testing"
)

func Test_StorageServiceKeysResponse_Unmarshal(t *testing.T) {
	// from https://msdn.microsoft.com/en-us/library/azure/ee460785.aspx
	response := []byte(`<?xml version="1.0" encoding="utf-8"?>
  <StorageService xmlns="http://schemas.microsoft.com/windowsazure">
    <Url>storage-service-url</Url>
    <StorageServiceKeys>
      <Primary>primary-key</Primary>
      <Secondary>secondary-key</Secondary>
    </StorageServiceKeys>
  </StorageService>`)

	keysResponse := GetStorageServiceKeysResponse{}
	err := xml.Unmarshal(response, &keysResponse)
	if err != nil {
		t.Fatal(err)
	}

	if expected := "primary-key"; keysResponse.PrimaryKey != expected {
		t.Fatalf("Expected %q but got %q", expected, keysResponse.PrimaryKey)
	}
	if expected := "secondary-key"; keysResponse.SecondaryKey != expected {
		t.Fatalf("Expected %q but got %q", expected, keysResponse.SecondaryKey)
	}
}
