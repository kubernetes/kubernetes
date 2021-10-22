// +build go1.7

package storageservice

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
