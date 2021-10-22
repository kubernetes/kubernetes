// +build go1.7

package virtualmachine

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
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/classic/management/testutils"
)

func TestAzureGetResourceExtensions(t *testing.T) {
	client := testutils.GetTestClient(t)

	list, err := NewClient(client).GetResourceExtensions()
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Found %d extensions", len(list))
	if len(list) == 0 {
		t.Fatal("Huh, no resource extensions at all? Something must be wrong.")
	}

	for _, extension := range list {
		if extension.Name == "" {
			t.Fatalf("Resource with empty name? Something must have gone wrong with serialization: %+v", extension)
		}
	}
}
