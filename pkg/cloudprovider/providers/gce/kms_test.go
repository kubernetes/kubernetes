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
	"testing"
)

const (
	testParentHash     = "testparenthash"
	testAltParentHash  = "testaltparenthash"
	testKeyVersionName = "testkeyname"
	testData           = "testdata"
)

// Tests creating and parsing headers used for on-disk representation of cipher text.
func TestDataHeaders(t *testing.T) {
	testService := gkmsService{
		parentHash: testParentHash,
	}

	testAltService := gkmsService{
		parentHash: testAltParentHash,
	}

	cipherWithHeader := testService.createDataWithHeader(testKeyVersionName, testData)

	cipherText, keyName, err := testService.parseDataWithHeader(cipherWithHeader)
	if err != nil {
		t.Fatal(err)
	}
	if keyName != testKeyVersionName {
		t.Fatalf("key name did not match after writing to header and reading back. Expected: %q, got %q", testKeyVersionName, keyName)
	}
	if cipherText != testData {
		t.Fatalf("cipher text did not match after writing to header and reading back. Expected: %q, got %q", testData, cipherText)
	}

	_, _, err = testAltService.parseDataWithHeader(cipherWithHeader)
	if err == nil {
		t.Fatalf("service did not throw an error when data was decrypted by a service having different parent hash")
	}
}
