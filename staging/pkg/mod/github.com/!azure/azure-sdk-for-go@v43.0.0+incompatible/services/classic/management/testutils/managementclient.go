// +build go1.7

// Package testutils contains some test utilities for the Azure SDK
package testutils

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
	"encoding/base64"
	"os"
	"testing"

	"github.com/Azure/azure-sdk-for-go/services/classic/management"
)

// GetTestClient returns a management Client for testing. Expects
// AZSUBSCRIPTIONID and AZCERTDATA to be present in the environment. AZCERTDATA
// is the base64encoded binary representation of the PEM certificate data.
func GetTestClient(t *testing.T) management.Client {
	subid := os.Getenv("AZSUBSCRIPTIONID")
	certdata := os.Getenv("AZCERTDATA")
	if subid == "" || certdata == "" {
		t.Skip("AZSUBSCRIPTIONID or AZCERTDATA not set, skipping test")
	}
	cert, err := base64.StdEncoding.DecodeString(certdata)
	if err != nil {
		t.Fatal(err)
	}

	client, err := management.NewClient(subid, cert)
	if err != nil {
		t.Fatal(err)
	}
	return testClient{client, t}
}

type testClient struct {
	management.Client
	t *testing.T
}

func chop(d []byte) string {
	const maxlen = 5000

	s := string(d)

	if len(s) > maxlen {
		return s[:maxlen] + "..."
	}
	return s
}

func (l testClient) SendAzureGetRequest(url string) ([]byte, error) {
	d, err := l.Client.SendAzureGetRequest(url)
	logOperation(l.t, "GET", url, nil, d, "", err)
	return d, err
}

func (l testClient) SendAzurePostRequest(url string, data []byte) (management.OperationID, error) {
	oid, err := l.Client.SendAzurePostRequest(url, data)
	logOperation(l.t, "POST", url, data, nil, oid, err)
	return oid, err
}

func (l testClient) SendAzurePutRequest(url string, contentType string, data []byte) (management.OperationID, error) {
	oid, err := l.Client.SendAzurePutRequest(url, contentType, data)
	logOperation(l.t, "PUT", url, data, nil, oid, err)
	return oid, err
}

func (l testClient) SendAzureDeleteRequest(url string) (management.OperationID, error) {
	oid, err := l.Client.SendAzureDeleteRequest(url)
	logOperation(l.t, "DELETE", url, nil, nil, oid, err)
	return oid, err
}

func logOperation(t *testing.T, method, url string, requestData, responseData []byte, oid management.OperationID, err error) {
	t.Logf("AZURE> %s %s\n", method, url)
	if requestData != nil {
		t.Logf("   >>> %s\n", chop(requestData))
	}
	if err != nil {
		t.Logf("   <<< ERROR: %+v\n", err)
	} else {
		if responseData != nil {
			t.Logf("   <<< %s\n", chop(responseData))
		} else {
			t.Logf("   <<< OperationID: %s\n", oid)
		}
	}
}
