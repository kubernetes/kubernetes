/*
Copyright 2020 The Kubernetes Authors.

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

package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGetNodenameForKernel(t *testing.T) {
	testcases := []struct {
		description       string
		hostname          string
		hostDomain        string
		setHostnameAsFQDN bool
		expectedHostname  string
		expectError       bool
	}{{
		description:       "no hostDomain, setHostnameAsFQDN false",
		hostname:          "test.pod.hostname",
		hostDomain:        "",
		setHostnameAsFQDN: false,
		expectedHostname:  "test.pod.hostname",
		expectError:       false,
	}, {
		description:       "no hostDomain, setHostnameAsFQDN true",
		hostname:          "test.pod.hostname",
		hostDomain:        "",
		setHostnameAsFQDN: true,
		expectedHostname:  "test.pod.hostname",
		expectError:       false,
	}, {
		description:       "valid hostDomain, setHostnameAsFQDN false",
		hostname:          "test.pod.hostname",
		hostDomain:        "svc.subdomain.local",
		setHostnameAsFQDN: false,
		expectedHostname:  "test.pod.hostname",
		expectError:       false,
	}, {
		description:       "valid hostDomain, setHostnameAsFQDN true",
		hostname:          "test.pod.hostname",
		hostDomain:        "svc.subdomain.local",
		setHostnameAsFQDN: true,
		expectedHostname:  "test.pod.hostname.svc.subdomain.local",
		expectError:       false,
	}, {
		description:       "FQDN is too long, setHostnameAsFQDN false",
		hostname:          "1234567.1234567",                                         //8*2-1=15 chars
		hostDomain:        "1234567.1234567.1234567.1234567.1234567.1234567.1234567", //8*7-1=55 chars
		setHostnameAsFQDN: false,                                                     //FQDN=15 + 1(dot) + 55 = 71 chars
		expectedHostname:  "1234567.1234567",
		expectError:       false,
	}, {
		description:       "FQDN is too long, setHostnameAsFQDN true",
		hostname:          "1234567.1234567",                                         //8*2-1=15 chars
		hostDomain:        "1234567.1234567.1234567.1234567.1234567.1234567.1234567", //8*7-1=55 chars
		setHostnameAsFQDN: true,                                                      //FQDN=15 + 1(dot) + 55 = 71 chars
		expectedHostname:  "",
		expectError:       true,
	}}

	for _, tc := range testcases {
		t.Logf("TestCase: %q", tc.description)
		outputHostname, err := GetNodenameForKernel(tc.hostname, tc.hostDomain, &tc.setHostnameAsFQDN)
		if tc.expectError {
			assert.Error(t, err)
		} else {
			assert.NoError(t, err)
		}
		assert.Equal(t, tc.expectedHostname, outputHostname)
	}

}
