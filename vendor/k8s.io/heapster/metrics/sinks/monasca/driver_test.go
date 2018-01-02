// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package monasca

import (
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
)

// test the transformation of timeseries to monasca metrics
func TestTimeseriesTransform(t *testing.T) {
	// setup
	sut := monascaSink{}

	// do
	metrics := sut.processMetrics(testInput)

	// assert
	set1 := map[string]metric{}
	set2 := map[string]metric{}
	for _, m := range expectedTransformed {
		set1[m.Name] = m
	}
	for _, m := range metrics {
		set2[m.Name] = m
	}
	assert.Equal(t, set1, set2)
}

// test if the sink creation fails when password is not provided
func TestMissingPasswordError(t *testing.T) {
	// setup
	uri, _ := url.Parse("monasca:?keystone-url=" + testConfig.IdentityEndpoint + "&user_id=" + testUserID)

	// do
	_, err := CreateMonascaSink(uri)

	// assert
	assert.Error(t, err)
}

// test if the sink creation fails when keystone-url is not provided
func TestMissingKeystoneURLError(t *testing.T) {
	// setup
	uri, _ := url.Parse("monasca:?user_id=" + testUserID + "&password=" + testPassword)

	// do
	_, err := CreateMonascaSink(uri)

	// assert
	assert.Error(t, err)
}

// test if the sink creation fails when neither user-id nor username are provided
func TestMissingUserError(t *testing.T) {
	// setup
	uri, _ := url.Parse("monasca:?keystone-url=" + testConfig.IdentityEndpoint + "&password=" + testPassword)

	// do
	_, err := CreateMonascaSink(uri)

	// assert
	assert.Error(t, err)
}

// test if the sink creation fails when domain_id and domainname are missing
// and username is provided
func TestMissingDomainWhenUsernameError(t *testing.T) {
	// setup
	uri, _ := url.Parse("monasca:?keystone-url=" + testConfig.IdentityEndpoint + "&password=" +
		testPassword + "&username=" + testUsername)

	// do
	_, err := CreateMonascaSink(uri)

	// assert
	assert.Error(t, err)
}

// test if the sink creation fails when password is not provided
func TestWrongMonascaURLError(t *testing.T) {
	// setup
	uri, _ := url.Parse("monasca:?keystone-url=" + testConfig.IdentityEndpoint + "&password=" +
		testConfig.Password + "&user-id=" + testConfig.UserID + "&monasca-url=_malformed")

	// do
	_, err := CreateMonascaSink(uri)

	// assert
	assert.Error(t, err)
}

// test the successful creation of the monasca
func TestMonascaSinkCreation(t *testing.T) {
	// setup
	uri, _ := url.Parse("monasca:?keystone-url=" + testConfig.IdentityEndpoint + "&password=" +
		testConfig.Password + "&user-id=" + testConfig.UserID)

	// do
	_, err := CreateMonascaSink(uri)

	// assert
	assert.NoError(t, err)
}

// integration test of storing metrics
func TestStoreMetrics(t *testing.T) {
	// setup
	ks, _ := NewKeystoneClient(testConfig)
	monURL, err := ks.MonascaURL()
	assert.NoError(t, err)
	sut := monascaSink{client: &ClientImpl{ksClient: ks, monascaURL: monURL}}

	// do
	sut.ExportData(testInput)

	// assert
	assert.Equal(t, 0, sut.numberOfFailures)
}

// integration test of failure to create metrics
func TestStoreMetricsFailure(t *testing.T) {
	// setup
	ks, _ := NewKeystoneClient(testConfig)
	monURL, _ := url.Parse("http://unexisting.monasca.com")
	sut := monascaSink{client: &ClientImpl{ksClient: ks, monascaURL: monURL}}

	// do
	sut.ExportData(testInput)

	// assert
	assert.Equal(t, 1, sut.numberOfFailures)
}
