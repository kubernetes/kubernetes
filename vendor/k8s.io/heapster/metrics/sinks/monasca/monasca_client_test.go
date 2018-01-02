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
	"net/http"
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
)

func initClientSUT() (*keystoneClientMock, Client) {
	ksClientMock := new(keystoneClientMock)
	monURL, _ := url.Parse(monascaAPIStub.URL)
	sut := &ClientImpl{ksClient: ksClientMock, monascaURL: monURL}
	return ksClientMock, sut
}

func TestValidRequest(t *testing.T) {
	// setup
	ksClientMock, sut := initClientSUT()
	ksClientMock.On("GetToken").Return(testToken, nil).Once()

	// do
	status, resp, err := sut.SendRequest("POST", "/metrics", expectedTransformed)

	// assert
	assert.NoError(t, err)
	assert.Equal(t, "", resp)
	assert.Equal(t, status, http.StatusNoContent)
	ksClientMock.AssertExpectations(t)
}

func TestBadTokenRequest(t *testing.T) {
	// setup
	ksClientMock, sut := initClientSUT()
	ksClientMock.On("GetToken").Return("blob", nil).Once()

	// do
	status, resp, err := sut.SendRequest("POST", "/metrics", expectedTransformed)

	// assert
	assert.NoError(t, err)
	assert.Equal(t, monUnauthorizedResp, resp)
	assert.Equal(t, http.StatusUnauthorized, status)
	ksClientMock.AssertExpectations(t)
}

func TestBadMetricsRequest(t *testing.T) {
	// setup
	ksClientMock, sut := initClientSUT()
	ksClientMock.On("GetToken").Return(testToken, nil).Once()

	// do
	status, _, err := sut.SendRequest("POST", "/metrics", "[1,2,3,4]")

	// assert
	assert.NoError(t, err)
	assert.Equal(t, status, http.StatusInternalServerError)
	ksClientMock.AssertExpectations(t)
}

func TestWrongURLRequest(t *testing.T) {
	// setup
	ksClientMock, sut := initClientSUT()
	ksClientMock.On("GetToken").Return(testToken, nil).Once()

	// do
	status, resp, err := sut.SendRequest("POST", "http:/malformed", expectedTransformed)

	// assert
	assert.Error(t, err)
	assert.Equal(t, "", resp)
	assert.Equal(t, status, 0)
	ksClientMock.AssertExpectations(t)
}

func TestMonascaHealthy(t *testing.T) {
	// setup
	ksClientMock, sut := initClientSUT()
	ksClientMock.On("GetToken").Return(testToken, nil).Once()

	// do
	healthy := sut.CheckHealth()

	// assert
	assert.True(t, healthy)
	ksClientMock.AssertExpectations(t)
}

func TestMonascaUnhealthy(t *testing.T) {
	// TODO: reenable once #1232 is fixed
	t.Skip("skipping test due to #1232")
	// setup
	ksClientMock := new(keystoneClientMock)
	monURL, _ := url.Parse("http://unexisting.monasca.com")
	sut := &ClientImpl{ksClient: ksClientMock, monascaURL: monURL}
	ksClientMock.On("GetToken").Return(testToken, nil).Once()

	// do
	healthy := sut.CheckHealth()

	// assert
	assert.False(t, healthy)
	ksClientMock.AssertExpectations(t)
}
