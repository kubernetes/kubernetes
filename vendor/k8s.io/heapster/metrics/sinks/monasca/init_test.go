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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"sync"
	"testing"

	"github.com/stretchr/testify/mock"
)

type keystoneClientMock struct {
	*KeystoneClientImpl
	mock.Mock
}

func (ks *keystoneClientMock) MonascaURL() (*url.URL, error) {
	args := ks.Called()
	return args.Get(0).(*url.URL), args.Error(1)
}

func (ks *keystoneClientMock) GetToken() (string, error) {
	args := ks.Called()
	return args.String(0), args.Error(1)
}

var monascaAPIStub *httptest.Server
var keystoneAPIStub *httptest.Server

type ksAuthRequest struct {
	Auth ksAuth `json:"auth"`
}

type ksAuth struct {
	Identity ksIdentity `json:"identity"`
}

type ksIdentity struct {
	Methods  []string   `json:"methods"`
	Password ksPassword `json:"password"`
}

type ksPassword struct {
	User ksUser `json:"user"`
}

type ksUser struct {
	ID       string `json:"id"`
	Password string `json:"password"`
}

// prepare before testing
func TestMain(m *testing.M) {
	// monasca stub
	monascaMutex := &sync.Mutex{}
	defer monascaAPIStub.Close()
	monascaAPIStub = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		monascaMutex.Lock()
		defer monascaMutex.Unlock()
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/metrics":
			defer r.Body.Close()
			contents, err := ioutil.ReadAll(r.Body)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				w.Write([]byte(fmt.Sprintf("%s", err)))
				break
			}

			// umarshal & do type checking on the fly
			metrics := []metric{}
			err = json.Unmarshal(contents, &metrics)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				w.Write([]byte(fmt.Sprintf("%s", err)))
				break
			}

			// check for empty dimensions
			for _, metric := range metrics {
				for _, dimVal := range metric.Dimensions {
					if dimVal == "" {
						w.WriteHeader(http.StatusInternalServerError)
						w.Write([]byte(monEmptyDimResp))
						break
					}
				}
			}

			// check token
			token := r.Header.Get("X-Auth-Token")
			if token != testToken {
				w.WriteHeader(http.StatusUnauthorized)
				w.Write([]byte(monUnauthorizedResp))
				break
			}
			w.WriteHeader(http.StatusNoContent)
			break
		case "/versions":
			w.WriteHeader(http.StatusOK)
			break
		}
	}))

	// keystone stub
	keystoneMutex := &sync.Mutex{}
	defer keystoneAPIStub.Close()
	keystoneAPIStub = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		keystoneMutex.Lock()
		defer keystoneMutex.Unlock()
		w.Header().Add("Content-Type", "application/json")
		switch r.URL.Path {
		case "/":
			w.Write([]byte(ksVersionResp))
			break
		case "/v3/auth/tokens":
			if r.Method == "HEAD" {
				ksToken := r.Header.Get("X-Auth-Token")
				if ksToken != testToken {
					w.WriteHeader(http.StatusUnauthorized)
					break
				}
				token := r.Header.Get("X-Subject-Token")
				if token == testToken {
					// token valid
					w.WriteHeader(http.StatusNoContent)
					break
				}
				// token invalid
				w.WriteHeader(http.StatusNotFound)
				break
			}

			// read request
			defer r.Body.Close()
			contents, err := ioutil.ReadAll(r.Body)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}
			req := ksAuthRequest{}
			err = json.Unmarshal(contents, &req)
			if err != nil {
				w.WriteHeader(http.StatusInternalServerError)
				return
			}

			// authenticate
			if req.Auth.Identity.Password.User.ID != testConfig.UserID ||
				req.Auth.Identity.Password.User.Password != testConfig.Password {
				w.WriteHeader(http.StatusUnauthorized)
				return
			}

			// return a token
			w.Header().Add("X-Subject-Token", testToken)
			w.WriteHeader(http.StatusAccepted)
			w.Write([]byte(ksAuthResp))
			break
		case "/v3/services":
			w.Write([]byte(ksServicesResp))
			break
		case "/v3/endpoints":
			w.Write([]byte(ksEndpointsResp))
		default:
			w.WriteHeader(http.StatusInternalServerError)
		}
	}))
	initKeystoneRespStubs()

	testConfig.Password = "bar"
	testConfig.UserID = "0ca8f6"
	testConfig.IdentityEndpoint = keystoneAPIStub.URL + "/v3"
	os.Exit(m.Run())
}
