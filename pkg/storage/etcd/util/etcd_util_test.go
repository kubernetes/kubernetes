/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	etcd "github.com/coreos/etcd/client"
	"github.com/stretchr/testify/assert"
)

const validEtcdVersion = "etcd 2.0.9"

func TestIsEtcdNotFound(t *testing.T) {
	try := func(err error, isNotFound bool) {
		if IsEtcdNotFound(err) != isNotFound {
			t.Errorf("Expected %#v to return %v, but it did not", err, isNotFound)
		}
	}
	try(&etcd.Error{Code: 101}, false)
	try(nil, false)
	try(fmt.Errorf("some other kind of error"), false)
}

func TestGetEtcdVersion_ValidVersion(t *testing.T) {
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, validEtcdVersion)
	}))
	defer testServer.Close()

	var version string
	var err error
	if version, err = GetEtcdVersion(testServer.URL); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	assert.Equal(t, validEtcdVersion, version, "Unexpected version")
	assert.Nil(t, err)
}

func TestGetEtcdVersion_ErrorStatus(t *testing.T) {
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer testServer.Close()

	_, err := GetEtcdVersion(testServer.URL)
	assert.NotNil(t, err)
}

func TestGetEtcdVersion_NotListening(t *testing.T) {
	portIsOpen := func(port int) bool {
		conn, err := net.DialTimeout("tcp", "127.0.0.1:"+strconv.Itoa(port), 1*time.Second)
		if err == nil {
			conn.Close()
			return true
		}
		return false
	}

	port := rand.Intn((1 << 16) - 1)
	for tried := 0; portIsOpen(port); tried++ {
		if tried >= 10 {
			t.Fatal("Couldn't find a closed TCP port to continue testing")
		}
		port++
	}

	_, err := GetEtcdVersion("http://127.0.0.1:" + strconv.Itoa(port))
	assert.NotNil(t, err)
}

func TestEtcdHealthCheck(t *testing.T) {
	tests := []struct {
		data      string
		expectErr bool
	}{
		{
			data:      "{\"health\": \"true\"}",
			expectErr: false,
		},
		{
			data:      "{\"health\": \"false\"}",
			expectErr: true,
		},
		{
			data:      "invalid json",
			expectErr: true,
		},
	}
	for _, test := range tests {
		err := EtcdHealthCheck([]byte(test.data))
		if err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v", err)
		}
		if err == nil && test.expectErr {
			t.Error("unexpected non-error")
		}
	}
}
