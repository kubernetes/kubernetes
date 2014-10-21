/*
Copyright 2014 Google Inc. All rights reserved.

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

package kubectl

import (
	"bytes"
	"io"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
)

func validateAction(expectedAction, actualAction client.FakeAction, t *testing.T) {
	if !reflect.DeepEqual(expectedAction, actualAction) {
		t.Errorf("Unexpected Action: %#v, expected: %#v", actualAction, expectedAction)
	}
}

func TestLoadNamespaceInfo(t *testing.T) {
	loadNamespaceInfoTests := []struct {
		nsData string
		nsInfo *NamespaceInfo
	}{
		{
			`{"Namespace":"test"}`,
			&NamespaceInfo{Namespace: "test"},
		},
		{
			"", nil,
		},
		{
			"missing",
			&NamespaceInfo{Namespace: "default"},
		},
	}
	for _, loadNamespaceInfoTest := range loadNamespaceInfoTests {
		tt := loadNamespaceInfoTest
		nsfile, err := ioutil.TempFile("", "testNamespaceInfo")
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if tt.nsData != "missing" {
			defer os.Remove(nsfile.Name())
			defer nsfile.Close()
			_, err := nsfile.WriteString(tt.nsData)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		} else {
			nsfile.Close()
			os.Remove(nsfile.Name())
		}
		nsInfo, err := LoadNamespaceInfo(nsfile.Name())
		if len(tt.nsData) == 0 && tt.nsData != "missing" {
			if err == nil {
				t.Error("LoadNamespaceInfo didn't fail on an empty file")
			}
			continue
		}
		if tt.nsData != "missing" {
			if err != nil {
				t.Errorf("Unexpected error: %v, %v", tt.nsData, err)
			}
			if !reflect.DeepEqual(nsInfo, tt.nsInfo) {
				t.Errorf("Expected %v, got %v", tt.nsInfo, nsInfo)
			}
		}
	}
}

func TestLoadAuthInfo(t *testing.T) {
	loadAuthInfoTests := []struct {
		authData string
		authInfo *AuthInfo
		r        io.Reader
	}{
		{
			`{"user": "user", "password": "pass"}`,
			&AuthInfo{User: "user", Password: "pass"},
			nil,
		},
		{
			"", nil, nil,
		},
		{
			"missing",
			&AuthInfo{User: "user", Password: "pass"},
			bytes.NewBufferString("user\npass"),
		},
	}
	for _, loadAuthInfoTest := range loadAuthInfoTests {
		tt := loadAuthInfoTest
		aifile, err := ioutil.TempFile("", "testAuthInfo")
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if tt.authData != "missing" {
			defer os.Remove(aifile.Name())
			defer aifile.Close()
			_, err = aifile.WriteString(tt.authData)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		} else {
			aifile.Close()
			os.Remove(aifile.Name())
		}
		authInfo, err := LoadAuthInfo(aifile.Name(), tt.r)
		if len(tt.authData) == 0 && tt.authData != "missing" {
			if err == nil {
				t.Error("LoadAuthInfo didn't fail on empty file")
			}
			continue
		}
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if !reflect.DeepEqual(authInfo, tt.authInfo) {
			t.Errorf("Expected %v, got %v", tt.authInfo, authInfo)
		}
	}
}
