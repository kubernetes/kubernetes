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

package auth_test

import (
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	clientauth "k8s.io/client-go/1.5/tools/auth"
)

func TestLoadFromFile(t *testing.T) {
	loadAuthInfoTests := []struct {
		authData  string
		authInfo  *clientauth.Info
		expectErr bool
	}{
		{
			`{"user": "user", "password": "pass"}`,
			&clientauth.Info{User: "user", Password: "pass"},
			false,
		},
		{
			"", nil, true,
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
		authInfo, err := clientauth.LoadFromFile(aifile.Name())
		gotErr := err != nil
		if gotErr != tt.expectErr {
			t.Errorf("expected errorness: %v, actual errorness: %v", tt.expectErr, gotErr)
		}
		if !reflect.DeepEqual(authInfo, tt.authInfo) {
			t.Errorf("Expected %v, got %v", tt.authInfo, authInfo)
		}
	}
}
