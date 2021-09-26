package adal

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"runtime"
	"strings"
	"testing"
)

const MockTokenJSON string = `{
	"access_token": "accessToken",
	"refresh_token": "refreshToken",
	"expires_in": "1000",
	"expires_on": "2000",
	"not_before": "3000",
	"resource": "resource",
	"token_type": "type"
}`

var TestToken = Token{
	AccessToken:  "accessToken",
	RefreshToken: "refreshToken",
	ExpiresIn:    "1000",
	ExpiresOn:    "2000",
	NotBefore:    "3000",
	Resource:     "resource",
	Type:         "type",
}

func writeTestTokenFile(t *testing.T, suffix string, contents string) *os.File {
	f, err := ioutil.TempFile(os.TempDir(), suffix)
	if err != nil {
		t.Fatalf("azure: unexpected error when creating temp file: %v", err)
	}
	defer f.Close()

	_, err = f.Write([]byte(contents))
	if err != nil {
		t.Fatalf("azure: unexpected error when writing temp test file: %v", err)
	}

	return f
}

func TestLoadToken(t *testing.T) {
	f := writeTestTokenFile(t, "testloadtoken", MockTokenJSON)
	defer os.Remove(f.Name())

	expectedToken := TestToken
	actualToken, err := LoadToken(f.Name())
	if err != nil {
		t.Fatalf("azure: unexpected error loading token from file: %v", err)
	}

	if *actualToken != expectedToken {
		t.Fatalf("azure: failed to decode properly expected(%v) actual(%v)", expectedToken, *actualToken)
	}

	// test that LoadToken closes the file properly
	err = SaveToken(f.Name(), 0600, *actualToken)
	if err != nil {
		t.Fatalf("azure: could not save token after LoadToken: %v", err)
	}
}

func TestLoadTokenFailsBadPath(t *testing.T) {
	_, err := LoadToken("/tmp/this_file_should_never_exist_really")
	expectedSubstring := "failed to open file"
	if err == nil || !strings.Contains(err.Error(), expectedSubstring) {
		t.Fatalf("azure: failed to get correct error expected(%s) actual(%s)", expectedSubstring, err.Error())
	}
}

func TestLoadTokenFailsBadJson(t *testing.T) {
	gibberishJSON := strings.Replace(MockTokenJSON, "expires_on", ";:\"gibberish", -1)
	f := writeTestTokenFile(t, "testloadtokenfailsbadjson", gibberishJSON)
	defer os.Remove(f.Name())

	_, err := LoadToken(f.Name())
	expectedSubstring := "failed to decode contents of file"
	if err == nil || !strings.Contains(err.Error(), expectedSubstring) {
		t.Fatalf("azure: failed to get correct error expected(%s) actual(%s)", expectedSubstring, err.Error())
	}
}

func token() *Token {
	var token Token
	json.Unmarshal([]byte(MockTokenJSON), &token)
	return &token
}

func TestSaveToken(t *testing.T) {
	f, err := ioutil.TempFile("", "testloadtoken")
	if err != nil {
		t.Fatalf("azure: unexpected error when creating temp file: %v", err)
	}
	defer os.Remove(f.Name())
	f.Close()

	mode := os.ModePerm & 0642
	err = SaveToken(f.Name(), mode, *token())
	if err != nil {
		t.Fatalf("azure: unexpected error saving token to file: %v", err)
	}
	fi, err := os.Stat(f.Name()) // open a new stat as held ones are not fresh
	if err != nil {
		t.Fatalf("azure: stat failed: %v", err)
	}
	if runtime.GOOS != "windows" { // permissions don't work on Windows
		if perm := fi.Mode().Perm(); perm != mode {
			t.Fatalf("azure: wrong file perm. got:%s; expected:%s file :%s", perm, mode, f.Name())
		}
	}

	var actualToken Token
	var expectedToken Token

	json.Unmarshal([]byte(MockTokenJSON), &expectedToken)

	contents, err := ioutil.ReadFile(f.Name())
	if err != nil {
		t.Fatal("!!")
	}
	json.Unmarshal(contents, &actualToken)

	if !reflect.DeepEqual(actualToken, expectedToken) {
		t.Fatal("azure: token was not serialized correctly")
	}
}

func TestSaveTokenFailsNoPermission(t *testing.T) {
	pathWhereWeShouldntHavePermission := "/usr/thiswontwork/atall"
	if runtime.GOOS == "windows" {
		pathWhereWeShouldntHavePermission = path.Join(os.Getenv("windir"), "system32\\mytokendir\\mytoken")
	}
	err := SaveToken(pathWhereWeShouldntHavePermission, 0644, *token())
	expectedSubstring := "failed to create directory"
	if err == nil || !strings.Contains(err.Error(), expectedSubstring) {
		t.Fatalf("azure: failed to get correct error expected(%s) actual(%v)", expectedSubstring, err)
	}
}

func TestSaveTokenFailsCantCreate(t *testing.T) {
	tokenPath := "/usr/thiswontwork"
	if runtime.GOOS == "windows" {
		tokenPath = path.Join(os.Getenv("windir"), "system32")
	}
	err := SaveToken(tokenPath, 0644, *token())
	expectedSubstring := "failed to create the temp file to write the token"
	if err == nil || !strings.Contains(err.Error(), expectedSubstring) {
		t.Fatalf("azure: failed to get correct error expected(%s) actual(%v)", expectedSubstring, err)
	}
}
