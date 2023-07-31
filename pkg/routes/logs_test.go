/*
Copyright 2021 The Kubernetes Authors.

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

package routes

import (
	"fmt"
	"github.com/emicklei/go-restful/v3"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

func TestLogFileHandler(t *testing.T) {
	oversizeFileName := fmt.Sprintf("%032768s", "a")
	tests := []struct {
		logpath      string
		expectedCode int
		expectedBody string
	}{
		{
			logpath:      oversizeFileName,
			expectedCode: http.StatusNotFound,
			expectedBody: "file not found\n",
		},
		{
			logpath:      "..",
			expectedCode: http.StatusBadRequest,
			expectedBody: "unsupported logpath\n",
		},
		{
			logpath:      "../logger/a",
			expectedCode: http.StatusBadRequest,
			expectedBody: "unsupported logpath\n",
		},
	}

	container := restful.NewContainer()
	logs := Logs{}
	logs.Install(container)

	for _, tc := range tests {
		request := httptest.NewRequest("GET", "/logs/"+tc.logpath, nil)
		response := httptest.NewRecorder()
		container.Dispatch(response, request)
		if response.Code != tc.expectedCode {
			t.Errorf("LogFileHandle failed expected status code %d actual %d", tc.expectedCode, response.Code)
		}
		if response.Body.String() != tc.expectedBody {
			t.Errorf("LogFileHandle failed expected response body %q actualt %q", tc.expectedBody, response.Body.String())
		}
	}

	emptyPath := "a/.."
	request := httptest.NewRequest("GET", "/logs/"+emptyPath, nil)
	response := httptest.NewRecorder()
	container.Dispatch(response, request)
	if response.Body.String() == "unsupported logpath\n" {
		t.Errorf("emptyPath should be allowed")
	}
}

func TestPreCheckLogFileNameLength(t *testing.T) {
	// In windows, with long file name support enabled, file names can have up to 32,767 characters.
	oversizeFileName := fmt.Sprintf("%032768s", "a")
	normalFileName := fmt.Sprintf("%0255s", "a")

	// check file with oversize name.
	if !logFileNameIsTooLong(oversizeFileName) {
		t.Error("failed to check oversize filename")
	}

	// check file with normal name which doesn't exist.
	if logFileNameIsTooLong(normalFileName) {
		t.Error("failed to check normal filename")
	}

	// check file with normal name which does exist.
	dir, err := os.MkdirTemp("", "logs")
	if err != nil {
		t.Fatal("failed to create temp dir")
	}
	defer os.RemoveAll(dir)

	normalFileName = filepath.Join(dir, normalFileName)
	f, err := os.Create(normalFileName)
	if err != nil {
		t.Error("failed to create test file")
	}
	defer os.Remove(normalFileName)
	defer f.Close()
	if logFileNameIsTooLong(normalFileName) {
		t.Error("failed to check normal filename")
	}
}
