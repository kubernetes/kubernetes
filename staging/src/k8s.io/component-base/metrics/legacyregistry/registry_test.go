/*
Copyright 2023 The Kubernetes Authors.

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

package legacyregistry

import (
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"
)

const (
	processStartTimeHeader = "Process-Start-Time-Unix"
)

func TestProcessStartTimeHeader(t *testing.T) {
	now := time.Now()
	handler := Handler()

	request, _ := http.NewRequest("GET", "/", nil)
	writer := httptest.NewRecorder()
	handler.ServeHTTP(writer, request)
	got := writer.Header().Get(processStartTimeHeader)
	gotInt, _ := strconv.ParseInt(got, 10, 64)
	if gotInt != now.Unix() {
		t.Errorf("got %d, wanted %d", gotInt, now.Unix())
	}
}
