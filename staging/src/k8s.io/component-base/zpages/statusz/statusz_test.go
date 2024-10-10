/*
Copyright 2024 The Kubernetes Authors.
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

package statusz

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/blang/semver/v4"
	"github.com/google/go-cmp/cmp"
)

const wantTmpl = `
----------------------------
title: Kubernetes Statusz
content_type: reference
auto_generated: true
description: Details of the status data that Kubernetes components report.
----------------------------

## Started: %v
## Up: %s

## Build Info
--------------
### Go version: %s
### Binary version: %v
### Emulation version: %v
### Minimum Compatibility version: %v

## List of useful endpoints
--------------
healthz:/healthz
livez:/livez
metrics:/metrics
readyz:/readyz
sli metrics:/metrics/slis
`

var (
	fakeStartTime               = time.Now()
	fakeUptime                  = uptime(fakeStartTime)
	fakeGoVersion               = "1.21"
	fakeBinaryVersion           = semver.Version{Major: 1, Minor: 31, Patch: 0}
	fakeEmulationVersion        = semver.Version{Major: 1, Minor: 30, Patch: 0}
	fakeMinCompatibilityVersion = semver.Version{Major: 1, Minor: 29, Patch: 0}
)

func TestStatusz(t *testing.T) {
	// Arrange
	mux := http.NewServeMux()
	reg = fakeRegistry{}
	Install(mux)

	// Act
	path := "/statusz"
	req, err := http.NewRequest("GET", fmt.Sprintf("http://example.com%s", path), nil)
	if err != nil {
		t.Fatalf("unexpected error while creating request: %v", err)
	}

	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	// Assert
	if w.Code != http.StatusOK {
		t.Fatalf("want status code: %v, got: %v", http.StatusOK, w.Code)
	}

	c := w.Header().Get("Content-Type")
	if c != "text/plain; charset=utf-8" {
		t.Fatalf("want header: %v, got: %v", "text/plain", c)
	}

	wantResp := fmt.Sprintf(
		wantTmpl,
		fakeStartTime.Format(time.UnixDate),
		fakeUptime,
		fakeGoVersion,
		fakeBinaryVersion,
		fakeEmulationVersion,
		fakeMinCompatibilityVersion,
	)
	if diff := cmp.Diff(wantResp, string(w.Body.String())); diff != "" {
		t.Errorf("Unexpected diff on response (-want,+got):\n%s", diff)
	}
}

type fakeRegistry struct {
	registry
}

func (fakeRegistry) processStartTime() time.Time {
	return fakeStartTime
}

func (fakeRegistry) goVersion() string {
	return fakeGoVersion
}

func (fakeRegistry) binaryVersion() semver.Version {
	return fakeBinaryVersion
}

func (fakeRegistry) emulationVersion() semver.Version {
	return fakeEmulationVersion
}

func (fakeRegistry) minCompatibilityVersion() semver.Version {
	return fakeMinCompatibilityVersion
}

func (fakeRegistry) usefulLinks() map[string]string {
	return map[string]string{
		"healthz":     "/healthz",
		"livez":       "/livez",
		"readyz":      "/readyz",
		"metrics":     "/metrics",
		"sli metrics": "/metrics/slis",
	}
}
