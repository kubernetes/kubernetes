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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/util/version"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	v1alpha1 "k8s.io/apiserver/pkg/server/statusz/api/v1alpha1"
)

const wantTmpl = `
%s statusz
Warning: This endpoint is not meant to be machine parseable, has no formatting compatibility guarantees and is for debugging purposes only.

Started: %v
Up: %s
Go version: %s
Binary version: %v
Emulation version: %v
Paths: /livez /readyz
`

const wantTmplWithoutEmulation = `
%s statusz
Warning: This endpoint is not meant to be machine parseable, has no formatting compatibility guarantees and is for debugging purposes only.

Started: %v
Up: %s
Go version: %s
Binary version: %v

Paths: /livez /readyz
`

func TestHandleStatusz(t *testing.T) {
	delimiters = []string{":"}
	fakeStartTime := time.Now()
	fakeUptime := uptime(fakeStartTime)
	fakeGoVersion := "1.21"
	fakeBvStr := "1.31"
	fakeEvStr := "1.30"
	fakeBinaryVersion := parseVersion(t, fakeBvStr)
	fakeEmulationVersion := parseVersion(t, fakeEvStr)
	fakeListedPaths := []string{"/livez/poststarthook/peer-discovery-cache-sync", "/livez/post", "/readyz/informer-sync", "/readyz/log", "/readyz/ping"}
	tests := []struct {
		name           string
		acceptHeader   string
		componentName  string
		registry       fakeRegistry
		wantStatusCode int
		wantBody       string
		wantJSONBody   *v1alpha1.Statusz
		wantWarning    bool
	}{
		{
			name:          "valid request for text/plain",
			acceptHeader:  "text/plain",
			componentName: "test-server",
			registry: fakeRegistry{
				startTime:    fakeStartTime,
				goVer:        fakeGoVersion,
				binaryVer:    fakeBinaryVersion,
				emulationVer: fakeEmulationVersion,
				listedPaths:  fakeListedPaths,
			},
			wantStatusCode: http.StatusOK,
			wantBody: fmt.Sprintf(
				wantTmpl,
				"test-server",
				fakeStartTime.Format(time.UnixDate),
				fakeUptime,
				fakeGoVersion,
				fakeBinaryVersion,
				fakeEmulationVersion,
			),
		},
		{
			name:          "valid request for v1alpha1",
			acceptHeader:  "application/json;v=v1alpha1;g=config.k8s.io;as=Statusz",
			componentName: "test-server",
			registry: fakeRegistry{
				startTime:    fakeStartTime,
				goVer:        fakeGoVersion,
				binaryVer:    fakeBinaryVersion,
				emulationVer: fakeEmulationVersion,
				listedPaths:  fakeListedPaths,
				deprecated:   map[string]bool{},
			},
			wantStatusCode: http.StatusOK,
			wantJSONBody: &v1alpha1.Statusz{
				TypeMeta: metav1.TypeMeta{
					Kind:       Kind,
					APIVersion: fmt.Sprintf("%s/%s", GroupName, Version),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-server",
				},
				StartTime:        metav1.Time{Time: fakeStartTime},
				UptimeSeconds:    int64(time.Since(fakeStartTime).Seconds()),
				GoVersion:        fakeGoVersion,
				BinaryVersion:    fakeBvStr,
				EmulationVersion: fakeEvStr,
				Paths:            []string{"/livez", "/readyz"},
			},
		},
		{
			name:          "no accept header",
			acceptHeader:  "",
			componentName: "test-server",
			registry: fakeRegistry{
				startTime:    fakeStartTime,
				goVer:        fakeGoVersion,
				binaryVer:    fakeBinaryVersion,
				emulationVer: fakeEmulationVersion,
				listedPaths:  fakeListedPaths,
			},
			wantStatusCode: http.StatusOK,
			wantBody: fmt.Sprintf(
				wantTmpl,
				"test-server",
				fakeStartTime.Format(time.UnixDate),
				fakeUptime,
				fakeGoVersion,
				fakeBinaryVersion,
				fakeEmulationVersion,
			),
		},
		{
			name:           "invalid accept header",
			acceptHeader:   "application/xml",
			componentName:  "test-server",
			wantStatusCode: http.StatusNotAcceptable,
		},
		{
			name:          "missing emulation version",
			acceptHeader:  "text/plain",
			componentName: "test-server",
			registry: fakeRegistry{
				startTime:    fakeStartTime,
				goVer:        fakeGoVersion,
				binaryVer:    fakeBinaryVersion,
				emulationVer: nil,
				listedPaths:  fakeListedPaths,
			},
			wantStatusCode: http.StatusOK,
			wantBody: fmt.Sprintf(
				wantTmplWithoutEmulation,
				"test-server",
				fakeStartTime.Format(time.UnixDate),
				fakeUptime,
				fakeGoVersion,
				fakeBinaryVersion,
			),
		},
		{
			name:           "application/json without params",
			acceptHeader:   "application/json",
			componentName:  "test-server",
			wantStatusCode: http.StatusNotAcceptable,
		},
		{
			name:           "application/json with missing as",
			acceptHeader:   "application/json;v=v1alpha1;g=config.k8s.io",
			componentName:  "test-server",
			wantStatusCode: http.StatusNotAcceptable,
		},
		{
			name:          "wildcard accept header",
			acceptHeader:  "*/*",
			componentName: "test-server",
			registry: fakeRegistry{
				startTime:    fakeStartTime,
				goVer:        fakeGoVersion,
				binaryVer:    fakeBinaryVersion,
				emulationVer: fakeEmulationVersion,
				listedPaths:  fakeListedPaths,
			},
			wantStatusCode: http.StatusOK,
			wantBody: fmt.Sprintf(
				wantTmpl,
				"test-server",
				fakeStartTime.Format(time.UnixDate),
				fakeUptime,
				fakeGoVersion,
				fakeBinaryVersion,
				fakeEmulationVersion,
			),
		},
		{
			name:          "bad json header fall back wildcard",
			acceptHeader:  "application/json;v=foo;g=config.k8s.io;as=Statusz,*/*",
			componentName: "test-server",
			registry: fakeRegistry{
				startTime:    fakeStartTime,
				goVer:        fakeGoVersion,
				binaryVer:    fakeBinaryVersion,
				emulationVer: fakeEmulationVersion,
				listedPaths:  fakeListedPaths,
			},
			wantStatusCode: http.StatusOK,
			wantBody: fmt.Sprintf(
				wantTmpl,
				"test-server",
				fakeStartTime.Format(time.UnixDate),
				fakeUptime,
				fakeGoVersion,
				fakeBinaryVersion,
				fakeEmulationVersion,
			),
		},
		{
			name:          "deprecated version request",
			acceptHeader:  "application/json;v=v1alpha1;g=config.k8s.io;as=Statusz",
			componentName: "test-server",
			registry: fakeRegistry{
				startTime:    fakeStartTime,
				goVer:        fakeGoVersion,
				binaryVer:    fakeBinaryVersion,
				emulationVer: fakeEmulationVersion,
				listedPaths:  fakeListedPaths,
				deprecated: map[string]bool{
					"v1alpha1": true,
				},
			},
			wantStatusCode: http.StatusOK,
			wantJSONBody: &v1alpha1.Statusz{
				TypeMeta: metav1.TypeMeta{
					Kind:       Kind,
					APIVersion: fmt.Sprintf("%s/%s", GroupName, Version),
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-server",
				},
				StartTime:        metav1.Time{Time: fakeStartTime},
				UptimeSeconds:    int64(time.Since(fakeStartTime).Seconds()),
				GoVersion:        fakeGoVersion,
				BinaryVersion:    fakeBvStr,
				EmulationVersion: fakeEvStr,
				Paths:            []string{"/livez", "/readyz"},
			},
			wantWarning: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Set the component name in the registry
			tt.registry.name = tt.componentName
			mux := http.NewServeMux()
			Install(mux, tt.registry)

			path := "/statusz"
			req, err := http.NewRequest(http.MethodGet, fmt.Sprintf("http://example.com%s", path), nil)
			if err != nil {
				t.Fatalf("unexpected error while creating request: %v", err)
			}
			if tt.acceptHeader != "" {
				req.Header.Set("Accept", tt.acceptHeader)
			}

			w := httptest.NewRecorder()
			mux.ServeHTTP(w, req)

			if w.Code != tt.wantStatusCode {
				t.Fatalf("want status code: %v, got: %v", tt.wantStatusCode, w.Code)
			}

			if tt.wantStatusCode == http.StatusOK {
				if tt.wantJSONBody != nil {
					var got v1alpha1.Statusz
					if err := json.Unmarshal(w.Body.Bytes(), &got); err != nil {
						t.Fatalf("unexpected error while unmarshalling response: %v", err)
					}
					if diff := cmp.Diff(*tt.wantJSONBody, got, timeEqual()); diff != "" {
						t.Errorf("Unexpected diff on response (-want,+got):\n%s", diff)
					}
					if tt.wantWarning {
						if !strings.Contains(w.Header().Get("Warning"), "deprecated") {
							t.Errorf("expected deprecation warning in header, but got: %s", w.Header().Get("Warning"))
						}
					}
				} else {
					if diff := cmp.Diff(tt.wantBody, string(w.Body.String())); diff != "" {
						t.Errorf("Unexpected diff on response (-want,+got):\n%s", diff)
					}
				}
			}
		})
	}
}

func parseVersion(t *testing.T, v string) *version.Version {
	parsed, err := version.ParseMajorMinor(v)
	if err != nil {
		t.Fatalf("error parsing binary version: %s", v)
	}

	return parsed
}

type fakeRegistry struct {
	name         string
	startTime    time.Time
	goVer        string
	binaryVer    *version.Version
	emulationVer *version.Version
	listedPaths  []string
	deprecated   map[string]bool
}

func (f fakeRegistry) componentName() string {
	return f.name
}

func (f fakeRegistry) deprecatedVersions() map[string]bool {
	return f.deprecated
}

func (f fakeRegistry) processStartTime() time.Time {
	return f.startTime
}

func (f fakeRegistry) goVersion() string {
	return f.goVer
}

func (f fakeRegistry) binaryVersion() *version.Version {
	return f.binaryVer
}

func (f fakeRegistry) emulationVersion() *version.Version {
	return f.emulationVer
}

func (f fakeRegistry) paths() []string {
	return f.listedPaths
}

func timeEqual() cmp.Option {
	return cmp.Comparer(func(expectedTime, actualTime metav1.Time) bool {
		return expectedTime.Truncate(time.Second).Equal(actualTime.Truncate(time.Second))
	})
}
