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
	"bytes"
	"fmt"
	"html/template"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestStatusz(t *testing.T) {
	tests := []struct {
		name             string
		opts             Options
		expectedTemplate *template.Template
		expectedStatus   int
	}{
		{
			name: "default",
			opts: Options{
				ComponentName: "test-component",
				StartTime:     time.Now(),
			},
			expectedTemplate: defaultTmp,
			expectedStatus:   http.StatusOK,
		},
	}

	for i, test := range tests {
		mux := http.NewServeMux()
		Statusz{}.Install(mux, test.opts)

		path := "/statusz"
		req, err := http.NewRequest("GET", fmt.Sprintf("http://example.com%s", path), nil)
		if err != nil {
			t.Fatalf("case[%d] Unexpected error: %v", i, err)
		}

		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)
		if w.Code != test.expectedStatus {
			t.Errorf("case[%d] Expected: %v, got: %v", i, test.expectedStatus, w.Code)
		}

		c := w.Header().Get("Content-Type")
		if c != "text/plain; charset=utf-8" {
			t.Errorf("case[%d] Expected: %v, got: %v", i, "text/plain", c)
		}

		data := prepareData(test.opts)
		want := new(bytes.Buffer)
		err = test.expectedTemplate.Execute(want, data)
		if err != nil {
			t.Fatalf("unexpected error while executing expected template: %v", err)
		}

		if w.Body.String() != want.String() {
			t.Errorf("case[%d] Expected:\n%v\ngot:\n%v\n", i, test.expectedTemplate, w.Body.String())
		}
	}

}

func prepareData(opts Options) struct {
	ServerName string
	StartTime  string
	Uptime     string
} {
	var data struct {
		ServerName string
		StartTime  string
		Uptime     string
	}

	data.ServerName = opts.ComponentName
	data.StartTime = opts.StartTime.Format(time.RFC1123)
	uptime := int64(time.Since(opts.StartTime).Seconds())
	data.Uptime = fmt.Sprintf("%d hr %02d min %02d sec",
		uptime/3600, (uptime/60)%60, uptime%60)

	return data
}
