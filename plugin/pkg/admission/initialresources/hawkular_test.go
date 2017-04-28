/*
Copyright 2015 The Kubernetes Authors.

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

package initialresources

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"

	assert "github.com/stretchr/testify/require"
)

const (
	testImageName    string = "hawkular/hawkular-metrics"
	testImageVersion string = "latest"
	testImageSHA     string = "b727ece3780cdd30e9a86226e520f26bcc396071ed7a86b7ef6684bb93a9f717"
	testPartialMatch string = "hawkular/hawkular-metrics:*"
)

func testImageWithVersion() string {
	return fmt.Sprintf("%s:%s", testImageName, testImageVersion)
}

func testImageWithReference() string {
	return fmt.Sprintf("%s@sha256:%s", testImageName, testImageSHA)
}

func TestTaqQuery(t *testing.T) {
	kind := api.ResourceCPU
	tQ := tagQuery(kind, testImageWithVersion(), false)

	assert.Equal(t, 2, len(tQ))
	assert.Equal(t, testPartialMatch, tQ[containerImageTag])
	assert.Equal(t, "cpu/usage", tQ[descriptorTag])

	tQe := tagQuery(kind, testImageWithVersion(), true)
	assert.Equal(t, 2, len(tQe))
	assert.Equal(t, testImageWithVersion(), tQe[containerImageTag])
	assert.Equal(t, "cpu/usage", tQe[descriptorTag])

	tQr := tagQuery(kind, testImageWithReference(), false)
	assert.Equal(t, 2, len(tQe))
	assert.Equal(t, testPartialMatch, tQr[containerImageTag])
	assert.Equal(t, "cpu/usage", tQr[descriptorTag])

	tQre := tagQuery(kind, testImageWithReference(), true)
	assert.Equal(t, 2, len(tQe))
	assert.Equal(t, testImageWithReference(), tQre[containerImageTag])
	assert.Equal(t, "cpu/usage", tQre[descriptorTag])

	kind = api.ResourceMemory
	tQ = tagQuery(kind, testImageWithReference(), true)
	assert.Equal(t, "memory/usage", tQ[descriptorTag])

	kind = api.ResourceStorage
	tQ = tagQuery(kind, testImageWithReference(), true)
	assert.Equal(t, "", tQ[descriptorTag])
}

func newSource(t *testing.T) (map[string]string, dataSource) {
	tenant := "16a8884e4c155457ee38a8901df6b536"
	reqs := make(map[string]string)

	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, tenant, r.Header.Get("Hawkular-Tenant"))
		assert.Equal(t, "Basic", r.Header.Get("Authorization")[:5])

		if strings.Contains(r.RequestURI, "counters/data") {
			assert.True(t, strings.Contains(r.RequestURI, url.QueryEscape(testImageWithVersion())))
			assert.True(t, strings.Contains(r.RequestURI, "cpu%2Fusage"))
			assert.True(t, strings.Contains(r.RequestURI, "percentiles=90"))

			reqs["counters/data"] = r.RequestURI
			fmt.Fprintf(w, ` [{"start":1444620095882,"end":1444648895882,"min":1.45,"avg":1.45,"median":1.45,"max":1.45,"percentile95th":1.45,"samples":123456,"percentiles":[{"value":7896.54,"quantile":0.9},{"value":1.45,"quantile":0.99}],"empty":false}]`)
		} else {
			reqs["unknown"] = r.RequestURI
		}
	}))

	paramUri := fmt.Sprintf("%s?user=test&pass=yep&tenant=foo&insecure=true", s.URL)

	hSource, err := newHawkularSource(paramUri)
	assert.NoError(t, err)

	return reqs, hSource
}

func TestInsecureMustBeBool(t *testing.T) {
	paramUri := fmt.Sprintf("localhost?user=test&pass=yep&insecure=foo")
	_, err := newHawkularSource(paramUri)
	if err == nil {
		t.Errorf("Expected error from newHawkularSource")
	}
}

func TestCAFileMustExist(t *testing.T) {
	paramUri := fmt.Sprintf("localhost?user=test&pass=yep&caCert=foo")
	_, err := newHawkularSource(paramUri)
	if err == nil {
		t.Errorf("Expected error from newHawkularSource")
	}
}

func TestServiceAccountIsMutuallyExclusiveWithAuth(t *testing.T) {
	paramUri := fmt.Sprintf("localhost?user=test&pass=yep&useServiceAccount=true")
	_, err := newHawkularSource(paramUri)
	if err == nil {
		t.Errorf("Expected error from newHawkularSource")
	}
}

func TestGetUsagePercentile(t *testing.T) {
	reqs, hSource := newSource(t)

	usage, samples, err := hSource.GetUsagePercentile(api.ResourceCPU, 90, testImageWithVersion(), "16a8884e4c155457ee38a8901df6b536", true, time.Now(), time.Now())
	assert.NoError(t, err)

	assert.Equal(t, 1, len(reqs))
	assert.Equal(t, "", reqs["unknown"])

	assert.Equal(t, int64(123456), int64(samples))
	assert.Equal(t, int64(7896), usage) // float64 -> int64
}
