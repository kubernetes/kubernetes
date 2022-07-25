/*
Copyright 2018 The Kubernetes Authors.

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

package exec

import (
	"fmt"
	"io"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/client-go/tools/metrics"
)

type mockExpiryGauge struct {
	v *time.Time
}

func (m *mockExpiryGauge) Set(t *time.Time) {
	m.v = t
}

func ptr(t time.Time) *time.Time {
	return &t
}

func TestCertificateExpirationTracker(t *testing.T) {
	now := time.Now()
	mockMetric := &mockExpiryGauge{}

	tracker := &certificateExpirationTracker{
		m:         map[*Authenticator]time.Time{},
		metricSet: mockMetric.Set,
	}

	firstAuthenticator := &Authenticator{}
	secondAuthenticator := &Authenticator{}
	for _, tc := range []struct {
		desc string
		auth *Authenticator
		time time.Time
		want *time.Time
	}{
		{
			desc: "ttl for one authenticator",
			auth: firstAuthenticator,
			time: now.Add(time.Minute * 10),
			want: ptr(now.Add(time.Minute * 10)),
		},
		{
			desc: "second authenticator shorter ttl",
			auth: secondAuthenticator,
			time: now.Add(time.Minute * 5),
			want: ptr(now.Add(time.Minute * 5)),
		},
		{
			desc: "update shorter to be longer",
			auth: secondAuthenticator,
			time: now.Add(time.Minute * 15),
			want: ptr(now.Add(time.Minute * 10)),
		},
		{
			desc: "update shorter to be zero time",
			auth: firstAuthenticator,
			time: time.Time{},
			want: ptr(now.Add(time.Minute * 15)),
		},
		{
			desc: "update last to be zero time records nil",
			auth: secondAuthenticator,
			time: time.Time{},
			want: nil,
		},
	} {
		// Must run in series as the tests build off each other.
		t.Run(tc.desc, func(t *testing.T) {
			tracker.set(tc.auth, tc.time)
			if mockMetric.v != nil && tc.want != nil {
				if !mockMetric.v.Equal(*tc.want) {
					t.Errorf("got: %s; want: %s", mockMetric.v, tc.want)
				}
			} else if mockMetric.v != tc.want {
				t.Errorf("got: %s; want: %s", mockMetric.v, tc.want)
			}
		})
	}
}

type mockCallsMetric struct {
	exitCode  int
	errorType string
}

type mockCallsMetricCounter struct {
	calls []mockCallsMetric
}

func (f *mockCallsMetricCounter) Increment(exitCode int, errorType string) {
	f.calls = append(f.calls, mockCallsMetric{exitCode: exitCode, errorType: errorType})
}

func TestCallsMetric(t *testing.T) {
	const (
		goodOutput = `{
			"kind": "ExecCredential",
			"apiVersion": "client.authentication.k8s.io/v1beta1",
			"status": {
				"token": "foo-bar"
			}
		}`
	)

	callsMetricCounter := &mockCallsMetricCounter{}
	originalExecPluginCalls := metrics.ExecPluginCalls
	t.Cleanup(func() { metrics.ExecPluginCalls = originalExecPluginCalls })
	metrics.ExecPluginCalls = callsMetricCounter

	exitCodes := []int{0, 1, 2, 0}
	var wantCallsMetrics []mockCallsMetric
	for _, exitCode := range exitCodes {
		c := api.ExecConfig{
			Command:    "./testdata/test-plugin.sh",
			APIVersion: "client.authentication.k8s.io/v1beta1",
			Env: []api.ExecEnvVar{
				{Name: "TEST_EXIT_CODE", Value: fmt.Sprintf("%d", exitCode)},
				{Name: "TEST_OUTPUT", Value: goodOutput},
			},
			InteractiveMode: api.IfAvailableExecInteractiveMode,
		}

		a, err := newAuthenticator(newCache(), func(_ int) bool { return false }, &c, nil)
		if err != nil {
			t.Fatal(err)
		}
		a.stderr = io.Discard

		// Run refresh creds twice so that our test validates that the metrics are set correctly twice
		// in a row with the same authenticator.
		refreshCreds := func() {
			if err := a.refreshCredsLocked(); (err == nil) != (exitCode == 0) {
				if err != nil {
					t.Fatalf("wanted no error, but got %q", err.Error())
				} else {
					t.Fatal("wanted error, but got nil")
				}
			}
			mockCallsMetric := mockCallsMetric{exitCode: exitCode, errorType: "no_error"}
			if exitCode != 0 {
				mockCallsMetric.errorType = "plugin_execution_error"
			}
			wantCallsMetrics = append(wantCallsMetrics, mockCallsMetric)
		}
		refreshCreds()
		refreshCreds()
	}

	// Run some iterations of the authenticator where the exec plugin fails to run to test special
	// metric values.
	refreshCreds := func(command string) {
		c := api.ExecConfig{
			Command:         command,
			APIVersion:      "client.authentication.k8s.io/v1beta1",
			InteractiveMode: api.IfAvailableExecInteractiveMode,
		}
		a, err := newAuthenticator(newCache(), func(_ int) bool { return false }, &c, nil)
		if err != nil {
			t.Fatal(err)
		}
		a.stderr = io.Discard
		if err := a.refreshCredsLocked(); err == nil {
			t.Fatal("expected the authenticator to fail because the plugin does not exist")
		}
		wantCallsMetrics = append(wantCallsMetrics, mockCallsMetric{exitCode: 1, errorType: "plugin_not_found_error"})
	}
	refreshCreds("does not exist without path slashes")
	refreshCreds("./does/not/exist/with/relative/path")
	refreshCreds("/does/not/exist/with/absolute/path")

	callsMetricComparer := cmp.Comparer(func(a, b mockCallsMetric) bool {
		return a.exitCode == b.exitCode && a.errorType == b.errorType
	})
	actuallCallsMetrics := callsMetricCounter.calls
	if diff := cmp.Diff(wantCallsMetrics, actuallCallsMetrics, callsMetricComparer); diff != "" {
		t.Fatalf("got unexpected metrics calls; -want, +got:\n%s", diff)
	}
}
