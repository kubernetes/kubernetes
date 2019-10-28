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
	"testing"
	"time"

	"k8s.io/client-go/tools/metrics"
)

type mockTTLGauge struct {
	v *time.Duration
}

func (m *mockTTLGauge) Set(d *time.Duration) {
	m.v = d
}

func ptr(d time.Duration) *time.Duration {
	return &d
}

func TestCertificateExpirationTracker(t *testing.T) {
	now := time.Now()
	nowFn := func() time.Time { return now }
	mockMetric := &mockTTLGauge{}
	realMetric := metrics.ClientCertTTL
	metrics.ClientCertTTL = mockMetric
	defer func() {
		metrics.ClientCertTTL = realMetric
	}()

	tracker := &certificateExpirationTracker{m: map[*Authenticator]time.Time{}}
	tracker.report(nowFn)
	if mockMetric.v != nil {
		t.Error("empty tracker should record nil value")
	}

	firstAuthenticator := &Authenticator{}
	secondAuthenticator := &Authenticator{}
	for _, tc := range []struct {
		desc string
		auth *Authenticator
		time time.Time
		want *time.Duration
	}{
		{
			desc: "ttl for one authenticator",
			auth: firstAuthenticator,
			time: now.Add(time.Minute * 10),
			want: ptr(time.Minute * 10),
		},
		{
			desc: "second authenticator shorter ttl",
			auth: secondAuthenticator,
			time: now.Add(time.Minute * 5),
			want: ptr(time.Minute * 5),
		},
		{
			desc: "update shorter to be longer",
			auth: secondAuthenticator,
			time: now.Add(time.Minute * 15),
			want: ptr(time.Minute * 10),
		},
		{
			desc: "update shorter to be zero time",
			auth: firstAuthenticator,
			time: time.Time{},
			want: ptr(time.Minute * 15),
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
			tracker.report(nowFn)
			if mockMetric.v != nil && tc.want != nil {
				if mockMetric.v.Seconds() != tc.want.Seconds() {
					t.Errorf("got: %v; want: %v", mockMetric.v, tc.want)
				}
			} else if mockMetric.v != tc.want {
				t.Errorf("got: %v; want: %v", mockMetric.v, tc.want)
			}
		})
	}
}
