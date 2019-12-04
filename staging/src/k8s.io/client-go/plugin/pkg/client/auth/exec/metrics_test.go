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
