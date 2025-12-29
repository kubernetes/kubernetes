//go:build linux
// +build linux

/*
Copyright 2025 The Kubernetes Authors.

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

package conntrack

import (
	"context"
	"errors"
	"fmt"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	proxyconfigapi "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestGetConntrackMax(t *testing.T) {
	ncores := runtime.NumCPU()
	testCases := []struct {
		min        int32
		maxPerCore int32
		expected   int
		err        string
	}{
		{
			expected: 0,
		},
		{
			maxPerCore: 67890, // use this if Max is 0
			min:        1,     // avoid 0 default
			expected:   67890 * ncores,
		},
		{
			maxPerCore: 1, // ensure that Min is considered
			min:        123456,
			expected:   123456,
		},
		{
			maxPerCore: 0, // leave system setting
			min:        123456,
			expected:   0,
		},
	}

	for i, tc := range testCases {
		cfg := proxyconfigapi.KubeProxyConntrackConfiguration{
			Min:        ptr.To(tc.min),
			MaxPerCore: ptr.To(tc.maxPerCore),
		}
		_, ctx := ktesting.NewTestContext(t)
		x, e := getConntrackMax(ctx, &cfg)
		if e != nil {
			if tc.err == "" {
				t.Errorf("[%d] unexpected error: %v", i, e)
			} else if !strings.Contains(e.Error(), tc.err) {
				t.Errorf("[%d] expected an error containing %q: %v", i, tc.err, e)
			}
		} else if x != tc.expected {
			t.Errorf("[%d] expected %d, got %d", i, tc.expected, x)
		}
	}
}

type fakeConntracker struct {
	called []string
	err    error
}

// SetMax value is calculated based on the number of CPUs by getConntrackMax()
func (fc *fakeConntracker) SetMax(ctx context.Context, max int) error {
	fc.called = append(fc.called, "SetMax")
	return fc.err
}
func (fc *fakeConntracker) SetTCPEstablishedTimeout(ctx context.Context, seconds int) error {
	fc.called = append(fc.called, fmt.Sprintf("SetTCPEstablishedTimeout(%d)", seconds))
	return fc.err
}
func (fc *fakeConntracker) SetTCPCloseWaitTimeout(ctx context.Context, seconds int) error {
	fc.called = append(fc.called, fmt.Sprintf("SetTCPCloseWaitTimeout(%d)", seconds))
	return fc.err
}
func (fc *fakeConntracker) SetTCPBeLiberal(ctx context.Context, value int) error {
	fc.called = append(fc.called, fmt.Sprintf("SetTCPBeLiberal(%d)", value))
	return fc.err
}
func (fc *fakeConntracker) SetUDPTimeout(ctx context.Context, seconds int) error {
	fc.called = append(fc.called, fmt.Sprintf("SetUDPTimeout(%d)", seconds))
	return fc.err
}
func (fc *fakeConntracker) SetUDPStreamTimeout(ctx context.Context, seconds int) error {
	fc.called = append(fc.called, fmt.Sprintf("SetUDPStreamTimeout(%d)", seconds))
	return fc.err
}

func TestSetupConntrack(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	tests := []struct {
		name         string
		config       proxyconfigapi.KubeProxyConntrackConfiguration
		expect       []string
		conntrackErr error
		wantErr      bool
	}{
		{
			name:   "do nothing if conntrack config is empty",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{},
			expect: nil,
		},
		{
			name: "SetMax is called if conntrack.maxPerCore is specified",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				MaxPerCore: ptr.To(int32(12)),
			},
			expect: []string{"SetMax"},
		},
		{
			name: "SetMax is not called if conntrack.maxPerCore is 0",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				MaxPerCore: ptr.To(int32(0)),
			},
			expect: nil,
		},
		{
			name: "SetTCPEstablishedTimeout is called if conntrack.tcpEstablishedTimeout is specified",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
			},
			expect: []string{"SetTCPEstablishedTimeout(5)"},
		},
		{
			name: "SetTCPEstablishedTimeout is not called if conntrack.tcpEstablishedTimeout is 0",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPEstablishedTimeout: &metav1.Duration{Duration: 0 * time.Second},
			},
			expect: nil,
		},
		{
			name: "SetTCPCloseWaitTimeout is called if conntrack.tcpCloseWaitTimeout is specified",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPCloseWaitTimeout: &metav1.Duration{Duration: 5 * time.Second},
			},
			expect: []string{"SetTCPCloseWaitTimeout(5)"},
		},
		{
			name: "SetTCPCloseWaitTimeout is not called if conntrack.tcpCloseWaitTimeout is 0",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPCloseWaitTimeout: &metav1.Duration{Duration: 0 * time.Second},
			},
			expect: nil,
		},
		{
			name: "SetTCPBeLiberal is called if conntrack.tcpBeLiberal is true",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPBeLiberal: true,
			},
			expect: []string{"SetTCPBeLiberal(1)"},
		},
		{
			name: "SetTCPBeLiberal is not called if conntrack.tcpBeLiberal is false",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPBeLiberal: false,
			},
			expect: nil,
		},
		{
			name: "SetUDPTimeout is called if conntrack.udpTimeout is specified",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				UDPTimeout: metav1.Duration{Duration: 5 * time.Second},
			},
			expect: []string{"SetUDPTimeout(5)"},
		},
		{
			name: "SetUDPTimeout is called if conntrack.udpTimeout is zero",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				UDPTimeout: metav1.Duration{Duration: 0 * time.Second},
			},
			expect: nil,
		},
		{
			name: "SetUDPStreamTimeout is called if conntrack.udpStreamTimeout is specified",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				UDPStreamTimeout: metav1.Duration{Duration: 5 * time.Second},
			},
			expect: []string{"SetUDPStreamTimeout(5)"},
		},
		{
			name: "SetUDPStreamTimeout is called if conntrack.udpStreamTimeout is zero",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				UDPStreamTimeout: metav1.Duration{Duration: 0 * time.Second},
			},
			expect: nil,
		},
		{
			name: "an error is returned if conntrack.SetTCPEstablishedTimeout fails",
			config: proxyconfigapi.KubeProxyConntrackConfiguration{
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
			},
			expect:       []string{"SetTCPEstablishedTimeout(5)"},
			conntrackErr: errors.New("random error"),
			wantErr:      true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fc := &fakeConntracker{err: test.conntrackErr}
			err := setSysctls(ctx, fc, &test.config)
			if test.wantErr && err == nil {
				t.Errorf("Test %q: Expected error, got nil", test.name)
			}
			if !test.wantErr && err != nil {
				t.Errorf("Test %q: Expected no error, got %v", test.name, err)
			}
			if !cmp.Equal(fc.called, test.expect) {
				t.Errorf("Test %q: Expected conntrack calls: %v, got: %v", test.name, test.expect, fc.called)
			}
		})
	}
}
