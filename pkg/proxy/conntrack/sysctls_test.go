//go:build linux

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
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestGetConntrackMax(t *testing.T) {
	ncores := runtime.NumCPU()
	const maxLimit = 1048576
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
			expected:   min(67890*ncores, maxLimit),
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
		cfg := kubeproxyconfig.KubeProxyConntrackConfiguration{
			Min:        ptr.To(tc.min),
			MaxPerCore: ptr.To(tc.maxPerCore),
		}
		_, ctx := ktesting.NewTestContext(t)
		x, e := getConntrackMax(ctx, &cfg, ncores)
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
	max      int
	hashsize int
	err      error

	called []string
}

func (fc *fakeConntracker) GetMax(ctx context.Context) (int, error) {
	return fc.max, fc.err
}
func (fc *fakeConntracker) SetMax(ctx context.Context, max int) error {
	fc.called = append(fc.called, fmt.Sprintf("SetMax(%d)", max))
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
func (fc *fakeConntracker) GetHashsize(ctx context.Context) (int, error) {
	return fc.hashsize, fc.err
}
func (fc *fakeConntracker) SetHashsize(ctx context.Context, value int) error {
	fc.called = append(fc.called, fmt.Sprintf("SetHashsize(%d)", value))
	return fc.err
}
func (fc *fakeConntracker) DetectNumCPU() int {
	return 8
}

func TestSetupConntrack(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	tests := []struct {
		name         string
		config       kubeproxyconfig.KubeProxyConntrackConfiguration
		max          int
		hashsize     int
		conntrackErr error
		expect       []string
		wantErr      bool
	}{
		{
			name:   "do nothing if conntrack config is empty",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{},
			expect: nil,
		},
		{
			name: "SetMax is called if conntrack.maxPerCore is specified and sysctl is unset",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore: ptr.To(int32(12)),
			},
			expect: []string{"SetMax(96)", "SetHashsize(24)"},
		},
		{
			name: "SetMax is not called if sysctl value is already correct",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore: ptr.To(int32(12)),
			},
			max:      96,
			hashsize: 24,
			expect:   nil,
		},
		{
			name: "SetMax is not called if sysctl value is higher than wanted",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore: ptr.To(int32(12)),
			},
			max:      192,
			hashsize: 48,
			expect:   nil,
		},
		{
			name: "SetMax is called if sysctl value is too low",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore: ptr.To(int32(12)),
			},
			max:      48,
			hashsize: 12,
			expect:   []string{"SetMax(96)", "SetHashsize(24)"},
		},
		{
			name: "SetMax is not called if conntrack.maxPerCore is 0",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore: ptr.To(int32(0)),
			},
			expect: nil,
		},
		{
			name: "SetHashsize is called if max is correct but hashsize isn't",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore: ptr.To(int32(12)),
			},
			max:      96,
			hashsize: 0,
			expect:   []string{"SetHashsize(24)"},
		},
		{
			name: "SetHashsize is not called if max is wrong but hashsize is correct",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				MaxPerCore: ptr.To(int32(12)),
			},
			max:      48,
			hashsize: 24,
			expect:   []string{"SetMax(96)"},
		},
		{
			name: "SetTCPEstablishedTimeout is called if conntrack.tcpEstablishedTimeout is specified",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
			},
			expect: []string{"SetTCPEstablishedTimeout(5)"},
		},
		{
			name: "SetTCPEstablishedTimeout is not called if conntrack.tcpEstablishedTimeout is 0",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				TCPEstablishedTimeout: &metav1.Duration{Duration: 0 * time.Second},
			},
			expect: nil,
		},
		{
			name: "SetTCPCloseWaitTimeout is called if conntrack.tcpCloseWaitTimeout is specified",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				TCPCloseWaitTimeout: &metav1.Duration{Duration: 5 * time.Second},
			},
			expect: []string{"SetTCPCloseWaitTimeout(5)"},
		},
		{
			name: "SetTCPCloseWaitTimeout is not called if conntrack.tcpCloseWaitTimeout is 0",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				TCPCloseWaitTimeout: &metav1.Duration{Duration: 0 * time.Second},
			},
			expect: nil,
		},
		{
			name: "SetTCPBeLiberal is called if conntrack.tcpBeLiberal is true",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				TCPBeLiberal: true,
			},
			expect: []string{"SetTCPBeLiberal(1)"},
		},
		{
			name: "SetTCPBeLiberal is not called if conntrack.tcpBeLiberal is false",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				TCPBeLiberal: false,
			},
			expect: nil,
		},
		{
			name: "SetUDPTimeout is called if conntrack.udpTimeout is specified",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				UDPTimeout: metav1.Duration{Duration: 5 * time.Second},
			},
			expect: []string{"SetUDPTimeout(5)"},
		},
		{
			name: "SetUDPTimeout is called if conntrack.udpTimeout is zero",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				UDPTimeout: metav1.Duration{Duration: 0 * time.Second},
			},
			expect: nil,
		},
		{
			name: "SetUDPStreamTimeout is called if conntrack.udpStreamTimeout is specified",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				UDPStreamTimeout: metav1.Duration{Duration: 5 * time.Second},
			},
			expect: []string{"SetUDPStreamTimeout(5)"},
		},
		{
			name: "SetUDPStreamTimeout is called if conntrack.udpStreamTimeout is zero",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				UDPStreamTimeout: metav1.Duration{Duration: 0 * time.Second},
			},
			expect: nil,
		},
		{
			name: "an error is returned if conntrack.SetTCPEstablishedTimeout fails",
			config: kubeproxyconfig.KubeProxyConntrackConfiguration{
				TCPEstablishedTimeout: &metav1.Duration{Duration: 5 * time.Second},
			},
			expect:       []string{"SetTCPEstablishedTimeout(5)"},
			conntrackErr: errors.New("random error"),
			wantErr:      true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fc := &fakeConntracker{max: test.max, hashsize: test.hashsize, err: test.conntrackErr}
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

func TestGetConntrackMax_Capped(t *testing.T) {
	// Simulate 512 cores
	numCPU := 512

	maxPerCore := int32(32768)
	cfg := kubeproxyconfig.KubeProxyConntrackConfiguration{
		MaxPerCore: ptr.To(maxPerCore),
		Min:        ptr.To(int32(0)),
	}

	_, ctx := ktesting.NewTestContext(t)
	val, err := getConntrackMax(ctx, &cfg, numCPU)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// 512 * 32768 = 16,777,216. Cap is 1,048,576
	expected := 1048576
	if val != expected {
		t.Errorf("Expected capped value %d, got %d", expected, val)
	}
}
