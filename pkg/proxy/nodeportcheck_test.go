/*
Copyright 2026 The Kubernetes Authors.

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

package proxy

import (
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	klogtesting "k8s.io/klog/v2/ktesting"
)

func TestParseEphemeralPortRange(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		wantLow int
		wantHi  int
		wantErr string
	}{
		{
			name:    "normal range",
			input:   "32768 60999\n",
			wantLow: 32768,
			wantHi:  60999,
		},
		{
			name:    "range without trailing newline",
			input:   "1024 65535",
			wantLow: 1024,
			wantHi:  65535,
		},
		{
			name:    "extra whitespace",
			input:   "  32768   60999  \n",
			wantLow: 32768,
			wantHi:  60999,
		},
		{
			name:    "empty input",
			input:   "",
			wantErr: "unexpected format",
		},
		{
			name:    "only one value",
			input:   "32768",
			wantErr: "unexpected format",
		},
		{
			name:    "non-numeric low",
			input:   "abc 60999",
			wantErr: "failed to parse low value",
		},
		{
			name:    "non-numeric high",
			input:   "32768 xyz",
			wantErr: "failed to parse high value",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			low, high, err := parseEphemeralPortRange(tc.input)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("expected error containing %q, got nil", tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Fatalf("expected error containing %q, got %q", tc.wantErr, err.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if low != tc.wantLow || high != tc.wantHi {
				t.Errorf("got (%d, %d), want (%d, %d)", low, high, tc.wantLow, tc.wantHi)
			}
		})
	}
}

func makeSvcPortName(namespace, name, port string) ServicePortName {
	return ServicePortName{
		NamespacedName: types.NamespacedName{Namespace: namespace, Name: name},
		Port:           port,
		Protocol:       v1.ProtocolTCP,
	}
}

func withNodePort(np int) func(*BaseServicePortInfo) {
	return func(b *BaseServicePortInfo) { b.nodePort = np }
}

func TestWarnIfNodePortsOverlapEphemeralRange(t *testing.T) {
	// ephemeral range: 32768-60999
	const low, high = 32768, 60999

	tests := []struct {
		name         string
		svcPortMap   ServicePortMap
		wantWarnings []string // substrings that must appear in the captured log output
	}{
		{
			name:         "empty map",
			svcPortMap:   ServicePortMap{},
			wantWarnings: nil,
		},
		{
			name: "no overlap - port below range",
			svcPortMap: ServicePortMap{
				makeSvcPortName("default", "svc1", "http"): makeTestServiceInfo("10.0.0.1", 80, "TCP", 0, withNodePort(30000)),
			},
			wantWarnings: nil,
		},
		{
			name: "no overlap - port above range",
			svcPortMap: ServicePortMap{
				makeSvcPortName("default", "svc1", "http"): makeTestServiceInfo("10.0.0.1", 80, "TCP", 0, withNodePort(61000)),
			},
			wantWarnings: nil,
		},
		{
			name: "NodePort at bottom of range",
			svcPortMap: ServicePortMap{
				makeSvcPortName("default", "svc1", "http"): makeTestServiceInfo("10.0.0.1", 80, "TCP", 0, withNodePort(32768)),
			},
			wantWarnings: []string{"NodePort overlaps with ephemeral port range"},
		},
		{
			name: "NodePort at top of range",
			svcPortMap: ServicePortMap{
				makeSvcPortName("default", "svc1", "http"): makeTestServiceInfo("10.0.0.1", 80, "TCP", 0, withNodePort(60999)),
			},
			wantWarnings: []string{"NodePort overlaps with ephemeral port range"},
		},
		{
			name: "HealthCheckNodePort overlaps",
			svcPortMap: ServicePortMap{
				makeSvcPortName("default", "svc1", "http"): makeTestServiceInfo("10.0.0.1", 80, "TCP", 40000),
			},
			wantWarnings: []string{"HealthCheckNodePort overlaps with ephemeral port range"},
		},
		{
			name: "both NodePort and HCNP overlap",
			svcPortMap: ServicePortMap{
				makeSvcPortName("default", "svc1", "http"): makeTestServiceInfo("10.0.0.1", 80, "TCP", 50001, withNodePort(50000)),
			},
			wantWarnings: []string{
				"NodePort overlaps with ephemeral port range",
				"HealthCheckNodePort overlaps with ephemeral port range",
			},
		},
		{
			name: "multiple services, only one overlaps",
			svcPortMap: ServicePortMap{
				makeSvcPortName("default", "svc1", "http"): makeTestServiceInfo("10.0.0.1", 80, "TCP", 0, withNodePort(30000)),
				makeSvcPortName("default", "svc2", "http"): makeTestServiceInfo("10.0.0.2", 80, "TCP", 0, withNodePort(45000)),
			},
			wantWarnings: []string{"NodePort overlaps with ephemeral port range"},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			buf := &klogtesting.BufferTL{}
			logger := klogtesting.NewLogger(buf, klogtesting.NewConfig())

			warnNodePortOverlaps(logger, tc.svcPortMap, low, high)

			output := buf.String()
			if len(tc.wantWarnings) == 0 && output != "" {
				t.Errorf("expected no log output, got: %s", output)
			}
			for _, want := range tc.wantWarnings {
				if !strings.Contains(output, want) {
					t.Errorf("expected log output containing %q, got: %s", want, output)
				}
			}
		})
	}
}
