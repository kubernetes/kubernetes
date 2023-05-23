/*
Copyright 2023 The Kubernetes Authors.

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

package spdy

import (
	"k8s.io/apimachinery/pkg/util/httpstream/spdy"
	restclient "k8s.io/client-go/rest"
	"testing"
	"time"
)

func TestRoundTripperForShouldCreateUpgraderThatUsesDefaultPingPeriod(t *testing.T) {
	_, upgrader, err := RoundTripperFor(&restclient.Config{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	spdyRoundTripper := upgrader.(*spdy.SpdyRoundTripper)
	if spdyRoundTripper.PingPeriod() != DefaultPingPeriod {
		t.Errorf("wrong ping period. expected: %v, got %v", DefaultPingPeriod, spdyRoundTripper.PingPeriod())
	}
}

func TestRoundTripperWithPingForShouldCreateUpgraderThatUsesSpecifiedPingPeriod(t *testing.T) {
	testCases := []struct {
		Name       string
		PingPeriod time.Duration
	}{
		{
			Name:       "10s ping period",
			PingPeriod: 10 * time.Second,
		},
		{
			Name:       "No pings",
			PingPeriod: 0,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.Name, func(t *testing.T) {
			_, upgrader, err := RoundTripperWithPingFor(&restclient.Config{}, tc.PingPeriod)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			spdyRoundTripper := upgrader.(*spdy.SpdyRoundTripper)
			if spdyRoundTripper.PingPeriod() != tc.PingPeriod {
				t.Errorf("wrong ping period. expected: %v, got %v", tc.PingPeriod, spdyRoundTripper.PingPeriod())
			}
		})
	}
}
