/*
Copyright 2021 The Kubernetes Authors.

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

package options

import (
	"errors"
	"net"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/client-go/rest"
	clocktesting "k8s.io/utils/clock/testing"
	netutils "k8s.io/utils/net"
)

func TestEmptyMainCert(t *testing.T) {
	secureServingInfo := &server.SecureServingInfo{}
	var loopbackClientConfig *rest.Config

	s := (&SecureServingOptions{
		BindAddress: netutils.ParseIPSloppy("127.0.0.1"),
	}).WithLoopback()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("failed to listen on 127.0.0.1:0")
	}
	defer ln.Close()
	s.Listener = ln
	s.BindPort = ln.Addr().(*net.TCPAddr).Port

	if err := s.ApplyTo(&secureServingInfo, &loopbackClientConfig); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if loopbackClientConfig == nil {
		t.Errorf("unexpected empty loopbackClientConfig")
	}
	if e, a := 1, len(secureServingInfo.SNICerts); e != a {
		t.Errorf("expected %d SNICert, got %d", e, a)
	}
}

func TestApplyToConfig(t *testing.T) {
	type testcase struct {
		name    string
		expired bool
	}

	testcases := []testcase{
		{
			name: "current time not after certificate expiry",
		},
		{
			name:    "current time after certificate expiry",
			expired: true,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := server.NewConfig(serializer.NewCodecFactory(runtime.NewScheme()))

			ssowlb := (&SecureServingOptions{
				BindAddress: netutils.ParseIPSloppy("127.0.0.1"),
			}).WithLoopback()
			now := time.Date(2026, time.January, 1, 0, 0, 0, 0, time.Local)
			fakeClock := clocktesting.NewFakeClock(now)
			ssowlb.clock = fakeClock

			ln, err := net.Listen("tcp", "127.0.0.1:0")
			if err != nil {
				t.Fatalf("failed to listen on 127.0.0.1:0")
			}
			defer ln.Close() //nolint:errcheck
			ssowlb.Listener = ln
			ssowlb.BindPort = ln.Addr().(*net.TCPAddr).Port

			if err := ssowlb.ApplyToConfig(cfg); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if tc.expired {
				fakeClock.Step(maxAge + 1)
			}

			checkSet := map[string][]healthz.HealthChecker{
				"/livez":   cfg.LivezChecks,
				"/healthz": cfg.HealthzChecks,
			}

			for endpoint, checks := range checkSet {
				var foundChecker healthz.HealthChecker

				for _, check := range checks {
					if check.Name() == "loopback-serving-certificate" {
						foundChecker = check
						break
					}
				}

				if foundChecker == nil {
					t.Fatalf("loopback-serving-certificate health checker not found for %s endpoint", endpoint)
				}

				err = foundChecker.Check(nil)
				if tc.expired {
					if !errors.Is(err, LoopbackCertificateExpiredError{}) {
						t.Errorf("%s endpoint check expected error and received error do not match. expected: %v , received: %v", endpoint, LoopbackCertificateExpiredError{}, err)
					}
					return
				}

				if err != nil {
					t.Errorf("%s endpoint check received an unexpected error: %v", endpoint, err)
				}
			}
		})
	}
}
