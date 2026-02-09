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

package options

import (
	"fmt"
	"net/http"
	"time"

	"github.com/google/uuid"

	"k8s.io/apiserver/pkg/server"
	"k8s.io/apiserver/pkg/server/dynamiccertificates"
	"k8s.io/apiserver/pkg/server/healthz"
	"k8s.io/client-go/rest"
	certutil "k8s.io/client-go/util/cert"
	"k8s.io/utils/clock"
)

type SecureServingOptionsWithLoopback struct {
	*SecureServingOptions
	clock clock.PassiveClock
}

func (o *SecureServingOptions) WithLoopback() *SecureServingOptionsWithLoopback {
	return &SecureServingOptionsWithLoopback{
		SecureServingOptions: o,
		clock:                clock.RealClock{},
	}
}

// Set a validity period of approximately 3 years for the loopback certificate
// to avoid kube-apiserver disruptions due to certificate expiration.
// When this certificate expires, restarting kube-apiserver will automatically
// regenerate a new certificate with fresh validity dates.
const maxAge = (3*365 + 1) * 24 * time.Hour

// ApplyTo fills up serving information in the server configuration.
func (s *SecureServingOptionsWithLoopback) ApplyTo(secureServingInfo **server.SecureServingInfo, loopbackClientConfig **rest.Config) error {
	return s.applyTo(secureServingInfo, loopbackClientConfig, nil)
}

type HealthzLivezHealthChecksAdder interface {
	AddHealthzChecks(checks ...healthz.HealthChecker)
	AddLivezChecks(checks ...healthz.HealthChecker)
}

func (s *SecureServingOptionsWithLoopback) applyTo(secureServingInfo **server.SecureServingInfo, loopbackClientConfig **rest.Config, healthCheckAdder HealthzLivezHealthChecksAdder) error {
	if s == nil || s.SecureServingOptions == nil || secureServingInfo == nil {
		return nil
	}

	if err := s.SecureServingOptions.ApplyTo(secureServingInfo); err != nil {
		return err
	}

	if *secureServingInfo == nil || loopbackClientConfig == nil {
		return nil
	}

	// create self-signed cert+key with the fake server.LoopbackClientServerNameOverride and
	// let the server return it when the loopback client connects.
	certPem, keyPem, err := certutil.GenerateSelfSignedCertKeyWithOptions(certutil.SelfSignedCertKeyOptions{
		Host:   server.LoopbackClientServerNameOverride,
		MaxAge: maxAge,
	})
	if err != nil {
		return fmt.Errorf("failed to generate self-signed certificate for loopback connection: %v", err)
	}
	certProvider, err := dynamiccertificates.NewStaticSNICertKeyContent("self-signed loopback", certPem, keyPem, server.LoopbackClientServerNameOverride)
	if err != nil {
		return fmt.Errorf("failed to generate self-signed certificate for loopback connection: %v", err)
	}

	// Write to the front of SNICerts so that this overrides any other certs with the same name
	(*secureServingInfo).SNICerts = append([]dynamiccertificates.SNICertKeyContentProvider{certProvider}, (*secureServingInfo).SNICerts...)

	secureLoopbackClientConfig, err := (*secureServingInfo).NewLoopbackClientConfig(uuid.New().String(), certPem)
	switch {
	// if we failed and there's no fallback loopback client config, we need to fail
	case err != nil && *loopbackClientConfig == nil:
		(*secureServingInfo).SNICerts = (*secureServingInfo).SNICerts[1:]
		return err

	// if we failed, but we already have a fallback loopback client config (usually insecure), allow it
	case err != nil && *loopbackClientConfig != nil:

	default:
		*loopbackClientConfig = secureLoopbackClientConfig
		if healthCheckAdder != nil {
			s.addLoopbackServingCertificateHealthCheck(healthCheckAdder)
		}
	}

	return nil
}

func (s *SecureServingOptionsWithLoopback) ApplyToConfig(cfg *server.Config) error {
	return s.applyTo(&cfg.SecureServing, &cfg.LoopbackClientConfig, cfg)
}

// addLoopbackServingCertificateHealthCheck adds a health check called `loopback-certificate-expiry` to the
// server that fails when the loopback client certificate has expired, enabling
// liveness probes to be used to automatically restart the apiserver.
func (s *SecureServingOptionsWithLoopback) addLoopbackServingCertificateHealthCheck(healthCheckAdder HealthzLivezHealthChecksAdder) {
	expirationDate := s.clock.Now().Add(maxAge)
	check := healthz.NamedCheck("loopback-serving-certificate", func(r *http.Request) error {
		if s.clock.Now().After(expirationDate) {
			return LoopbackCertificateExpiredError{}
		}

		return nil
	})

	healthCheckAdder.AddHealthzChecks(check)
	healthCheckAdder.AddLivezChecks(check)
}

type LoopbackCertificateExpiredError struct{}

func (lcee LoopbackCertificateExpiredError) Error() string {
	return "loopback serving certificate is expired"
}
