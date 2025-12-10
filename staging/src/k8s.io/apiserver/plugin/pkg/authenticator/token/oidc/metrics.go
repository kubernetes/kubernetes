/*
Copyright 2024 The Kubernetes Authors.

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

package oidc

import (
	"context"
	"crypto/sha256"
	"fmt"
	"sync"
	"time"

	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/utils/clock"
)

const (
	namespace = "apiserver"
	subsystem = "authentication"
)

var (
	jwtAuthenticatorLatencyMetric = metrics.NewHistogramVec(
		&metrics.HistogramOpts{
			Namespace:      namespace,
			Subsystem:      subsystem,
			Name:           "jwt_authenticator_latency_seconds",
			Help:           "Latency of jwt authentication operations in seconds. This is the time spent authenticating a token for cache miss only (i.e. when the token is not found in the cache).",
			StabilityLevel: metrics.ALPHA,
			// default histogram buckets with a 1ms starting point
			Buckets: []float64{.001, .005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10},
		},
		[]string{"result", "jwt_issuer_hash"},
	)

	jwksFetchLastTimestampSeconds = metrics.NewGaugeVec(
		&metrics.GaugeOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "jwt_authenticator_jwks_fetch_last_timestamp_seconds",
			Help: "Timestamp of the last successful or failed JWKS fetch split by result, api server identity " +
				"and jwt issuer for the JWT authenticator.",
			StabilityLevel: metrics.ALPHA,
		},
		[]string{"result", "jwt_issuer_hash", "apiserver_id_hash"},
	)

	jwksFetchLastKeySetInfo = metrics.NewDesc(
		metrics.BuildFQName(namespace, subsystem, "jwt_authenticator_jwks_fetch_last_key_set_info"),
		"Information about the last JWKS fetched by the JWT authenticator with hash as label, split by api server identity and jwt issuer.",
		[]string{"jwt_issuer_hash", "apiserver_id_hash", "hash"},
		nil,
		metrics.ALPHA,
		"",
	)
)

// jwksHashKey uniquely identifies a JWKS by issuer and API server ID
type jwksHashKey struct {
	jwtIssuerHash   string
	apiServerIDHash string
}

// jwksHashProvider manages JWKS hashes for all authenticators
type jwksHashProvider struct {
	hashes sync.Map // map[jwksHashKey]string
}

func newJWKSHashProvider() *jwksHashProvider {
	return &jwksHashProvider{}
}

func (p *jwksHashProvider) setHash(jwtIssuer, apiServerID, keySet string) {
	key := jwksHashKey{
		jwtIssuerHash:   getHash(jwtIssuer),
		apiServerIDHash: getHash(apiServerID),
	}
	jwksHash := getHash(keySet)
	p.hashes.Store(key, jwksHash)
}

func (p *jwksHashProvider) getHashes() map[jwksHashKey]string {
	result := make(map[jwksHashKey]string)
	p.hashes.Range(func(k, v interface{}) bool {
		result[k.(jwksHashKey)] = v.(string)
		return true
	})
	return result
}

func (p *jwksHashProvider) reset() {
	p.hashes.Range(func(k, v interface{}) bool {
		p.hashes.Delete(k)
		return true
	})
}

func (p *jwksHashProvider) deleteHash(jwtIssuer, apiServerID string) {
	key := jwksHashKey{
		jwtIssuerHash:   getHash(jwtIssuer),
		apiServerIDHash: getHash(apiServerID),
	}
	p.hashes.Delete(key)
}

// jwksHashCollector is a custom collector that emits JWKS hash metrics
type jwksHashCollector struct {
	metrics.BaseStableCollector
	desc         *metrics.Desc
	hashProvider *jwksHashProvider
}

func newJWKSHashCollector(desc *metrics.Desc, hashProvider *jwksHashProvider) metrics.StableCollector {
	return &jwksHashCollector{
		desc:         desc,
		hashProvider: hashProvider,
	}
}

func (c *jwksHashCollector) DescribeWithStability(ch chan<- *metrics.Desc) {
	ch <- c.desc
}

func (c *jwksHashCollector) CollectWithStability(ch chan<- metrics.Metric) {
	hashes := c.hashProvider.getHashes()
	for key, hash := range hashes {
		ch <- metrics.NewLazyConstMetric(
			c.desc,
			metrics.GaugeValue,
			1,
			key.jwtIssuerHash,
			key.apiServerIDHash,
			hash,
		)
	}
}

var registerMetrics sync.Once
var hashProvider = newJWKSHashProvider()

func RegisterMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(jwtAuthenticatorLatencyMetric)
		legacyregistry.MustRegister(jwksFetchLastTimestampSeconds)
		legacyregistry.CustomMustRegister(newJWKSHashCollector(jwksFetchLastKeySetInfo, hashProvider))
	})
}

func recordAuthenticationLatency(result, jwtIssuerHash string, duration time.Duration) {
	jwtAuthenticatorLatencyMetric.WithLabelValues(result, jwtIssuerHash).Observe(duration.Seconds())
}

func recordJWKSFetchTimestamp(result, jwtIssuer, apiServerID string) {
	jwksFetchLastTimestampSeconds.WithLabelValues(result, getHash(jwtIssuer), getHash(apiServerID)).SetToCurrentTime()
}

func recordJWKSFetchKeySetSuccess(jwtIssuer, apiServerID, keySet string) {
	recordJWKSFetchKeySetHash(jwtIssuer, apiServerID, keySet)
	recordJWKSFetchTimestamp("success", jwtIssuer, apiServerID)
}

func recordJWKSFetchKeySetFailure(jwtIssuer, apiServerID string) {
	recordJWKSFetchTimestamp("failure", jwtIssuer, apiServerID)
}

func recordJWKSFetchKeySetHash(jwtIssuer, apiServerID, keySet string) {
	hashProvider.setHash(jwtIssuer, apiServerID, keySet)
}

// DeleteJWKSFetchMetrics deletes all JWKS-related metrics for a specific issuer and API server.
// This includes the hash metric and timestamp metrics (both success and failure).
// This should be called when an issuer is removed from the configuration to clean up stale metrics.
func DeleteJWKSFetchMetrics(jwtIssuer, apiServerID string) {
	jwtIssuerHash := getHash(jwtIssuer)
	apiServerIDHash := getHash(apiServerID)

	hashProvider.deleteHash(jwtIssuer, apiServerID)

	jwksFetchLastTimestampSeconds.DeleteLabelValues("success", jwtIssuerHash, apiServerIDHash)
	jwksFetchLastTimestampSeconds.DeleteLabelValues("failure", jwtIssuerHash, apiServerIDHash)
}

func ResetMetrics() {
	jwtAuthenticatorLatencyMetric.Reset()
	jwksFetchLastTimestampSeconds.Reset()
	hashProvider.reset()
}

func getHash(data string) string {
	if len(data) > 0 {
		return fmt.Sprintf("sha256:%x", sha256.Sum256([]byte(data)))
	}
	return ""
}

func newInstrumentedAuthenticator(jwtIssuer string, delegate AuthenticatorTokenWithHealthCheck) AuthenticatorTokenWithHealthCheck {
	return newInstrumentedAuthenticatorWithClock(jwtIssuer, delegate, clock.RealClock{})
}

func newInstrumentedAuthenticatorWithClock(jwtIssuer string, delegate AuthenticatorTokenWithHealthCheck, clock clock.PassiveClock) *instrumentedAuthenticator {
	return &instrumentedAuthenticator{
		jwtIssuerHash: getHash(jwtIssuer),
		delegate:      delegate,
		clock:         clock,
	}
}

type instrumentedAuthenticator struct {
	jwtIssuerHash string
	delegate      AuthenticatorTokenWithHealthCheck
	clock         clock.PassiveClock
}

func (a *instrumentedAuthenticator) AuthenticateToken(ctx context.Context, token string) (*authenticator.Response, bool, error) {
	start := a.clock.Now()
	response, ok, err := a.delegate.AuthenticateToken(ctx, token)
	// this only happens when issuer doesn't match the authenticator
	// we don't want to record metrics for this case
	if !ok && err == nil {
		return response, ok, err
	}

	duration := a.clock.Since(start)
	if err != nil {
		recordAuthenticationLatency("failure", a.jwtIssuerHash, duration)
	} else {
		recordAuthenticationLatency("success", a.jwtIssuerHash, duration)
	}
	return response, ok, err
}

func (a *instrumentedAuthenticator) HealthCheck() error {
	return a.delegate.HealthCheck()
}
