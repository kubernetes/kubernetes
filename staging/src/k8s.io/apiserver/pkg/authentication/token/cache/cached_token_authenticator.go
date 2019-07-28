/*
Copyright 2017 The Kubernetes Authors.

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

package cache

import (
	"context"
	"fmt"
	"time"

	utilclock "k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apiserver/pkg/authentication/authenticator"
)

// cacheRecord holds the three return values of the authenticator.Token AuthenticateToken method
type cacheRecord struct {
	resp *authenticator.Response
	ok   bool
	err  error
}

type cachedTokenAuthenticator struct {
	authenticator authenticator.Token

	cacheErrs  bool
	successTTL time.Duration
	failureTTL time.Duration

	cache cache
}

type cache interface {
	// given a key, return the record, and whether or not it existed
	get(key string) (value *cacheRecord, exists bool)
	// caches the record for the key
	set(key string, value *cacheRecord, ttl time.Duration)
	// removes the record for the key
	remove(key string)
}

// New returns a token authenticator that caches the results of the specified authenticator. A ttl of 0 bypasses the cache.
func New(authenticator authenticator.Token, cacheErrs bool, successTTL, failureTTL time.Duration) authenticator.Token {
	return newWithClock(authenticator, cacheErrs, successTTL, failureTTL, utilclock.RealClock{})
}

func newWithClock(authenticator authenticator.Token, cacheErrs bool, successTTL, failureTTL time.Duration, clock utilclock.Clock) authenticator.Token {
	return &cachedTokenAuthenticator{
		authenticator: authenticator,
		cacheErrs:     cacheErrs,
		successTTL:    successTTL,
		failureTTL:    failureTTL,
		cache:         newStripedCache(32, fnvHashFunc, func() cache { return newSimpleCache(128, clock) }),
	}
}

// AuthenticateToken implements authenticator.Token
func (a *cachedTokenAuthenticator) AuthenticateToken(ctx context.Context, token string) (*authenticator.Response, bool, error) {
	auds, _ := authenticator.AudiencesFrom(ctx)

	key := keyFunc(auds, token)
	if record, ok := a.cache.get(key); ok {
		return record.resp, record.ok, record.err
	}

	resp, ok, err := a.authenticator.AuthenticateToken(ctx, token)
	if !a.cacheErrs && err != nil {
		return resp, ok, err
	}

	switch {
	case ok && a.successTTL > 0:
		a.cache.set(key, &cacheRecord{resp: resp, ok: ok, err: err}, a.successTTL)
	case !ok && a.failureTTL > 0:
		a.cache.set(key, &cacheRecord{resp: resp, ok: ok, err: err}, a.failureTTL)
	}

	return resp, ok, err
}

func keyFunc(auds []string, token string) string {
	return fmt.Sprintf("%#v|%v", auds, token)
}
