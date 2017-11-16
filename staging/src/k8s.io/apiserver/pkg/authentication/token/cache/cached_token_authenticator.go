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
	"time"

	utilclock "k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
)

// cacheRecord holds the three return values of the authenticator.Token AuthenticateToken method
type cacheRecord struct {
	user user.Info
	ok   bool
	err  error
}

type cachedTokenAuthenticator struct {
	authenticator authenticator.Token

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
func New(authenticator authenticator.Token, successTTL, failureTTL time.Duration) authenticator.Token {
	return newWithClock(authenticator, successTTL, failureTTL, utilclock.RealClock{})
}

func newWithClock(authenticator authenticator.Token, successTTL, failureTTL time.Duration, clock utilclock.Clock) authenticator.Token {
	return &cachedTokenAuthenticator{
		authenticator: authenticator,
		successTTL:    successTTL,
		failureTTL:    failureTTL,
		cache:         newStripedCache(32, fnvKeyFunc, func() cache { return newSimpleCache(128, clock) }),
	}
}

// AuthenticateToken implements authenticator.Token
func (a *cachedTokenAuthenticator) AuthenticateToken(token string) (user.Info, bool, error) {
	if record, ok := a.cache.get(token); ok {
		return record.user, record.ok, record.err
	}

	user, ok, err := a.authenticator.AuthenticateToken(token)

	switch {
	case ok && a.successTTL > 0:
		a.cache.set(token, &cacheRecord{user: user, ok: ok, err: err}, a.successTTL)
	case !ok && a.failureTTL > 0:
		a.cache.set(token, &cacheRecord{user: user, ok: ok, err: err}, a.failureTTL)
	}

	return user, ok, err
}
