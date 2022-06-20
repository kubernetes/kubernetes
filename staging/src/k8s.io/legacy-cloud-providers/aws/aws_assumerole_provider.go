/*
Copyright 2014 The Kubernetes Authors.

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

package aws

import (
	"sync"
	"time"

	"github.com/aws/aws-sdk-go/aws/credentials"
)

const (
	invalidateCredsAfter = 1 * time.Second
)

// assumeRoleProviderWithRateLimiting makes sure we call the underlying provider only
// once after `invalidateCredsAfter` period
type assumeRoleProviderWithRateLimiting struct {
	provider             credentials.Provider
	invalidateCredsAfter time.Duration
	sync.RWMutex
	lastError        error
	lastValue        credentials.Value
	lastRetrieveTime time.Time
}

func assumeRoleProvider(provider credentials.Provider) credentials.Provider {
	return &assumeRoleProviderWithRateLimiting{provider: provider,
		invalidateCredsAfter: invalidateCredsAfter}
}

func (l *assumeRoleProviderWithRateLimiting) Retrieve() (credentials.Value, error) {
	l.Lock()
	defer l.Unlock()
	if time.Since(l.lastRetrieveTime) < l.invalidateCredsAfter {
		if l.lastError != nil {
			return credentials.Value{}, l.lastError
		}
		return l.lastValue, nil
	}
	l.lastValue, l.lastError = l.provider.Retrieve()
	l.lastRetrieveTime = time.Now()
	return l.lastValue, l.lastError
}

func (l *assumeRoleProviderWithRateLimiting) IsExpired() bool {
	return l.provider.IsExpired()
}
