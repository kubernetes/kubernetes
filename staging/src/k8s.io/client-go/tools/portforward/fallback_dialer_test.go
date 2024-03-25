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

package portforward

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/util/httpstream"
)

func TestFallbackDialer(t *testing.T) {
	primaryProtocol := "primary.fake.protocol"
	secondaryProtocol := "secondary.fake.protocol"
	protocols := []string{primaryProtocol, secondaryProtocol}
	// If primary dialer error is nil, then no fallback and primary negotiated protocol returned.
	primary := &fakeDialer{dialed: false, negotiatedProtocol: primaryProtocol}
	secondary := &fakeDialer{dialed: false, negotiatedProtocol: secondaryProtocol}
	fallbackDialer := NewFallbackDialer(primary, secondary, notCalled)
	_, negotiated, err := fallbackDialer.Dial(protocols...)
	assert.True(t, primary.dialed, "no fallback; primary should have dialed")
	assert.False(t, secondary.dialed, "no fallback; secondary should *not* have dialed")
	assert.Equal(t, primaryProtocol, negotiated, "primary negotiated protocol returned")
	assert.Nil(t, err, "error from primary dialer should be nil")
	// If primary dialer error is upgrade error, then fallback returning secondary dial response.
	primary = &fakeDialer{dialed: false, negotiatedProtocol: primaryProtocol, err: &httpstream.UpgradeFailureError{}}
	secondary = &fakeDialer{dialed: false, negotiatedProtocol: secondaryProtocol}
	fallbackDialer = NewFallbackDialer(primary, secondary, httpstream.IsUpgradeFailure)
	_, negotiated, err = fallbackDialer.Dial(protocols...)
	assert.True(t, primary.dialed, "fallback; primary should have dialed")
	assert.True(t, secondary.dialed, "fallback; secondary should have dialed")
	assert.Equal(t, secondaryProtocol, negotiated, "negotiated protocol is from secondary dialer")
	assert.Nil(t, err, "error from secondary dialer should be nil")
	// If primary dialer returns non-upgrade error, then primary error is returned.
	nonUpgradeErr := fmt.Errorf("This is a non-upgrade error")
	primary = &fakeDialer{dialed: false, err: nonUpgradeErr}
	secondary = &fakeDialer{dialed: false}
	fallbackDialer = NewFallbackDialer(primary, secondary, httpstream.IsUpgradeFailure)
	_, _, err = fallbackDialer.Dial(protocols...)
	assert.True(t, primary.dialed, "no fallback; primary should have dialed")
	assert.False(t, secondary.dialed, "no fallback; secondary should *not* have dialed")
	assert.Equal(t, nonUpgradeErr, err, "error is from primary dialer")
}

func notCalled(err error) bool { return false }
