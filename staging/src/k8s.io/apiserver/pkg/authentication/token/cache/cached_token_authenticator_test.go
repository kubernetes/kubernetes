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
	"reflect"
	"testing"
	"time"

	utilclock "k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
)

func TestCachedTokenAuthenticator(t *testing.T) {
	var (
		calledWithToken []string

		resultUsers map[string]user.Info
		resultOk    bool
		resultErr   error
	)
	fakeAuth := authenticator.TokenFunc(func(token string) (user.Info, bool, error) {
		calledWithToken = append(calledWithToken, token)
		return resultUsers[token], resultOk, resultErr
	})
	fakeClock := utilclock.NewFakeClock(time.Now())

	a := newWithClock(fakeAuth, time.Minute, 0, fakeClock)

	calledWithToken, resultUsers, resultOk, resultErr = []string{}, nil, false, nil
	a.AuthenticateToken("bad1")
	a.AuthenticateToken("bad2")
	a.AuthenticateToken("bad3")
	a.AuthenticateToken("bad1")
	a.AuthenticateToken("bad2")
	a.AuthenticateToken("bad3")
	if !reflect.DeepEqual(calledWithToken, []string{"bad1", "bad2", "bad3", "bad1", "bad2", "bad3"}) {
		t.Errorf("Expected failing calls to bypass cache, got %v", calledWithToken)
	}

	// reset calls, make the backend return success for three user tokens
	calledWithToken = []string{}
	resultUsers, resultOk, resultErr = map[string]user.Info{}, true, nil
	resultUsers["usertoken1"] = &user.DefaultInfo{Name: "user1"}
	resultUsers["usertoken2"] = &user.DefaultInfo{Name: "user2"}
	resultUsers["usertoken3"] = &user.DefaultInfo{Name: "user3"}

	// populate cache
	if user, ok, err := a.AuthenticateToken("usertoken1"); err != nil || !ok || user.GetName() != "user1" {
		t.Errorf("Expected user1")
	}
	if user, ok, err := a.AuthenticateToken("usertoken2"); err != nil || !ok || user.GetName() != "user2" {
		t.Errorf("Expected user2")
	}
	if user, ok, err := a.AuthenticateToken("usertoken3"); err != nil || !ok || user.GetName() != "user3" {
		t.Errorf("Expected user3")
	}
	if !reflect.DeepEqual(calledWithToken, []string{"usertoken1", "usertoken2", "usertoken3"}) {
		t.Errorf("Expected token calls, got %v", calledWithToken)
	}

	// reset calls, make the backend return failures
	calledWithToken = []string{}
	resultUsers, resultOk, resultErr = nil, false, nil

	// authenticate calls still succeed and backend is not hit
	if user, ok, err := a.AuthenticateToken("usertoken1"); err != nil || !ok || user.GetName() != "user1" {
		t.Errorf("Expected user1")
	}
	if user, ok, err := a.AuthenticateToken("usertoken2"); err != nil || !ok || user.GetName() != "user2" {
		t.Errorf("Expected user2")
	}
	if user, ok, err := a.AuthenticateToken("usertoken3"); err != nil || !ok || user.GetName() != "user3" {
		t.Errorf("Expected user3")
	}
	if !reflect.DeepEqual(calledWithToken, []string{}) {
		t.Errorf("Expected no token calls, got %v", calledWithToken)
	}

	// skip forward in time
	fakeClock.Step(2 * time.Minute)

	// backend is consulted again and fails
	a.AuthenticateToken("usertoken1")
	a.AuthenticateToken("usertoken2")
	a.AuthenticateToken("usertoken3")
	if !reflect.DeepEqual(calledWithToken, []string{"usertoken1", "usertoken2", "usertoken3"}) {
		t.Errorf("Expected token calls, got %v", calledWithToken)
	}
}
