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
	fakeAuth := authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
		calledWithToken = append(calledWithToken, token)
		return &authenticator.Response{User: resultUsers[token]}, resultOk, resultErr
	})
	fakeClock := utilclock.NewFakeClock(time.Now())

	a := newWithClock(fakeAuth, true, time.Minute, 0, fakeClock)

	calledWithToken, resultUsers, resultOk, resultErr = []string{}, nil, false, nil
	a.AuthenticateToken(context.Background(), "bad1")
	a.AuthenticateToken(context.Background(), "bad2")
	a.AuthenticateToken(context.Background(), "bad3")
	a.AuthenticateToken(context.Background(), "bad1")
	a.AuthenticateToken(context.Background(), "bad2")
	a.AuthenticateToken(context.Background(), "bad3")
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
	if resp, ok, err := a.AuthenticateToken(context.Background(), "usertoken1"); err != nil || !ok || resp.User.GetName() != "user1" {
		t.Errorf("Expected user1")
	}
	if resp, ok, err := a.AuthenticateToken(context.Background(), "usertoken2"); err != nil || !ok || resp.User.GetName() != "user2" {
		t.Errorf("Expected user2")
	}
	if resp, ok, err := a.AuthenticateToken(context.Background(), "usertoken3"); err != nil || !ok || resp.User.GetName() != "user3" {
		t.Errorf("Expected user3")
	}
	if !reflect.DeepEqual(calledWithToken, []string{"usertoken1", "usertoken2", "usertoken3"}) {
		t.Errorf("Expected token calls, got %v", calledWithToken)
	}

	// reset calls, make the backend return failures
	calledWithToken = []string{}
	resultUsers, resultOk, resultErr = nil, false, nil

	// authenticate calls still succeed and backend is not hit
	if resp, ok, err := a.AuthenticateToken(context.Background(), "usertoken1"); err != nil || !ok || resp.User.GetName() != "user1" {
		t.Errorf("Expected user1")
	}
	if resp, ok, err := a.AuthenticateToken(context.Background(), "usertoken2"); err != nil || !ok || resp.User.GetName() != "user2" {
		t.Errorf("Expected user2")
	}
	if resp, ok, err := a.AuthenticateToken(context.Background(), "usertoken3"); err != nil || !ok || resp.User.GetName() != "user3" {
		t.Errorf("Expected user3")
	}
	if !reflect.DeepEqual(calledWithToken, []string{}) {
		t.Errorf("Expected no token calls, got %v", calledWithToken)
	}

	// skip forward in time
	fakeClock.Step(2 * time.Minute)

	// backend is consulted again and fails
	a.AuthenticateToken(context.Background(), "usertoken1")
	a.AuthenticateToken(context.Background(), "usertoken2")
	a.AuthenticateToken(context.Background(), "usertoken3")
	if !reflect.DeepEqual(calledWithToken, []string{"usertoken1", "usertoken2", "usertoken3"}) {
		t.Errorf("Expected token calls, got %v", calledWithToken)
	}
}

func TestCachedTokenAuthenticatorWithAudiences(t *testing.T) {
	resultUsers := make(map[string]user.Info)
	fakeAuth := authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
		auds, _ := authenticator.AudiencesFrom(ctx)
		return &authenticator.Response{User: resultUsers[auds[0]+token]}, true, nil
	})
	fakeClock := utilclock.NewFakeClock(time.Now())

	a := newWithClock(fakeAuth, true, time.Minute, 0, fakeClock)

	resultUsers["audAusertoken1"] = &user.DefaultInfo{Name: "user1"}
	resultUsers["audBusertoken1"] = &user.DefaultInfo{Name: "user1-different"}

	if u, ok, _ := a.AuthenticateToken(authenticator.WithAudiences(context.Background(), []string{"audA"}), "usertoken1"); !ok || u.User.GetName() != "user1" {
		t.Errorf("Expected user1")
	}
	if u, ok, _ := a.AuthenticateToken(authenticator.WithAudiences(context.Background(), []string{"audB"}), "usertoken1"); !ok || u.User.GetName() != "user1-different" {
		t.Errorf("Expected user1-different")
	}
}
