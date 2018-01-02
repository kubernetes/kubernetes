/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package session

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"runtime"
	"testing"
	"time"

	"github.com/vmware/govmomi/test"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type testKeepAlive int

func (t *testKeepAlive) Func(soap.RoundTripper) error {
	*t++
	return nil
}

func newManager(t *testing.T) (*Manager, *url.URL) {
	u := test.URL()
	if u == nil {
		t.SkipNow()
	}

	soapClient := soap.NewClient(u, true)
	vimClient, err := vim25.NewClient(context.Background(), soapClient)
	if err != nil {
		t.Fatal(err)
	}

	return NewManager(vimClient), u
}

func TestKeepAlive(t *testing.T) {
	var i testKeepAlive
	var j int

	m, u := newManager(t)
	k := KeepAlive(m.client.RoundTripper, time.Millisecond)
	k.(*keepAlive).keepAlive = i.Func
	m.client.RoundTripper = k

	// Expect keep alive to not have triggered yet
	if i != 0 {
		t.Errorf("Expected i == 0, got i: %d", i)
	}

	// Logging in starts keep alive
	err := m.Login(context.Background(), u.User)
	if err != nil {
		t.Error(err)
	}

	time.Sleep(2 * time.Millisecond)

	// Expect keep alive to triggered at least once
	if i == 0 {
		t.Errorf("Expected i != 0, got i: %d", i)
	}

	j = int(i)
	time.Sleep(2 * time.Millisecond)

	// Expect keep alive to triggered at least once more
	if int(i) <= j {
		t.Errorf("Expected i > j, got i: %d, j: %d", i, j)
	}

	// Logging out stops keep alive
	err = m.Logout(context.Background())
	if err != nil {
		t.Error(err)
	}

	j = int(i)
	time.Sleep(2 * time.Millisecond)

	// Expect keep alive to have stopped
	if int(i) != j {
		t.Errorf("Expected i == j, got i: %d, j: %d", i, j)
	}
}

func testSessionOK(t *testing.T, m *Manager, ok bool) {
	s, err := m.UserSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	_, file, line, _ := runtime.Caller(1)
	prefix := fmt.Sprintf("%s:%d", file, line)

	if ok && s == nil {
		t.Fatalf("%s: Expected session to be OK, but is invalid", prefix)
	}

	if !ok && s != nil {
		t.Fatalf("%s: Expected session to be invalid, but is OK", prefix)
	}
}

// Run with:
//
//   env GOVMOMI_KEEPALIVE_TEST=1 go test -timeout=60m -run TestRealKeepAlive
//
func TestRealKeepAlive(t *testing.T) {
	if os.Getenv("GOVMOMI_KEEPALIVE_TEST") != "1" {
		t.SkipNow()
	}

	m1, u1 := newManager(t)
	m2, u2 := newManager(t)

	// Enable keepalive on m2
	k := KeepAlive(m2.client.RoundTripper, 10*time.Minute)
	m2.client.RoundTripper = k

	// Expect both sessions to be invalid
	testSessionOK(t, m1, false)
	testSessionOK(t, m2, false)

	// Logging in starts keep alive
	if err := m1.Login(context.Background(), u1.User); err != nil {
		t.Error(err)
	}
	if err := m2.Login(context.Background(), u2.User); err != nil {
		t.Error(err)
	}

	// Expect both sessions to be valid
	testSessionOK(t, m1, true)
	testSessionOK(t, m2, true)

	// Wait for m1 to time out
	delay := 31 * time.Minute
	fmt.Printf("%s: Waiting %d minutes for session to time out...\n", time.Now(), int(delay.Minutes()))
	time.Sleep(delay)

	// Expect m1's session to be invalid, m2's session to be valid
	testSessionOK(t, m1, false)
	testSessionOK(t, m2, true)
}

func isNotAuthenticated(err error) bool {
	if soap.IsSoapFault(err) {
		switch soap.ToSoapFault(err).VimFault().(type) {
		case types.NotAuthenticated:
			return true
		}
	}
	return false
}

func isInvalidLogin(err error) bool {
	if soap.IsSoapFault(err) {
		switch soap.ToSoapFault(err).VimFault().(type) {
		case types.InvalidLogin:
			return true
		}
	}
	return false
}

func TestKeepAliveHandler(t *testing.T) {
	u := test.URL()
	if u == nil {
		t.SkipNow()
	}

	m1, u1 := newManager(t)
	m2, u2 := newManager(t)

	reauth := make(chan bool)

	// Keep alive handler that will re-login.
	// Real-world case: connectivity to ESX/VC is down long enough for the session to expire
	// Test-world case: we call TerminateSession below
	k := KeepAliveHandler(m2.client.RoundTripper, 2*time.Second, func(roundTripper soap.RoundTripper) error {
		_, err := methods.GetCurrentTime(context.Background(), roundTripper)
		if err != nil {
			if isNotAuthenticated(err) {
				err = m2.Login(context.Background(), u2.User)

				if err != nil {
					if isInvalidLogin(err) {
						reauth <- false
						t.Log("failed to re-authenticate, quitting keep alive handler")
						return err
					}
				} else {
					reauth <- true
				}
			}
		}

		return nil
	})

	m2.client.RoundTripper = k

	// Logging in starts keep alive
	if err := m1.Login(context.Background(), u1.User); err != nil {
		t.Error(err)
	}
	if err := m2.Login(context.Background(), u2.User); err != nil {
		t.Error(err)
	}

	// Terminate session for m2.  Note that self terminate fails, so we need 2 sessions for this test.
	s, err := m2.UserSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	err = m1.TerminateSession(context.Background(), []string{s.Key})
	if err != nil {
		t.Fatal(err)
	}

	_, err = methods.GetCurrentTime(context.Background(), m2.client)
	if err == nil {
		t.Error("expected to fail")
	}

	// Wait for keepalive to re-authenticate
	<-reauth

	_, err = methods.GetCurrentTime(context.Background(), m2.client)
	if err != nil {
		t.Fatal(err)
	}

	// Clear credentials to test re-authentication failure
	u2.User = nil

	s, err = m2.UserSession(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	err = m1.TerminateSession(context.Background(), []string{s.Key})
	if err != nil {
		t.Fatal(err)
	}

	// Wait for keepalive re-authenticate attempt
	result := <-reauth

	_, err = methods.GetCurrentTime(context.Background(), m2.client)
	if err == nil {
		t.Error("expected to fail")
	}

	if result {
		t.Errorf("expected reauth to fail")
	}
}
