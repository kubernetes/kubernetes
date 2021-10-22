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
	"net/url"
	"testing"

	"github.com/vmware/govmomi/test"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
)

func sessionClient(u *url.URL, t *testing.T) *Manager {
	soapClient := soap.NewClient(u, true)
	vimClient, err := vim25.NewClient(context.Background(), soapClient)
	if err != nil {
		t.Fatal(err)
	}

	return NewManager(vimClient)
}

func TestLogin(t *testing.T) {
	u := test.URL()
	if u == nil {
		t.SkipNow()
	}

	session := sessionClient(u, t)
	err := session.Login(context.Background(), u.User)
	if err != nil {
		t.Errorf("Expected no error, got %v", err)
	}
}

func TestLogout(t *testing.T) {
	u := test.URL()
	if u == nil {
		t.SkipNow()
	}

	session := sessionClient(u, t)
	err := session.Login(context.Background(), u.User)
	if err != nil {
		t.Error("Login Error: ", err)
	}

	err = session.Logout(context.Background())
	if err != nil {
		t.Errorf("Expected nil, got %v", err)
	}

	err = session.Logout(context.Background())
	if err == nil {
		t.Errorf("Expected NotAuthenticated, got nil")
	}
}

func TestSessionIsActive(t *testing.T) {
	u := test.URL()
	if u == nil {
		t.SkipNow()
	}

	session := sessionClient(u, t)

	// Skip test against ESXi -- SessionIsActive is not implemented
	if session.client.ServiceContent.About.ApiType != "VirtualCenter" {
		t.Skipf("Talking to %s instead of %s", session.client.ServiceContent.About.ApiType, "VirtualCenter")
	}

	err := session.Login(context.Background(), u.User)
	if err != nil {
		t.Error("Login Error: ", err)
	}

	active, err := session.SessionIsActive(context.Background())
	if err != nil || !active {
		t.Errorf("Expected %t, got %t", true, active)
		t.Errorf("Expected nil, got %v", err)
	}

	session.Logout(context.Background())

	active, err = session.SessionIsActive(context.Background())
	if err == nil || active {
		t.Errorf("Expected %t, got %t", false, active)
		t.Errorf("Expected NotAuthenticated, got %v", err)
	}
}
