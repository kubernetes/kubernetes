/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package test

import (
	"context"
	"net/url"
	"os"
	"testing"

	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

// URL parses the GOVMOMI_TEST_URL environment variable if set.
func URL() *url.URL {
	s := os.Getenv("GOVMOMI_TEST_URL")
	if s == "" {
		return nil
	}
	u, err := soap.ParseURL(s)
	if err != nil {
		panic(err)
	}
	return u
}

// NewAuthenticatedClient creates a new vim25.Client, authenticates the user
// specified in the test URL, and returns it.
func NewAuthenticatedClient(t *testing.T) *vim25.Client {
	u := URL()
	if u == nil {
		t.SkipNow()
	}

	soapClient := soap.NewClient(u, true)
	vimClient, err := vim25.NewClient(context.Background(), soapClient)
	if err != nil {
		t.Fatal(err)
	}

	req := types.Login{
		This: *vimClient.ServiceContent.SessionManager,
	}

	req.UserName = u.User.Username()
	if pw, ok := u.User.Password(); ok {
		req.Password = pw
	}

	_, err = methods.Login(context.Background(), vimClient, &req)
	if err != nil {
		t.Fatal(err)
	}

	return vimClient
}
