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

package vim25

import (
	"context"
	"encoding/json"
	"net/url"
	"os"
	"testing"

	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

// Duplicated to prevent cyclic dependency...
func testURL(t *testing.T) *url.URL {
	s := os.Getenv("GOVMOMI_TEST_URL")
	if s == "" {
		t.SkipNow()
	}
	u, err := soap.ParseURL(s)
	if err != nil {
		panic(err)
	}
	return u
}

func sessionLogin(t *testing.T, c *Client) {
	req := types.Login{
		This: *c.ServiceContent.SessionManager,
	}

	u := testURL(t).User
	req.UserName = u.Username()
	if pw, ok := u.Password(); ok {
		req.Password = pw
	}

	_, err := methods.Login(context.Background(), c, &req)
	if err != nil {
		t.Fatal(err)
	}
}

func sessionCheck(t *testing.T, c *Client) {
	var mgr mo.SessionManager

	err := mo.RetrieveProperties(context.Background(), c, c.ServiceContent.PropertyCollector, *c.ServiceContent.SessionManager, &mgr)
	if err != nil {
		t.Fatal(err)
	}
}

func TestClientSerialization(t *testing.T) {
	var c1, c2 *Client

	soapClient := soap.NewClient(testURL(t), true)
	c1, err := NewClient(context.Background(), soapClient)
	if err != nil {
		t.Fatal(err)
	}

	// Login
	sessionLogin(t, c1)
	sessionCheck(t, c1)

	// Serialize/deserialize
	b, err := json.Marshal(c1)
	if err != nil {
		t.Fatal(err)
	}
	c2 = &Client{}
	err = json.Unmarshal(b, c2)
	if err != nil {
		t.Fatal(err)
	}

	// Check the session is still valid
	sessionCheck(t, c2)
}
