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

package govmomi

import (
	"context"
	"errors"
	"net/http"
	"net/url"
	"testing"

	"github.com/vmware/govmomi/test"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/types"
)

func TestNewClient(t *testing.T) {
	u := test.URL()
	if u == nil {
		t.SkipNow()
	}

	c, err := NewClient(context.Background(), u, true)
	if err != nil {
		t.Fatal(err)
	}

	f := func() error {
		var x mo.Folder
		err = mo.RetrieveProperties(context.Background(), c, c.ServiceContent.PropertyCollector, c.ServiceContent.RootFolder, &x)
		if err != nil {
			return err
		}
		if len(x.Name) == 0 {
			return errors.New("empty response")
		}
		return nil
	}

	// check cookie is valid with an sdk request
	if err := f(); err != nil {
		t.Fatal(err)
	}

	// check cookie is valid with a non-sdk request
	u.User = nil // turn off Basic auth
	u.Path = "/folder"
	r, err := c.Client.Get(u.String())
	if err != nil {
		t.Fatal(err)
	}
	if r.StatusCode != http.StatusOK {
		t.Fatal(r)
	}

	// sdk request should fail w/o a valid cookie
	c.Client.Jar = nil
	if err := f(); err == nil {
		t.Fatal("should fail")
	}

	// invalid login
	u.Path = vim25.Path
	u.User = url.UserPassword("ENOENT", "EINVAL")
	_, err = NewClient(context.Background(), u, true)
	if err == nil {
		t.Fatal("should fail")
	}
}

func TestInvalidSdk(t *testing.T) {
	u := test.URL()
	if u == nil {
		t.SkipNow()
	}

	// a URL other than a valid /sdk should error, not panic
	u.Path = "/mob"
	_, err := NewClient(context.Background(), u, true)
	if err == nil {
		t.Fatal("should fail")
	}
}

func TestPropertiesN(t *testing.T) {
	u := test.URL()
	if u == nil {
		t.SkipNow()
	}

	c, err := NewClient(context.Background(), u, true)
	if err != nil {
		t.Fatal(err)
	}

	var f mo.Folder
	err = c.RetrieveOne(context.Background(), c.ServiceContent.RootFolder, nil, &f)
	if err != nil {
		t.Fatal(err)
	}

	var dc mo.Datacenter
	err = c.RetrieveOne(context.Background(), f.ChildEntity[0], nil, &dc)
	if err != nil {
		t.Fatal(err)
	}

	var folderReferences = []types.ManagedObjectReference{
		dc.DatastoreFolder,
		dc.HostFolder,
		dc.NetworkFolder,
		dc.VmFolder,
	}

	var folders []mo.Folder
	err = c.Retrieve(context.Background(), folderReferences, []string{"name"}, &folders)
	if err != nil {
		t.Fatal(err)
	}

	if len(folders) != len(folderReferences) {
		t.Fatalf("Expected %d, got %d", len(folderReferences), len(folders))
	}
}
