/*
Copyright 2016 The Kubernetes Authors All rights reserved.
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

package openstack

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
)

// NewClient constructs a new provider client with the given auth options, url
// tells the keystone endpoint, and username/password just the ones used to login
// to openstack, the tenantid is the tenant which could grant all the privileges.
func NewClient(url, username, password, tenantid string) (*gophercloud.ProviderClient, error) {
	authOpts := gophercloud.AuthOptions{
		IdentityEndpoint: url,
		Username:         username,
		Password:         password,
		TenantID:         tenantid,
		AllowReauth:      true,
	}
	return openstack.AuthenticatedClient(authOpts)
}

// NewClientFromFile constructs a new provider client with configurations read
// from file, the file is supposed to be json format, which looks like this:
//   {
//     "auth-url": "",
//     "user-name": "",
//     "password": "",
//     "tenant-id": ""
//   }
func NewClientFromFile(path string) (*gophercloud.ProviderClient, error) {
	cfg := &struct {
		URL      string `json:"auth-url"`
		Username string `json:"user-name"`
		Password string `json:"password"`
		TenantID string `json:"tenant-id"`
	}{}
	file, err := os.Open(path)
	defer file.Close()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize client: %s", err)
	}
	bytes, err := ioutil.ReadAll(file)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize client: %s", err)
	}
	err = json.Unmarshal(bytes, cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize client: %s", err)
	}
	return NewClient(cfg.URL, cfg.Username, cfg.Password, cfg.TenantID)
}
