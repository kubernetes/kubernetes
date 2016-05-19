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

package keystone

import (
	"fmt"

	"github.com/mitchellh/mapstructure"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	"github.com/rackspace/gophercloud/openstack/identity/v2/tokens"
	ops "k8s.io/kubernetes/pkg/util/openstack"
)

type apiCallValidator struct {
	client *gophercloud.ProviderClient
}

func newAPICallValidator(path string) (*apiCallValidator, error) {
	client, err := ops.NewClientFromFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize openstack client: %s", err)
	}
	return &apiCallValidator{
		client: client,
	}, nil
}

func (a *apiCallValidator) support(token string) bool {
	return true
}

func (a *apiCallValidator) validate(token string) (*response, error) {
	result := tokens.Get(openstack.NewIdentityV2(a.client), token)
	if result.Err != nil {
		return nil, fmt.Errorf("failed to validate: %s", result.Err)
	}
	r := &response{}
	if err := mapstructure.Decode(result.Body, r); err != nil {
		return nil, fmt.Errorf("decode to response failed: %s", err)
	}
	return r, nil
}
