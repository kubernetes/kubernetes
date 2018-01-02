// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package monasca

import (
	"net/url"

	"github.com/rackspace/gophercloud"
)

// Config represents the configuration of the Monasca sink.
type Config struct {
	gophercloud.AuthOptions
	MonascaURL string
}

// NewConfig builds a configuration object from the parameters of the monasca sink.
func NewConfig(opts url.Values) Config {
	config := Config{}
	if len(opts["keystone-url"]) >= 1 {
		config.IdentityEndpoint = opts["keystone-url"][0]
	}
	if len(opts["tenant-id"]) >= 1 {
		config.TenantID = opts["tenant-id"][0]
	}
	if len(opts["username"]) >= 1 {
		config.Username = opts["username"][0]
	}
	if len(opts["user-id"]) >= 1 {
		config.UserID = opts["user-id"][0]
	}
	if len(opts["password"]) >= 1 {
		config.Password = opts["password"][0]
	}
	if len(opts["api-key"]) >= 1 {
		config.APIKey = opts["api-key"][0]
	}
	if len(opts["domain-id"]) >= 1 {
		config.DomainID = opts["domain-id"][0]
	}
	if len(opts["domain-name"]) >= 1 {
		config.DomainName = opts["domain-name"][0]
	}
	if len(opts["monasca-url"]) >= 1 {
		config.MonascaURL = opts["monasca-url"][0]
	}
	return config
}
