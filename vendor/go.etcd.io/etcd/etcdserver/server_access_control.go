// Copyright 2018 The etcd Authors
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

package etcdserver

import "sync"

// AccessController controls etcd server HTTP request access.
type AccessController struct {
	corsMu          sync.RWMutex
	CORS            map[string]struct{}
	hostWhitelistMu sync.RWMutex
	HostWhitelist   map[string]struct{}
}

// NewAccessController returns a new "AccessController" with default "*" values.
func NewAccessController() *AccessController {
	return &AccessController{
		CORS:          map[string]struct{}{"*": {}},
		HostWhitelist: map[string]struct{}{"*": {}},
	}
}

// OriginAllowed determines whether the server will allow a given CORS origin.
// If CORS is empty, allow all.
func (ac *AccessController) OriginAllowed(origin string) bool {
	ac.corsMu.RLock()
	defer ac.corsMu.RUnlock()
	if len(ac.CORS) == 0 { // allow all
		return true
	}
	_, ok := ac.CORS["*"]
	if ok {
		return true
	}
	_, ok = ac.CORS[origin]
	return ok
}

// IsHostWhitelisted returns true if the host is whitelisted.
// If whitelist is empty, allow all.
func (ac *AccessController) IsHostWhitelisted(host string) bool {
	ac.hostWhitelistMu.RLock()
	defer ac.hostWhitelistMu.RUnlock()
	if len(ac.HostWhitelist) == 0 { // allow all
		return true
	}
	_, ok := ac.HostWhitelist["*"]
	if ok {
		return true
	}
	_, ok = ac.HostWhitelist[host]
	return ok
}
