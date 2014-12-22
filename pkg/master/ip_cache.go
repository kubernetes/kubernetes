/*
Copyright 2014 Google Inc. All rights reserved.

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

package master

import (
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
)

type ipCacheEntry struct {
	ip         string
	lastUpdate time.Time
}

type ipCache struct {
	clock         util.Clock
	cloudProvider cloudprovider.Interface
	cache         map[string]ipCacheEntry
	lock          sync.Mutex
	ttl           time.Duration
}

// NewIPCache makes a new ip caching layer, which will get IP addresses from cp,
// and use clock for deciding when to re-get an IP address.
// Thread-safe.
//
// TODO: when we switch to go1.4, this class would be a good candidate for something
// that could be produced from a template and a type via `go generate`.
func NewIPCache(cp cloudprovider.Interface, clock util.Clock, ttl time.Duration) *ipCache {
	return &ipCache{
		clock:         clock,
		cloudProvider: cp,
		cache:         map[string]ipCacheEntry{},
		ttl:           ttl,
	}
}

// GetInstanceIP returns the IP address of host, from the cache
// if possible, otherwise it asks the cloud provider.
func (c *ipCache) GetInstanceIP(host string) string {
	c.lock.Lock()
	defer c.lock.Unlock()
	data, ok := c.cache[host]
	now := c.clock.Now()

	if !ok || now.Sub(data.lastUpdate) > c.ttl {
		ip := getInstanceIPFromCloud(c.cloudProvider, host)
		data = ipCacheEntry{
			ip:         ip,
			lastUpdate: now,
		}
		c.cache[host] = data
	}
	return data.ip
}

func getInstanceIPFromCloud(cloud cloudprovider.Interface, host string) string {
	if cloud == nil {
		return ""
	}
	instances, ok := cloud.Instances()
	if instances == nil || !ok {
		return ""
	}
	addr, err := instances.IPAddress(host)
	if err != nil {
		glog.Errorf("Error getting instance IP for %q: %v", host, err)
		return ""
	}
	return addr.String()
}
