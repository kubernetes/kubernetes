// Copyright 2016 The etcd Authors
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

package cache

import (
	"sync"

	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/golang/groupcache/lru"
)

var (
	DefaultMaxEntries = 2048
	ErrCompacted      = rpctypes.ErrGRPCCompacted
)

type Cache interface {
	Add(req *pb.RangeRequest, resp *pb.RangeResponse)
	Get(req *pb.RangeRequest) (*pb.RangeResponse, error)
	Compact(revision int64)
}

// keyFunc returns the key of an request, which is used to look up in the cache for it's caching response.
func keyFunc(req *pb.RangeRequest) string {
	// TODO: use marshalTo to reduce allocation
	b, err := req.Marshal()
	if err != nil {
		panic(err)
	}
	return string(b)
}

func NewCache(maxCacheEntries int) Cache {
	return &cache{
		lru: lru.New(maxCacheEntries),
	}
}

// cache implements Cache
type cache struct {
	mu           sync.RWMutex
	lru          *lru.Cache
	compactedRev int64
}

// Add adds the response of a request to the cache if its revision is larger than the compacted revision of the cache.
func (c *cache) Add(req *pb.RangeRequest, resp *pb.RangeResponse) {
	key := keyFunc(req)

	c.mu.Lock()
	defer c.mu.Unlock()

	if req.Revision > c.compactedRev {
		c.lru.Add(key, resp)
	}
}

// Get looks up the caching response for a given request.
// Get is also responsible for lazy eviction when accessing compacted entries.
func (c *cache) Get(req *pb.RangeRequest) (*pb.RangeResponse, error) {
	key := keyFunc(req)

	c.mu.Lock()
	defer c.mu.Unlock()

	if req.Revision > c.compactedRev {
		c.lru.Remove(key)
		return nil, ErrCompacted
	}

	if resp, ok := c.lru.Get(key); ok {
		return resp.(*pb.RangeResponse), nil
	}
	return nil, nil
}

// Compact invalidate all caching response before the given rev.
// Replace with the invalidation is lazy. The actual removal happens when the entries is accessed.
func (c *cache) Compact(revision int64) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if revision > c.compactedRev {
		c.compactedRev = revision
	}
}
