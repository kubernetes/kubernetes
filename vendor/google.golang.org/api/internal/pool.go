// Copyright 2016 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"errors"

	"google.golang.org/grpc/naming"
)

// PoolResolver provides a fixed list of addresses to load balance between
// and does not provide further updates.
type PoolResolver struct {
	poolSize int
	dialOpt  *DialSettings
	ch       chan []*naming.Update
}

// NewPoolResolver returns a PoolResolver
// This is an EXPERIMENTAL API and may be changed or removed in the future.
func NewPoolResolver(size int, o *DialSettings) *PoolResolver {
	return &PoolResolver{poolSize: size, dialOpt: o}
}

// Resolve returns a Watcher for the endpoint defined by the DialSettings
// provided to NewPoolResolver.
func (r *PoolResolver) Resolve(target string) (naming.Watcher, error) {
	if r.dialOpt.Endpoint == "" {
		return nil, errors.New("no endpoint configured")
	}
	addrs := make([]*naming.Update, 0, r.poolSize)
	for i := 0; i < r.poolSize; i++ {
		addrs = append(addrs, &naming.Update{Op: naming.Add, Addr: r.dialOpt.Endpoint, Metadata: i})
	}
	r.ch = make(chan []*naming.Update, 1)
	r.ch <- addrs
	return r, nil
}

// Next returns a static list of updates on the first call,
// and blocks indefinitely until Close is called on subsequent calls.
func (r *PoolResolver) Next() ([]*naming.Update, error) {
	return <-r.ch, nil
}

// Close releases resources associated with the pool and causes Next to unblock.
func (r *PoolResolver) Close() {
	close(r.ch)
}
