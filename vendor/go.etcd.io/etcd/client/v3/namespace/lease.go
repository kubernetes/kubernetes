// Copyright 2017 The etcd Authors
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

package namespace

import (
	"bytes"
	"context"

	"go.etcd.io/etcd/client/v3"
)

type leasePrefix struct {
	clientv3.Lease
	pfx []byte
}

// NewLease wraps a Lease interface to filter for only keys with a prefix
// and remove that prefix when fetching attached keys through TimeToLive.
func NewLease(l clientv3.Lease, prefix string) clientv3.Lease {
	return &leasePrefix{l, []byte(prefix)}
}

func (l *leasePrefix) TimeToLive(ctx context.Context, id clientv3.LeaseID, opts ...clientv3.LeaseOption) (*clientv3.LeaseTimeToLiveResponse, error) {
	resp, err := l.Lease.TimeToLive(ctx, id, opts...)
	if err != nil {
		return nil, err
	}
	if len(resp.Keys) > 0 {
		var outKeys [][]byte
		for i := range resp.Keys {
			if len(resp.Keys[i]) < len(l.pfx) {
				// too short
				continue
			}
			if !bytes.Equal(resp.Keys[i][:len(l.pfx)], l.pfx) {
				// doesn't match prefix
				continue
			}
			// strip prefix
			outKeys = append(outKeys, resp.Keys[i][len(l.pfx):])
		}
		resp.Keys = outKeys
	}
	return resp, nil
}
