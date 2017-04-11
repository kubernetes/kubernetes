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

// Package mirror implements etcd mirroring operations.
package mirror

import (
	"github.com/coreos/etcd/clientv3"
	"golang.org/x/net/context"
)

const (
	batchLimit = 1000
)

// Syncer syncs with the key-value state of an etcd cluster.
type Syncer interface {
	// SyncBase syncs the base state of the key-value state.
	// The key-value state are sent through the returned chan.
	SyncBase(ctx context.Context) (<-chan clientv3.GetResponse, chan error)
	// SyncUpdates syncs the updates of the key-value state.
	// The update events are sent through the returned chan.
	SyncUpdates(ctx context.Context) clientv3.WatchChan
}

// NewSyncer creates a Syncer.
func NewSyncer(c *clientv3.Client, prefix string, rev int64) Syncer {
	return &syncer{c: c, prefix: prefix, rev: rev}
}

type syncer struct {
	c      *clientv3.Client
	rev    int64
	prefix string
}

func (s *syncer) SyncBase(ctx context.Context) (<-chan clientv3.GetResponse, chan error) {
	respchan := make(chan clientv3.GetResponse, 1024)
	errchan := make(chan error, 1)

	// if rev is not specified, we will choose the most recent revision.
	if s.rev == 0 {
		resp, err := s.c.Get(ctx, "foo")
		if err != nil {
			errchan <- err
			close(respchan)
			close(errchan)
			return respchan, errchan
		}
		s.rev = resp.Header.Revision
	}

	go func() {
		defer close(respchan)
		defer close(errchan)

		var key string

		opts := []clientv3.OpOption{clientv3.WithLimit(batchLimit), clientv3.WithRev(s.rev)}

		if len(s.prefix) == 0 {
			// If len(s.prefix) == 0, we will sync the entire key-value space.
			// We then range from the smallest key (0x00) to the end.
			opts = append(opts, clientv3.WithFromKey())
			key = "\x00"
		} else {
			// If len(s.prefix) != 0, we will sync key-value space with given prefix.
			// We then range from the prefix to the next prefix if exists. Or we will
			// range from the prefix to the end if the next prefix does not exists.
			opts = append(opts, clientv3.WithRange(clientv3.GetPrefixRangeEnd(s.prefix)))
			key = s.prefix
		}

		for {
			resp, err := s.c.Get(ctx, key, opts...)
			if err != nil {
				errchan <- err
				return
			}

			respchan <- (clientv3.GetResponse)(*resp)

			if !resp.More {
				return
			}
			// move to next key
			key = string(append(resp.Kvs[len(resp.Kvs)-1].Key, 0))
		}
	}()

	return respchan, errchan
}

func (s *syncer) SyncUpdates(ctx context.Context) clientv3.WatchChan {
	if s.rev == 0 {
		panic("unexpected revision = 0. Calling SyncUpdates before SyncBase finishes?")
	}
	return s.c.Watch(ctx, s.prefix, clientv3.WithPrefix(), clientv3.WithRev(s.rev+1))
}
