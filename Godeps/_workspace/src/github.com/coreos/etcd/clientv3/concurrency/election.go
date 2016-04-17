// Copyright 2016 CoreOS, Inc.
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

package concurrency

import (
	"errors"

	v3 "github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/storage/storagepb"
	"golang.org/x/net/context"
)

var (
	ErrElectionNotLeader = errors.New("election: not leader")
	ErrElectionNoLeader  = errors.New("election: no leader")
)

type Election struct {
	client *v3.Client

	keyPrefix string

	leaderKey     string
	leaderRev     int64
	leaderSession *Session
}

// NewElection returns a new election on a given key prefix.
func NewElection(client *v3.Client, pfx string) *Election {
	return &Election{client: client, keyPrefix: pfx}
}

// Campaign puts a value as eligible for the election. It blocks until
// it is elected, an error occurs, or the context is cancelled.
func (e *Election) Campaign(ctx context.Context, val string) error {
	s, serr := NewSession(e.client)
	if serr != nil {
		return serr
	}

	k, rev, err := NewUniqueKV(ctx, e.client, e.keyPrefix, val, v3.WithLease(s.Lease()))
	if err == nil {
		err = waitDeletes(ctx, e.client, e.keyPrefix, v3.WithPrefix(), v3.WithRev(rev-1))
	}

	if err != nil {
		// clean up in case of context cancel
		select {
		case <-ctx.Done():
			e.client.Delete(e.client.Ctx(), k)
		default:
		}
		return err
	}

	e.leaderKey, e.leaderRev, e.leaderSession = k, rev, s
	return nil
}

// Proclaim lets the leader announce a new value without another election.
func (e *Election) Proclaim(ctx context.Context, val string) error {
	if e.leaderSession == nil {
		return ErrElectionNotLeader
	}
	cmp := v3.Compare(v3.CreateRevision(e.leaderKey), "=", e.leaderRev)
	txn := e.client.Txn(ctx).If(cmp)
	txn = txn.Then(v3.OpPut(e.leaderKey, val, v3.WithLease(e.leaderSession.Lease())))
	tresp, terr := txn.Commit()
	if terr != nil {
		return terr
	}
	if !tresp.Succeeded {
		e.leaderKey = ""
		return ErrElectionNotLeader
	}
	return nil
}

// Resign lets a leader start a new election.
func (e *Election) Resign() (err error) {
	if e.leaderSession == nil {
		return nil
	}
	_, err = e.client.Delete(e.client.Ctx(), e.leaderKey)
	e.leaderKey = ""
	e.leaderSession = nil
	return err
}

// Leader returns the leader value for the current election.
func (e *Election) Leader() (string, error) {
	resp, err := e.client.Get(e.client.Ctx(), e.keyPrefix, v3.WithFirstCreate()...)
	if err != nil {
		return "", err
	} else if len(resp.Kvs) == 0 {
		// no leader currently elected
		return "", ErrElectionNoLeader
	}
	return string(resp.Kvs[0].Value), nil
}

// Observe returns a channel that observes all leader proposal values as
// GetResponse values on the current leader key. The channel closes when
// the context is cancelled or the underlying watcher is otherwise disrupted.
func (e *Election) Observe(ctx context.Context) <-chan v3.GetResponse {
	retc := make(chan v3.GetResponse)
	go e.observe(ctx, retc)
	return retc
}

func (e *Election) observe(ctx context.Context, ch chan<- v3.GetResponse) {
	defer close(ch)
	for {
		resp, err := e.client.Get(ctx, e.keyPrefix, v3.WithFirstCreate()...)
		if err != nil {
			return
		}

		var kv *storagepb.KeyValue

		cctx, cancel := context.WithCancel(ctx)
		if len(resp.Kvs) == 0 {
			// wait for first key put on prefix
			opts := []v3.OpOption{v3.WithRev(resp.Header.Revision), v3.WithPrefix()}
			wch := e.client.Watch(cctx, e.keyPrefix, opts...)

			for kv == nil {
				wr, ok := <-wch
				if !ok || wr.Err() != nil {
					cancel()
					return
				}
				// only accept PUTs; a DELETE will make observe() spin
				for _, ev := range wr.Events {
					if ev.Type == storagepb.PUT {
						kv = ev.Kv
						break
					}
				}
			}
		} else {
			kv = resp.Kvs[0]
		}

		wch := e.client.Watch(cctx, string(kv.Key), v3.WithRev(kv.ModRevision))
		keyDeleted := false
		for !keyDeleted {
			wr, ok := <-wch
			if !ok {
				return
			}
			for _, ev := range wr.Events {
				if ev.Type == storagepb.DELETE {
					keyDeleted = true
					break
				}
				resp.Header = &wr.Header
				resp.Kvs = []*storagepb.KeyValue{ev.Kv}
				select {
				case ch <- *resp:
				case <-cctx.Done():
					return
				}
			}
		}
		cancel()
	}
}

// Key returns the leader key if elected, empty string otherwise.
func (e *Election) Key() string { return e.leaderKey }
