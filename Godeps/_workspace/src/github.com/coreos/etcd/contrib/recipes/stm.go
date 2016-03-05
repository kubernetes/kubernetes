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

package recipe

import (
	v3 "github.com/coreos/etcd/clientv3"
	"golang.org/x/net/context"
)

// STM implements software transactional memory over etcd
type STM struct {
	client *v3.Client
	// rset holds the read key's value and revision of read
	rset map[string]*RemoteKV
	// wset holds the write key and its value
	wset map[string]string
	// aborted is whether user aborted the txn
	aborted bool
	apply   func(*STM) error
}

// NewSTM creates new transaction loop for a given apply function.
func NewSTM(client *v3.Client, apply func(*STM) error) <-chan error {
	s := &STM{client: client, apply: apply}
	errc := make(chan error, 1)
	go func() {
		var err error
		for {
			s.clear()
			if err = apply(s); err != nil || s.aborted {
				break
			}
			if ok, cerr := s.commit(); ok || cerr != nil {
				err = cerr
				break
			}
		}
		errc <- err
	}()
	return errc
}

// Abort abandons the apply loop, letting the transaction close without a commit.
func (s *STM) Abort() { s.aborted = true }

// Get returns the value for a given key, inserting the key into the txn's rset.
func (s *STM) Get(key string) (string, error) {
	if wv, ok := s.wset[key]; ok {
		return wv, nil
	}
	if rk, ok := s.rset[key]; ok {
		return rk.Value(), nil
	}
	rk, err := GetRemoteKV(s.client, key)
	if err != nil {
		return "", err
	}
	// TODO: setup watchers to abort txn early
	s.rset[key] = rk
	return rk.Value(), nil
}

// Put adds a value for a key to the write set.
func (s *STM) Put(key string, val string) { s.wset[key] = val }

// commit attempts to apply the txn's changes to the server.
func (s *STM) commit() (ok bool, rr error) {
	// read set must not change
	cmps := make([]v3.Cmp, 0, len(s.rset))
	for k, rk := range s.rset {
		// use < to support updating keys that don't exist yet
		cmp := v3.Compare(v3.ModifiedRevision(k), "<", rk.Revision()+1)
		cmps = append(cmps, cmp)
	}

	// apply all writes
	puts := make([]v3.Op, 0, len(s.wset))
	for k, v := range s.wset {
		puts = append(puts, v3.OpPut(k, v))
	}
	txnresp, err := s.client.Txn(context.TODO()).If(cmps...).Then(puts...).Commit()
	if err != nil {
		return false, err
	}
	return txnresp.Succeeded, err
}

func (s *STM) clear() {
	s.rset = make(map[string]*RemoteKV)
	s.wset = make(map[string]string)
}
