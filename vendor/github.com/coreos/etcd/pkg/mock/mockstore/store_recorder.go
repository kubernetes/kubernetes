// Copyright 2015 The etcd Authors
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

package mockstore

import (
	"time"

	"github.com/coreos/etcd/pkg/testutil"
	"github.com/coreos/etcd/store"
)

// StoreRecorder provides a Store interface with a testutil.Recorder
type StoreRecorder struct {
	store.Store
	testutil.Recorder
}

// storeRecorder records all the methods it receives.
// storeRecorder DOES NOT work as a actual store.
// It always returns invalid empty response and no error.
type storeRecorder struct {
	store.Store
	testutil.Recorder
}

func NewNop() store.Store { return &storeRecorder{Recorder: &testutil.RecorderBuffered{}} }
func NewRecorder() *StoreRecorder {
	sr := &storeRecorder{Recorder: &testutil.RecorderBuffered{}}
	return &StoreRecorder{Store: sr, Recorder: sr.Recorder}
}
func NewRecorderStream() *StoreRecorder {
	sr := &storeRecorder{Recorder: testutil.NewRecorderStream()}
	return &StoreRecorder{Store: sr, Recorder: sr.Recorder}
}

func (s *storeRecorder) Version() int  { return 0 }
func (s *storeRecorder) Index() uint64 { return 0 }
func (s *storeRecorder) Get(path string, recursive, sorted bool) (*store.Event, error) {
	s.Record(testutil.Action{
		Name:   "Get",
		Params: []interface{}{path, recursive, sorted},
	})
	return &store.Event{}, nil
}
func (s *storeRecorder) Set(path string, dir bool, val string, expireOpts store.TTLOptionSet) (*store.Event, error) {
	s.Record(testutil.Action{
		Name:   "Set",
		Params: []interface{}{path, dir, val, expireOpts},
	})
	return &store.Event{}, nil
}
func (s *storeRecorder) Update(path, val string, expireOpts store.TTLOptionSet) (*store.Event, error) {
	s.Record(testutil.Action{
		Name:   "Update",
		Params: []interface{}{path, val, expireOpts},
	})
	return &store.Event{}, nil
}
func (s *storeRecorder) Create(path string, dir bool, val string, uniq bool, expireOpts store.TTLOptionSet) (*store.Event, error) {
	s.Record(testutil.Action{
		Name:   "Create",
		Params: []interface{}{path, dir, val, uniq, expireOpts},
	})
	return &store.Event{}, nil
}
func (s *storeRecorder) CompareAndSwap(path, prevVal string, prevIdx uint64, val string, expireOpts store.TTLOptionSet) (*store.Event, error) {
	s.Record(testutil.Action{
		Name:   "CompareAndSwap",
		Params: []interface{}{path, prevVal, prevIdx, val, expireOpts},
	})
	return &store.Event{}, nil
}
func (s *storeRecorder) Delete(path string, dir, recursive bool) (*store.Event, error) {
	s.Record(testutil.Action{
		Name:   "Delete",
		Params: []interface{}{path, dir, recursive},
	})
	return &store.Event{}, nil
}
func (s *storeRecorder) CompareAndDelete(path, prevVal string, prevIdx uint64) (*store.Event, error) {
	s.Record(testutil.Action{
		Name:   "CompareAndDelete",
		Params: []interface{}{path, prevVal, prevIdx},
	})
	return &store.Event{}, nil
}
func (s *storeRecorder) Watch(_ string, _, _ bool, _ uint64) (store.Watcher, error) {
	s.Record(testutil.Action{Name: "Watch"})
	return store.NewNopWatcher(), nil
}
func (s *storeRecorder) Save() ([]byte, error) {
	s.Record(testutil.Action{Name: "Save"})
	return nil, nil
}
func (s *storeRecorder) Recovery(b []byte) error {
	s.Record(testutil.Action{Name: "Recovery"})
	return nil
}

func (s *storeRecorder) SaveNoCopy() ([]byte, error) {
	s.Record(testutil.Action{Name: "SaveNoCopy"})
	return nil, nil
}

func (s *storeRecorder) Clone() store.Store {
	s.Record(testutil.Action{Name: "Clone"})
	return s
}

func (s *storeRecorder) JsonStats() []byte { return nil }
func (s *storeRecorder) DeleteExpiredKeys(cutoff time.Time) {
	s.Record(testutil.Action{
		Name:   "DeleteExpiredKeys",
		Params: []interface{}{cutoff},
	})
}

func (s *storeRecorder) HasTTLKeys() bool {
	s.Record(testutil.Action{
		Name: "HasTTLKeys",
	})
	return true
}

// errStoreRecorder is a storeRecorder, but returns the given error on
// Get, Watch methods.
type errStoreRecorder struct {
	storeRecorder
	err error
}

func NewErrRecorder(err error) *StoreRecorder {
	sr := &errStoreRecorder{err: err}
	sr.Recorder = &testutil.RecorderBuffered{}
	return &StoreRecorder{Store: sr, Recorder: sr.Recorder}
}

func (s *errStoreRecorder) Get(path string, recursive, sorted bool) (*store.Event, error) {
	s.storeRecorder.Get(path, recursive, sorted)
	return nil, s.err
}
func (s *errStoreRecorder) Watch(path string, recursive, sorted bool, index uint64) (store.Watcher, error) {
	s.storeRecorder.Watch(path, recursive, sorted, index)
	return nil, s.err
}
