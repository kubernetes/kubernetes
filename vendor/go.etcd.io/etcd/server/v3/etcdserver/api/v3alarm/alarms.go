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

// Package v3alarm manages health status alarms in etcd.
package v3alarm

import (
	"sync"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/client/pkg/v3/types"
	"go.etcd.io/etcd/server/v3/mvcc/backend"
	"go.etcd.io/etcd/server/v3/mvcc/buckets"

	"go.uber.org/zap"
)

type BackendGetter interface {
	Backend() backend.Backend
}

type alarmSet map[types.ID]*pb.AlarmMember

// AlarmStore persists alarms to the backend.
type AlarmStore struct {
	lg    *zap.Logger
	mu    sync.Mutex
	types map[pb.AlarmType]alarmSet

	bg BackendGetter
}

func NewAlarmStore(lg *zap.Logger, bg BackendGetter) (*AlarmStore, error) {
	if lg == nil {
		lg = zap.NewNop()
	}
	ret := &AlarmStore{lg: lg, types: make(map[pb.AlarmType]alarmSet), bg: bg}
	err := ret.restore()
	return ret, err
}

func (a *AlarmStore) Activate(id types.ID, at pb.AlarmType) *pb.AlarmMember {
	a.mu.Lock()
	defer a.mu.Unlock()

	newAlarm := &pb.AlarmMember{MemberID: uint64(id), Alarm: at}
	if m := a.addToMap(newAlarm); m != newAlarm {
		return m
	}

	v, err := newAlarm.Marshal()
	if err != nil {
		a.lg.Panic("failed to marshal alarm member", zap.Error(err))
	}

	b := a.bg.Backend()
	b.BatchTx().LockInsideApply()
	b.BatchTx().UnsafePut(buckets.Alarm, v, nil)
	b.BatchTx().Unlock()

	return newAlarm
}

func (a *AlarmStore) Deactivate(id types.ID, at pb.AlarmType) *pb.AlarmMember {
	a.mu.Lock()
	defer a.mu.Unlock()

	t := a.types[at]
	if t == nil {
		t = make(alarmSet)
		a.types[at] = t
	}
	m := t[id]
	if m == nil {
		return nil
	}

	delete(t, id)

	v, err := m.Marshal()
	if err != nil {
		a.lg.Panic("failed to marshal alarm member", zap.Error(err))
	}

	b := a.bg.Backend()
	b.BatchTx().LockInsideApply()
	b.BatchTx().UnsafeDelete(buckets.Alarm, v)
	b.BatchTx().Unlock()

	return m
}

func (a *AlarmStore) Get(at pb.AlarmType) (ret []*pb.AlarmMember) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if at == pb.AlarmType_NONE {
		for _, t := range a.types {
			for _, m := range t {
				ret = append(ret, m)
			}
		}
		return ret
	}
	for _, m := range a.types[at] {
		ret = append(ret, m)
	}
	return ret
}

func (a *AlarmStore) restore() error {
	b := a.bg.Backend()
	tx := b.BatchTx()

	tx.LockOutsideApply()
	tx.UnsafeCreateBucket(buckets.Alarm)
	err := tx.UnsafeForEach(buckets.Alarm, func(k, v []byte) error {
		var m pb.AlarmMember
		if err := m.Unmarshal(k); err != nil {
			return err
		}
		a.addToMap(&m)
		return nil
	})
	tx.Unlock()

	b.ForceCommit()
	return err
}

func (a *AlarmStore) addToMap(newAlarm *pb.AlarmMember) *pb.AlarmMember {
	t := a.types[newAlarm.Alarm]
	if t == nil {
		t = make(alarmSet)
		a.types[newAlarm.Alarm] = t
	}
	m := t[types.ID(newAlarm.MemberID)]
	if m != nil {
		return m
	}
	t[types.ID(newAlarm.MemberID)] = newAlarm
	return newAlarm
}
