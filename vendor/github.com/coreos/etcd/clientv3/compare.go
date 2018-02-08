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

package clientv3

import (
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
)

type CompareTarget int
type CompareResult int

const (
	CompareVersion CompareTarget = iota
	CompareCreated
	CompareModified
	CompareValue
)

type Cmp pb.Compare

func Compare(cmp Cmp, result string, v interface{}) Cmp {
	var r pb.Compare_CompareResult

	switch result {
	case "=":
		r = pb.Compare_EQUAL
	case "!=":
		r = pb.Compare_NOT_EQUAL
	case ">":
		r = pb.Compare_GREATER
	case "<":
		r = pb.Compare_LESS
	default:
		panic("Unknown result op")
	}

	cmp.Result = r
	switch cmp.Target {
	case pb.Compare_VALUE:
		val, ok := v.(string)
		if !ok {
			panic("bad compare value")
		}
		cmp.TargetUnion = &pb.Compare_Value{Value: []byte(val)}
	case pb.Compare_VERSION:
		cmp.TargetUnion = &pb.Compare_Version{Version: mustInt64(v)}
	case pb.Compare_CREATE:
		cmp.TargetUnion = &pb.Compare_CreateRevision{CreateRevision: mustInt64(v)}
	case pb.Compare_MOD:
		cmp.TargetUnion = &pb.Compare_ModRevision{ModRevision: mustInt64(v)}
	default:
		panic("Unknown compare type")
	}
	return cmp
}

func Value(key string) Cmp {
	return Cmp{Key: []byte(key), Target: pb.Compare_VALUE}
}

func Version(key string) Cmp {
	return Cmp{Key: []byte(key), Target: pb.Compare_VERSION}
}

func CreateRevision(key string) Cmp {
	return Cmp{Key: []byte(key), Target: pb.Compare_CREATE}
}

func ModRevision(key string) Cmp {
	return Cmp{Key: []byte(key), Target: pb.Compare_MOD}
}

// KeyBytes returns the byte slice holding with the comparison key.
func (cmp *Cmp) KeyBytes() []byte { return cmp.Key }

// WithKeyBytes sets the byte slice for the comparison key.
func (cmp *Cmp) WithKeyBytes(key []byte) { cmp.Key = key }

// ValueBytes returns the byte slice holding the comparison value, if any.
func (cmp *Cmp) ValueBytes() []byte {
	if tu, ok := cmp.TargetUnion.(*pb.Compare_Value); ok {
		return tu.Value
	}
	return nil
}

// WithValueBytes sets the byte slice for the comparison's value.
func (cmp *Cmp) WithValueBytes(v []byte) { cmp.TargetUnion.(*pb.Compare_Value).Value = v }

// mustInt64 panics if val isn't an int or int64. It returns an int64 otherwise.
func mustInt64(val interface{}) int64 {
	if v, ok := val.(int64); ok {
		return v
	}
	if v, ok := val.(int); ok {
		return int64(v)
	}
	panic("bad value")
}

// mustInt64orLeaseID panics if val isn't a LeaseID, int or int64. It returns an
// int64 otherwise.
func mustInt64orLeaseID(val interface{}) int64 {
	if v, ok := val.(LeaseID); ok {
		return int64(v)
	}
	return mustInt64(val)
}
