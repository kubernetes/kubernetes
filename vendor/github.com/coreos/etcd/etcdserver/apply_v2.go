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

package etcdserver

import (
	"encoding/json"
	"path"
	"time"

	"github.com/coreos/etcd/etcdserver/api"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/etcdserver/membership"
	"github.com/coreos/etcd/pkg/pbutil"
	"github.com/coreos/etcd/store"
	"github.com/coreos/go-semver/semver"
)

// ApplierV2 is the interface for processing V2 raft messages
type ApplierV2 interface {
	Delete(r *pb.Request) Response
	Post(r *pb.Request) Response
	Put(r *pb.Request) Response
	QGet(r *pb.Request) Response
	Sync(r *pb.Request) Response
}

func NewApplierV2(s store.Store, c *membership.RaftCluster) ApplierV2 {
	return &applierV2store{store: s, cluster: c}
}

type applierV2store struct {
	store   store.Store
	cluster *membership.RaftCluster
}

func (a *applierV2store) Delete(r *pb.Request) Response {
	switch {
	case r.PrevIndex > 0 || r.PrevValue != "":
		return toResponse(a.store.CompareAndDelete(r.Path, r.PrevValue, r.PrevIndex))
	default:
		return toResponse(a.store.Delete(r.Path, r.Dir, r.Recursive))
	}
}

func (a *applierV2store) Post(r *pb.Request) Response {
	return toResponse(a.store.Create(r.Path, r.Dir, r.Val, true, toTTLOptions(r)))
}

func (a *applierV2store) Put(r *pb.Request) Response {
	ttlOptions := toTTLOptions(r)
	exists, existsSet := pbutil.GetBool(r.PrevExist)
	switch {
	case existsSet:
		if exists {
			if r.PrevIndex == 0 && r.PrevValue == "" {
				return toResponse(a.store.Update(r.Path, r.Val, ttlOptions))
			}
			return toResponse(a.store.CompareAndSwap(r.Path, r.PrevValue, r.PrevIndex, r.Val, ttlOptions))
		}
		return toResponse(a.store.Create(r.Path, r.Dir, r.Val, false, ttlOptions))
	case r.PrevIndex > 0 || r.PrevValue != "":
		return toResponse(a.store.CompareAndSwap(r.Path, r.PrevValue, r.PrevIndex, r.Val, ttlOptions))
	default:
		if storeMemberAttributeRegexp.MatchString(r.Path) {
			id := membership.MustParseMemberIDFromKey(path.Dir(r.Path))
			var attr membership.Attributes
			if err := json.Unmarshal([]byte(r.Val), &attr); err != nil {
				plog.Panicf("unmarshal %s should never fail: %v", r.Val, err)
			}
			if a.cluster != nil {
				a.cluster.UpdateAttributes(id, attr)
			}
			// return an empty response since there is no consumer.
			return Response{}
		}
		if r.Path == membership.StoreClusterVersionKey() {
			if a.cluster != nil {
				a.cluster.SetVersion(semver.Must(semver.NewVersion(r.Val)), api.UpdateCapability)
			}
			// return an empty response since there is no consumer.
			return Response{}
		}
		return toResponse(a.store.Set(r.Path, r.Dir, r.Val, ttlOptions))
	}
}

func (a *applierV2store) QGet(r *pb.Request) Response {
	return toResponse(a.store.Get(r.Path, r.Recursive, r.Sorted))
}

func (a *applierV2store) Sync(r *pb.Request) Response {
	a.store.DeleteExpiredKeys(time.Unix(0, r.Time))
	return Response{}
}

// applyV2Request interprets r as a call to store.X and returns a Response interpreted
// from store.Event
func (s *EtcdServer) applyV2Request(r *pb.Request) Response {
	toTTLOptions(r)
	switch r.Method {
	case "POST":
		return s.applyV2.Post(r)
	case "PUT":
		return s.applyV2.Put(r)
	case "DELETE":
		return s.applyV2.Delete(r)
	case "QGET":
		return s.applyV2.QGet(r)
	case "SYNC":
		return s.applyV2.Sync(r)
	default:
		// This should never be reached, but just in case:
		return Response{err: ErrUnknownMethod}
	}
}

func toTTLOptions(r *pb.Request) store.TTLOptionSet {
	refresh, _ := pbutil.GetBool(r.Refresh)
	ttlOptions := store.TTLOptionSet{Refresh: refresh}
	if r.Expiration != 0 {
		ttlOptions.ExpireTime = time.Unix(0, r.Expiration)
	}
	return ttlOptions
}

func toResponse(ev *store.Event, err error) Response {
	return Response{Event: ev, err: err}
}
