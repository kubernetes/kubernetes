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
	"fmt"
	"path"
	"strconv"
	"time"

	"github.com/coreos/go-semver/semver"
	"go.etcd.io/etcd/pkg/v3/pbutil"
	"go.etcd.io/etcd/server/v3/etcdserver/api"
	"go.etcd.io/etcd/server/v3/etcdserver/api/membership"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2store"

	"go.uber.org/zap"
)

const v2Version = "v2"

// ApplierV2 is the interface for processing V2 raft messages
type ApplierV2 interface {
	Delete(r *RequestV2) Response
	Post(r *RequestV2) Response
	Put(r *RequestV2, shouldApplyV3 membership.ShouldApplyV3) Response
	QGet(r *RequestV2) Response
	Sync(r *RequestV2) Response
}

func NewApplierV2(lg *zap.Logger, s v2store.Store, c *membership.RaftCluster) ApplierV2 {
	if lg == nil {
		lg = zap.NewNop()
	}
	return &applierV2store{lg: lg, store: s, cluster: c}
}

type applierV2store struct {
	lg      *zap.Logger
	store   v2store.Store
	cluster *membership.RaftCluster
}

func (a *applierV2store) Delete(r *RequestV2) Response {
	switch {
	case r.PrevIndex > 0 || r.PrevValue != "":
		return toResponse(a.store.CompareAndDelete(r.Path, r.PrevValue, r.PrevIndex))
	default:
		return toResponse(a.store.Delete(r.Path, r.Dir, r.Recursive))
	}
}

func (a *applierV2store) Post(r *RequestV2) Response {
	return toResponse(a.store.Create(r.Path, r.Dir, r.Val, true, r.TTLOptions()))
}

func (a *applierV2store) Put(r *RequestV2, shouldApplyV3 membership.ShouldApplyV3) Response {
	ttlOptions := r.TTLOptions()
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
			id := membership.MustParseMemberIDFromKey(a.lg, path.Dir(r.Path))
			var attr membership.Attributes
			if err := json.Unmarshal([]byte(r.Val), &attr); err != nil {
				a.lg.Panic("failed to unmarshal", zap.String("value", r.Val), zap.Error(err))
			}
			if a.cluster != nil {
				a.cluster.UpdateAttributes(id, attr, shouldApplyV3)
			}
			// return an empty response since there is no consumer.
			return Response{}
		}
		// TODO remove v2 version set to avoid the conflict between v2 and v3 in etcd 3.6
		if r.Path == membership.StoreClusterVersionKey() {
			if a.cluster != nil {
				// persist to backend given v2store can be very stale
				a.cluster.SetVersion(semver.Must(semver.NewVersion(r.Val)), api.UpdateCapability, shouldApplyV3)
			}
			return Response{}
		}
		return toResponse(a.store.Set(r.Path, r.Dir, r.Val, ttlOptions))
	}
}

func (a *applierV2store) QGet(r *RequestV2) Response {
	return toResponse(a.store.Get(r.Path, r.Recursive, r.Sorted))
}

func (a *applierV2store) Sync(r *RequestV2) Response {
	a.store.DeleteExpiredKeys(time.Unix(0, r.Time))
	return Response{}
}

// applyV2Request interprets r as a call to v2store.X
// and returns a Response interpreted from v2store.Event
func (s *EtcdServer) applyV2Request(r *RequestV2, shouldApplyV3 membership.ShouldApplyV3) (resp Response) {
	stringer := panicAlternativeStringer{
		stringer:    r,
		alternative: func() string { return fmt.Sprintf("id:%d,method:%s,path:%s", r.ID, r.Method, r.Path) },
	}
	defer func(start time.Time) {
		success := resp.Err == nil
		applySec.WithLabelValues(v2Version, r.Method, strconv.FormatBool(success)).Observe(time.Since(start).Seconds())
		warnOfExpensiveRequest(s.Logger(), s.Cfg.WarningApplyDuration, start, stringer, nil, nil)
	}(time.Now())

	switch r.Method {
	case "POST":
		return s.applyV2.Post(r)
	case "PUT":
		return s.applyV2.Put(r, shouldApplyV3)
	case "DELETE":
		return s.applyV2.Delete(r)
	case "QGET":
		return s.applyV2.QGet(r)
	case "SYNC":
		return s.applyV2.Sync(r)
	default:
		// This should never be reached, but just in case:
		return Response{Err: ErrUnknownMethod}
	}
}

func (r *RequestV2) TTLOptions() v2store.TTLOptionSet {
	refresh, _ := pbutil.GetBool(r.Refresh)
	ttlOptions := v2store.TTLOptionSet{Refresh: refresh}
	if r.Expiration != 0 {
		ttlOptions.ExpireTime = time.Unix(0, r.Expiration)
	}
	return ttlOptions
}

func toResponse(ev *v2store.Event, err error) Response {
	return Response{Event: ev, Err: err}
}
