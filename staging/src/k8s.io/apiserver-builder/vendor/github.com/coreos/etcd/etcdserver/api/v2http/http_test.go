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

package v2http

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"sort"
	"testing"

	etcdErr "github.com/coreos/etcd/error"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/etcdserver/membership"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/go-semver/semver"
	"golang.org/x/net/context"
)

type fakeCluster struct {
	id         uint64
	clientURLs []string
	members    map[uint64]*membership.Member
}

func (c *fakeCluster) ID() types.ID         { return types.ID(c.id) }
func (c *fakeCluster) ClientURLs() []string { return c.clientURLs }
func (c *fakeCluster) Members() []*membership.Member {
	var ms membership.MembersByID
	for _, m := range c.members {
		ms = append(ms, m)
	}
	sort.Sort(ms)
	return []*membership.Member(ms)
}
func (c *fakeCluster) Member(id types.ID) *membership.Member { return c.members[uint64(id)] }
func (c *fakeCluster) IsIDRemoved(id types.ID) bool          { return false }
func (c *fakeCluster) Version() *semver.Version              { return nil }

// errServer implements the etcd.Server interface for testing.
// It returns the given error from any Do/Process/AddMember/RemoveMember calls.
type errServer struct {
	err error
}

func (fs *errServer) Start()           {}
func (fs *errServer) Stop()            {}
func (fs *errServer) ID() types.ID     { return types.ID(1) }
func (fs *errServer) Leader() types.ID { return types.ID(1) }
func (fs *errServer) Do(ctx context.Context, r etcdserverpb.Request) (etcdserver.Response, error) {
	return etcdserver.Response{}, fs.err
}
func (fs *errServer) Process(ctx context.Context, m raftpb.Message) error {
	return fs.err
}
func (fs *errServer) AddMember(ctx context.Context, m membership.Member) error {
	return fs.err
}
func (fs *errServer) RemoveMember(ctx context.Context, id uint64) error {
	return fs.err
}
func (fs *errServer) UpdateMember(ctx context.Context, m membership.Member) error {
	return fs.err
}

func (fs *errServer) ClusterVersion() *semver.Version { return nil }

func TestWriteError(t *testing.T) {
	// nil error should not panic
	rec := httptest.NewRecorder()
	r := new(http.Request)
	writeError(rec, r, nil)
	h := rec.Header()
	if len(h) > 0 {
		t.Fatalf("unexpected non-empty headers: %#v", h)
	}
	b := rec.Body.String()
	if len(b) > 0 {
		t.Fatalf("unexpected non-empty body: %q", b)
	}

	tests := []struct {
		err   error
		wcode int
		wi    string
	}{
		{
			etcdErr.NewError(etcdErr.EcodeKeyNotFound, "/foo/bar", 123),
			http.StatusNotFound,
			"123",
		},
		{
			etcdErr.NewError(etcdErr.EcodeTestFailed, "/foo/bar", 456),
			http.StatusPreconditionFailed,
			"456",
		},
		{
			err:   errors.New("something went wrong"),
			wcode: http.StatusInternalServerError,
		},
	}

	for i, tt := range tests {
		rw := httptest.NewRecorder()
		writeError(rw, r, tt.err)
		if code := rw.Code; code != tt.wcode {
			t.Errorf("#%d: code=%d, want %d", i, code, tt.wcode)
		}
		if idx := rw.Header().Get("X-Etcd-Index"); idx != tt.wi {
			t.Errorf("#%d: X-Etcd-Index=%q, want %q", i, idx, tt.wi)
		}
	}
}

func TestAllowMethod(t *testing.T) {
	tests := []struct {
		m  string
		ms []string
		w  bool
		wh string
	}{
		// Accepted methods
		{
			m:  "GET",
			ms: []string{"GET", "POST", "PUT"},
			w:  true,
		},
		{
			m:  "POST",
			ms: []string{"POST"},
			w:  true,
		},
		// Made-up methods no good
		{
			m:  "FAKE",
			ms: []string{"GET", "POST", "PUT"},
			w:  false,
			wh: "GET,POST,PUT",
		},
		// Empty methods no good
		{
			m:  "",
			ms: []string{"GET", "POST"},
			w:  false,
			wh: "GET,POST",
		},
		// Empty accepted methods no good
		{
			m:  "GET",
			ms: []string{""},
			w:  false,
			wh: "",
		},
		// No methods accepted
		{
			m:  "GET",
			ms: []string{},
			w:  false,
			wh: "",
		},
	}

	for i, tt := range tests {
		rw := httptest.NewRecorder()
		g := allowMethod(rw, tt.m, tt.ms...)
		if g != tt.w {
			t.Errorf("#%d: got allowMethod()=%t, want %t", i, g, tt.w)
		}
		if !tt.w {
			if rw.Code != http.StatusMethodNotAllowed {
				t.Errorf("#%d: code=%d, want %d", i, rw.Code, http.StatusMethodNotAllowed)
			}
			gh := rw.Header().Get("Allow")
			if gh != tt.wh {
				t.Errorf("#%d: Allow header=%q, want %q", i, gh, tt.wh)
			}
		}
	}
}
