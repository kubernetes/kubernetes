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
	"encoding/json"
	"net/http"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api"
	"github.com/coreos/etcd/lease/leasehttp"
	"github.com/coreos/etcd/rafthttp"
)

const (
	peerMembersPrefix = "/members"
)

// NewPeerHandler generates an http.Handler to handle etcd peer requests.
func NewPeerHandler(s *etcdserver.EtcdServer) http.Handler {
	var lh http.Handler
	l := s.Lessor()
	if l != nil {
		lh = leasehttp.NewHandler(l, func() <-chan struct{} { return s.ApplyWait() })
	}
	return newPeerHandler(s.Cluster(), s.RaftHandler(), lh)
}

func newPeerHandler(cluster api.Cluster, raftHandler http.Handler, leaseHandler http.Handler) http.Handler {
	mh := &peerMembersHandler{
		cluster: cluster,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", http.NotFound)
	mux.Handle(rafthttp.RaftPrefix, raftHandler)
	mux.Handle(rafthttp.RaftPrefix+"/", raftHandler)
	mux.Handle(peerMembersPrefix, mh)
	if leaseHandler != nil {
		mux.Handle(leasehttp.LeasePrefix, leaseHandler)
		mux.Handle(leasehttp.LeaseInternalPrefix, leaseHandler)
	}
	mux.HandleFunc(versionPath, versionHandler(cluster, serveVersion))
	return mux
}

type peerMembersHandler struct {
	cluster api.Cluster
}

func (h *peerMembersHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r.Method, "GET") {
		return
	}
	w.Header().Set("X-Etcd-Cluster-ID", h.cluster.ID().String())

	if r.URL.Path != peerMembersPrefix {
		http.Error(w, "bad path", http.StatusBadRequest)
		return
	}
	ms := h.cluster.Members()
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(ms); err != nil {
		plog.Warningf("failed to encode members response (%v)", err)
	}
}
