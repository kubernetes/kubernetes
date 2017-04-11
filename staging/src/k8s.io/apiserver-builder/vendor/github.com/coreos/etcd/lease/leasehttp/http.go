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

package leasehttp

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/lease"
	"golang.org/x/net/context"
)

// NewHandler returns an http Handler for lease renewals
func NewHandler(l lease.Lessor) http.Handler {
	return &leaseHandler{l}
}

type leaseHandler struct{ l lease.Lessor }

func (h *leaseHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}

	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "error reading body", http.StatusBadRequest)
		return
	}

	lreq := pb.LeaseKeepAliveRequest{}
	if err := lreq.Unmarshal(b); err != nil {
		http.Error(w, "error unmarshalling request", http.StatusBadRequest)
		return
	}

	ttl, err := h.l.Renew(lease.LeaseID(lreq.ID))
	if err != nil {
		if err == lease.ErrLeaseNotFound {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}

		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	// TODO: fill out ResponseHeader
	resp := &pb.LeaseKeepAliveResponse{ID: lreq.ID, TTL: ttl}
	v, err := resp.Marshal()
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/protobuf")
	w.Write(v)
}

// RenewHTTP renews a lease at a given primary server.
// TODO: Batch request in future?
func RenewHTTP(ctx context.Context, id lease.LeaseID, url string, rt http.RoundTripper) (int64, error) {
	// will post lreq protobuf to leader
	lreq, err := (&pb.LeaseKeepAliveRequest{ID: int64(id)}).Marshal()
	if err != nil {
		return -1, err
	}

	cc := &http.Client{Transport: rt}
	req, err := http.NewRequest("POST", url, bytes.NewReader(lreq))
	if err != nil {
		return -1, err
	}
	req.Header.Set("Content-Type", "application/protobuf")
	req.Cancel = ctx.Done()

	resp, err := cc.Do(req)
	if err != nil {
		// TODO detect if leader failed and retry?
		return -1, err
	}
	b, err := ioutil.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return -1, err
	}

	if resp.StatusCode == http.StatusNotFound {
		return -1, lease.ErrLeaseNotFound
	}

	if resp.StatusCode != http.StatusOK {
		return -1, fmt.Errorf("lease: unknown error(%s)", string(b))
	}

	lresp := &pb.LeaseKeepAliveResponse{}
	if err := lresp.Unmarshal(b); err != nil {
		return -1, fmt.Errorf(`lease: %v. data = "%s"`, err, string(b))
	}
	if lresp.ID != int64(id) {
		return -1, fmt.Errorf("lease: renew id mismatch")
	}
	return lresp.TTL, nil
}
