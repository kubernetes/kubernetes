// Copyright 2015 CoreOS, Inc.
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

package remote

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strconv"

	"github.com/coreos/flannel/Godeps/_workspace/src/github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/flannel/Godeps/_workspace/src/github.com/coreos/go-systemd/activation"
	log "github.com/coreos/flannel/Godeps/_workspace/src/github.com/golang/glog"
	"github.com/coreos/flannel/Godeps/_workspace/src/github.com/gorilla/mux"
	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"

	"github.com/coreos/flannel/subnet"
)

type handler func(context.Context, subnet.Manager, http.ResponseWriter, *http.Request)

func jsonResponse(w http.ResponseWriter, code int, v interface{}) {
	w.Header().Set("Content-Type", "application/json; charset=utf-8")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Error("Error JSON encoding response: %v", err)
	}
}

// GET /{network}/config
func handleGetNetworkConfig(ctx context.Context, sm subnet.Manager, w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	network := mux.Vars(r)["network"]
	if network == "_" {
		network = ""
	}

	c, err := sm.GetNetworkConfig(ctx, network)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, err)
		return
	}

	jsonResponse(w, http.StatusOK, c)
}

// POST /{network}/leases
func handleAcquireLease(ctx context.Context, sm subnet.Manager, w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	network := mux.Vars(r)["network"]
	if network == "_" {
		network = ""
	}

	attrs := subnet.LeaseAttrs{}
	if err := json.NewDecoder(r.Body).Decode(&attrs); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, "JSON decoding error: ", err)
		return
	}

	lease, err := sm.AcquireLease(ctx, network, &attrs)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, err)
		return
	}

	jsonResponse(w, http.StatusOK, lease)
}

// PUT /{network}/{lease.network}
func handleRenewLease(ctx context.Context, sm subnet.Manager, w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	network := mux.Vars(r)["network"]
	if network == "_" {
		network = ""
	}

	lease := subnet.Lease{}
	if err := json.NewDecoder(r.Body).Decode(&lease); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		fmt.Fprint(w, "JSON decoding error: ", err)
		return
	}

	if err := sm.RenewLease(ctx, network, &lease); err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, err)
		return
	}

	jsonResponse(w, http.StatusOK, lease)
}

func getCursor(u *url.URL) interface{} {
	vals, ok := u.Query()["next"]
	if !ok {
		return nil
	}
	return vals[0]
}

// GET /{network}/leases?next=cursor
func handleWatchLeases(ctx context.Context, sm subnet.Manager, w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	network := mux.Vars(r)["network"]
	if network == "_" {
		network = ""
	}

	cursor := getCursor(r.URL)

	wr, err := sm.WatchLeases(ctx, network, cursor)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, err)
		return
	}

	switch wr.Cursor.(type) {
	case string:
	case fmt.Stringer:
		wr.Cursor = wr.Cursor.(fmt.Stringer).String()
	default:
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, fmt.Errorf("internal error: watch cursor is of unknown type"))
		return
	}

	jsonResponse(w, http.StatusOK, wr)
}

// GET /?next=cursor watches
// GET / retrieves all networks
func handleNetworks(ctx context.Context, sm subnet.Manager, w http.ResponseWriter, r *http.Request) {
	defer r.Body.Close()

	cursor := getCursor(r.URL)
	wr, err := sm.WatchNetworks(ctx, cursor)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, err)
		return
	}

	switch wr.Cursor.(type) {
	case string:
	case fmt.Stringer:
		wr.Cursor = wr.Cursor.(fmt.Stringer).String()
	default:
		w.WriteHeader(http.StatusInternalServerError)
		fmt.Fprint(w, fmt.Errorf("internal error: watch cursor is of unknown type"))
		return
	}

	jsonResponse(w, http.StatusOK, wr)
}

func bindHandler(h handler, ctx context.Context, sm subnet.Manager) http.HandlerFunc {
	return func(resp http.ResponseWriter, req *http.Request) {
		h(ctx, sm, resp, req)
	}
}

func fdListener(addr string) (net.Listener, error) {
	fdOffset := 0
	if addr != "" {
		fd, err := strconv.Atoi(addr)
		if err != nil {
			return nil, fmt.Errorf("fd index is not a number")
		}
		fdOffset = fd - 3
	}

	listeners, err := activation.Listeners(false)
	if err != nil {
		return nil, err
	}

	if fdOffset >= len(listeners) {
		return nil, fmt.Errorf("fd %v is out of range (%v)", addr, len(listeners)+3)
	}

	if listeners[fdOffset] == nil {
		return nil, fmt.Errorf("fd %v was not socket activated", addr)
	}

	return listeners[fdOffset], nil
}

func listener(addr, cafile, certfile, keyfile string) (net.Listener, error) {
	rex := regexp.MustCompile("(?:([a-z]+)://)?(.*)")
	groups := rex.FindStringSubmatch(addr)

	var l net.Listener
	var err error

	switch {
	case groups == nil:
		return nil, fmt.Errorf("bad listener address")

	case groups[1] == "", groups[1] == "tcp":
		if l, err = net.Listen("tcp", groups[2]); err != nil {
			return nil, err
		}

	case groups[1] == "fd":
		if l, err = fdListener(groups[2]); err != nil {
			return nil, err
		}

	default:
		return nil, fmt.Errorf("bad listener scheme")
	}

	tlsinfo := transport.TLSInfo{
		CAFile:   cafile,
		CertFile: certfile,
		KeyFile:  keyfile,
	}

	if !tlsinfo.Empty() {
		cfg, err := tlsinfo.ServerConfig()
		if err != nil {
			return nil, err
		}

		l = tls.NewListener(l, cfg)
	}

	return l, nil
}

func RunServer(ctx context.Context, sm subnet.Manager, listenAddr, cafile, certfile, keyfile string) {
	// {network} is always required a the API level but to
	// keep backward compat, special "_" network is allowed
	// that means "no network"

	r := mux.NewRouter()
	r.HandleFunc("/v1/{network}/config", bindHandler(handleGetNetworkConfig, ctx, sm)).Methods("GET")
	r.HandleFunc("/v1/{network}/leases", bindHandler(handleAcquireLease, ctx, sm)).Methods("POST")
	r.HandleFunc("/v1/{network}/leases/{subnet}", bindHandler(handleRenewLease, ctx, sm)).Methods("PUT")
	r.HandleFunc("/v1/{network}/leases", bindHandler(handleWatchLeases, ctx, sm)).Methods("GET")
	r.HandleFunc("/v1/", bindHandler(handleNetworks, ctx, sm)).Methods("GET")

	l, err := listener(listenAddr, cafile, certfile, keyfile)
	if err != nil {
		log.Errorf("Error listening on %v: %v", listenAddr, err)
		return
	}

	c := make(chan error, 1)
	go func() {
		c <- http.Serve(l, httpLogger(r))
	}()

	select {
	case <-ctx.Done():
		l.Close()
		<-c

	case err := <-c:
		log.Errorf("Error serving on %v: %v", listenAddr, err)
	}
}
