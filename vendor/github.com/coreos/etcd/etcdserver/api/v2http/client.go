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
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"

	etcdErr "github.com/coreos/etcd/error"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api"
	"github.com/coreos/etcd/etcdserver/api/etcdhttp"
	"github.com/coreos/etcd/etcdserver/api/v2http/httptypes"
	"github.com/coreos/etcd/etcdserver/auth"
	"github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/etcdserver/membership"
	"github.com/coreos/etcd/etcdserver/stats"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/store"
	"github.com/jonboulle/clockwork"
	"golang.org/x/net/context"
)

const (
	authPrefix     = "/v2/auth"
	keysPrefix     = "/v2/keys"
	machinesPrefix = "/v2/machines"
	membersPrefix  = "/v2/members"
	statsPrefix    = "/v2/stats"
)

// NewClientHandler generates a muxed http.Handler with the given parameters to serve etcd client requests.
func NewClientHandler(server *etcdserver.EtcdServer, timeout time.Duration) http.Handler {
	mux := http.NewServeMux()
	etcdhttp.HandleBasic(mux, server)
	handleV2(mux, server, timeout)
	return requestLogger(mux)
}

func handleV2(mux *http.ServeMux, server *etcdserver.EtcdServer, timeout time.Duration) {
	sec := auth.NewStore(server, timeout)
	kh := &keysHandler{
		sec:                   sec,
		server:                server,
		cluster:               server.Cluster(),
		timer:                 server,
		timeout:               timeout,
		clientCertAuthEnabled: server.Cfg.ClientCertAuthEnabled,
	}

	sh := &statsHandler{
		stats: server,
	}

	mh := &membersHandler{
		sec:     sec,
		server:  server,
		cluster: server.Cluster(),
		timeout: timeout,
		clock:   clockwork.NewRealClock(),
		clientCertAuthEnabled: server.Cfg.ClientCertAuthEnabled,
	}

	mah := &machinesHandler{cluster: server.Cluster()}

	sech := &authHandler{
		sec:                   sec,
		cluster:               server.Cluster(),
		clientCertAuthEnabled: server.Cfg.ClientCertAuthEnabled,
	}
	mux.HandleFunc("/", http.NotFound)
	mux.Handle(keysPrefix, kh)
	mux.Handle(keysPrefix+"/", kh)
	mux.HandleFunc(statsPrefix+"/store", sh.serveStore)
	mux.HandleFunc(statsPrefix+"/self", sh.serveSelf)
	mux.HandleFunc(statsPrefix+"/leader", sh.serveLeader)
	mux.Handle(membersPrefix, mh)
	mux.Handle(membersPrefix+"/", mh)
	mux.Handle(machinesPrefix, mah)
	handleAuth(mux, sech)
}

type keysHandler struct {
	sec                   auth.Store
	server                etcdserver.Server
	cluster               api.Cluster
	timer                 etcdserver.RaftTimer
	timeout               time.Duration
	clientCertAuthEnabled bool
}

func (h *keysHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r.Method, "HEAD", "GET", "PUT", "POST", "DELETE") {
		return
	}

	w.Header().Set("X-Etcd-Cluster-ID", h.cluster.ID().String())

	ctx, cancel := context.WithTimeout(context.Background(), h.timeout)
	defer cancel()
	clock := clockwork.NewRealClock()
	startTime := clock.Now()
	rr, noValueOnSuccess, err := parseKeyRequest(r, clock)
	if err != nil {
		writeKeyError(w, err)
		return
	}
	// The path must be valid at this point (we've parsed the request successfully).
	if !hasKeyPrefixAccess(h.sec, r, r.URL.Path[len(keysPrefix):], rr.Recursive, h.clientCertAuthEnabled) {
		writeKeyNoAuth(w)
		return
	}
	if !rr.Wait {
		reportRequestReceived(rr)
	}
	resp, err := h.server.Do(ctx, rr)
	if err != nil {
		err = trimErrorPrefix(err, etcdserver.StoreKeysPrefix)
		writeKeyError(w, err)
		reportRequestFailed(rr, err)
		return
	}
	switch {
	case resp.Event != nil:
		if err := writeKeyEvent(w, resp.Event, noValueOnSuccess, h.timer); err != nil {
			// Should never be reached
			plog.Errorf("error writing event (%v)", err)
		}
		reportRequestCompleted(rr, resp, startTime)
	case resp.Watcher != nil:
		ctx, cancel := context.WithTimeout(context.Background(), defaultWatchTimeout)
		defer cancel()
		handleKeyWatch(ctx, w, resp.Watcher, rr.Stream, h.timer)
	default:
		writeKeyError(w, errors.New("received response with no Event/Watcher!"))
	}
}

type machinesHandler struct {
	cluster api.Cluster
}

func (h *machinesHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r.Method, "GET", "HEAD") {
		return
	}
	endpoints := h.cluster.ClientURLs()
	w.Write([]byte(strings.Join(endpoints, ", ")))
}

type membersHandler struct {
	sec                   auth.Store
	server                etcdserver.Server
	cluster               api.Cluster
	timeout               time.Duration
	clock                 clockwork.Clock
	clientCertAuthEnabled bool
}

func (h *membersHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r.Method, "GET", "POST", "DELETE", "PUT") {
		return
	}
	if !hasWriteRootAccess(h.sec, r, h.clientCertAuthEnabled) {
		writeNoAuth(w, r)
		return
	}
	w.Header().Set("X-Etcd-Cluster-ID", h.cluster.ID().String())

	ctx, cancel := context.WithTimeout(context.Background(), h.timeout)
	defer cancel()

	switch r.Method {
	case "GET":
		switch trimPrefix(r.URL.Path, membersPrefix) {
		case "":
			mc := newMemberCollection(h.cluster.Members())
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(mc); err != nil {
				plog.Warningf("failed to encode members response (%v)", err)
			}
		case "leader":
			id := h.server.Leader()
			if id == 0 {
				writeError(w, r, httptypes.NewHTTPError(http.StatusServiceUnavailable, "During election"))
				return
			}
			m := newMember(h.cluster.Member(id))
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(m); err != nil {
				plog.Warningf("failed to encode members response (%v)", err)
			}
		default:
			writeError(w, r, httptypes.NewHTTPError(http.StatusNotFound, "Not found"))
		}
	case "POST":
		req := httptypes.MemberCreateRequest{}
		if ok := unmarshalRequest(r, &req, w); !ok {
			return
		}
		now := h.clock.Now()
		m := membership.NewMember("", req.PeerURLs, "", &now)
		_, err := h.server.AddMember(ctx, *m)
		switch {
		case err == membership.ErrIDExists || err == membership.ErrPeerURLexists:
			writeError(w, r, httptypes.NewHTTPError(http.StatusConflict, err.Error()))
			return
		case err != nil:
			plog.Errorf("error adding member %s (%v)", m.ID, err)
			writeError(w, r, err)
			return
		}
		res := newMember(m)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		if err := json.NewEncoder(w).Encode(res); err != nil {
			plog.Warningf("failed to encode members response (%v)", err)
		}
	case "DELETE":
		id, ok := getID(r.URL.Path, w)
		if !ok {
			return
		}
		_, err := h.server.RemoveMember(ctx, uint64(id))
		switch {
		case err == membership.ErrIDRemoved:
			writeError(w, r, httptypes.NewHTTPError(http.StatusGone, fmt.Sprintf("Member permanently removed: %s", id)))
		case err == membership.ErrIDNotFound:
			writeError(w, r, httptypes.NewHTTPError(http.StatusNotFound, fmt.Sprintf("No such member: %s", id)))
		case err != nil:
			plog.Errorf("error removing member %s (%v)", id, err)
			writeError(w, r, err)
		default:
			w.WriteHeader(http.StatusNoContent)
		}
	case "PUT":
		id, ok := getID(r.URL.Path, w)
		if !ok {
			return
		}
		req := httptypes.MemberUpdateRequest{}
		if ok := unmarshalRequest(r, &req, w); !ok {
			return
		}
		m := membership.Member{
			ID:             id,
			RaftAttributes: membership.RaftAttributes{PeerURLs: req.PeerURLs.StringSlice()},
		}
		_, err := h.server.UpdateMember(ctx, m)
		switch {
		case err == membership.ErrPeerURLexists:
			writeError(w, r, httptypes.NewHTTPError(http.StatusConflict, err.Error()))
		case err == membership.ErrIDNotFound:
			writeError(w, r, httptypes.NewHTTPError(http.StatusNotFound, fmt.Sprintf("No such member: %s", id)))
		case err != nil:
			plog.Errorf("error updating member %s (%v)", m.ID, err)
			writeError(w, r, err)
		default:
			w.WriteHeader(http.StatusNoContent)
		}
	}
}

type statsHandler struct {
	stats stats.Stats
}

func (h *statsHandler) serveStore(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r.Method, "GET") {
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(h.stats.StoreStats())
}

func (h *statsHandler) serveSelf(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r.Method, "GET") {
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(h.stats.SelfStats())
}

func (h *statsHandler) serveLeader(w http.ResponseWriter, r *http.Request) {
	if !allowMethod(w, r.Method, "GET") {
		return
	}
	stats := h.stats.LeaderStats()
	if stats == nil {
		etcdhttp.WriteError(w, r, httptypes.NewHTTPError(http.StatusForbidden, "not current leader"))
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(stats)
}

// parseKeyRequest converts a received http.Request on keysPrefix to
// a server Request, performing validation of supplied fields as appropriate.
// If any validation fails, an empty Request and non-nil error is returned.
func parseKeyRequest(r *http.Request, clock clockwork.Clock) (etcdserverpb.Request, bool, error) {
	noValueOnSuccess := false
	emptyReq := etcdserverpb.Request{}

	err := r.ParseForm()
	if err != nil {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodeInvalidForm,
			err.Error(),
		)
	}

	if !strings.HasPrefix(r.URL.Path, keysPrefix) {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodeInvalidForm,
			"incorrect key prefix",
		)
	}
	p := path.Join(etcdserver.StoreKeysPrefix, r.URL.Path[len(keysPrefix):])

	var pIdx, wIdx uint64
	if pIdx, err = getUint64(r.Form, "prevIndex"); err != nil {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodeIndexNaN,
			`invalid value for "prevIndex"`,
		)
	}
	if wIdx, err = getUint64(r.Form, "waitIndex"); err != nil {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodeIndexNaN,
			`invalid value for "waitIndex"`,
		)
	}

	var rec, sort, wait, dir, quorum, stream bool
	if rec, err = getBool(r.Form, "recursive"); err != nil {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodeInvalidField,
			`invalid value for "recursive"`,
		)
	}
	if sort, err = getBool(r.Form, "sorted"); err != nil {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodeInvalidField,
			`invalid value for "sorted"`,
		)
	}
	if wait, err = getBool(r.Form, "wait"); err != nil {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodeInvalidField,
			`invalid value for "wait"`,
		)
	}
	// TODO(jonboulle): define what parameters dir is/isn't compatible with?
	if dir, err = getBool(r.Form, "dir"); err != nil {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodeInvalidField,
			`invalid value for "dir"`,
		)
	}
	if quorum, err = getBool(r.Form, "quorum"); err != nil {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodeInvalidField,
			`invalid value for "quorum"`,
		)
	}
	if stream, err = getBool(r.Form, "stream"); err != nil {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodeInvalidField,
			`invalid value for "stream"`,
		)
	}

	if wait && r.Method != "GET" {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodeInvalidField,
			`"wait" can only be used with GET requests`,
		)
	}

	pV := r.FormValue("prevValue")
	if _, ok := r.Form["prevValue"]; ok && pV == "" {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodePrevValueRequired,
			`"prevValue" cannot be empty`,
		)
	}

	if noValueOnSuccess, err = getBool(r.Form, "noValueOnSuccess"); err != nil {
		return emptyReq, false, etcdErr.NewRequestError(
			etcdErr.EcodeInvalidField,
			`invalid value for "noValueOnSuccess"`,
		)
	}

	// TTL is nullable, so leave it null if not specified
	// or an empty string
	var ttl *uint64
	if len(r.FormValue("ttl")) > 0 {
		i, err := getUint64(r.Form, "ttl")
		if err != nil {
			return emptyReq, false, etcdErr.NewRequestError(
				etcdErr.EcodeTTLNaN,
				`invalid value for "ttl"`,
			)
		}
		ttl = &i
	}

	// prevExist is nullable, so leave it null if not specified
	var pe *bool
	if _, ok := r.Form["prevExist"]; ok {
		bv, err := getBool(r.Form, "prevExist")
		if err != nil {
			return emptyReq, false, etcdErr.NewRequestError(
				etcdErr.EcodeInvalidField,
				"invalid value for prevExist",
			)
		}
		pe = &bv
	}

	// refresh is nullable, so leave it null if not specified
	var refresh *bool
	if _, ok := r.Form["refresh"]; ok {
		bv, err := getBool(r.Form, "refresh")
		if err != nil {
			return emptyReq, false, etcdErr.NewRequestError(
				etcdErr.EcodeInvalidField,
				"invalid value for refresh",
			)
		}
		refresh = &bv
		if refresh != nil && *refresh {
			val := r.FormValue("value")
			if _, ok := r.Form["value"]; ok && val != "" {
				return emptyReq, false, etcdErr.NewRequestError(
					etcdErr.EcodeRefreshValue,
					`A value was provided on a refresh`,
				)
			}
			if ttl == nil {
				return emptyReq, false, etcdErr.NewRequestError(
					etcdErr.EcodeRefreshTTLRequired,
					`No TTL value set`,
				)
			}
		}
	}

	rr := etcdserverpb.Request{
		Method:    r.Method,
		Path:      p,
		Val:       r.FormValue("value"),
		Dir:       dir,
		PrevValue: pV,
		PrevIndex: pIdx,
		PrevExist: pe,
		Wait:      wait,
		Since:     wIdx,
		Recursive: rec,
		Sorted:    sort,
		Quorum:    quorum,
		Stream:    stream,
	}

	if pe != nil {
		rr.PrevExist = pe
	}

	if refresh != nil {
		rr.Refresh = refresh
	}

	// Null TTL is equivalent to unset Expiration
	if ttl != nil {
		expr := time.Duration(*ttl) * time.Second
		rr.Expiration = clock.Now().Add(expr).UnixNano()
	}

	return rr, noValueOnSuccess, nil
}

// writeKeyEvent trims the prefix of key path in a single Event under
// StoreKeysPrefix, serializes it and writes the resulting JSON to the given
// ResponseWriter, along with the appropriate headers.
func writeKeyEvent(w http.ResponseWriter, ev *store.Event, noValueOnSuccess bool, rt etcdserver.RaftTimer) error {
	if ev == nil {
		return errors.New("cannot write empty Event!")
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Etcd-Index", fmt.Sprint(ev.EtcdIndex))
	w.Header().Set("X-Raft-Index", fmt.Sprint(rt.Index()))
	w.Header().Set("X-Raft-Term", fmt.Sprint(rt.Term()))

	if ev.IsCreated() {
		w.WriteHeader(http.StatusCreated)
	}

	ev = trimEventPrefix(ev, etcdserver.StoreKeysPrefix)
	if noValueOnSuccess &&
		(ev.Action == store.Set || ev.Action == store.CompareAndSwap ||
			ev.Action == store.Create || ev.Action == store.Update) {
		ev.Node = nil
		ev.PrevNode = nil
	}
	return json.NewEncoder(w).Encode(ev)
}

func writeKeyNoAuth(w http.ResponseWriter) {
	e := etcdErr.NewError(etcdErr.EcodeUnauthorized, "Insufficient credentials", 0)
	e.WriteTo(w)
}

// writeKeyError logs and writes the given Error to the ResponseWriter.
// If Error is not an etcdErr, the error will be converted to an etcd error.
func writeKeyError(w http.ResponseWriter, err error) {
	if err == nil {
		return
	}
	switch e := err.(type) {
	case *etcdErr.Error:
		e.WriteTo(w)
	default:
		switch err {
		case etcdserver.ErrTimeoutDueToLeaderFail, etcdserver.ErrTimeoutDueToConnectionLost:
			mlog.MergeError(err)
		default:
			mlog.MergeErrorf("got unexpected response error (%v)", err)
		}
		ee := etcdErr.NewError(etcdErr.EcodeRaftInternal, err.Error(), 0)
		ee.WriteTo(w)
	}
}

func handleKeyWatch(ctx context.Context, w http.ResponseWriter, wa store.Watcher, stream bool, rt etcdserver.RaftTimer) {
	defer wa.Remove()
	ech := wa.EventChan()
	var nch <-chan bool
	if x, ok := w.(http.CloseNotifier); ok {
		nch = x.CloseNotify()
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Etcd-Index", fmt.Sprint(wa.StartIndex()))
	w.Header().Set("X-Raft-Index", fmt.Sprint(rt.Index()))
	w.Header().Set("X-Raft-Term", fmt.Sprint(rt.Term()))
	w.WriteHeader(http.StatusOK)

	// Ensure headers are flushed early, in case of long polling
	w.(http.Flusher).Flush()

	for {
		select {
		case <-nch:
			// Client closed connection. Nothing to do.
			return
		case <-ctx.Done():
			// Timed out. net/http will close the connection for us, so nothing to do.
			return
		case ev, ok := <-ech:
			if !ok {
				// If the channel is closed this may be an indication of
				// that notifications are much more than we are able to
				// send to the client in time. Then we simply end streaming.
				return
			}
			ev = trimEventPrefix(ev, etcdserver.StoreKeysPrefix)
			if err := json.NewEncoder(w).Encode(ev); err != nil {
				// Should never be reached
				plog.Warningf("error writing event (%v)", err)
				return
			}
			if !stream {
				return
			}
			w.(http.Flusher).Flush()
		}
	}
}

func trimEventPrefix(ev *store.Event, prefix string) *store.Event {
	if ev == nil {
		return nil
	}
	// Since the *Event may reference one in the store history
	// history, we must copy it before modifying
	e := ev.Clone()
	trimNodeExternPrefix(e.Node, prefix)
	trimNodeExternPrefix(e.PrevNode, prefix)
	return e
}

func trimNodeExternPrefix(n *store.NodeExtern, prefix string) {
	if n == nil {
		return
	}
	n.Key = strings.TrimPrefix(n.Key, prefix)
	for _, nn := range n.Nodes {
		trimNodeExternPrefix(nn, prefix)
	}
}

func trimErrorPrefix(err error, prefix string) error {
	if e, ok := err.(*etcdErr.Error); ok {
		e.Cause = strings.TrimPrefix(e.Cause, prefix)
	}
	return err
}

func unmarshalRequest(r *http.Request, req json.Unmarshaler, w http.ResponseWriter) bool {
	ctype := r.Header.Get("Content-Type")
	semicolonPosition := strings.Index(ctype, ";")
	if semicolonPosition != -1 {
		ctype = strings.TrimSpace(strings.ToLower(ctype[0:semicolonPosition]))
	}
	if ctype != "application/json" {
		writeError(w, r, httptypes.NewHTTPError(http.StatusUnsupportedMediaType, fmt.Sprintf("Bad Content-Type %s, accept application/json", ctype)))
		return false
	}
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		writeError(w, r, httptypes.NewHTTPError(http.StatusBadRequest, err.Error()))
		return false
	}
	if err := req.UnmarshalJSON(b); err != nil {
		writeError(w, r, httptypes.NewHTTPError(http.StatusBadRequest, err.Error()))
		return false
	}
	return true
}

func getID(p string, w http.ResponseWriter) (types.ID, bool) {
	idStr := trimPrefix(p, membersPrefix)
	if idStr == "" {
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return 0, false
	}
	id, err := types.IDFromString(idStr)
	if err != nil {
		writeError(w, nil, httptypes.NewHTTPError(http.StatusNotFound, fmt.Sprintf("No such member: %s", idStr)))
		return 0, false
	}
	return id, true
}

// getUint64 extracts a uint64 by the given key from a Form. If the key does
// not exist in the form, 0 is returned. If the key exists but the value is
// badly formed, an error is returned. If multiple values are present only the
// first is considered.
func getUint64(form url.Values, key string) (i uint64, err error) {
	if vals, ok := form[key]; ok {
		i, err = strconv.ParseUint(vals[0], 10, 64)
	}
	return
}

// getBool extracts a bool by the given key from a Form. If the key does not
// exist in the form, false is returned. If the key exists but the value is
// badly formed, an error is returned. If multiple values are present only the
// first is considered.
func getBool(form url.Values, key string) (b bool, err error) {
	if vals, ok := form[key]; ok {
		b, err = strconv.ParseBool(vals[0])
	}
	return
}

// trimPrefix removes a given prefix and any slash following the prefix
// e.g.: trimPrefix("foo", "foo") == trimPrefix("foo/", "foo") == ""
func trimPrefix(p, prefix string) (s string) {
	s = strings.TrimPrefix(p, prefix)
	s = strings.TrimPrefix(s, "/")
	return
}

func newMemberCollection(ms []*membership.Member) *httptypes.MemberCollection {
	c := httptypes.MemberCollection(make([]httptypes.Member, len(ms)))

	for i, m := range ms {
		c[i] = newMember(m)
	}

	return &c
}

func newMember(m *membership.Member) httptypes.Member {
	tm := httptypes.Member{
		ID:         m.ID.String(),
		Name:       m.Name,
		PeerURLs:   make([]string, len(m.PeerURLs)),
		ClientURLs: make([]string, len(m.ClientURLs)),
	}

	copy(tm.PeerURLs, m.PeerURLs)
	copy(tm.ClientURLs, m.ClientURLs)

	return tm
}
