/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package bulk

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/websocket"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/internalversion"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	bulkapi "k8s.io/apiserver/pkg/apis/bulk"
	bulkvalidation "k8s.io/apiserver/pkg/apis/bulk/validation"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/server/httplog"
	"k8s.io/apiserver/pkg/util/wsstream"
)

// HTTP handler for bulk-watch, stateless.
type watchHTTPHandler struct {
	*APIManager
}

// bulkConnection represents single WS connection, this interface
// used by get/watch handlers to send responses / errors.
type bulkConnection interface {
	Done() chan struct{}
	Abort(error)
	SendResponse(req *bulkapi.BulkRequest, resp *bulkapi.BulkResponse) bool
	SendError(req *bulkapi.BulkRequest, err error) bool
	GroupVersion() schema.GroupVersion
	Context() request.Context
	SerializerInfo() runtime.SerializerInfo
	OriginalRequest() *http.Request
}

// Handles single bulkwatch connection, keeps connection state.
type bulkConnectionImpl struct {

	// Link to bulk.APIManager
	*APIManager

	request *http.Request

	// Protect mutable state (watches map etc)
	sync.Mutex

	// Active request context
	context request.Context

	serializerInfo runtime.SerializerInfo
	encoder        runtime.Encoder
	decoder        runtime.Decoder

	quitOnce  sync.Once
	quit      chan struct{}
	responses chan *bulkapi.BulkResponse

	// map 'watch-id' -> single watch state
	watches map[string]singleWatch

	proxyPool proxyConnectionsPool
}

func (h *watchHTTPHandler) responseError(err error, w http.ResponseWriter, req *http.Request) {
	if ctx, ok := h.mapper.Get(req); !ok {
		panic("request context required")
	} else {
		responsewriters.ErrorNegotiated(ctx, err, h.negotiatedSerializer, h.GroupVersion, w, req)
	}
}

// ServeHTTP ... DOCME
func (h watchHTTPHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	m := h.APIManager
	ctx, ok := m.mapper.Get(req)
	if !ok {
		responsewriters.InternalError(w, req, errors.New("no context found for request"))
		return
	}

	// negotiate for the stream serializerInfo - TODO: should we create new fn?
	si, err := negotiation.NegotiateOutputStreamSerializer(req, m.negotiatedSerializer)
	if err != nil {
		h.responseError(apierrors.NewInternalError(err), w, req)
		return
	}

	// TODO: use correct mediaType here
	// - Respect 'Accept' header?
	// - Return err NewNotAcceptableError([]string{"**;stream=bulkwatch"})
	mediaType := si.MediaType
	if mediaType != runtime.ContentTypeJSON {
		mediaType += ";stream=bulk-watch"
	}

	if !wsstream.IsWebSocketRequest(req) {
		err := fmt.Errorf("bulk watch supports only websocket")
		h.responseError(apierrors.NewInternalError(err), w, req)
		return
	}

	w = httplog.Unlogged(w)
	w.Header().Set("Content-Type", mediaType)

	wh := &bulkConnectionImpl{
		APIManager: m,
		request:    req,
		context:    ctx,

		serializerInfo: si,
		encoder:        m.negotiatedSerializer.EncoderForVersion(si.StreamSerializer, m.GroupVersion),
		decoder:        m.negotiatedSerializer.DecoderToVersion(si.StreamSerializer, m.GroupVersion),

		watches:   make(map[string]singleWatch),
		responses: make(chan *bulkapi.BulkResponse, 10),
		quit:      make(chan struct{}),
	}
	wh.proxyPool = h.newProxyConnectionsPool(wh)

	handler := websocket.Handler(wh.HandleWS)
	websocket.Server{Handler: handler}.ServeHTTP(w, req)
}

func (s *bulkConnectionImpl) GroupVersion() schema.GroupVersion { return s.APIManager.GroupVersion }

func (s *bulkConnectionImpl) Context() request.Context { return s.context }

func (s *bulkConnectionImpl) Done() chan struct{} { return s.quit }

func (s *bulkConnectionImpl) SerializerInfo() runtime.SerializerInfo { return s.serializerInfo }

func (s *bulkConnectionImpl) OriginalRequest() *http.Request { return s.request }

func (s *bulkConnectionImpl) Abort(err error) {
	if err != nil {
		utilruntime.HandleError(err)
	}
	s.quitOnce.Do(func() { close(s.quit) })
}

func (s *bulkConnectionImpl) resetTimeout(ws *websocket.Conn) {
	if s.wsTimeout > 0 {
		if err := ws.SetDeadline(time.Now().Add(s.wsTimeout)); err != nil {
			utilruntime.HandleError(err)
		}
	}
}

func (s *bulkConnectionImpl) SendResponse(req *bulkapi.BulkRequest, resp *bulkapi.BulkResponse) bool {
	if req != nil && req.RequestID != "" {
		resp.RequestID = &req.RequestID
	}
	select {
	case s.responses <- resp:
		return true
	case <-s.quit:
		return false
	}
}

func (s *bulkConnectionImpl) SendError(req *bulkapi.BulkRequest, err error) bool {
	status := responsewriters.ErrorToAPIStatus(err)
	return s.SendResponse(req, &bulkapi.BulkResponse{Failure: status})
}

// Reads incoming requests from WS, validates them and run `handleRequest`
func (s *bulkConnectionImpl) readRequestsLoop(ws *websocket.Conn) {

	defer utilruntime.HandleCrash()
	groupKind := bulkapi.Kind("BulkRequest")
	defaultGVK := s.GroupVersion().WithKind("BulkRequest")
	var data []byte

	for {
		s.resetTimeout(ws)
		if err := websocket.Message.Receive(ws, &data); err != nil {
			if err == io.EOF {
				s.Abort(nil)
				return
			}
			s.Abort(fmt.Errorf("unable to receive message: %v", err))
			return
		}
		if len(data) == 0 {
			continue
		}
		reqRaw, _, err := s.decoder.Decode(data, &defaultGVK, &bulkapi.BulkRequest{})
		if err != nil {
			s.Abort(fmt.Errorf("unable to decode bulk request: %v", err))
			return
		}
		req, ok := reqRaw.(*bulkapi.BulkRequest)
		if !ok {
			s.Abort(fmt.Errorf("unable to decode bulk request: cast error"))
			return
		}
		if errs := bulkvalidation.ValidateBulkRequest(req); len(errs) > 0 {
			s.SendError(req, apierrors.NewInvalid(groupKind, "", errs))
			continue
		}
		if err = s.handleRequest(req); err != nil {
			s.SendError(req, err)
			continue
		}
	}
}

func (s *bulkConnectionImpl) handleRequest(r *bulkapi.BulkRequest) error {
	switch {
	case r.Watch != nil:
		return s.handleNewWatch(r)
	case r.StopWatch != nil:
		return s.handleStopWatch(r)
	default:
		return fmt.Errorf("unknown operation")
	}
}

func (s *bulkConnectionImpl) handleStopWatch(r *bulkapi.BulkRequest) error {
	wid := r.StopWatch.WatchID
	if w, ok := s.watches[wid]; ok {
		w.StopWatch(r)
		return nil
	}
	if proxy, ok := s.proxyPool.FindConnectionByWatch(wid); ok {
		return proxy.ForwardRequest(r)
	}
	return fmt.Errorf("watch not found")
}

func (s *bulkConnectionImpl) handleNewWatch(r *bulkapi.BulkRequest) error {
	s.Lock()
	defer s.Unlock()
	rs := &r.Watch.Selector
	s.normalizeSelector(rs)

	groupInfo, err := s.findGroupInfo(rs)
	if err != nil {
		return err
	}
	if groupInfo.Proxied != nil {
		// Route to proxy.
		return s.forwardRequest(r, groupInfo.Proxied)
	}

	watchID := r.Watch.WatchID
	if _, ok := s.watches[watchID]; ok {
		return fmt.Errorf("watch %v already exists", watchID)
	}

	// Watch list or single object, respect namespace.
	ctx := request.WithNamespace(s.Context, rs.Namespace)

	permChecker := newAuthorizationCheckerForWatch(groupInfo, ctx, rs)
	if err = permChecker(); err != nil {
		return err
	}
	w, err := makeSingleWatch(s, r, groupInfo.Local)
	if err != nil {
		return err
	}

	s.watches[w.WatchID()] = w
	go func() {
		defer utilruntime.HandleCrash()
		defer s.stopAndRemoveWatch(w)
		w.RunWatchLoop()
	}()
	return nil
}

func (s *bulkConnectionImpl) findGroupInfo(rs *bulkapi.ResourceSelector) (*registeredAPIGroup, error) {
	gv := schema.GroupVersion{Group: rs.Group, Version: rs.Version}
	groupInfo, ok := s.apiGroups[gv]
	if !ok {
		return nil, fmt.Errorf("unsupported group '%s/%s'", rs.Group, rs.Version)
	}
	return groupInfo, nil
}

func (s *bulkConnectionImpl) normalizeSelector(rs *bulkapi.ResourceSelector) {
	if rs.Version == "" {
		rs.Version = s.APIManager.preferredVersion[rs.Group]
	}
	if rs.Name != "" {
		nameSelector := fields.OneTermEqualSelector("metadata.name", rs.Name)
		if rs.Options == nil {
			rs.Options = &internalversion.ListOptions{}
		}
		rs.Options.FieldSelector = nameSelector
	}
}

func (s *bulkConnectionImpl) forwardRequest(r *bulkapi.BulkRequest, proxyInfo *ProxiedAPIGroupInfo) error {
	proxy, err := s.proxyPool.SpawnProxyConnection(proxyInfo)
	if err != nil {
		return fmt.Errorf("unable to open proxy connection to %s: %v", proxyInfo.GroupVersion, err)
	}
	// Just route request without additional process (authorization etc)
	// all responses automatically routed directly into current websocket.
	return proxy.ForwardRequest(r)
}

func (s *bulkConnectionImpl) stopAndRemoveWatch(w singleWatch) {
	s.Lock()
	defer s.Unlock()
	w.StopWatch(nil)
	delete(s.watches, w.WatchID())
}

// Consumes objects from 'responses' channel and writes them into the websocket.
func (s *bulkConnectionImpl) runResponsesLoop(ws *websocket.Conn) {
	defer utilruntime.HandleCrash()
	streamBuf := &bytes.Buffer{}
	for {
		select {
		case response, ok := <-s.responses:
			if !ok {
				return
			}
			s.resetTimeout(ws)
			if err := s.encoder.Encode(response, streamBuf); err != nil {
				s.Abort(fmt.Errorf("unable to encode event: %v, %v", err, response))
				return
			}
			var data interface{}
			if s.serializerInfo.EncodesAsText {
				data = streamBuf.String()
			} else {
				data = streamBuf.Bytes()
			}
			if err := websocket.Message.Send(ws, data); err != nil {
				s.Abort(err)
				return
			}
			streamBuf.Reset()
		case <-s.quit:
			return
		}
	}
}

func maybeHandleError(err error) {
	if err != nil {
		utilruntime.HandleError(err)
	}
}

func (s *bulkConnectionImpl) HandleWS(ws *websocket.Conn) {
	defer maybeHandleError(ws.Close())
	defer maybeHandleError(s.proxyPool.Close())

	go s.readRequestsLoop(ws)
	go s.runResponsesLoop(ws)

	select {
	case <-s.quit:
		glog.V(10).Infof("bulkConnectionImpl{} was quit")
		return
	case <-s.Context().Done():
		err := s.Context().Err()
		glog.V(10).Infof("Context was Done due to %v", err)
		s.Abort(err)
		return
	}
}
