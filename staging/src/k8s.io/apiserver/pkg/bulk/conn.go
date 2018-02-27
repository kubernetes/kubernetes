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
	"fmt"
	"net/http"
	"sync"

	"golang.org/x/net/context"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	bulkapi "k8s.io/apiserver/pkg/apis/bulk"
	bulkvalidation "k8s.io/apiserver/pkg/apis/bulk/validation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// bulkConnection represents single connection, this interface
// used by get/watch handlers to send responses/errors.
type bulkConnection interface {
	APIManager() *APIManager
	SerializerInfo() runtime.SerializerInfo
	StartProcessing(requests <-chan *bulkapi.ClientMessage, rw responseWriter) (done <-chan struct{})
	SendResponse(req *bulkapi.ClientMessage, resp *bulkapi.ServerMessage) bool
}

// responseWriter serialize and write response object into output sink (http or websocket connection)
type responseWriter interface {
	WriteResponse(resp *bulkapi.ServerMessage) error
}

// Handles single bulkwatch connection, keeps connection state.
type bulkConnectionImpl struct {

	// Output (http writer, wsocket etc)
	rwriter responseWriter

	// Input (chan of requests)
	requests <-chan *bulkapi.ClientMessage

	// Link to bulk.APIManager
	apiManager *APIManager

	// Protect mutable state (watches map etc)
	sync.Mutex

	// context for all gets/lits/watches, canceled when connection is closed
	context       context.Context
	contextCancel context.CancelFunc

	tasks sync.WaitGroup
	loops sync.WaitGroup

	// map 'watch-id' -> single watch state
	watches map[string]singleWatch

	// Outgoing responses
	responses chan *bulkapi.ServerMessage

	// connections to proxied apiservers
	proxyPool proxyConnectionsPool

	serializerInfo runtime.SerializerInfo
}

func newBulkConnectionImpl(m *APIManager, req *http.Request, ctx request.Context, si runtime.SerializerInfo) *bulkConnectionImpl {
	ctx, cancel := context.WithCancel(ctx)
	bc := &bulkConnectionImpl{
		apiManager:     m,
		context:        ctx,
		contextCancel:  cancel,
		serializerInfo: si,
		watches:        make(map[string]singleWatch),
		responses:      make(chan *bulkapi.ServerMessage, 100),
	}
	haveProxy := false
	for _, gi := range m.apiGroups {
		if gi.Proxied != nil {
			haveProxy = true
			break
		}
	}
	if haveProxy {
		bc.proxyPool = newProxyConnectionsPool(ctx, bc, req)
	}
	return bc
}

func (s *bulkConnectionImpl) StartProcessing(requests <-chan *bulkapi.ClientMessage, rw responseWriter) <-chan struct{} {
	processingDone := make(chan struct{})
	s.requests = requests
	s.rwriter = rw
	s.loops.Add(1)
	go func() {
		defer utilruntime.HandleCrash()
		defer s.loops.Done()
		s.runWriteResponsesLoop()
	}()
	s.loops.Add(1)
	go func() {
		defer utilruntime.HandleCrash()
		defer s.loops.Done()

		s.runProcessRequestsLoop()
		s.tasks.Wait()
		close(processingDone)
	}()
	return processingDone
}

func (s *bulkConnectionImpl) APIManager() *APIManager                { return s.apiManager }
func (s *bulkConnectionImpl) SerializerInfo() runtime.SerializerInfo { return s.serializerInfo }

func (s *bulkConnectionImpl) Close() {
	s.contextCancel() // ask all sub-goroutines to stop & wait for them
	if s.proxyPool != nil {
		s.proxyPool.Close()
	}
	close(s.responses)
	s.loops.Wait() // wait for processing loops
}

func (s *bulkConnectionImpl) handleRequest(r *bulkapi.ClientMessage) {
	err := s.handleRequestImpl(r)
	if err != nil {
		s.SendResponse(r, errorResponse(err))
	}
}

func (s *bulkConnectionImpl) handleRequestImpl(r *bulkapi.ClientMessage) error {
	if errs := bulkvalidation.ValidateClientMessage(r); len(errs) > 0 {
		groupKind := bulkapi.Kind("ClientMessage")
		return apierrors.NewInvalid(groupKind, "", errs)
	}
	switch {
	case r.Watch != nil:
		return s.handleNewWatch(r)
	case r.WatchList != nil:
		return s.handleNewWatchList(r)
	case r.StopWatch != nil:
		return s.handleStopWatch(r)
	case r.Get != nil:
		return s.handleGet(r)
	case r.List != nil:
		return s.handleList(r)
	default:
		return fmt.Errorf("unknown operation")
	}
}

func (s *bulkConnectionImpl) handleStopWatch(r *bulkapi.ClientMessage) error {
	wid := r.StopWatch.WatchID
	if w, ok := s.watches[wid]; ok {
		w.StopWatch(r)
		return nil
	}
	if s.proxyPool != nil {
		if proxy, ok := s.proxyPool.FindConnectionByWatch(wid); ok {
			return proxy.ForwardRequest(r)
		}
	}
	return fmt.Errorf("watch not found")
}

func (s *bulkConnectionImpl) handleNewWatch(r *bulkapi.ClientMessage) error {
	s.Lock()
	defer s.Unlock()

	s.normalizeGVK(&r.Watch.ItemSelector.GroupVersionResource)
	groupInfo, err := s.findGroupInfo(r.Watch.ItemSelector.GroupVersionResource)
	if err != nil {
		return err
	}
	if err = s.checkWatchAlreadyExists(r.Watch.WatchID); err != nil {
		return err
	}
	if groupInfo.Proxied != nil {
		return s.forwardRequest(r, groupInfo.Proxied)
	}
	w, err := makeSingleWatch(s.context, s, r, groupInfo.Local)
	if err != nil {
		return err
	}
	return s.spawnWatchLoop(w)
}

func (s *bulkConnectionImpl) handleNewWatchList(r *bulkapi.ClientMessage) error {
	s.Lock()
	defer s.Unlock()
	s.normalizeGVK(&r.WatchList.ListSelector.GroupVersionResource)
	groupInfo, err := s.findGroupInfo(r.WatchList.ListSelector.GroupVersionResource)
	if err != nil {
		return err
	}
	if err = s.checkWatchAlreadyExists(r.WatchList.WatchID); err != nil {
		return err
	}
	if groupInfo.Proxied != nil {
		return s.forwardRequest(r, groupInfo.Proxied)
	}
	w, err := makeSingleWatchList(s.context, s, r, groupInfo.Local)
	if err != nil {
		return err
	}
	return s.spawnWatchLoop(w)
}

func (s *bulkConnectionImpl) handleGet(r *bulkapi.ClientMessage) error {
	s.Lock()
	defer s.Unlock()
	s.normalizeGVK(&r.Get.ItemSelector.GroupVersionResource)
	groupInfo, err := s.findGroupInfo(r.Get.ItemSelector.GroupVersionResource)
	if err != nil {
		return err
	}
	if groupInfo.Proxied != nil {
		return s.forwardRequest(r, groupInfo.Proxied)
	}
	return s.spawnAsyncOperation(func() {
		performSingleGet(s.context, s, r, groupInfo.Local)
	})
}

func (s *bulkConnectionImpl) handleList(r *bulkapi.ClientMessage) error {
	s.Lock()
	defer s.Unlock()
	s.normalizeGVK(&r.List.ListSelector.GroupVersionResource)
	groupInfo, err := s.findGroupInfo(r.List.ListSelector.GroupVersionResource)
	if err != nil {
		return err
	}
	if groupInfo.Proxied != nil {
		return s.forwardRequest(r, groupInfo.Proxied)
	}
	return s.spawnAsyncOperation(func() {
		performSingleList(s.context, s, r, groupInfo.Local)
	})
}

func (s *bulkConnectionImpl) checkWatchAlreadyExists(watchID string) error {
	if _, ok := s.watches[watchID]; ok {
		return fmt.Errorf("watch %v already exists", watchID)
	}
	return nil
}

func (s *bulkConnectionImpl) spawnAsyncOperation(operation func()) error {
	s.tasks.Add(1)
	go func() {
		defer utilruntime.HandleCrash()
		defer s.tasks.Done()
		operation()
	}()
	return nil
}

func (s *bulkConnectionImpl) spawnWatchLoop(w singleWatch) error {
	s.watches[w.WatchID()] = w
	s.tasks.Add(1)
	go func() {
		defer utilruntime.HandleCrash()
		defer s.tasks.Done()
		defer s.stopAndRemoveWatch(w)
		w.RunWatchLoop()
	}()
	return nil
}

func (s *bulkConnectionImpl) normalizeGVK(gvk *bulkapi.GroupVersionResource) {
	if gvk.Version == "" {
		gvk.Version = s.apiManager.preferredVersion[gvk.Group]
	}
}

func (s *bulkConnectionImpl) findGroupInfo(rs bulkapi.GroupVersionResource) (*registeredAPIGroup, error) {
	version := rs.Version
	gv := schema.GroupVersion{Group: rs.Group, Version: version}
	groupInfo, ok := s.apiManager.apiGroups[gv]
	if !ok {
		return nil, fmt.Errorf("unsupported group '%s/%s'", rs.Group, version)
	}
	return groupInfo, nil
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

func (s *bulkConnectionImpl) runProcessRequestsLoop() {
	for {
		select {
		case req, ok := <-s.requests:
			if !ok {
				s.tasks.Wait()
				s.contextCancel()
				return
			}
			s.handleRequest(req)
		case <-s.context.Done():
			s.tasks.Wait()
			return
		}
	}

}

func (s *bulkConnectionImpl) runWriteResponsesLoop() {
	for {
		select {
		case resp, ok := <-s.responses:
			if !ok {
				return
			}
			if err := s.rwriter.WriteResponse(resp); err != nil {
				utilruntime.HandleError(err)
				return
			}
		case <-s.context.Done():
			return
		}
	}
}

func (s *bulkConnectionImpl) SendResponse(req *bulkapi.ClientMessage, resp *bulkapi.ServerMessage) bool {
	if req != nil && req.RequestID != "" {
		resp.RequestID = &req.RequestID
	}
	select {
	case s.responses <- resp:
		return true
	case <-s.context.Done():
		return false
	}
}

func errorResponse(err error) *bulkapi.ServerMessage {
	return &bulkapi.ServerMessage{Failure: responsewriters.ErrorToAPIStatus(err)}
}
