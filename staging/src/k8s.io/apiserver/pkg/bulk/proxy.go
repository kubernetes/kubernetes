/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"
	"net/http"
	"net/url"
	"sync"
	"time"

	"crypto/tls"
	"github.com/golang/glog"
	"golang.org/x/net/context"
	"golang.org/x/net/websocket"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	bulkapi "k8s.io/apiserver/pkg/apis/bulk"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/transport"
)

type proxyConnection interface {
	ForwardRequest(req *bulkapi.ClientMessage) error
}

type proxyConnectionsPool interface {
	SpawnProxyConnection(pi *ProxiedAPIGroupInfo) (c proxyConnection, err error)
	FindConnectionByWatch(wid string) (c proxyConnection, ok bool)
	Close()
}

type watchesRegistry interface {
	RegisterWatch(wid watchKey, pk proxyKey)
	UnregisterWatch(wid watchKey)
}

type proxyKey schema.GroupVersion
type watchKey string

// Holds state for single watch.
type proxyConnectionsPoolImpl struct {

	// Root connection
	bulkConnection

	originalRequest *http.Request

	context context.Context
	done    chan struct{}

	// Serializer (derived from ws connection)
	serializerInfo runtime.SerializerInfo
	encoder        runtime.Encoder
	decoder        runtime.Decoder

	// Protect internal state
	sync.Mutex

	// All active proxies
	proxies map[proxyKey]*proxyConnectionImpl

	// Waiter for active proxies
	proxiesCounter sync.WaitGroup

	// Mapping watchID -> proxy id (group-version)
	watches map[watchKey]proxyKey
}

// Single proxy connection
type proxyConnectionImpl struct {
	bulkConnection
	proxyKey
	watchesRegistry

	serializerInfo runtime.SerializerInfo
	encoder        runtime.Encoder
	decoder        runtime.Decoder

	ws        *websocket.Conn
	wsTimeout time.Duration

	activeWatches  map[watchKey]bool
	activeRequests map[string]int
}

func newProxyConnectionsPool(ctx context.Context, bc bulkConnection, req *http.Request) *proxyConnectionsPoolImpl {
	encoder, decoder := createNonstreamEncoderDecoder(bc)
	pool := &proxyConnectionsPoolImpl{
		bulkConnection:  bc,
		serializerInfo:  bc.SerializerInfo(),
		originalRequest: req,
		context:         ctx,
		encoder:         encoder,
		decoder:         decoder,
		proxies:         make(map[proxyKey]*proxyConnectionImpl),
		watches:         make(map[watchKey]proxyKey),
		done:            make(chan struct{}),
	}
	return pool
}

func (p *proxyConnectionsPoolImpl) Close() {
	close(p.done)
	p.proxiesCounter.Wait() // wait until all proxies are closed & cleaned
}

func (p *proxyConnectionsPoolImpl) RegisterWatch(wid watchKey, pk proxyKey) {
	p.Lock()
	defer p.Unlock()
	p.watches[wid] = pk
}

func (p *proxyConnectionsPoolImpl) UnregisterWatch(wid watchKey) {
	p.Lock()
	defer p.Unlock()
	delete(p.watches, wid)
}

func (p *proxyConnectionsPoolImpl) SpawnProxyConnection(pi *ProxiedAPIGroupInfo) (pc proxyConnection, err error) {
	p.Lock()
	defer p.Unlock()
	glog.V(9).Infof("get or spawn proxy connection for %v", pi.GroupVersion)

	pk := proxyKey(pi.GroupVersion)
	if proxy, ok := p.proxies[pk]; ok {
		glog.V(9).Infof("reuse existing connection for %v", pi.GroupVersion)
		return proxy, nil
	}

	err = pi.transportBuildErr
	if err != nil {
		return
	}

	user, ok := genericapirequest.UserFrom(p.context)
	if !ok {
		err = fmt.Errorf("missing user")
		return
	}
	rloc, err := pi.ServiceResolver.ResolveEndpoint(pi.ServiceNamespace, pi.ServiceName)
	if err != nil {
		err = fmt.Errorf("missing route (%s)", err)
		return
	}

	// write a new location based on the existing request pointed at the target service
	location := &url.URL{}
	location.Scheme = "wss"
	location.Path = "/bulk" // TODO: add api version.
	location.Host = rloc.Host

	origin := p.originalRequest
	wsconf, err := websocket.NewConfig(location.String(), origin.URL.String())
	if err != nil {
		return
	}

	wsconf.TlsConfig = pi.tlsConfig

	fakeReq := &http.Request{Header: utilnet.CloneHeader(origin.Header), URL: location}
	transport.SetAuthProxyHeaders(fakeReq, user.GetName(), user.GetGroups(), user.GetExtra())
	wsconf.Header = fakeReq.Header

	addr := location.Host
	glog.V(8).Infof("tls-dial to %s", addr)
	rawConn, err := pi.dialContext(p.context, "tcp", addr)
	if err != nil {
		return
	}

	// close raw tcp connection if context is cancelled during handshaking
	chanCloser := sync.Once{}
	go func() {
		select {
		case <-p.done:
		case <-p.context.Done():
		}
		chanCloser.Do(func() { rawConn.Close() })
	}()

	conn := tls.Client(rawConn, wsconf.TlsConfig)
	if err = conn.Handshake(); err != nil {
		return
	}
	ws, err := websocket.NewClient(wsconf, conn)
	if err != nil {
		return
	}

	proxy := &proxyConnectionImpl{
		bulkConnection:  p.bulkConnection,
		watchesRegistry: p,
		proxyKey:        pk,
		serializerInfo:  p.serializerInfo,
		encoder:         p.encoder,
		decoder:         p.decoder,
		ws:              ws,
		wsTimeout:       p.APIManager().wsTimeout,
		activeWatches:   make(map[watchKey]bool),
		activeRequests:  make(map[string]int),
	}
	p.proxies[pk] = proxy
	p.proxiesCounter.Add(1)
	go func() {
		defer utilruntime.HandleCrash()
		defer p.proxiesCounter.Done()
		defer chanCloser.Do(func() { ws.Close() })
		proxy.cleanupProxy(proxy.ForwardResponsesLoop())
	}()
	return
}

func (p *proxyConnectionsPoolImpl) FindConnectionByWatch(watchID string) (proxy proxyConnection, ok bool) {
	p.Lock()
	defer p.Unlock()
	pk, ok := p.watches[watchKey(watchID)]
	if !ok {
		return
	}
	proxy, ok = p.proxies[pk]
	return
}

func (p *proxyConnectionImpl) ForwardRequest(req *bulkapi.ClientMessage) (err error) {
	p.resetTimeout()

	streamBuf := &bytes.Buffer{}
	if err := p.encoder.Encode(req, streamBuf); err != nil {
		return fmt.Errorf("unable to encode event: %v", err)
	}

	var data interface{}
	if p.serializerInfo.EncodesAsText {
		data = streamBuf.String()
	} else {
		data = streamBuf.Bytes()
	}
	if req.RequestID != "" {
		p.activeRequests[req.RequestID]++
	}
	return websocket.Message.Send(p.ws, data)
}

func (p *proxyConnectionImpl) cleanupProxy(err error) {
	if err != nil {
		utilruntime.HandleError(err)
	}
	if len(p.activeWatches) == 0 && len(p.activeRequests) == 0 {
		return
	}
	for wid := range p.activeWatches {
		p.watchesRegistry.UnregisterWatch(wid)
	}

	// Notify client -- send 'watch failed' events for all active watches / requests
	var obj *metav1.Status
	if err == nil {
		err = errors.NewInternalError(fmt.Errorf("proxy connection was done"))
	} else {
		err = errors.NewInternalError(fmt.Errorf("proxy connection done due to internal error: %v", err))
	}
	obj = responsewriters.ErrorToAPIStatus(err)

	event := watch.Event{Type: watch.Error, Object: obj}
	for wid := range p.activeWatches {
		resp := &bulkapi.ServerMessage{WatchEvent: &bulkapi.BulkWatchEvent{WatchID: string(wid), WatchEvent: event}}
		p.bulkConnection.SendResponse(nil, resp)
	}
	for rid, cnt := range p.activeRequests {
		requestid := string(rid)
		resp := &bulkapi.ServerMessage{RequestID: &requestid, Failure: obj}
		for i := 0; i < cnt; i++ {
			p.bulkConnection.SendResponse(nil, resp)
		}
	}
}

func (p *proxyConnectionImpl) preprocessResponse(resp *bulkapi.ServerMessage) {
	if resp.RequestID != nil {
		rid := *resp.RequestID
		pv := p.activeRequests[rid]
		if pv <= 1 {
			delete(p.activeRequests, rid)
		} else {
			p.activeRequests[rid] = pv - 1
		}
	}
	if resp.WatchStarted != nil {
		wid := resp.WatchStarted.WatchID
		p.activeWatches[watchKey(wid)] = true
		p.watchesRegistry.RegisterWatch(watchKey(resp.WatchStarted.WatchID), p.proxyKey)
	}
	if resp.WatchStopped != nil {
		wid := resp.WatchStopped.WatchID
		p.watchesRegistry.UnregisterWatch(watchKey(resp.WatchStopped.WatchID))
		delete(p.activeWatches, watchKey(wid))
	}
}

func (p *proxyConnectionImpl) ForwardResponsesLoop() (err error) {
	defaultGVK := p.bulkConnection.APIManager().GroupVersion.WithKind("ServerMessage")
	var data []byte
	for {
		if err = websocket.Message.Receive(p.ws, &data); err != nil {
			return
		}
		if len(data) == 0 {
			continue
		}

		obj, _, err1 := p.decoder.Decode(data, &defaultGVK, &bulkapi.ServerMessage{})
		if err1 != nil {
			err = fmt.Errorf("unable to decode bulk response: %v", err)
			return
		}
		resp, ok := obj.(*bulkapi.ServerMessage)
		if !ok {
			err = fmt.Errorf("unable to decode bulk response: cast error")
			return
		}

		p.preprocessResponse(resp)
		if ok := p.bulkConnection.SendResponse(nil, resp); !ok {
			return
		}
		p.resetTimeout()
	}
}

func (p *proxyConnectionImpl) resetTimeout() {
	if p.wsTimeout > 0 {
		if err := p.ws.SetDeadline(time.Now().Add(p.wsTimeout)); err != nil {
			utilruntime.HandleError(err)
		}
	}
}
