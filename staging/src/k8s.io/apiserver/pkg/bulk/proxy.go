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
	"net/url"
	"sync"
	"time"

	"crypto/tls"
	"github.com/golang/glog"
	"golang.org/x/net/websocket"
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
	"net/http"
)

type proxyConnection interface {
	ForwardRequest(req *bulkapi.BulkRequest) error
}

type proxyConnectionsPool interface {
	SpawnProxyConnection(pi *ProxiedAPIGroupInfo) (c proxyConnection, err error)
	FindConnectionByWatch(wid string) (c proxyConnection, ok bool)
	Close() error
}

type proxyKey schema.GroupVersion
type watchKey string

// Holds state for single watch.
type proxyConnectionsPoolImpl struct {

	// Link to bulk.APIManager
	*APIManager

	// Root connection
	bulkConnection

	// Serializer (derived from ws connection)
	serializerInfo runtime.SerializerInfo
	encoder        runtime.Encoder
	decoder        runtime.Decoder

	// Protect internal state
	sync.Mutex

	// All active proxies
	proxies map[proxyKey]*proxyConnectionImpl

	// Mapping watchID -> proxy id (group-version)
	watches map[watchKey]proxyKey
}

func (m *APIManager) newProxyConnectionsPool(bc *bulkConnectionImpl) *proxyConnectionsPoolImpl {
	return &proxyConnectionsPoolImpl{
		APIManager:     m,
		bulkConnection: bc,
		serializerInfo: bc.serializerInfo,
		encoder:        bc.encoder,
		decoder:        bc.decoder,
		proxies:        make(map[proxyKey]*proxyConnectionImpl),
		watches:        make(map[watchKey]proxyKey),
	}
}

func (p *proxyConnectionsPoolImpl) Close() error {
	p.Lock()
	defer p.Unlock()

	var errors []error
	for _, proxy := range p.proxies {
		err := proxy.Close()
		if err != nil {
			errors = append(errors, err)
		}
	}
	if len(errors) == 1 {
		return errors[0]
	}
	if errors != nil {
		return fmt.Errorf("errors %v", errors)
	}
	return nil
}

func (p *proxyConnectionsPoolImpl) RegisterWatch(wid watchKey, pk proxyKey) {
	// p.Lock()
	// defer p.Unlock()
	p.watches[wid] = pk
}

func (p *proxyConnectionsPoolImpl) UnregisterWatch(wid watchKey) {
	// p.Lock()
	// defer p.Unlock()
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

	user, ok := genericapirequest.UserFrom(p.bulkConnection.Context())
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
	location.Path = "/bulk/watch" // TODO: add api version.
	location.Host = rloc.Host

	origin := p.bulkConnection.OriginalRequest().URL
	wsconf, err := websocket.NewConfig(location.String(), origin.String())
	if err != nil {
		return
	}

	fakeReq := &http.Request{Header: utilnet.CloneHeader(p.bulkConnection.OriginalRequest().Header), URL: location}
	transport.SetAuthProxyHeaders(fakeReq, user.GetName(), user.GetGroups(), user.GetExtra())

	wsconf.TlsConfig = pi.tlsConfig
	wsconf.Header = fakeReq.Header

	// TODO: integrate with existing http machinery (httptrace, tls cache from client-go etc)
	addr := location.Host
	glog.V(8).Infof("tls-dial to %s", addr)

	// FIXME: use dialer from `pi`
	conn, err := tls.Dial("tcp", addr, wsconf.TlsConfig)
	if err != nil {
		return
	}

	ws, err := websocket.NewClient(wsconf, conn)
	if err != nil {
		return
	}

	proxy := &proxyConnectionImpl{
		bulkConnection: p.bulkConnection,
		proxyKey:       pk,
		pool:           p,
		serializerInfo: p.serializerInfo,
		encoder:        p.encoder,
		decoder:        p.decoder,
		ws:             ws,
		wsTimeout:      p.wsTimeout,
		activeWatches:  make(map[watchKey]bool),
	}
	p.proxies[pk] = proxy

	go func() {
		defer ws.Close()
		proxy.ForwardResponsesLoop()
	}()
	return proxy, nil
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

// Single proxy connection
type proxyConnectionImpl struct {

	// Ref to master bulk-api connection
	bulkConnection
	proxyKey

	pool *proxyConnectionsPoolImpl

	serializerInfo runtime.SerializerInfo
	encoder        runtime.Encoder
	decoder        runtime.Decoder

	ws        *websocket.Conn
	wsTimeout time.Duration

	activeWatches map[watchKey]bool
}

func (p *proxyConnectionImpl) Close() error {
	return p.ws.Close()
}

func (p *proxyConnectionImpl) ForwardRequest(req *bulkapi.BulkRequest) (err error) {
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
	return websocket.Message.Send(p.ws, data)
}

func (p *proxyConnectionImpl) cleanupProxy(err error) {
	if len(p.activeWatches) == 0 {
		return
	}
	for wid := range p.activeWatches {
		p.pool.UnregisterWatch(wid)
	}
	// Notify client -- send 'watch failed' events for all active watches
	var obj *metav1.Status
	if err != nil {
		obj = responsewriters.ErrorToAPIStatus(err)
	} else {
		obj = &metav1.Status{}
	}
	event := watch.Event{Type: watch.Error, Object: obj}
	for wid := range p.activeWatches {
		resp := &bulkapi.BulkResponse{WatchEvent: &bulkapi.BulkWatchEvent{WatchID: string(wid), Event: event}}
		p.bulkConnection.SendResponse(nil, resp)
	}
}

func (p *proxyConnectionImpl) handleResponse(resp *bulkapi.BulkResponse) {
	if resp.WatchStarted != nil {
		wid := resp.WatchStarted.WatchID
		p.activeWatches[watchKey(wid)] = true
		p.pool.RegisterWatch(watchKey(resp.WatchStarted.WatchID), p.proxyKey)
	}
	if resp.WatchStopped != nil {
		wid := resp.WatchStopped.WatchID
		p.pool.UnregisterWatch(watchKey(resp.WatchStopped.WatchID))
		delete(p.activeWatches, watchKey(wid))
	}
}

func (p *proxyConnectionImpl) ForwardResponsesLoop() (err error) {
	defer func() { p.cleanupProxy(err) }()

	defaultGVK := p.bulkConnection.GroupVersion().WithKind("BulkResponse")
	var data []byte

	for {
		if err = websocket.Message.Receive(p.ws, &data); err != nil {
			return
		}
		if len(data) == 0 {
			continue
		}

		obj, _, err1 := p.decoder.Decode(data, &defaultGVK, &bulkapi.BulkResponse{})
		if err1 != nil {
			err = fmt.Errorf("unable to decode bulk response: %v", err)
			return
		}
		resp, ok := obj.(*bulkapi.BulkResponse)
		if !ok {
			err = fmt.Errorf("unable to decode bulk response: cast error")
			return
		}

		p.handleResponse(resp)
		p.bulkConnection.SendResponse(nil, resp)
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
