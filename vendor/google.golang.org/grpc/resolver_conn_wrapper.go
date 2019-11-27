/*
 *
 * Copyright 2017 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package grpc

import (
	"fmt"
	"strings"
	"sync/atomic"

	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/channelz"
	"google.golang.org/grpc/resolver"
)

// ccResolverWrapper is a wrapper on top of cc for resolvers.
// It implements resolver.ClientConnection interface.
type ccResolverWrapper struct {
	cc       *ClientConn
	resolver resolver.Resolver
	addrCh   chan []resolver.Address
	scCh     chan string
	done     uint32 // accessed atomically; set to 1 when closed.
	curState resolver.State
}

// split2 returns the values from strings.SplitN(s, sep, 2).
// If sep is not found, it returns ("", "", false) instead.
func split2(s, sep string) (string, string, bool) {
	spl := strings.SplitN(s, sep, 2)
	if len(spl) < 2 {
		return "", "", false
	}
	return spl[0], spl[1], true
}

// parseTarget splits target into a struct containing scheme, authority and
// endpoint.
//
// If target is not a valid scheme://authority/endpoint, it returns {Endpoint:
// target}.
func parseTarget(target string) (ret resolver.Target) {
	var ok bool
	ret.Scheme, ret.Endpoint, ok = split2(target, "://")
	if !ok {
		return resolver.Target{Endpoint: target}
	}
	ret.Authority, ret.Endpoint, ok = split2(ret.Endpoint, "/")
	if !ok {
		return resolver.Target{Endpoint: target}
	}
	return ret
}

// newCCResolverWrapper parses cc.target for scheme and gets the resolver
// builder for this scheme and builds the resolver. The monitoring goroutine
// for it is not started yet and can be created by calling start().
//
// If withResolverBuilder dial option is set, the specified resolver will be
// used instead.
func newCCResolverWrapper(cc *ClientConn) (*ccResolverWrapper, error) {
	rb := cc.dopts.resolverBuilder
	if rb == nil {
		return nil, fmt.Errorf("could not get resolver for scheme: %q", cc.parsedTarget.Scheme)
	}

	ccr := &ccResolverWrapper{
		cc:     cc,
		addrCh: make(chan []resolver.Address, 1),
		scCh:   make(chan string, 1),
	}

	var err error
	ccr.resolver, err = rb.Build(cc.parsedTarget, ccr, resolver.BuildOption{DisableServiceConfig: cc.dopts.disableServiceConfig})
	if err != nil {
		return nil, err
	}
	return ccr, nil
}

func (ccr *ccResolverWrapper) resolveNow(o resolver.ResolveNowOption) {
	ccr.resolver.ResolveNow(o)
}

func (ccr *ccResolverWrapper) close() {
	ccr.resolver.Close()
	atomic.StoreUint32(&ccr.done, 1)
}

func (ccr *ccResolverWrapper) isDone() bool {
	return atomic.LoadUint32(&ccr.done) == 1
}

func (ccr *ccResolverWrapper) UpdateState(s resolver.State) {
	if ccr.isDone() {
		return
	}
	grpclog.Infof("ccResolverWrapper: sending update to cc: %v", s)
	if channelz.IsOn() {
		ccr.addChannelzTraceEvent(s)
	}
	ccr.cc.updateResolverState(s)
	ccr.curState = s
}

// NewAddress is called by the resolver implementation to send addresses to gRPC.
func (ccr *ccResolverWrapper) NewAddress(addrs []resolver.Address) {
	if ccr.isDone() {
		return
	}
	grpclog.Infof("ccResolverWrapper: sending new addresses to cc: %v", addrs)
	if channelz.IsOn() {
		ccr.addChannelzTraceEvent(resolver.State{Addresses: addrs, ServiceConfig: ccr.curState.ServiceConfig})
	}
	ccr.curState.Addresses = addrs
	ccr.cc.updateResolverState(ccr.curState)
}

// NewServiceConfig is called by the resolver implementation to send service
// configs to gRPC.
func (ccr *ccResolverWrapper) NewServiceConfig(sc string) {
	if ccr.isDone() {
		return
	}
	grpclog.Infof("ccResolverWrapper: got new service config: %v", sc)
	c, err := parseServiceConfig(sc)
	if err != nil {
		return
	}
	if channelz.IsOn() {
		ccr.addChannelzTraceEvent(resolver.State{Addresses: ccr.curState.Addresses, ServiceConfig: c})
	}
	ccr.curState.ServiceConfig = c
	ccr.cc.updateResolverState(ccr.curState)
}

func (ccr *ccResolverWrapper) addChannelzTraceEvent(s resolver.State) {
	var updates []string
	oldSC, oldOK := ccr.curState.ServiceConfig.(*ServiceConfig)
	newSC, newOK := s.ServiceConfig.(*ServiceConfig)
	if oldOK != newOK || (oldOK && newOK && oldSC.rawJSONString != newSC.rawJSONString) {
		updates = append(updates, "service config updated")
	}
	if len(ccr.curState.Addresses) > 0 && len(s.Addresses) == 0 {
		updates = append(updates, "resolver returned an empty address list")
	} else if len(ccr.curState.Addresses) == 0 && len(s.Addresses) > 0 {
		updates = append(updates, "resolver returned new addresses")
	}
	channelz.AddTraceEvent(ccr.cc.channelzID, &channelz.TraceEventDesc{
		Desc:     fmt.Sprintf("Resolver state updated: %+v (%v)", s, strings.Join(updates, "; ")),
		Severity: channelz.CtINFO,
	})
}
