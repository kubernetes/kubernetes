/*
 *
 * Copyright 2024 gRPC authors.
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

// Package delegatingresolver implements a resolver capable of resolving both
// target URIs and proxy addresses.
package delegatingresolver

import (
	"fmt"
	"net/http"
	"net/url"
	"sync"

	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/internal/proxyattributes"
	"google.golang.org/grpc/internal/transport"
	"google.golang.org/grpc/internal/transport/networktype"
	"google.golang.org/grpc/resolver"
	"google.golang.org/grpc/serviceconfig"
)

var (
	logger = grpclog.Component("delegating-resolver")
	// HTTPSProxyFromEnvironment will be overwritten in the tests
	HTTPSProxyFromEnvironment = http.ProxyFromEnvironment
)

// delegatingResolver manages both target URI and proxy address resolution by
// delegating these tasks to separate child resolvers. Essentially, it acts as
// an intermediary between the gRPC ClientConn and the child resolvers.
//
// It implements the [resolver.Resolver] interface.
type delegatingResolver struct {
	target   resolver.Target     // parsed target URI to be resolved
	cc       resolver.ClientConn // gRPC ClientConn
	proxyURL *url.URL            // proxy URL, derived from proxy environment and target

	// We do not hold both mu and childMu in the same goroutine. Avoid holding
	// both locks when calling into the child, as the child resolver may
	// synchronously callback into the channel.
	mu                  sync.Mutex         // protects all the fields below
	targetResolverState *resolver.State    // state of the target resolver
	proxyAddrs          []resolver.Address // resolved proxy addresses; empty if no proxy is configured

	// childMu serializes calls into child resolvers. It also protects access to
	// the following fields.
	childMu        sync.Mutex
	targetResolver resolver.Resolver // resolver for the target URI, based on its scheme
	proxyResolver  resolver.Resolver // resolver for the proxy URI; nil if no proxy is configured
}

// nopResolver is a resolver that does nothing.
type nopResolver struct{}

func (nopResolver) ResolveNow(resolver.ResolveNowOptions) {}

func (nopResolver) Close() {}

// proxyURLForTarget determines the proxy URL for the given address based on the
// environment. It can return the following:
//   - nil URL, nil error: No proxy is configured or the address is excluded
//     using the `NO_PROXY` environment variable or if req.URL.Host is
//     "localhost" (with or without // a port number)
//   - nil URL, non-nil error: An error occurred while retrieving the proxy URL.
//   - non-nil URL, nil error: A proxy is configured, and the proxy URL was
//     retrieved successfully without any errors.
func proxyURLForTarget(address string) (*url.URL, error) {
	req := &http.Request{URL: &url.URL{
		Scheme: "https",
		Host:   address,
	}}
	return HTTPSProxyFromEnvironment(req)
}

// New creates a new delegating resolver that can create up to two child
// resolvers:
//   - one to resolve the proxy address specified using the supported
//     environment variables. This uses the registered resolver for the "dns"
//     scheme. It is lazily built when a target resolver update contains at least
//     one TCP address.
//   - one to resolve the target URI using the resolver specified by the scheme
//     in the target URI or specified by the user using the WithResolvers dial
//     option. As a special case, if the target URI's scheme is "dns" and a
//     proxy is specified using the supported environment variables, the target
//     URI's path portion is used as the resolved address unless target
//     resolution is enabled using the dial option.
func New(target resolver.Target, cc resolver.ClientConn, opts resolver.BuildOptions, targetResolverBuilder resolver.Builder, targetResolutionEnabled bool) (resolver.Resolver, error) {
	r := &delegatingResolver{
		target:         target,
		cc:             cc,
		proxyResolver:  nopResolver{},
		targetResolver: nopResolver{},
	}

	var err error
	r.proxyURL, err = proxyURLForTarget(target.Endpoint())
	if err != nil {
		return nil, fmt.Errorf("delegating_resolver: failed to determine proxy URL for target %s: %v", target, err)
	}

	// proxy is not configured or proxy address excluded using `NO_PROXY` env
	// var, so only target resolver is used.
	if r.proxyURL == nil {
		return targetResolverBuilder.Build(target, cc, opts)
	}

	if logger.V(2) {
		logger.Infof("Proxy URL detected : %s", r.proxyURL)
	}

	// Resolver updates from one child may trigger calls into the other. Block
	// updates until the children are initialized.
	r.childMu.Lock()
	defer r.childMu.Unlock()
	// When the scheme is 'dns' and target resolution on client is not enabled,
	// resolution should be handled by the proxy, not the client. Therefore, we
	// bypass the target resolver and store the unresolved target address.
	if target.URL.Scheme == "dns" && !targetResolutionEnabled {
		r.targetResolverState = &resolver.State{
			Addresses: []resolver.Address{{Addr: target.Endpoint()}},
			Endpoints: []resolver.Endpoint{{Addresses: []resolver.Address{{Addr: target.Endpoint()}}}},
		}
		r.updateTargetResolverState(*r.targetResolverState)
		return r, nil
	}
	wcc := &wrappingClientConn{
		stateListener: r.updateTargetResolverState,
		parent:        r,
	}
	if r.targetResolver, err = targetResolverBuilder.Build(target, wcc, opts); err != nil {
		return nil, fmt.Errorf("delegating_resolver: unable to build the resolver for target %s: %v", target, err)
	}
	return r, nil
}

// proxyURIResolver creates a resolver for resolving proxy URIs using the "dns"
// scheme. It adjusts the proxyURL to conform to the "dns:///" format and builds
// a resolver with a wrappingClientConn to capture resolved addresses.
func (r *delegatingResolver) proxyURIResolver(opts resolver.BuildOptions) (resolver.Resolver, error) {
	proxyBuilder := resolver.Get("dns")
	if proxyBuilder == nil {
		panic("delegating_resolver: resolver for proxy not found for scheme dns")
	}
	url := *r.proxyURL
	url.Scheme = "dns"
	url.Path = "/" + r.proxyURL.Host
	url.Host = "" // Clear the Host field to conform to the "dns:///" format

	proxyTarget := resolver.Target{URL: url}
	wcc := &wrappingClientConn{
		stateListener: r.updateProxyResolverState,
		parent:        r,
	}
	return proxyBuilder.Build(proxyTarget, wcc, opts)
}

func (r *delegatingResolver) ResolveNow(o resolver.ResolveNowOptions) {
	r.childMu.Lock()
	defer r.childMu.Unlock()
	r.targetResolver.ResolveNow(o)
	r.proxyResolver.ResolveNow(o)
}

func (r *delegatingResolver) Close() {
	r.childMu.Lock()
	defer r.childMu.Unlock()
	r.targetResolver.Close()
	r.targetResolver = nil

	r.proxyResolver.Close()
	r.proxyResolver = nil
}

func networkTypeFromAddr(addr resolver.Address) string {
	networkType, ok := networktype.Get(addr)
	if !ok {
		networkType, _ = transport.ParseDialTarget(addr.Addr)
	}
	return networkType
}

func isTCPAddressPresent(state *resolver.State) bool {
	for _, addr := range state.Addresses {
		if networkType := networkTypeFromAddr(addr); networkType == "tcp" {
			return true
		}
	}
	for _, endpoint := range state.Endpoints {
		for _, addr := range endpoint.Addresses {
			if networktype := networkTypeFromAddr(addr); networktype == "tcp" {
				return true
			}
		}
	}
	return false
}

// updateClientConnStateLocked constructs a combined list of addresses by
// pairing each proxy address with every target address of type TCP. For each
// pair, it creates a new [resolver.Address] using the proxy address and
// attaches the corresponding target address and user info as attributes. Target
// addresses that are not of type TCP are appended to the list as-is. The
// function returns nil if either resolver has not yet provided an update, and
// returns the result of ClientConn.UpdateState once both resolvers have
// provided at least one update.
func (r *delegatingResolver) updateClientConnStateLocked() error {
	if r.targetResolverState == nil || r.proxyAddrs == nil {
		return nil
	}

	// If multiple resolved proxy addresses are present, we send only the
	// unresolved proxy host and let net.Dial handle the proxy host name
	// resolution when creating the transport. Sending all resolved addresses
	// would increase the number of addresses passed to the ClientConn and
	// subsequently to load balancing (LB) policies like Round Robin, leading
	// to additional TCP connections. However, if there's only one resolved
	// proxy address, we send it directly, as it doesn't affect the address
	// count returned by the target resolver and the address count sent to the
	// ClientConn.
	var proxyAddr resolver.Address
	if len(r.proxyAddrs) == 1 {
		proxyAddr = r.proxyAddrs[0]
	} else {
		proxyAddr = resolver.Address{Addr: r.proxyURL.Host}
	}
	var addresses []resolver.Address
	for _, targetAddr := range (*r.targetResolverState).Addresses {
		// Avoid proxy when network is not tcp.
		if networkType := networkTypeFromAddr(targetAddr); networkType != "tcp" {
			addresses = append(addresses, targetAddr)
			continue
		}
		addresses = append(addresses, proxyattributes.Set(proxyAddr, proxyattributes.Options{
			User:        r.proxyURL.User,
			ConnectAddr: targetAddr.Addr,
		}))
	}

	// For each target endpoint, construct a new [resolver.Endpoint] that
	// includes all addresses from all proxy endpoints and the addresses from
	// that target endpoint, preserving the number of target endpoints.
	var endpoints []resolver.Endpoint
	for _, endpt := range (*r.targetResolverState).Endpoints {
		var addrs []resolver.Address
		for _, targetAddr := range endpt.Addresses {
			// Avoid proxy when network is not tcp.
			if networkType := networkTypeFromAddr(targetAddr); networkType != "tcp" {
				addrs = append(addrs, targetAddr)
				continue
			}
			for _, proxyAddr := range r.proxyAddrs {
				addrs = append(addrs, proxyattributes.Set(proxyAddr, proxyattributes.Options{
					User:        r.proxyURL.User,
					ConnectAddr: targetAddr.Addr,
				}))
			}
		}
		endpoints = append(endpoints, resolver.Endpoint{Addresses: addrs})
	}
	// Use the targetResolverState for its service config and attributes
	// contents. The state update is only sent after both the target and proxy
	// resolvers have sent their updates, and curState has been updated with the
	// combined addresses.
	curState := *r.targetResolverState
	curState.Addresses = addresses
	curState.Endpoints = endpoints
	return r.cc.UpdateState(curState)
}

// updateProxyResolverState updates the proxy resolver state by storing proxy
// addresses and endpoints, marking the resolver as ready, and triggering a
// state update if both proxy and target resolvers are ready. If the ClientConn
// returns a non-nil error, it calls `ResolveNow()` on the target resolver.  It
// is a StateListener function of wrappingClientConn passed to the proxy
// resolver.
func (r *delegatingResolver) updateProxyResolverState(state resolver.State) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if logger.V(2) {
		logger.Infof("Addresses received from proxy resolver: %s", state.Addresses)
	}
	if len(state.Endpoints) > 0 {
		// We expect exactly one address per endpoint because the proxy resolver
		// uses "dns" resolution.
		r.proxyAddrs = make([]resolver.Address, 0, len(state.Endpoints))
		for _, endpoint := range state.Endpoints {
			r.proxyAddrs = append(r.proxyAddrs, endpoint.Addresses...)
		}
	} else if state.Addresses != nil {
		r.proxyAddrs = state.Addresses
	} else {
		r.proxyAddrs = []resolver.Address{} // ensure proxyAddrs is non-nil to indicate an update has been received
	}
	err := r.updateClientConnStateLocked()
	// Another possible approach was to block until updates are received from
	// both resolvers. But this is not used because calling `New()` triggers
	// `Build()` for the first resolver, which calls `UpdateState()`. And the
	// second resolver hasn't sent an update yet, so it would cause `New()` to
	// block indefinitely.
	if err != nil {
		go func() {
			r.childMu.Lock()
			defer r.childMu.Unlock()
			if r.targetResolver != nil {
				r.targetResolver.ResolveNow(resolver.ResolveNowOptions{})
			}
		}()
	}
	return err
}

// updateTargetResolverState is the StateListener function provided to the
// target resolver via wrappingClientConn. It updates the resolver state and
// marks the target resolver as ready. If the update includes at least one TCP
// address and the proxy resolver has not yet been constructed, it initializes
// the proxy resolver. A combined state update is triggered once both resolvers
// are ready. If all addresses are non-TCP, it proceeds without waiting for the
// proxy resolver. If ClientConn.UpdateState returns a non-nil error,
// ResolveNow() is called on the proxy resolver.
func (r *delegatingResolver) updateTargetResolverState(state resolver.State) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if logger.V(2) {
		logger.Infof("Addresses received from target resolver: %v", state.Addresses)
	}
	r.targetResolverState = &state
	// If no addresses returned by resolver have network type as tcp , do not
	// wait for proxy update.
	if !isTCPAddressPresent(r.targetResolverState) {
		return r.cc.UpdateState(*r.targetResolverState)
	}

	// The proxy resolver may be rebuilt multiple times, specifically each time
	// the target resolver sends an update, even if the target resolver is built
	// successfully but building the proxy resolver fails.
	if len(r.proxyAddrs) == 0 {
		go func() {
			r.childMu.Lock()
			defer r.childMu.Unlock()
			if _, ok := r.proxyResolver.(nopResolver); !ok {
				return
			}
			proxyResolver, err := r.proxyURIResolver(resolver.BuildOptions{})
			if err != nil {
				r.cc.ReportError(fmt.Errorf("delegating_resolver: unable to build the proxy resolver: %v", err))
				return
			}
			r.proxyResolver = proxyResolver
		}()
	}

	err := r.updateClientConnStateLocked()
	if err != nil {
		go func() {
			r.childMu.Lock()
			defer r.childMu.Unlock()
			if r.proxyResolver != nil {
				r.proxyResolver.ResolveNow(resolver.ResolveNowOptions{})
			}
		}()
	}
	return nil
}

// wrappingClientConn serves as an intermediary between the parent ClientConn
// and the child resolvers created here. It implements the resolver.ClientConn
// interface and is passed in that capacity to the child resolvers.
type wrappingClientConn struct {
	// Callback to deliver resolver state updates
	stateListener func(state resolver.State) error
	parent        *delegatingResolver
}

// UpdateState receives resolver state updates and forwards them to the
// appropriate listener function (either for the proxy or target resolver).
func (wcc *wrappingClientConn) UpdateState(state resolver.State) error {
	return wcc.stateListener(state)
}

// ReportError intercepts errors from the child resolvers and passes them to
// ClientConn.
func (wcc *wrappingClientConn) ReportError(err error) {
	wcc.parent.cc.ReportError(err)
}

// NewAddress intercepts the new resolved address from the child resolvers and
// passes them to ClientConn.
func (wcc *wrappingClientConn) NewAddress(addrs []resolver.Address) {
	wcc.UpdateState(resolver.State{Addresses: addrs})
}

// ParseServiceConfig parses the provided service config and returns an object
// that provides the parsed config.
func (wcc *wrappingClientConn) ParseServiceConfig(serviceConfigJSON string) *serviceconfig.ParseResult {
	return wcc.parent.cc.ParseServiceConfig(serviceConfigJSON)
}
