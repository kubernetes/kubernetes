// Copyright 2018 The etcd Authors
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

// Package endpoint resolves etcd entpoints using grpc targets of the form 'endpoint://<id>/<endpoint>'.
package endpoint

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"strings"
	"sync"

	"google.golang.org/grpc/resolver"
)

const scheme = "endpoint"

var (
	targetPrefix = fmt.Sprintf("%s://", scheme)

	bldr *builder
)

func init() {
	bldr = &builder{
		resolverGroups: make(map[string]*ResolverGroup),
	}
	resolver.Register(bldr)
}

type builder struct {
	mu             sync.RWMutex
	resolverGroups map[string]*ResolverGroup
}

// NewResolverGroup creates a new ResolverGroup with the given id.
func NewResolverGroup(id string) (*ResolverGroup, error) {
	return bldr.newResolverGroup(id)
}

// ResolverGroup keeps all endpoints of resolvers using a common endpoint://<id>/ target
// up-to-date.
type ResolverGroup struct {
	mu        sync.RWMutex
	id        string
	endpoints []string
	resolvers []*Resolver
}

func (e *ResolverGroup) addResolver(r *Resolver) {
	e.mu.Lock()
	addrs := epsToAddrs(e.endpoints...)
	e.resolvers = append(e.resolvers, r)
	e.mu.Unlock()
	r.cc.NewAddress(addrs)
}

func (e *ResolverGroup) removeResolver(r *Resolver) {
	e.mu.Lock()
	for i, er := range e.resolvers {
		if er == r {
			e.resolvers = append(e.resolvers[:i], e.resolvers[i+1:]...)
			break
		}
	}
	e.mu.Unlock()
}

// SetEndpoints updates the endpoints for ResolverGroup. All registered resolver are updated
// immediately with the new endpoints.
func (e *ResolverGroup) SetEndpoints(endpoints []string) {
	addrs := epsToAddrs(endpoints...)
	e.mu.Lock()
	e.endpoints = endpoints
	for _, r := range e.resolvers {
		r.cc.NewAddress(addrs)
	}
	e.mu.Unlock()
}

// Target constructs a endpoint target using the endpoint id of the ResolverGroup.
func (e *ResolverGroup) Target(endpoint string) string {
	return Target(e.id, endpoint)
}

// Target constructs a endpoint resolver target.
func Target(id, endpoint string) string {
	return fmt.Sprintf("%s://%s/%s", scheme, id, endpoint)
}

// IsTarget checks if a given target string in an endpoint resolver target.
func IsTarget(target string) bool {
	return strings.HasPrefix(target, "endpoint://")
}

func (e *ResolverGroup) Close() {
	bldr.close(e.id)
}

// Build creates or reuses an etcd resolver for the etcd cluster name identified by the authority part of the target.
func (b *builder) Build(target resolver.Target, cc resolver.ClientConn, opts resolver.BuildOption) (resolver.Resolver, error) {
	if len(target.Authority) < 1 {
		return nil, fmt.Errorf("'etcd' target scheme requires non-empty authority identifying etcd cluster being routed to")
	}
	id := target.Authority
	es, err := b.getResolverGroup(id)
	if err != nil {
		return nil, fmt.Errorf("failed to build resolver: %v", err)
	}
	r := &Resolver{
		endpointID: id,
		cc:         cc,
	}
	es.addResolver(r)
	return r, nil
}

func (b *builder) newResolverGroup(id string) (*ResolverGroup, error) {
	b.mu.RLock()
	_, ok := b.resolverGroups[id]
	b.mu.RUnlock()
	if ok {
		return nil, fmt.Errorf("Endpoint already exists for id: %s", id)
	}

	es := &ResolverGroup{id: id}
	b.mu.Lock()
	b.resolverGroups[id] = es
	b.mu.Unlock()
	return es, nil
}

func (b *builder) getResolverGroup(id string) (*ResolverGroup, error) {
	b.mu.RLock()
	es, ok := b.resolverGroups[id]
	b.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("ResolverGroup not found for id: %s", id)
	}
	return es, nil
}

func (b *builder) close(id string) {
	b.mu.Lock()
	delete(b.resolverGroups, id)
	b.mu.Unlock()
}

func (b *builder) Scheme() string {
	return scheme
}

// Resolver provides a resolver for a single etcd cluster, identified by name.
type Resolver struct {
	endpointID string
	cc         resolver.ClientConn
	sync.RWMutex
}

// TODO: use balancer.epsToAddrs
func epsToAddrs(eps ...string) (addrs []resolver.Address) {
	addrs = make([]resolver.Address, 0, len(eps))
	for _, ep := range eps {
		addrs = append(addrs, resolver.Address{Addr: ep})
	}
	return addrs
}

func (*Resolver) ResolveNow(o resolver.ResolveNowOption) {}

func (r *Resolver) Close() {
	es, err := bldr.getResolverGroup(r.endpointID)
	if err != nil {
		return
	}
	es.removeResolver(r)
}

// ParseEndpoint endpoint parses an endpoint of the form
// (http|https)://<host>*|(unix|unixs)://<path>)
// and returns a protocol ('tcp' or 'unix'),
// host (or filepath if a unix socket),
// scheme (http, https, unix, unixs).
func ParseEndpoint(endpoint string) (proto string, host string, scheme string) {
	proto = "tcp"
	host = endpoint
	url, uerr := url.Parse(endpoint)
	if uerr != nil || !strings.Contains(endpoint, "://") {
		return proto, host, scheme
	}
	scheme = url.Scheme

	// strip scheme:// prefix since grpc dials by host
	host = url.Host
	switch url.Scheme {
	case "http", "https":
	case "unix", "unixs":
		proto = "unix"
		host = url.Host + url.Path
	default:
		proto, host = "", ""
	}
	return proto, host, scheme
}

// ParseTarget parses a endpoint://<id>/<endpoint> string and returns the parsed id and endpoint.
// If the target is malformed, an error is returned.
func ParseTarget(target string) (string, string, error) {
	noPrefix := strings.TrimPrefix(target, targetPrefix)
	if noPrefix == target {
		return "", "", fmt.Errorf("malformed target, %s prefix is required: %s", targetPrefix, target)
	}
	parts := strings.SplitN(noPrefix, "/", 2)
	if len(parts) != 2 {
		return "", "", fmt.Errorf("malformed target, expected %s://<id>/<endpoint>, but got %s", scheme, target)
	}
	return parts[0], parts[1], nil
}

// Dialer dials a endpoint using net.Dialer.
// Context cancelation and timeout are supported.
func Dialer(ctx context.Context, dialEp string) (net.Conn, error) {
	proto, host, _ := ParseEndpoint(dialEp)
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	default:
	}
	dialer := &net.Dialer{}
	if deadline, ok := ctx.Deadline(); ok {
		dialer.Deadline = deadline
	}
	return dialer.DialContext(ctx, proto, host)
}
