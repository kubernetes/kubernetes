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

package proxy

import (
	"net"
	"strconv"

	"k8s.io/apimachinery/pkg/util/sets"
)

// Endpoint in an interface which abstracts information about an endpoint.
type Endpoint interface {
	// String returns endpoint string.  An example format can be: `IP:Port`.
	// We take the returned value as ServiceEndpoint.Endpoint.
	String() string
	// IP returns IP part of the endpoint.
	IP() string
	// Port returns the Port part of the endpoint.
	Port() int

	// IsLocal returns true if the endpoint is running on the same host as kube-proxy.
	IsLocal() bool
	// IsReady returns true if an endpoint is ready and not terminating, or
	// if PublishNotReadyAddresses is set on the service.
	IsReady() bool
	// IsServing returns true if an endpoint is ready. It does not account
	// for terminating state.
	IsServing() bool
	// IsTerminating returns true if an endpoint is terminating. For pods,
	// that is any pod with a deletion timestamp.
	IsTerminating() bool

	// ZoneHints returns the zone hint for the endpoint. This is based on
	// endpoint.hints.forZones[*].name in the EndpointSlice API.
	ZoneHints() sets.Set[string]
	// NodeHints returns the node hint for the endpoint. This is based on
	// endpoint.hints.forNodes[*].name in the EndpointSlice API.
	NodeHints() sets.Set[string]
}

// BaseEndpointInfo contains base information that defines an endpoint.
// This could be used directly by proxier while processing endpoints,
// or can be used for constructing a more specific EndpointInfo struct
// defined by the proxier if needed.
type BaseEndpointInfo struct {
	// Cache this values to improve performance
	ip   string
	port int
	// endpoint is the same as net.JoinHostPort(ip,port)
	endpoint string

	// isLocal indicates whether the endpoint is running on same host as kube-proxy.
	isLocal bool

	// ready indicates whether this endpoint is ready and NOT terminating, unless
	// PublishNotReadyAddresses is set on the service, in which case it will just
	// always be true.
	ready bool
	// serving indicates whether this endpoint is ready regardless of its terminating state.
	// For pods this is true if it has a ready status regardless of its deletion timestamp.
	serving bool
	// terminating indicates whether this endpoint is terminating.
	// For pods this is true if it has a non-nil deletion timestamp.
	terminating bool

	// zoneHints represent the zone hints for the endpoint. This is based on
	// endpoint.hints.forZones[*].name in the EndpointSlice API.
	zoneHints sets.Set[string]
	// nodeHints represent the node hints for the endpoint. This is based on
	// endpoint.hints.forNodes[*].name in the EndpointSlice API.
	nodeHints sets.Set[string]
}

var _ Endpoint = &BaseEndpointInfo{}

// String is part of proxy.Endpoint interface.
func (info *BaseEndpointInfo) String() string {
	return info.endpoint
}

// IP returns just the IP part of the endpoint, it's a part of proxy.Endpoint interface.
func (info *BaseEndpointInfo) IP() string {
	return info.ip
}

// Port returns just the Port part of the endpoint.
func (info *BaseEndpointInfo) Port() int {
	return info.port
}

// IsLocal is part of proxy.Endpoint interface.
func (info *BaseEndpointInfo) IsLocal() bool {
	return info.isLocal
}

// IsReady returns true if an endpoint is ready and not terminating.
func (info *BaseEndpointInfo) IsReady() bool {
	return info.ready
}

// IsServing returns true if an endpoint is ready, regardless of if the
// endpoint is terminating.
func (info *BaseEndpointInfo) IsServing() bool {
	return info.serving
}

// IsTerminating retruns true if an endpoint is terminating. For pods,
// that is any pod with a deletion timestamp.
func (info *BaseEndpointInfo) IsTerminating() bool {
	return info.terminating
}

// ZoneHints returns the zone hints for the endpoint.
func (info *BaseEndpointInfo) ZoneHints() sets.Set[string] {
	return info.zoneHints
}

// NodeHints returns the node hints for the endpoint.
func (info *BaseEndpointInfo) NodeHints() sets.Set[string] {
	return info.nodeHints
}

func newBaseEndpointInfo(ip string, port int, isLocal, ready, serving, terminating bool, zoneHints, nodeHints sets.Set[string]) *BaseEndpointInfo {
	return &BaseEndpointInfo{
		ip:          ip,
		port:        port,
		endpoint:    net.JoinHostPort(ip, strconv.Itoa(port)),
		isLocal:     isLocal,
		ready:       ready,
		serving:     serving,
		terminating: terminating,
		zoneHints:   zoneHints,
		nodeHints:   nodeHints,
	}
}
