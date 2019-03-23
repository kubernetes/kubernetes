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

// Package resolver defines APIs for name resolution in gRPC.
// All APIs in this package are experimental.
package resolver

var (
	// m is a map from scheme to resolver builder.
	m = make(map[string]Builder)
	// defaultScheme is the default scheme to use.
	defaultScheme = "passthrough"
)

// TODO(bar) install dns resolver in init(){}.

// Register registers the resolver builder to the resolver map. b.Scheme will be
// used as the scheme registered with this builder.
//
// NOTE: this function must only be called during initialization time (i.e. in
// an init() function), and is not thread-safe. If multiple Resolvers are
// registered with the same name, the one registered last will take effect.
func Register(b Builder) {
	m[b.Scheme()] = b
}

// Get returns the resolver builder registered with the given scheme.
//
// If no builder is register with the scheme, nil will be returned.
func Get(scheme string) Builder {
	if b, ok := m[scheme]; ok {
		return b
	}
	return nil
}

// SetDefaultScheme sets the default scheme that will be used.
// The default default scheme is "passthrough".
func SetDefaultScheme(scheme string) {
	defaultScheme = scheme
}

// GetDefaultScheme gets the default scheme that will be used.
func GetDefaultScheme() string {
	return defaultScheme
}

// AddressType indicates the address type returned by name resolution.
type AddressType uint8

const (
	// Backend indicates the address is for a backend server.
	Backend AddressType = iota
	// GRPCLB indicates the address is for a grpclb load balancer.
	GRPCLB
)

// Address represents a server the client connects to.
// This is the EXPERIMENTAL API and may be changed or extended in the future.
type Address struct {
	// Addr is the server address on which a connection will be established.
	Addr string
	// Type is the type of this address.
	Type AddressType
	// ServerName is the name of this address.
	//
	// e.g. if Type is GRPCLB, ServerName should be the name of the remote load
	// balancer, not the name of the backend.
	ServerName string
	// Metadata is the information associated with Addr, which may be used
	// to make load balancing decision.
	Metadata interface{}
}

// BuildOption includes additional information for the builder to create
// the resolver.
type BuildOption struct {
	// DisableServiceConfig indicates whether resolver should fetch service config data.
	DisableServiceConfig bool
}

// ClientConn contains the callbacks for resolver to notify any updates
// to the gRPC ClientConn.
//
// This interface is to be implemented by gRPC. Users should not need a
// brand new implementation of this interface. For the situations like
// testing, the new implementation should embed this interface. This allows
// gRPC to add new methods to this interface.
type ClientConn interface {
	// NewAddress is called by resolver to notify ClientConn a new list
	// of resolved addresses.
	// The address list should be the complete list of resolved addresses.
	NewAddress(addresses []Address)
	// NewServiceConfig is called by resolver to notify ClientConn a new
	// service config. The service config should be provided as a json string.
	NewServiceConfig(serviceConfig string)
}

// Target represents a target for gRPC, as specified in:
// https://github.com/grpc/grpc/blob/master/doc/naming.md.
type Target struct {
	Scheme    string
	Authority string
	Endpoint  string
}

// Builder creates a resolver that will be used to watch name resolution updates.
type Builder interface {
	// Build creates a new resolver for the given target.
	//
	// gRPC dial calls Build synchronously, and fails if the returned error is
	// not nil.
	Build(target Target, cc ClientConn, opts BuildOption) (Resolver, error)
	// Scheme returns the scheme supported by this resolver.
	// Scheme is defined at https://github.com/grpc/grpc/blob/master/doc/naming.md.
	Scheme() string
}

// ResolveNowOption includes additional information for ResolveNow.
type ResolveNowOption struct{}

// Resolver watches for the updates on the specified target.
// Updates include address updates and service config updates.
type Resolver interface {
	// ResolveNow will be called by gRPC to try to resolve the target name
	// again. It's just a hint, resolver can ignore this if it's not necessary.
	//
	// It could be called multiple times concurrently.
	ResolveNow(ResolveNowOption)
	// Close closes the resolver.
	Close()
}

// UnregisterForTesting removes the resolver builder with the given scheme from the
// resolver map.
// This function is for testing only.
func UnregisterForTesting(scheme string) {
	delete(m, scheme)
}
