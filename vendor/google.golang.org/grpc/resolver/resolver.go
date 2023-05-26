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

import (
	"context"
	"net"
	"net/url"
	"strings"

	"google.golang.org/grpc/attributes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/internal/pretty"
	"google.golang.org/grpc/serviceconfig"
)

var (
	// m is a map from scheme to resolver builder.
	m = make(map[string]Builder)
	// defaultScheme is the default scheme to use.
	defaultScheme = "passthrough"
)

// TODO(bar) install dns resolver in init(){}.

// Register registers the resolver builder to the resolver map. b.Scheme will
// be used as the scheme registered with this builder. The registry is case
// sensitive, and schemes should not contain any uppercase characters.
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

// SetDefaultScheme sets the default scheme that will be used. The default
// default scheme is "passthrough".
//
// NOTE: this function must only be called during initialization time (i.e. in
// an init() function), and is not thread-safe. The scheme set last overrides
// previously set values.
func SetDefaultScheme(scheme string) {
	defaultScheme = scheme
}

// GetDefaultScheme gets the default scheme that will be used.
func GetDefaultScheme() string {
	return defaultScheme
}

// AddressType indicates the address type returned by name resolution.
//
// Deprecated: use Attributes in Address instead.
type AddressType uint8

const (
	// Backend indicates the address is for a backend server.
	//
	// Deprecated: use Attributes in Address instead.
	Backend AddressType = iota
	// GRPCLB indicates the address is for a grpclb load balancer.
	//
	// Deprecated: to select the GRPCLB load balancing policy, use a service
	// config with a corresponding loadBalancingConfig.  To supply balancer
	// addresses to the GRPCLB load balancing policy, set State.Attributes
	// using balancer/grpclb/state.Set.
	GRPCLB
)

// Address represents a server the client connects to.
//
// # Experimental
//
// Notice: This type is EXPERIMENTAL and may be changed or removed in a
// later release.
type Address struct {
	// Addr is the server address on which a connection will be established.
	Addr string

	// ServerName is the name of this address.
	// If non-empty, the ServerName is used as the transport certification authority for
	// the address, instead of the hostname from the Dial target string. In most cases,
	// this should not be set.
	//
	// If Type is GRPCLB, ServerName should be the name of the remote load
	// balancer, not the name of the backend.
	//
	// WARNING: ServerName must only be populated with trusted values. It
	// is insecure to populate it with data from untrusted inputs since untrusted
	// values could be used to bypass the authority checks performed by TLS.
	ServerName string

	// Attributes contains arbitrary data about this address intended for
	// consumption by the SubConn.
	Attributes *attributes.Attributes

	// BalancerAttributes contains arbitrary data about this address intended
	// for consumption by the LB policy.  These attribes do not affect SubConn
	// creation, connection establishment, handshaking, etc.
	BalancerAttributes *attributes.Attributes

	// Type is the type of this address.
	//
	// Deprecated: use Attributes instead.
	Type AddressType

	// Metadata is the information associated with Addr, which may be used
	// to make load balancing decision.
	//
	// Deprecated: use Attributes instead.
	Metadata interface{}
}

// Equal returns whether a and o are identical.  Metadata is compared directly,
// not with any recursive introspection.
func (a Address) Equal(o Address) bool {
	return a.Addr == o.Addr && a.ServerName == o.ServerName &&
		a.Attributes.Equal(o.Attributes) &&
		a.BalancerAttributes.Equal(o.BalancerAttributes) &&
		a.Type == o.Type && a.Metadata == o.Metadata
}

// String returns JSON formatted string representation of the address.
func (a Address) String() string {
	return pretty.ToJSON(a)
}

// BuildOptions includes additional information for the builder to create
// the resolver.
type BuildOptions struct {
	// DisableServiceConfig indicates whether a resolver implementation should
	// fetch service config data.
	DisableServiceConfig bool
	// DialCreds is the transport credentials used by the ClientConn for
	// communicating with the target gRPC service (set via
	// WithTransportCredentials). In cases where a name resolution service
	// requires the same credentials, the resolver may use this field. In most
	// cases though, it is not appropriate, and this field may be ignored.
	DialCreds credentials.TransportCredentials
	// CredsBundle is the credentials bundle used by the ClientConn for
	// communicating with the target gRPC service (set via
	// WithCredentialsBundle). In cases where a name resolution service
	// requires the same credentials, the resolver may use this field. In most
	// cases though, it is not appropriate, and this field may be ignored.
	CredsBundle credentials.Bundle
	// Dialer is the custom dialer used by the ClientConn for dialling the
	// target gRPC service (set via WithDialer). In cases where a name
	// resolution service requires the same dialer, the resolver may use this
	// field. In most cases though, it is not appropriate, and this field may
	// be ignored.
	Dialer func(context.Context, string) (net.Conn, error)
}

// State contains the current Resolver state relevant to the ClientConn.
type State struct {
	// Addresses is the latest set of resolved addresses for the target.
	Addresses []Address

	// ServiceConfig contains the result from parsing the latest service
	// config.  If it is nil, it indicates no service config is present or the
	// resolver does not provide service configs.
	ServiceConfig *serviceconfig.ParseResult

	// Attributes contains arbitrary data about the resolver intended for
	// consumption by the load balancing policy.
	Attributes *attributes.Attributes
}

// ClientConn contains the callbacks for resolver to notify any updates
// to the gRPC ClientConn.
//
// This interface is to be implemented by gRPC. Users should not need a
// brand new implementation of this interface. For the situations like
// testing, the new implementation should embed this interface. This allows
// gRPC to add new methods to this interface.
type ClientConn interface {
	// UpdateState updates the state of the ClientConn appropriately.
	//
	// If an error is returned, the resolver should try to resolve the
	// target again. The resolver should use a backoff timer to prevent
	// overloading the server with requests. If a resolver is certain that
	// reresolving will not change the result, e.g. because it is
	// a watch-based resolver, returned errors can be ignored.
	//
	// If the resolved State is the same as the last reported one, calling
	// UpdateState can be omitted.
	UpdateState(State) error
	// ReportError notifies the ClientConn that the Resolver encountered an
	// error.  The ClientConn will notify the load balancer and begin calling
	// ResolveNow on the Resolver with exponential backoff.
	ReportError(error)
	// NewAddress is called by resolver to notify ClientConn a new list
	// of resolved addresses.
	// The address list should be the complete list of resolved addresses.
	//
	// Deprecated: Use UpdateState instead.
	NewAddress(addresses []Address)
	// NewServiceConfig is called by resolver to notify ClientConn a new
	// service config. The service config should be provided as a json string.
	//
	// Deprecated: Use UpdateState instead.
	NewServiceConfig(serviceConfig string)
	// ParseServiceConfig parses the provided service config and returns an
	// object that provides the parsed config.
	ParseServiceConfig(serviceConfigJSON string) *serviceconfig.ParseResult
}

// Target represents a target for gRPC, as specified in:
// https://github.com/grpc/grpc/blob/master/doc/naming.md.
// It is parsed from the target string that gets passed into Dial or DialContext
// by the user. And gRPC passes it to the resolver and the balancer.
//
// If the target follows the naming spec, and the parsed scheme is registered
// with gRPC, we will parse the target string according to the spec. If the
// target does not contain a scheme or if the parsed scheme is not registered
// (i.e. no corresponding resolver available to resolve the endpoint), we will
// apply the default scheme, and will attempt to reparse it.
//
// Examples:
//
//   - "dns://some_authority/foo.bar"
//     Target{Scheme: "dns", Authority: "some_authority", Endpoint: "foo.bar"}
//   - "foo.bar"
//     Target{Scheme: resolver.GetDefaultScheme(), Endpoint: "foo.bar"}
//   - "unknown_scheme://authority/endpoint"
//     Target{Scheme: resolver.GetDefaultScheme(), Endpoint: "unknown_scheme://authority/endpoint"}
type Target struct {
	// Deprecated: use URL.Scheme instead.
	Scheme string
	// Deprecated: use URL.Host instead.
	Authority string
	// URL contains the parsed dial target with an optional default scheme added
	// to it if the original dial target contained no scheme or contained an
	// unregistered scheme. Any query params specified in the original dial
	// target can be accessed from here.
	URL url.URL
}

// Endpoint retrieves endpoint without leading "/" from either `URL.Path`
// or `URL.Opaque`. The latter is used when the former is empty.
func (t Target) Endpoint() string {
	endpoint := t.URL.Path
	if endpoint == "" {
		endpoint = t.URL.Opaque
	}
	// For targets of the form "[scheme]://[authority]/endpoint, the endpoint
	// value returned from url.Parse() contains a leading "/". Although this is
	// in accordance with RFC 3986, we do not want to break existing resolver
	// implementations which expect the endpoint without the leading "/". So, we
	// end up stripping the leading "/" here. But this will result in an
	// incorrect parsing for something like "unix:///path/to/socket". Since we
	// own the "unix" resolver, we can workaround in the unix resolver by using
	// the `URL` field.
	return strings.TrimPrefix(endpoint, "/")
}

// Builder creates a resolver that will be used to watch name resolution updates.
type Builder interface {
	// Build creates a new resolver for the given target.
	//
	// gRPC dial calls Build synchronously, and fails if the returned error is
	// not nil.
	Build(target Target, cc ClientConn, opts BuildOptions) (Resolver, error)
	// Scheme returns the scheme supported by this resolver.  Scheme is defined
	// at https://github.com/grpc/grpc/blob/master/doc/naming.md.  The returned
	// string should not contain uppercase characters, as they will not match
	// the parsed target's scheme as defined in RFC 3986.
	Scheme() string
}

// ResolveNowOptions includes additional information for ResolveNow.
type ResolveNowOptions struct{}

// Resolver watches for the updates on the specified target.
// Updates include address updates and service config updates.
type Resolver interface {
	// ResolveNow will be called by gRPC to try to resolve the target name
	// again. It's just a hint, resolver can ignore this if it's not necessary.
	//
	// It could be called multiple times concurrently.
	ResolveNow(ResolveNowOptions)
	// Close closes the resolver.
	Close()
}

// UnregisterForTesting removes the resolver builder with the given scheme from the
// resolver map.
// This function is for testing only.
func UnregisterForTesting(scheme string) {
	delete(m, scheme)
}
