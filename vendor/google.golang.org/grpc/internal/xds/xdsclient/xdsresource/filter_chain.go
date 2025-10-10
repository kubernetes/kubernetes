/*
 *
 * Copyright 2021 gRPC authors.
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
 */

package xdsresource

import (
	"errors"
	"fmt"
	"net"
	"sync/atomic"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/internal/resolver"
	"google.golang.org/grpc/internal/xds/httpfilter"
	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource/version"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	v3listenerpb "github.com/envoyproxy/go-control-plane/envoy/config/listener/v3"
	v3httppb "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/network/http_connection_manager/v3"
	v3tlspb "github.com/envoyproxy/go-control-plane/envoy/extensions/transport_sockets/tls/v3"
)

const (
	// Used as the map key for unspecified prefixes. The actual value of this
	// key is immaterial.
	unspecifiedPrefixMapKey = "unspecified"

	// An unspecified destination or source prefix should be considered a less
	// specific match than a wildcard prefix, `0.0.0.0/0` or `::/0`. Also, an
	// unspecified prefix should match most v4 and v6 addresses compared to the
	// wildcard prefixes which match only a specific network (v4 or v6).
	//
	// We use these constants when looking up the most specific prefix match. A
	// wildcard prefix will match 0 bits, and to make sure that a wildcard
	// prefix is considered a more specific match than an unspecified prefix, we
	// use a value of -1 for the latter.
	noPrefixMatch          = -2
	unspecifiedPrefixMatch = -1
)

// FilterChain captures information from within a FilterChain message in a
// Listener resource.
type FilterChain struct {
	// SecurityCfg contains transport socket security configuration.
	SecurityCfg *SecurityConfig
	// HTTPFilters represent the HTTP Filters that comprise this FilterChain.
	HTTPFilters []HTTPFilter
	// RouteConfigName is the route configuration name for this FilterChain.
	//
	// Exactly one of RouteConfigName and InlineRouteConfig is set.
	RouteConfigName string
	// InlineRouteConfig is the inline route configuration (RDS response)
	// returned for this filter chain.
	//
	// Exactly one of RouteConfigName and InlineRouteConfig is set.
	InlineRouteConfig *RouteConfigUpdate
	// UsableRouteConfiguration is the routing configuration for this filter
	// chain (LDS + RDS).
	UsableRouteConfiguration *atomic.Pointer[UsableRouteConfiguration]
}

// VirtualHostWithInterceptors captures information present in a VirtualHost
// update, and also contains routes with instantiated HTTP Filters.
type VirtualHostWithInterceptors struct {
	// Domains are the domain names which map to this Virtual Host. On the
	// server side, this will be dictated by the :authority header of the
	// incoming RPC.
	Domains []string
	// Routes are the Routes for this Virtual Host.
	Routes []RouteWithInterceptors
}

// RouteWithInterceptors captures information in a Route, and contains
// a usable matcher and also instantiated HTTP Filters.
type RouteWithInterceptors struct {
	// M is the matcher used to match to this route.
	M *CompositeMatcher
	// ActionType is the type of routing action to initiate once matched to.
	ActionType RouteActionType
	// Interceptors are interceptors instantiated for this route. These will be
	// constructed from a combination of the top level configuration and any
	// HTTP Filter overrides present in Virtual Host or Route.
	Interceptors []resolver.ServerInterceptor
}

// UsableRouteConfiguration contains a matchable route configuration, with
// instantiated HTTP Filters per route.
type UsableRouteConfiguration struct {
	VHS    []VirtualHostWithInterceptors
	Err    error
	NodeID string // For logging purposes. Populated by the listener wrapper.
}

// StatusErrWithNodeID returns an error produced by the status package with the
// specified code and message, and includes the xDS node ID.
func (rc *UsableRouteConfiguration) StatusErrWithNodeID(c codes.Code, msg string, args ...any) error {
	return status.Error(c, fmt.Sprintf("[xDS node id: %v]: %s", rc.NodeID, fmt.Sprintf(msg, args...)))
}

// ConstructUsableRouteConfiguration takes Route Configuration and converts it
// into matchable route configuration, with instantiated HTTP Filters per route.
func (fc *FilterChain) ConstructUsableRouteConfiguration(config RouteConfigUpdate) *UsableRouteConfiguration {
	vhs := make([]VirtualHostWithInterceptors, 0, len(config.VirtualHosts))
	for _, vh := range config.VirtualHosts {
		vhwi, err := fc.convertVirtualHost(vh)
		if err != nil {
			// Non nil if (lds + rds) fails, shouldn't happen since validated by
			// xDS Client, treat as L7 error but shouldn't happen.
			return &UsableRouteConfiguration{Err: fmt.Errorf("virtual host construction: %v", err)}
		}
		vhs = append(vhs, vhwi)
	}
	return &UsableRouteConfiguration{VHS: vhs}
}

func (fc *FilterChain) convertVirtualHost(virtualHost *VirtualHost) (VirtualHostWithInterceptors, error) {
	rs := make([]RouteWithInterceptors, len(virtualHost.Routes))
	for i, r := range virtualHost.Routes {
		rs[i].ActionType = r.ActionType
		rs[i].M = RouteToMatcher(r)
		for _, filter := range fc.HTTPFilters {
			// Route is highest priority on server side, as there is no concept
			// of an upstream cluster on server side.
			override := r.HTTPFilterConfigOverride[filter.Name]
			if override == nil {
				// Virtual Host is second priority.
				override = virtualHost.HTTPFilterConfigOverride[filter.Name]
			}
			sb, ok := filter.Filter.(httpfilter.ServerInterceptorBuilder)
			if !ok {
				// Should not happen if it passed xdsClient validation.
				return VirtualHostWithInterceptors{}, fmt.Errorf("filter does not support use in server")
			}
			si, err := sb.BuildServerInterceptor(filter.Config, override)
			if err != nil {
				return VirtualHostWithInterceptors{}, fmt.Errorf("filter construction: %v", err)
			}
			if si != nil {
				rs[i].Interceptors = append(rs[i].Interceptors, si)
			}
		}
	}
	return VirtualHostWithInterceptors{Domains: virtualHost.Domains, Routes: rs}, nil
}

// SourceType specifies the connection source IP match type.
type SourceType int

const (
	// SourceTypeAny matches connection attempts from any source.
	SourceTypeAny SourceType = iota
	// SourceTypeSameOrLoopback matches connection attempts from the same host.
	SourceTypeSameOrLoopback
	// SourceTypeExternal matches connection attempts from a different host.
	SourceTypeExternal
)

// FilterChainManager contains all the match criteria specified through all
// filter chains in a single Listener resource. It also contains the default
// filter chain specified in the Listener resource. It provides two important
// pieces of functionality:
//  1. Validate the filter chains in an incoming Listener resource to make sure
//     that there aren't filter chains which contain the same match criteria.
//  2. As part of performing the above validation, it builds an internal data
//     structure which will if used to look up the matching filter chain at
//     connection time.
//
// The logic specified in the documentation around the xDS FilterChainMatch
// proto mentions 8 criteria to match on.
// The following order applies:
//
// 1. Destination port.
// 2. Destination IP address.
// 3. Server name (e.g. SNI for TLS protocol),
// 4. Transport protocol.
// 5. Application protocols (e.g. ALPN for TLS protocol).
// 6. Source type (e.g. any, local or external network).
// 7. Source IP address.
// 8. Source port.
type FilterChainManager struct {
	// Destination prefix is the first match criteria that we support.
	// Therefore, this multi-stage map is indexed on destination prefixes
	// specified in the match criteria.
	// Unspecified destination prefix matches end up as a wildcard entry here
	// with a key of 0.0.0.0/0.
	dstPrefixMap map[string]*destPrefixEntry

	// At connection time, we do not have the actual destination prefix to match
	// on. We only have the real destination address of the incoming connection.
	// This means that we cannot use the above map at connection time. This list
	// contains the map entries from the above map that we can use at connection
	// time to find matching destination prefixes in O(n) time.
	//
	// TODO: Implement LC-trie to support logarithmic time lookups. If that
	// involves too much time/effort, sort this slice based on the netmask size.
	dstPrefixes []*destPrefixEntry

	def *FilterChain // Default filter chain, if specified.

	// Slice of filter chains managed by this filter chain manager.
	fcs []*FilterChain

	// RouteConfigNames are the route configuration names which need to be
	// dynamically queried for RDS Configuration for any FilterChains which
	// specify to load RDS Configuration dynamically.
	RouteConfigNames map[string]bool
}

// destPrefixEntry is the value type of the map indexed on destination prefixes.
type destPrefixEntry struct {
	// The actual destination prefix. Set to nil for unspecified prefixes.
	net *net.IPNet
	// We need to keep track of the transport protocols seen as part of the
	// config validation (and internal structure building) phase. The only two
	// values that we support are empty string and "raw_buffer", with the latter
	// taking preference. Once we have seen one filter chain with "raw_buffer",
	// we can drop everything other filter chain with an empty transport
	// protocol.
	rawBufferSeen bool
	// For each specified source type in the filter chain match criteria, this
	// array points to the set of specified source prefixes.
	// Unspecified source type matches end up as a wildcard entry here with an
	// index of 0, which actually represents the source type `ANY`.
	srcTypeArr sourceTypesArray
}

// An array for the fixed number of source types that we have.
type sourceTypesArray [3]*sourcePrefixes

// sourcePrefixes contains source prefix related information specified in the
// match criteria. These are pointed to by the array of source types.
type sourcePrefixes struct {
	// These are very similar to the 'dstPrefixMap' and 'dstPrefixes' field of
	// FilterChainManager. Go there for more info.
	srcPrefixMap map[string]*sourcePrefixEntry
	srcPrefixes  []*sourcePrefixEntry
}

// sourcePrefixEntry contains match criteria per source prefix.
type sourcePrefixEntry struct {
	// The actual destination prefix. Set to nil for unspecified prefixes.
	net *net.IPNet
	// Mapping from source ports specified in the match criteria to the actual
	// filter chain. Unspecified source port matches en up as a wildcard entry
	// here with a key of 0.
	srcPortMap map[int]*FilterChain
}

// NewFilterChainManager parses the received Listener resource and builds a
// FilterChainManager. Returns a non-nil error on validation failures.
//
// This function is only exported so that tests outside of this package can
// create a FilterChainManager.
func NewFilterChainManager(lis *v3listenerpb.Listener) (*FilterChainManager, error) {
	// Parse all the filter chains and build the internal data structures.
	fci := &FilterChainManager{
		dstPrefixMap:     make(map[string]*destPrefixEntry),
		RouteConfigNames: make(map[string]bool),
	}
	if err := fci.addFilterChains(lis.GetFilterChains()); err != nil {
		return nil, err
	}
	// Build the source and dest prefix slices used by Lookup().
	fcSeen := false
	for _, dstPrefix := range fci.dstPrefixMap {
		fci.dstPrefixes = append(fci.dstPrefixes, dstPrefix)
		for _, st := range dstPrefix.srcTypeArr {
			if st == nil {
				continue
			}
			for _, srcPrefix := range st.srcPrefixMap {
				st.srcPrefixes = append(st.srcPrefixes, srcPrefix)
				for _, fc := range srcPrefix.srcPortMap {
					if fc != nil {
						fcSeen = true
					}
				}
			}
		}
	}

	// Retrieve the default filter chain. The match criteria specified on the
	// default filter chain is never used. The default filter chain simply gets
	// used when none of the other filter chains match.
	var def *FilterChain
	if dfc := lis.GetDefaultFilterChain(); dfc != nil {
		var err error
		if def, err = fci.filterChainFromProto(dfc); err != nil {
			return nil, err
		}
	}
	fci.def = def
	if fci.def != nil {
		fci.fcs = append(fci.fcs, fci.def)
	}

	// If there are no supported filter chains and no default filter chain, we
	// fail here. This will call the Listener resource to be NACK'ed.
	if !fcSeen && fci.def == nil {
		return nil, fmt.Errorf("no supported filter chains and no default filter chain")
	}
	return fci, nil
}

// addFilterChains parses the filter chains in fcs and adds the required
// internal data structures corresponding to the match criteria.
func (fcm *FilterChainManager) addFilterChains(fcs []*v3listenerpb.FilterChain) error {
	for _, fc := range fcs {
		fcMatch := fc.GetFilterChainMatch()
		if fcMatch.GetDestinationPort().GetValue() != 0 {
			// Destination port is the first match criteria and we do not
			// support filter chains which contains this match criteria.
			logger.Warningf("Dropping filter chain %+v since it contains unsupported destination_port match field", fc)
			continue
		}

		// Build the internal representation of the filter chain match fields.
		if err := fcm.addFilterChainsForDestPrefixes(fc); err != nil {
			return err
		}
	}

	return nil
}

func (fcm *FilterChainManager) addFilterChainsForDestPrefixes(fc *v3listenerpb.FilterChain) error {
	ranges := fc.GetFilterChainMatch().GetPrefixRanges()
	dstPrefixes := make([]*net.IPNet, 0, len(ranges))
	for _, pr := range ranges {
		cidr := fmt.Sprintf("%s/%d", pr.GetAddressPrefix(), pr.GetPrefixLen().GetValue())
		_, ipnet, err := net.ParseCIDR(cidr)
		if err != nil {
			return fmt.Errorf("failed to parse destination prefix range: %+v", pr)
		}
		dstPrefixes = append(dstPrefixes, ipnet)
	}

	if len(dstPrefixes) == 0 {
		// Use the unspecified entry when destination prefix is unspecified, and
		// set the `net` field to nil.
		if fcm.dstPrefixMap[unspecifiedPrefixMapKey] == nil {
			fcm.dstPrefixMap[unspecifiedPrefixMapKey] = &destPrefixEntry{}
		}
		return fcm.addFilterChainsForServerNames(fcm.dstPrefixMap[unspecifiedPrefixMapKey], fc)
	}
	for _, prefix := range dstPrefixes {
		p := prefix.String()
		if fcm.dstPrefixMap[p] == nil {
			fcm.dstPrefixMap[p] = &destPrefixEntry{net: prefix}
		}
		if err := fcm.addFilterChainsForServerNames(fcm.dstPrefixMap[p], fc); err != nil {
			return err
		}
	}
	return nil
}

func (fcm *FilterChainManager) addFilterChainsForServerNames(dstEntry *destPrefixEntry, fc *v3listenerpb.FilterChain) error {
	// Filter chains specifying server names in their match criteria always fail
	// a match at connection time. So, these filter chains can be dropped now.
	if len(fc.GetFilterChainMatch().GetServerNames()) != 0 {
		logger.Warningf("Dropping filter chain %+v since it contains unsupported server_names match field", fc)
		return nil
	}

	return fcm.addFilterChainsForTransportProtocols(dstEntry, fc)
}

func (fcm *FilterChainManager) addFilterChainsForTransportProtocols(dstEntry *destPrefixEntry, fc *v3listenerpb.FilterChain) error {
	tp := fc.GetFilterChainMatch().GetTransportProtocol()
	switch {
	case tp != "" && tp != "raw_buffer":
		// Only allow filter chains with transport protocol set to empty string
		// or "raw_buffer".
		logger.Warningf("Dropping filter chain %+v since it contains unsupported value for transport_protocols match field", fc)
		return nil
	case tp == "" && dstEntry.rawBufferSeen:
		// If we have already seen filter chains with transport protocol set to
		// "raw_buffer", we can drop filter chains with transport protocol set
		// to empty string, since the former takes precedence.
		logger.Warningf("Dropping filter chain %+v since it contains unsupported value for transport_protocols match field", fc)
		return nil
	case tp != "" && !dstEntry.rawBufferSeen:
		// This is the first "raw_buffer" that we are seeing. Set the bit and
		// reset the source types array which might contain entries for filter
		// chains with transport protocol set to empty string.
		dstEntry.rawBufferSeen = true
		dstEntry.srcTypeArr = sourceTypesArray{}
	}
	return fcm.addFilterChainsForApplicationProtocols(dstEntry, fc)
}

func (fcm *FilterChainManager) addFilterChainsForApplicationProtocols(dstEntry *destPrefixEntry, fc *v3listenerpb.FilterChain) error {
	if len(fc.GetFilterChainMatch().GetApplicationProtocols()) != 0 {
		logger.Warningf("Dropping filter chain %+v since it contains unsupported application_protocols match field", fc)
		return nil
	}
	return fcm.addFilterChainsForSourceType(dstEntry, fc)
}

// addFilterChainsForSourceType adds source types to the internal data
// structures and delegates control to addFilterChainsForSourcePrefixes to
// continue building the internal data structure.
func (fcm *FilterChainManager) addFilterChainsForSourceType(dstEntry *destPrefixEntry, fc *v3listenerpb.FilterChain) error {
	var srcType SourceType
	switch st := fc.GetFilterChainMatch().GetSourceType(); st {
	case v3listenerpb.FilterChainMatch_ANY:
		srcType = SourceTypeAny
	case v3listenerpb.FilterChainMatch_SAME_IP_OR_LOOPBACK:
		srcType = SourceTypeSameOrLoopback
	case v3listenerpb.FilterChainMatch_EXTERNAL:
		srcType = SourceTypeExternal
	default:
		return fmt.Errorf("unsupported source type: %v", st)
	}

	st := int(srcType)
	if dstEntry.srcTypeArr[st] == nil {
		dstEntry.srcTypeArr[st] = &sourcePrefixes{srcPrefixMap: make(map[string]*sourcePrefixEntry)}
	}
	return fcm.addFilterChainsForSourcePrefixes(dstEntry.srcTypeArr[st].srcPrefixMap, fc)
}

// addFilterChainsForSourcePrefixes adds source prefixes to the internal data
// structures and delegates control to addFilterChainsForSourcePorts to continue
// building the internal data structure.
func (fcm *FilterChainManager) addFilterChainsForSourcePrefixes(srcPrefixMap map[string]*sourcePrefixEntry, fc *v3listenerpb.FilterChain) error {
	ranges := fc.GetFilterChainMatch().GetSourcePrefixRanges()
	srcPrefixes := make([]*net.IPNet, 0, len(ranges))
	for _, pr := range fc.GetFilterChainMatch().GetSourcePrefixRanges() {
		cidr := fmt.Sprintf("%s/%d", pr.GetAddressPrefix(), pr.GetPrefixLen().GetValue())
		_, ipnet, err := net.ParseCIDR(cidr)
		if err != nil {
			return fmt.Errorf("failed to parse source prefix range: %+v", pr)
		}
		srcPrefixes = append(srcPrefixes, ipnet)
	}

	if len(srcPrefixes) == 0 {
		// Use the unspecified entry when destination prefix is unspecified, and
		// set the `net` field to nil.
		if srcPrefixMap[unspecifiedPrefixMapKey] == nil {
			srcPrefixMap[unspecifiedPrefixMapKey] = &sourcePrefixEntry{
				srcPortMap: make(map[int]*FilterChain),
			}
		}
		return fcm.addFilterChainsForSourcePorts(srcPrefixMap[unspecifiedPrefixMapKey], fc)
	}
	for _, prefix := range srcPrefixes {
		p := prefix.String()
		if srcPrefixMap[p] == nil {
			srcPrefixMap[p] = &sourcePrefixEntry{
				net:        prefix,
				srcPortMap: make(map[int]*FilterChain),
			}
		}
		if err := fcm.addFilterChainsForSourcePorts(srcPrefixMap[p], fc); err != nil {
			return err
		}
	}
	return nil
}

// addFilterChainsForSourcePorts adds source ports to the internal data
// structures and completes the process of building the internal data structure.
// It is here that we determine if there are multiple filter chains with
// overlapping matching rules.
func (fcm *FilterChainManager) addFilterChainsForSourcePorts(srcEntry *sourcePrefixEntry, fcProto *v3listenerpb.FilterChain) error {
	ports := fcProto.GetFilterChainMatch().GetSourcePorts()
	srcPorts := make([]int, 0, len(ports))
	for _, port := range ports {
		srcPorts = append(srcPorts, int(port))
	}

	fc, err := fcm.filterChainFromProto(fcProto)
	if err != nil {
		return err
	}

	if len(srcPorts) == 0 {
		// Use the wildcard port '0', when source ports are unspecified.
		if curFC := srcEntry.srcPortMap[0]; curFC != nil {
			return errors.New("multiple filter chains with overlapping matching rules are defined")
		}
		srcEntry.srcPortMap[0] = fc
		fcm.fcs = append(fcm.fcs, fc)
		return nil
	}
	for _, port := range srcPorts {
		if curFC := srcEntry.srcPortMap[port]; curFC != nil {
			return errors.New("multiple filter chains with overlapping matching rules are defined")
		}
		srcEntry.srcPortMap[port] = fc
	}
	fcm.fcs = append(fcm.fcs, fc)
	return nil
}

// FilterChains returns the filter chains for this filter chain manager.
func (fcm *FilterChainManager) FilterChains() []*FilterChain {
	return fcm.fcs
}

// filterChainFromProto extracts the relevant information from the FilterChain
// proto and stores it in our internal representation. It also persists any
// RouteNames which need to be queried dynamically via RDS.
func (fcm *FilterChainManager) filterChainFromProto(fc *v3listenerpb.FilterChain) (*FilterChain, error) {
	filterChain, err := processNetworkFilters(fc.GetFilters())
	if err != nil {
		return nil, err
	}
	// These route names will be dynamically queried via RDS in the wrapped
	// listener, which receives the LDS response, if specified for the filter
	// chain.
	if filterChain.RouteConfigName != "" {
		fcm.RouteConfigNames[filterChain.RouteConfigName] = true
	}
	// If the transport_socket field is not specified, it means that the control
	// plane has not sent us any security config. This is fine and the server
	// will use the fallback credentials configured as part of the
	// xdsCredentials.
	ts := fc.GetTransportSocket()
	if ts == nil {
		return filterChain, nil
	}
	if name := ts.GetName(); name != transportSocketName {
		return nil, fmt.Errorf("transport_socket field has unexpected name: %s", name)
	}
	tc := ts.GetTypedConfig()
	if typeURL := tc.GetTypeUrl(); typeURL != version.V3DownstreamTLSContextURL {
		return nil, fmt.Errorf("transport_socket missing typed_config or wrong type_url: %q", typeURL)
	}
	downstreamCtx := &v3tlspb.DownstreamTlsContext{}
	if err := proto.Unmarshal(tc.GetValue(), downstreamCtx); err != nil {
		return nil, fmt.Errorf("failed to unmarshal DownstreamTlsContext in LDS response: %v", err)
	}
	if downstreamCtx.GetRequireSni().GetValue() {
		return nil, fmt.Errorf("require_sni field set to true in DownstreamTlsContext message: %v", downstreamCtx)
	}
	if downstreamCtx.GetOcspStaplePolicy() != v3tlspb.DownstreamTlsContext_LENIENT_STAPLING {
		return nil, fmt.Errorf("ocsp_staple_policy field set to unsupported value in DownstreamTlsContext message: %v", downstreamCtx)
	}
	// The following fields from `DownstreamTlsContext` are ignore:
	// - disable_stateless_session_resumption
	// - session_ticket_keys
	// - session_ticket_keys_sds_secret_config
	// - session_timeout
	if downstreamCtx.GetCommonTlsContext() == nil {
		return nil, errors.New("DownstreamTlsContext in LDS response does not contain a CommonTlsContext")
	}
	sc, err := securityConfigFromCommonTLSContext(downstreamCtx.GetCommonTlsContext(), true)
	if err != nil {
		return nil, err
	}
	if sc == nil {
		// sc == nil is a valid case where the control plane has not sent us any
		// security configuration. xDS creds will use fallback creds.
		return filterChain, nil
	}
	sc.RequireClientCert = downstreamCtx.GetRequireClientCertificate().GetValue()
	if sc.RequireClientCert && sc.RootInstanceName == "" {
		return nil, errors.New("security configuration on the server-side does not contain root certificate provider instance name, but require_client_cert field is set")
	}
	filterChain.SecurityCfg = sc
	return filterChain, nil
}

// Validate takes a function to validate the FilterChains in this manager.
func (fcm *FilterChainManager) Validate(f func(fc *FilterChain) error) error {
	for _, dst := range fcm.dstPrefixMap {
		for _, srcType := range dst.srcTypeArr {
			if srcType == nil {
				continue
			}
			for _, src := range srcType.srcPrefixMap {
				for _, fc := range src.srcPortMap {
					if err := f(fc); err != nil {
						return err
					}
				}
			}
		}
	}
	return f(fcm.def)
}

func processNetworkFilters(filters []*v3listenerpb.Filter) (*FilterChain, error) {
	rc := &UsableRouteConfiguration{}
	filterChain := &FilterChain{
		UsableRouteConfiguration: &atomic.Pointer[UsableRouteConfiguration]{},
	}
	filterChain.UsableRouteConfiguration.Store(rc)
	seenNames := make(map[string]bool, len(filters))
	seenHCM := false
	for _, filter := range filters {
		name := filter.GetName()
		if name == "" {
			return nil, fmt.Errorf("network filters {%+v} is missing name field in filter: {%+v}", filters, filter)
		}
		if seenNames[name] {
			return nil, fmt.Errorf("network filters {%+v} has duplicate filter name %q", filters, name)
		}
		seenNames[name] = true

		// Network filters have a oneof field named `config_type` where we
		// only support `TypedConfig` variant.
		switch typ := filter.GetConfigType().(type) {
		case *v3listenerpb.Filter_TypedConfig:
			// The typed_config field has an `anypb.Any` proto which could
			// directly contain the serialized bytes of the actual filter
			// configuration, or it could be encoded as a `TypedStruct`.
			// TODO: Add support for `TypedStruct`.
			tc := filter.GetTypedConfig()

			// The only network filter that we currently support is the v3
			// HttpConnectionManager. So, we can directly check the type_url
			// and unmarshal the config.
			// TODO: Implement a registry of supported network filters (like
			// we have for HTTP filters), when we have to support network
			// filters other than HttpConnectionManager.
			if tc.GetTypeUrl() != version.V3HTTPConnManagerURL {
				return nil, fmt.Errorf("network filters {%+v} has unsupported network filter %q in filter {%+v}", filters, tc.GetTypeUrl(), filter)
			}
			hcm := &v3httppb.HttpConnectionManager{}
			if err := tc.UnmarshalTo(hcm); err != nil {
				return nil, fmt.Errorf("network filters {%+v} failed unmarshalling of network filter {%+v}: %v", filters, filter, err)
			}
			// "Any filters after HttpConnectionManager should be ignored during
			// connection processing but still be considered for validity.
			// HTTPConnectionManager must have valid http_filters." - A36
			filters, err := processHTTPFilters(hcm.GetHttpFilters(), true)
			if err != nil {
				return nil, fmt.Errorf("network filters {%+v} had invalid server side HTTP Filters {%+v}: %v", filters, hcm.GetHttpFilters(), err)
			}
			if !seenHCM {
				// Validate for RBAC in only the HCM that will be used, since this isn't a logical validation failure,
				// it's simply a validation to support RBAC HTTP Filter.
				// "HttpConnectionManager.xff_num_trusted_hops must be unset or zero and
				// HttpConnectionManager.original_ip_detection_extensions must be empty. If
				// either field has an incorrect value, the Listener must be NACKed." - A41
				if hcm.XffNumTrustedHops != 0 {
					return nil, fmt.Errorf("xff_num_trusted_hops must be unset or zero %+v", hcm)
				}
				if len(hcm.OriginalIpDetectionExtensions) != 0 {
					return nil, fmt.Errorf("original_ip_detection_extensions must be empty %+v", hcm)
				}

				// TODO: Implement terminal filter logic, as per A36.
				filterChain.HTTPFilters = filters
				seenHCM = true
				switch hcm.RouteSpecifier.(type) {
				case *v3httppb.HttpConnectionManager_Rds:
					if hcm.GetRds().GetConfigSource().GetAds() == nil {
						return nil, fmt.Errorf("ConfigSource is not ADS: %+v", hcm)
					}
					name := hcm.GetRds().GetRouteConfigName()
					if name == "" {
						return nil, fmt.Errorf("empty route_config_name: %+v", hcm)
					}
					filterChain.RouteConfigName = name
				case *v3httppb.HttpConnectionManager_RouteConfig:
					// "RouteConfiguration validation logic inherits all
					// previous validations made for client-side usage as RDS
					// does not distinguish between client-side and
					// server-side." - A36
					// Can specify v3 here, as will never get to this function
					// if v2.
					routeU, err := generateRDSUpdateFromRouteConfiguration(hcm.GetRouteConfig())
					if err != nil {
						return nil, fmt.Errorf("failed to parse inline RDS resp: %v", err)
					}
					filterChain.InlineRouteConfig = &routeU
				case nil:
					return nil, fmt.Errorf("no RouteSpecifier: %+v", hcm)
				default:
					return nil, fmt.Errorf("unsupported type %T for RouteSpecifier", hcm.RouteSpecifier)
				}
			}
		default:
			return nil, fmt.Errorf("network filters {%+v} has unsupported config_type %T in filter %s", filters, typ, filter.GetName())
		}
	}
	if !seenHCM {
		return nil, fmt.Errorf("network filters {%+v} missing HttpConnectionManager filter", filters)
	}
	return filterChain, nil
}

// FilterChainLookupParams wraps parameters to be passed to Lookup.
type FilterChainLookupParams struct {
	// IsUnspecified indicates whether the server is listening on a wildcard
	// address, "0.0.0.0" for IPv4 and "::" for IPv6. Only when this is set to
	// true, do we consider the destination prefixes specified in the filter
	// chain match criteria.
	IsUnspecifiedListener bool
	// DestAddr is the local address of an incoming connection.
	DestAddr net.IP
	// SourceAddr is the remote address of an incoming connection.
	SourceAddr net.IP
	// SourcePort is the remote port of an incoming connection.
	SourcePort int
}

// Lookup returns the most specific matching filter chain to be used for an
// incoming connection on the server side.
//
// Returns a non-nil error if no matching filter chain could be found or
// multiple matching filter chains were found, and in both cases, the incoming
// connection must be dropped.
func (fcm *FilterChainManager) Lookup(params FilterChainLookupParams) (*FilterChain, error) {
	dstPrefixes := filterByDestinationPrefixes(fcm.dstPrefixes, params.IsUnspecifiedListener, params.DestAddr)
	if len(dstPrefixes) == 0 {
		if fcm.def != nil {
			return fcm.def, nil
		}
		return nil, fmt.Errorf("no matching filter chain based on destination prefix match for %+v", params)
	}

	srcType := SourceTypeExternal
	if params.SourceAddr.Equal(params.DestAddr) || params.SourceAddr.IsLoopback() {
		srcType = SourceTypeSameOrLoopback
	}
	srcPrefixes := filterBySourceType(dstPrefixes, srcType)
	if len(srcPrefixes) == 0 {
		if fcm.def != nil {
			return fcm.def, nil
		}
		return nil, fmt.Errorf("no matching filter chain based on source type match for %+v", params)
	}
	srcPrefixEntry, err := filterBySourcePrefixes(srcPrefixes, params.SourceAddr)
	if err != nil {
		return nil, err
	}
	if fc := filterBySourcePorts(srcPrefixEntry, params.SourcePort); fc != nil {
		return fc, nil
	}
	if fcm.def != nil {
		return fcm.def, nil
	}
	return nil, fmt.Errorf("no matching filter chain after all match criteria for %+v", params)
}

// filterByDestinationPrefixes is the first stage of the filter chain
// matching algorithm. It takes the complete set of configured filter chain
// matchers and returns the most specific matchers based on the destination
// prefix match criteria (the prefixes which match the most number of bits).
func filterByDestinationPrefixes(dstPrefixes []*destPrefixEntry, isUnspecified bool, dstAddr net.IP) []*destPrefixEntry {
	if !isUnspecified {
		// Destination prefix matchers are considered only when the listener is
		// bound to the wildcard address.
		return dstPrefixes
	}

	var matchingDstPrefixes []*destPrefixEntry
	maxSubnetMatch := noPrefixMatch
	for _, prefix := range dstPrefixes {
		if prefix.net != nil && !prefix.net.Contains(dstAddr) {
			// Skip prefixes which don't match.
			continue
		}
		// For unspecified prefixes, since we do not store a real net.IPNet
		// inside prefix, we do not perform a match. Instead we simply set
		// the matchSize to -1, which is less than the matchSize (0) for a
		// wildcard prefix.
		matchSize := unspecifiedPrefixMatch
		if prefix.net != nil {
			matchSize, _ = prefix.net.Mask.Size()
		}
		if matchSize < maxSubnetMatch {
			continue
		}
		if matchSize > maxSubnetMatch {
			maxSubnetMatch = matchSize
			matchingDstPrefixes = make([]*destPrefixEntry, 0, 1)
		}
		matchingDstPrefixes = append(matchingDstPrefixes, prefix)
	}
	return matchingDstPrefixes
}

// filterBySourceType is the second stage of the matching algorithm. It
// trims the filter chains based on the most specific source type match.
func filterBySourceType(dstPrefixes []*destPrefixEntry, srcType SourceType) []*sourcePrefixes {
	var (
		srcPrefixes      []*sourcePrefixes
		bestSrcTypeMatch int
	)
	for _, prefix := range dstPrefixes {
		var (
			srcPrefix *sourcePrefixes
			match     int
		)
		switch srcType {
		case SourceTypeExternal:
			match = int(SourceTypeExternal)
			srcPrefix = prefix.srcTypeArr[match]
		case SourceTypeSameOrLoopback:
			match = int(SourceTypeSameOrLoopback)
			srcPrefix = prefix.srcTypeArr[match]
		}
		if srcPrefix == nil {
			match = int(SourceTypeAny)
			srcPrefix = prefix.srcTypeArr[match]
		}
		if match < bestSrcTypeMatch {
			continue
		}
		if match > bestSrcTypeMatch {
			bestSrcTypeMatch = match
			srcPrefixes = make([]*sourcePrefixes, 0)
		}
		if srcPrefix != nil {
			// The source type array always has 3 entries, but these could be
			// nil if the appropriate source type match was not specified.
			srcPrefixes = append(srcPrefixes, srcPrefix)
		}
	}
	return srcPrefixes
}

// filterBySourcePrefixes is the third stage of the filter chain matching
// algorithm. It trims the filter chains based on the source prefix. At most one
// filter chain with the most specific match progress to the next stage.
func filterBySourcePrefixes(srcPrefixes []*sourcePrefixes, srcAddr net.IP) (*sourcePrefixEntry, error) {
	var matchingSrcPrefixes []*sourcePrefixEntry
	maxSubnetMatch := noPrefixMatch
	for _, sp := range srcPrefixes {
		for _, prefix := range sp.srcPrefixes {
			if prefix.net != nil && !prefix.net.Contains(srcAddr) {
				// Skip prefixes which don't match.
				continue
			}
			// For unspecified prefixes, since we do not store a real net.IPNet
			// inside prefix, we do not perform a match. Instead we simply set
			// the matchSize to -1, which is less than the matchSize (0) for a
			// wildcard prefix.
			matchSize := unspecifiedPrefixMatch
			if prefix.net != nil {
				matchSize, _ = prefix.net.Mask.Size()
			}
			if matchSize < maxSubnetMatch {
				continue
			}
			if matchSize > maxSubnetMatch {
				maxSubnetMatch = matchSize
				matchingSrcPrefixes = make([]*sourcePrefixEntry, 0, 1)
			}
			matchingSrcPrefixes = append(matchingSrcPrefixes, prefix)
		}
	}
	if len(matchingSrcPrefixes) == 0 {
		// Finding no match is not an error condition. The caller will end up
		// using the default filter chain if one was configured.
		return nil, nil
	}
	// We expect at most a single matching source prefix entry at this point. If
	// we have multiple entries here, and some of their source port matchers had
	// wildcard entries, we could be left with more than one matching filter
	// chain and hence would have been flagged as an invalid configuration at
	// config validation time.
	if len(matchingSrcPrefixes) != 1 {
		return nil, errors.New("multiple matching filter chains")
	}
	return matchingSrcPrefixes[0], nil
}

// filterBySourcePorts is the last stage of the filter chain matching
// algorithm. It trims the filter chains based on the source ports.
func filterBySourcePorts(spe *sourcePrefixEntry, srcPort int) *FilterChain {
	if spe == nil {
		return nil
	}
	// A match could be a wildcard match (this happens when the match
	// criteria does not specify source ports) or a specific port match (this
	// happens when the match criteria specifies a set of ports and the source
	// port of the incoming connection matches one of the specified ports). The
	// latter is considered to be a more specific match.
	if fc := spe.srcPortMap[srcPort]; fc != nil {
		return fc
	}
	if fc := spe.srcPortMap[0]; fc != nil {
		return fc
	}
	return nil
}
