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
	"regexp"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/internal/xds/clusterspecifier"
	"google.golang.org/grpc/internal/xds/httpfilter"
	"google.golang.org/grpc/internal/xds/matcher"
	"google.golang.org/protobuf/types/known/anypb"
)

// RouteConfigUpdate contains information received in an RDS response, which is
// of interest to the registered RDS watcher.
type RouteConfigUpdate struct {
	VirtualHosts []*VirtualHost
	// ClusterSpecifierPlugins are the LB Configurations for any
	// ClusterSpecifierPlugins referenced by the Route Table.
	ClusterSpecifierPlugins map[string]clusterspecifier.BalancerConfig
	// Raw is the resource from the xds response.
	Raw *anypb.Any
}

// VirtualHost contains the routes for a list of Domains.
//
// Note that the domains in this slice can be a wildcard, not an exact string.
// The consumer of this struct needs to find the best match for its hostname.
type VirtualHost struct {
	Domains []string
	// Routes contains a list of routes, each containing matchers and
	// corresponding action.
	Routes []*Route
	// HTTPFilterConfigOverride contains any HTTP filter config overrides for
	// the virtual host which may be present.  An individual filter's override
	// may be unused if the matching Route contains an override for that
	// filter.
	HTTPFilterConfigOverride map[string]httpfilter.FilterConfig
	RetryConfig              *RetryConfig
}

// RetryConfig contains all retry-related configuration in either a VirtualHost
// or Route.
type RetryConfig struct {
	// RetryOn is a set of status codes on which to retry.  Only Canceled,
	// DeadlineExceeded, Internal, ResourceExhausted, and Unavailable are
	// supported; any other values will be omitted.
	RetryOn      map[codes.Code]bool
	NumRetries   uint32       // maximum number of retry attempts
	RetryBackoff RetryBackoff // retry backoff policy
}

// RetryBackoff describes the backoff policy for retries.
type RetryBackoff struct {
	BaseInterval time.Duration // initial backoff duration between attempts
	MaxInterval  time.Duration // maximum backoff duration
}

// HashPolicyType specifies the type of HashPolicy from a received RDS Response.
type HashPolicyType int

const (
	// HashPolicyTypeHeader specifies to hash a Header in the incoming request.
	HashPolicyTypeHeader HashPolicyType = iota
	// HashPolicyTypeChannelID specifies to hash a unique Identifier of the
	// Channel. This is a 64-bit random int computed at initialization time.
	HashPolicyTypeChannelID
)

// HashPolicy specifies the HashPolicy if the upstream cluster uses a hashing
// load balancer.
type HashPolicy struct {
	HashPolicyType HashPolicyType
	Terminal       bool
	// Fields used for type HEADER.
	HeaderName        string
	Regex             *regexp.Regexp
	RegexSubstitution string
}

// RouteActionType is the action of the route from a received RDS response.
type RouteActionType int

const (
	// RouteActionUnsupported are routing types currently unsupported by grpc.
	// According to A36, "A Route with an inappropriate action causes RPCs
	// matching that route to fail."
	RouteActionUnsupported RouteActionType = iota
	// RouteActionRoute is the expected route type on the client side. Route
	// represents routing a request to some upstream cluster. On the client
	// side, if an RPC matches to a route that is not RouteActionRoute, the RPC
	// will fail according to A36.
	RouteActionRoute
	// RouteActionNonForwardingAction is the expected route type on the server
	// side. NonForwardingAction represents when a route will generate a
	// response directly, without forwarding to an upstream host.
	RouteActionNonForwardingAction
)

// Route is both a specification of how to match a request as well as an
// indication of the action to take upon match.
type Route struct {
	Path   *string
	Prefix *string
	Regex  *regexp.Regexp
	// Indicates if prefix/path matching should be case insensitive. The default
	// is false (case sensitive).
	CaseInsensitive bool
	Headers         []*HeaderMatcher
	Fraction        *uint32

	HashPolicies []*HashPolicy

	// If the matchers above indicate a match, the below configuration is used.
	// If MaxStreamDuration is nil, it indicates neither of the route action's
	// max_stream_duration fields (grpc_timeout_header_max nor
	// max_stream_duration) were set.  In this case, the ListenerUpdate's
	// MaxStreamDuration field should be used.  If MaxStreamDuration is set to
	// an explicit zero duration, the application's deadline should be used.
	MaxStreamDuration *time.Duration
	// HTTPFilterConfigOverride contains any HTTP filter config overrides for
	// the route which may be present.  An individual filter's override may be
	// unused if the matching WeightedCluster contains an override for that
	// filter.
	HTTPFilterConfigOverride map[string]httpfilter.FilterConfig
	RetryConfig              *RetryConfig

	ActionType RouteActionType

	// Only one of the following fields (WeightedClusters or
	// ClusterSpecifierPlugin) will be set for a route.
	WeightedClusters map[string]WeightedCluster
	// ClusterSpecifierPlugin is the name of the Cluster Specifier Plugin that
	// this Route is linked to, if specified by xDS.
	ClusterSpecifierPlugin string
}

// WeightedCluster contains settings for an xds ActionType.WeightedCluster.
type WeightedCluster struct {
	// Weight is the relative weight of the cluster.  It will never be zero.
	Weight uint32
	// HTTPFilterConfigOverride contains any HTTP filter config overrides for
	// the weighted cluster which may be present.
	HTTPFilterConfigOverride map[string]httpfilter.FilterConfig
}

// HeaderMatcher represents header matchers.
type HeaderMatcher struct {
	Name         string
	InvertMatch  *bool
	ExactMatch   *string
	RegexMatch   *regexp.Regexp
	PrefixMatch  *string
	SuffixMatch  *string
	RangeMatch   *Int64Range
	PresentMatch *bool
	StringMatch  *matcher.StringMatcher
}

// Int64Range is a range for header range match.
type Int64Range struct {
	Start int64
	End   int64
}
