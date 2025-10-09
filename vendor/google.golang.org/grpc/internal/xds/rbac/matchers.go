/*
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

package rbac

import (
	"errors"
	"fmt"
	"net"
	"net/netip"
	"regexp"

	v3corepb "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	v3rbacpb "github.com/envoyproxy/go-control-plane/envoy/config/rbac/v3"
	v3route_componentspb "github.com/envoyproxy/go-control-plane/envoy/config/route/v3"
	v3matcherpb "github.com/envoyproxy/go-control-plane/envoy/type/matcher/v3"
	internalmatcher "google.golang.org/grpc/internal/xds/matcher"
)

// matcher is an interface that takes data about incoming RPC's and returns
// whether it matches with whatever matcher implements this interface.
type matcher interface {
	match(data *rpcData) bool
}

// policyMatcher helps determine whether an incoming RPC call matches a policy.
// A policy is a logical role (e.g. Service Admin), which is comprised of
// permissions and principals. A principal is an identity (or identities) for a
// downstream subject which are assigned the policy (role), and a permission is
// an action(s) that a principal(s) can take. A policy matches if both a
// permission and a principal match, which will be determined by the child or
// permissions and principal matchers. policyMatcher implements the matcher
// interface.
type policyMatcher struct {
	permissions *orMatcher
	principals  *orMatcher
}

func newPolicyMatcher(policy *v3rbacpb.Policy) (*policyMatcher, error) {
	permissions, err := matchersFromPermissions(policy.Permissions)
	if err != nil {
		return nil, err
	}
	principals, err := matchersFromPrincipals(policy.Principals)
	if err != nil {
		return nil, err
	}
	return &policyMatcher{
		permissions: &orMatcher{matchers: permissions},
		principals:  &orMatcher{matchers: principals},
	}, nil
}

func (pm *policyMatcher) match(data *rpcData) bool {
	// A policy matches if and only if at least one of its permissions match the
	// action taking place AND at least one if its principals match the
	// downstream peer.
	return pm.permissions.match(data) && pm.principals.match(data)
}

// matchersFromPermissions takes a list of permissions (can also be
// a single permission, e.g. from a not matcher which is logically !permission)
// and returns a list of matchers which correspond to that permission. This will
// be called in many instances throughout the initial construction of the RBAC
// engine from the AND and OR matchers and also from the NOT matcher.
func matchersFromPermissions(permissions []*v3rbacpb.Permission) ([]matcher, error) {
	var matchers []matcher
	for _, permission := range permissions {
		switch permission.GetRule().(type) {
		case *v3rbacpb.Permission_AndRules:
			mList, err := matchersFromPermissions(permission.GetAndRules().Rules)
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, &andMatcher{matchers: mList})
		case *v3rbacpb.Permission_OrRules:
			mList, err := matchersFromPermissions(permission.GetOrRules().Rules)
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, &orMatcher{matchers: mList})
		case *v3rbacpb.Permission_Any:
			matchers = append(matchers, &alwaysMatcher{})
		case *v3rbacpb.Permission_Header:
			m, err := newHeaderMatcher(permission.GetHeader())
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, m)
		case *v3rbacpb.Permission_UrlPath:
			m, err := newURLPathMatcher(permission.GetUrlPath())
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, m)
		case *v3rbacpb.Permission_DestinationIp:
			// Due to this being on server side, the destination IP is the local
			// IP.
			m, err := newLocalIPMatcher(permission.GetDestinationIp())
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, m)
		case *v3rbacpb.Permission_DestinationPort:
			matchers = append(matchers, newPortMatcher(permission.GetDestinationPort()))
		case *v3rbacpb.Permission_NotRule:
			mList, err := matchersFromPermissions([]*v3rbacpb.Permission{{Rule: permission.GetNotRule().Rule}})
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, &notMatcher{matcherToNot: mList[0]})
		case *v3rbacpb.Permission_Metadata:
			// Never matches - so no-op if not inverted, always match if
			// inverted.
			if permission.GetMetadata().GetInvert() { // Test metadata being no-op and also metadata with invert always matching
				matchers = append(matchers, &alwaysMatcher{})
			}
		case *v3rbacpb.Permission_RequestedServerName:
			// Not supported in gRPC RBAC currently - a permission typed as
			// requested server name in the initial config will be a no-op.
		}
	}
	return matchers, nil
}

func matchersFromPrincipals(principals []*v3rbacpb.Principal) ([]matcher, error) {
	var matchers []matcher
	for _, principal := range principals {
		switch principal.GetIdentifier().(type) {
		case *v3rbacpb.Principal_AndIds:
			mList, err := matchersFromPrincipals(principal.GetAndIds().Ids)
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, &andMatcher{matchers: mList})
		case *v3rbacpb.Principal_OrIds:
			mList, err := matchersFromPrincipals(principal.GetOrIds().Ids)
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, &orMatcher{matchers: mList})
		case *v3rbacpb.Principal_Any:
			matchers = append(matchers, &alwaysMatcher{})
		case *v3rbacpb.Principal_Authenticated_:
			authenticatedMatcher, err := newAuthenticatedMatcher(principal.GetAuthenticated())
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, authenticatedMatcher)
		case *v3rbacpb.Principal_DirectRemoteIp:
			m, err := newRemoteIPMatcher(principal.GetDirectRemoteIp())
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, m)
		case *v3rbacpb.Principal_Header:
			// Do we need an error here?
			m, err := newHeaderMatcher(principal.GetHeader())
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, m)
		case *v3rbacpb.Principal_UrlPath:
			m, err := newURLPathMatcher(principal.GetUrlPath())
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, m)
		case *v3rbacpb.Principal_NotId:
			mList, err := matchersFromPrincipals([]*v3rbacpb.Principal{{Identifier: principal.GetNotId().Identifier}})
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, &notMatcher{matcherToNot: mList[0]})
		case *v3rbacpb.Principal_SourceIp:
			// The source ip principal identifier is deprecated. Thus, a
			// principal typed as a source ip in the identifier will be a no-op.
			// The config should use DirectRemoteIp instead.
		case *v3rbacpb.Principal_RemoteIp:
			// RBAC in gRPC treats direct_remote_ip and remote_ip as logically
			// equivalent, as per A41.
			m, err := newRemoteIPMatcher(principal.GetRemoteIp())
			if err != nil {
				return nil, err
			}
			matchers = append(matchers, m)
		case *v3rbacpb.Principal_Metadata:
			// Not supported in gRPC RBAC currently - a principal typed as
			// Metadata in the initial config will be a no-op.
		}
	}
	return matchers, nil
}

// orMatcher is a matcher where it successfully matches if one of it's
// children successfully match. It also logically represents a principal or
// permission, but can also be it's own entity further down the tree of
// matchers. orMatcher implements the matcher interface.
type orMatcher struct {
	matchers []matcher
}

func (om *orMatcher) match(data *rpcData) bool {
	// Range through child matchers and pass in data about incoming RPC, and
	// only one child matcher has to match to be logically successful.
	for _, m := range om.matchers {
		if m.match(data) {
			return true
		}
	}
	return false
}

// andMatcher is a matcher that is successful if every child matcher
// matches. andMatcher implements the matcher interface.
type andMatcher struct {
	matchers []matcher
}

func (am *andMatcher) match(data *rpcData) bool {
	for _, m := range am.matchers {
		if !m.match(data) {
			return false
		}
	}
	return true
}

// alwaysMatcher is a matcher that will always match. This logically
// represents an any rule for a permission or a principal. alwaysMatcher
// implements the matcher interface.
type alwaysMatcher struct {
}

func (am *alwaysMatcher) match(*rpcData) bool {
	return true
}

// notMatcher is a matcher that nots an underlying matcher. notMatcher
// implements the matcher interface.
type notMatcher struct {
	matcherToNot matcher
}

func (nm *notMatcher) match(data *rpcData) bool {
	return !nm.matcherToNot.match(data)
}

// headerMatcher is a matcher that matches on incoming HTTP Headers present
// in the incoming RPC. headerMatcher implements the matcher interface.
type headerMatcher struct {
	matcher internalmatcher.HeaderMatcher
}

func newHeaderMatcher(headerMatcherConfig *v3route_componentspb.HeaderMatcher) (*headerMatcher, error) {
	var m internalmatcher.HeaderMatcher
	switch headerMatcherConfig.HeaderMatchSpecifier.(type) {
	case *v3route_componentspb.HeaderMatcher_ExactMatch:
		m = internalmatcher.NewHeaderExactMatcher(headerMatcherConfig.Name, headerMatcherConfig.GetExactMatch(), headerMatcherConfig.InvertMatch)
	case *v3route_componentspb.HeaderMatcher_SafeRegexMatch:
		regex, err := regexp.Compile(headerMatcherConfig.GetSafeRegexMatch().Regex)
		if err != nil {
			return nil, err
		}
		m = internalmatcher.NewHeaderRegexMatcher(headerMatcherConfig.Name, regex, headerMatcherConfig.InvertMatch)
	case *v3route_componentspb.HeaderMatcher_RangeMatch:
		m = internalmatcher.NewHeaderRangeMatcher(headerMatcherConfig.Name, headerMatcherConfig.GetRangeMatch().Start, headerMatcherConfig.GetRangeMatch().End, headerMatcherConfig.InvertMatch)
	case *v3route_componentspb.HeaderMatcher_PresentMatch:
		m = internalmatcher.NewHeaderPresentMatcher(headerMatcherConfig.Name, headerMatcherConfig.GetPresentMatch(), headerMatcherConfig.InvertMatch)
	case *v3route_componentspb.HeaderMatcher_PrefixMatch:
		m = internalmatcher.NewHeaderPrefixMatcher(headerMatcherConfig.Name, headerMatcherConfig.GetPrefixMatch(), headerMatcherConfig.InvertMatch)
	case *v3route_componentspb.HeaderMatcher_SuffixMatch:
		m = internalmatcher.NewHeaderSuffixMatcher(headerMatcherConfig.Name, headerMatcherConfig.GetSuffixMatch(), headerMatcherConfig.InvertMatch)
	case *v3route_componentspb.HeaderMatcher_ContainsMatch:
		m = internalmatcher.NewHeaderContainsMatcher(headerMatcherConfig.Name, headerMatcherConfig.GetContainsMatch(), headerMatcherConfig.InvertMatch)
	case *v3route_componentspb.HeaderMatcher_StringMatch:
		sm, err := internalmatcher.StringMatcherFromProto(headerMatcherConfig.GetStringMatch())
		if err != nil {
			return nil, fmt.Errorf("invalid string matcher %+v: %v", headerMatcherConfig.GetStringMatch(), err)
		}
		m = internalmatcher.NewHeaderStringMatcher(headerMatcherConfig.Name, sm, headerMatcherConfig.InvertMatch)
	default:
		return nil, errors.New("unknown header matcher type")
	}
	return &headerMatcher{matcher: m}, nil
}

func (hm *headerMatcher) match(data *rpcData) bool {
	return hm.matcher.Match(data.md)
}

// urlPathMatcher matches on the URL Path of the incoming RPC. In gRPC, this
// logically maps to the full method name the RPC is calling on the server side.
// urlPathMatcher implements the matcher interface.
type urlPathMatcher struct {
	stringMatcher internalmatcher.StringMatcher
}

func newURLPathMatcher(pathMatcher *v3matcherpb.PathMatcher) (*urlPathMatcher, error) {
	stringMatcher, err := internalmatcher.StringMatcherFromProto(pathMatcher.GetPath())
	if err != nil {
		return nil, err
	}
	return &urlPathMatcher{stringMatcher: stringMatcher}, nil
}

func (upm *urlPathMatcher) match(data *rpcData) bool {
	return upm.stringMatcher.Match(data.fullMethod)
}

// remoteIPMatcher and localIPMatcher both are matchers that match against
// a CIDR Range. Two different matchers are needed as the remote and destination
// ip addresses come from different parts of the data about incoming RPC's
// passed in. Matching a CIDR Range means to determine whether the IP Address
// falls within the CIDR Range or not. They both implement the matcher
// interface.
type remoteIPMatcher struct {
	// ipNet represents the CidrRange that this matcher was configured with.
	// This is what will remote and destination IP's will be matched against.
	ipNet *net.IPNet
}

func newRemoteIPMatcher(cidrRange *v3corepb.CidrRange) (*remoteIPMatcher, error) {
	// Convert configuration to a cidrRangeString, as Go standard library has
	// methods that parse cidr string.
	cidrRangeString := fmt.Sprintf("%s/%d", cidrRange.AddressPrefix, cidrRange.PrefixLen.Value)
	_, ipNet, err := net.ParseCIDR(cidrRangeString)
	if err != nil {
		return nil, err
	}
	return &remoteIPMatcher{ipNet: ipNet}, nil
}

func (sim *remoteIPMatcher) match(data *rpcData) bool {
	ip, _ := netip.ParseAddr(data.peerInfo.Addr.String())
	return sim.ipNet.Contains(net.IP(ip.AsSlice()))
}

type localIPMatcher struct {
	ipNet *net.IPNet
}

func newLocalIPMatcher(cidrRange *v3corepb.CidrRange) (*localIPMatcher, error) {
	cidrRangeString := fmt.Sprintf("%s/%d", cidrRange.AddressPrefix, cidrRange.PrefixLen.Value)
	_, ipNet, err := net.ParseCIDR(cidrRangeString)
	if err != nil {
		return nil, err
	}
	return &localIPMatcher{ipNet: ipNet}, nil
}

func (dim *localIPMatcher) match(data *rpcData) bool {
	ip, _ := netip.ParseAddr(data.localAddr.String())
	return dim.ipNet.Contains(net.IP(ip.AsSlice()))
}

// portMatcher matches on whether the destination port of the RPC matches the
// destination port this matcher was instantiated with. portMatcher
// implements the matcher interface.
type portMatcher struct {
	destinationPort uint32
}

func newPortMatcher(destinationPort uint32) *portMatcher {
	return &portMatcher{destinationPort: destinationPort}
}

func (pm *portMatcher) match(data *rpcData) bool {
	return data.destinationPort == pm.destinationPort
}

// authenticatedMatcher matches on the name of the Principal. If set, the URI
// SAN or DNS SAN in that order is used from the certificate, otherwise the
// subject field is used. If unset, it applies to any user that is
// authenticated. authenticatedMatcher implements the matcher interface.
type authenticatedMatcher struct {
	stringMatcher *internalmatcher.StringMatcher
}

func newAuthenticatedMatcher(authenticatedMatcherConfig *v3rbacpb.Principal_Authenticated) (*authenticatedMatcher, error) {
	// Represents this line in the RBAC documentation = "If unset, it applies to
	// any user that is authenticated" (see package-level comments).
	if authenticatedMatcherConfig.PrincipalName == nil {
		return &authenticatedMatcher{}, nil
	}
	stringMatcher, err := internalmatcher.StringMatcherFromProto(authenticatedMatcherConfig.PrincipalName)
	if err != nil {
		return nil, err
	}
	return &authenticatedMatcher{stringMatcher: &stringMatcher}, nil
}

func (am *authenticatedMatcher) match(data *rpcData) bool {
	if data.authType != "tls" {
		// Connection is not authenticated.
		return false
	}
	if am.stringMatcher == nil {
		// Allows any authenticated user.
		return true
	}
	// "If there is no client certificate (thus no SAN nor Subject), check if ""
	// (empty string) matches. If it matches, the principal_name is said to
	// match" - A41
	if len(data.certs) == 0 {
		return am.stringMatcher.Match("")
	}
	cert := data.certs[0]
	// The order of matching as per the RBAC documentation (see package-level comments)
	// is as follows: URI SANs, DNS SANs, and then subject name.
	for _, uriSAN := range cert.URIs {
		if am.stringMatcher.Match(uriSAN.String()) {
			return true
		}
	}
	for _, dnsSAN := range cert.DNSNames {
		if am.stringMatcher.Match(dnsSAN) {
			return true
		}
	}
	return am.stringMatcher.Match(cert.Subject.String())
}
