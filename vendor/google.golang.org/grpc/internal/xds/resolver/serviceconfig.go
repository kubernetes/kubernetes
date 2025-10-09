/*
 *
 * Copyright 2020 gRPC authors.
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

package resolver

import (
	"context"
	"encoding/json"
	"fmt"
	"math/bits"
	rand "math/rand/v2"
	"strings"
	"sync/atomic"
	"time"

	xxhash "github.com/cespare/xxhash/v2"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/internal/grpcutil"
	iresolver "google.golang.org/grpc/internal/resolver"
	iringhash "google.golang.org/grpc/internal/ringhash"
	"google.golang.org/grpc/internal/serviceconfig"
	"google.golang.org/grpc/internal/wrr"
	"google.golang.org/grpc/internal/xds/balancer/clustermanager"
	"google.golang.org/grpc/internal/xds/httpfilter"
	"google.golang.org/grpc/internal/xds/xdsclient/xdsresource"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

const (
	cdsName                      = "cds_experimental"
	xdsClusterManagerName        = "xds_cluster_manager_experimental"
	clusterPrefix                = "cluster:"
	clusterSpecifierPluginPrefix = "cluster_specifier_plugin:"
)

type serviceConfig struct {
	LoadBalancingConfig balancerConfig `json:"loadBalancingConfig"`
}

type balancerConfig []map[string]any

func newBalancerConfig(name string, config any) balancerConfig {
	return []map[string]any{{name: config}}
}

type cdsBalancerConfig struct {
	Cluster string `json:"cluster"`
}

type xdsChildConfig struct {
	ChildPolicy balancerConfig `json:"childPolicy"`
}

type xdsClusterManagerConfig struct {
	Children map[string]xdsChildConfig `json:"children"`
}

// serviceConfigJSON produces a service config in JSON format that contains LB
// policy config for the "xds_cluster_manager" LB policy, with entries in the
// children map for all active clusters.
func serviceConfigJSON(activeClusters map[string]*clusterInfo) []byte {
	// Generate children (all entries in activeClusters).
	children := make(map[string]xdsChildConfig)
	for cluster, ci := range activeClusters {
		children[cluster] = ci.cfg
	}

	sc := serviceConfig{
		LoadBalancingConfig: newBalancerConfig(
			xdsClusterManagerName, xdsClusterManagerConfig{Children: children},
		),
	}

	// This is not expected to fail as we have constructed the service config by
	// hand right above, and therefore ok to panic.
	bs, err := json.Marshal(sc)
	if err != nil {
		panic(fmt.Sprintf("failed to marshal service config %+v: %v", sc, err))
	}
	return bs
}

type virtualHost struct {
	// map from filter name to its config
	httpFilterConfigOverride map[string]httpfilter.FilterConfig
	// retry policy present in virtual host
	retryConfig *xdsresource.RetryConfig
}

// routeCluster holds information about a cluster as referenced by a route.
type routeCluster struct {
	name string
	// map from filter name to its config
	httpFilterConfigOverride map[string]httpfilter.FilterConfig
}

type route struct {
	m                 *xdsresource.CompositeMatcher // converted from route matchers
	actionType        xdsresource.RouteActionType   // holds route action type
	clusters          wrr.WRR                       // holds *routeCluster entries
	maxStreamDuration time.Duration
	// map from filter name to its config
	httpFilterConfigOverride map[string]httpfilter.FilterConfig
	retryConfig              *xdsresource.RetryConfig
	hashPolicies             []*xdsresource.HashPolicy
}

func (r route) String() string {
	return fmt.Sprintf("%s -> { clusters: %v, maxStreamDuration: %v }", r.m.String(), r.clusters, r.maxStreamDuration)
}

// stoppableConfigSelector extends the iresolver.ConfigSelector interface with a
// stop() method. This makes it possible to swap the current config selector
// with an erroring config selector when the LDS or RDS resource is not found on
// the management server.
type stoppableConfigSelector interface {
	iresolver.ConfigSelector
	stop()
}

// erroringConfigSelector always returns an error, with the xDS node ID included
// in the error message. It is used to swap out the current config selector
// when the LDS or RDS resource is not found on the management server.
type erroringConfigSelector struct {
	err error
}

func newErroringConfigSelector(err error, xdsNodeID string) *erroringConfigSelector {
	return &erroringConfigSelector{err: annotateErrorWithNodeID(status.Error(codes.Unavailable, err.Error()), xdsNodeID)}
}

func (cs *erroringConfigSelector) SelectConfig(iresolver.RPCInfo) (*iresolver.RPCConfig, error) {
	return nil, cs.err
}
func (cs *erroringConfigSelector) stop() {}

type configSelector struct {
	r                *xdsResolver
	xdsNodeID        string
	virtualHost      virtualHost
	routes           []route
	clusters         map[string]*clusterInfo
	httpFilterConfig []xdsresource.HTTPFilter
}

var errNoMatchedRouteFound = status.Errorf(codes.Unavailable, "no matched route was found")
var errUnsupportedClientRouteAction = status.Errorf(codes.Unavailable, "matched route does not have a supported route action type")

// annotateErrorWithNodeID annotates the given error with the provided xDS node
// ID. This is used by the real config selector when it runs into errors, and
// also by the erroring config selector.
func annotateErrorWithNodeID(err error, nodeID string) error {
	return fmt.Errorf("[xDS node id: %s]: %w", nodeID, err)
}

func (cs *configSelector) SelectConfig(rpcInfo iresolver.RPCInfo) (*iresolver.RPCConfig, error) {
	var rt *route
	// Loop through routes in order and select first match.
	for _, r := range cs.routes {
		if r.m.Match(rpcInfo) {
			rt = &r
			break
		}
	}

	if rt == nil || rt.clusters == nil {
		return nil, annotateErrorWithNodeID(errNoMatchedRouteFound, cs.xdsNodeID)
	}

	if rt.actionType != xdsresource.RouteActionRoute {
		return nil, annotateErrorWithNodeID(errUnsupportedClientRouteAction, cs.xdsNodeID)
	}

	cluster, ok := rt.clusters.Next().(*routeCluster)
	if !ok {
		return nil, annotateErrorWithNodeID(status.Errorf(codes.Internal, "error retrieving cluster for match: %v (%T)", cluster, cluster), cs.xdsNodeID)
	}

	// Add a ref to the selected cluster, as this RPC needs this cluster until
	// it is committed.
	ref := &cs.clusters[cluster.name].refCount
	atomic.AddInt32(ref, 1)

	interceptor, err := cs.newInterceptor(rt, cluster)
	if err != nil {
		return nil, annotateErrorWithNodeID(err, cs.xdsNodeID)
	}

	lbCtx := clustermanager.SetPickedCluster(rpcInfo.Context, cluster.name)
	lbCtx = iringhash.SetXDSRequestHash(lbCtx, cs.generateHash(rpcInfo, rt.hashPolicies))

	config := &iresolver.RPCConfig{
		// Communicate to the LB policy the chosen cluster and request hash, if Ring Hash LB policy.
		Context: lbCtx,
		OnCommitted: func() {
			// When the RPC is committed, the cluster is no longer required.
			// Decrease its ref.
			if v := atomic.AddInt32(ref, -1); v == 0 {
				// This entry will be removed from activeClusters when
				// producing the service config for the empty update.
				cs.r.serializer.TrySchedule(func(context.Context) {
					cs.r.onClusterRefDownToZero()
				})
			}
		},
		Interceptor: interceptor,
	}

	if rt.maxStreamDuration != 0 {
		config.MethodConfig.Timeout = &rt.maxStreamDuration
	}
	if rt.retryConfig != nil {
		config.MethodConfig.RetryPolicy = retryConfigToPolicy(rt.retryConfig)
	} else if cs.virtualHost.retryConfig != nil {
		config.MethodConfig.RetryPolicy = retryConfigToPolicy(cs.virtualHost.retryConfig)
	}

	return config, nil
}

func retryConfigToPolicy(config *xdsresource.RetryConfig) *serviceconfig.RetryPolicy {
	return &serviceconfig.RetryPolicy{
		MaxAttempts:          int(config.NumRetries) + 1,
		InitialBackoff:       config.RetryBackoff.BaseInterval,
		MaxBackoff:           config.RetryBackoff.MaxInterval,
		BackoffMultiplier:    2,
		RetryableStatusCodes: config.RetryOn,
	}
}

func (cs *configSelector) generateHash(rpcInfo iresolver.RPCInfo, hashPolicies []*xdsresource.HashPolicy) uint64 {
	var hash uint64
	var generatedHash bool
	var md, emd metadata.MD
	var mdRead bool
	for _, policy := range hashPolicies {
		var policyHash uint64
		var generatedPolicyHash bool
		switch policy.HashPolicyType {
		case xdsresource.HashPolicyTypeHeader:
			if strings.HasSuffix(policy.HeaderName, "-bin") {
				continue
			}
			if !mdRead {
				md, _ = metadata.FromOutgoingContext(rpcInfo.Context)
				emd, _ = grpcutil.ExtraMetadata(rpcInfo.Context)
				mdRead = true
			}
			values := emd.Get(policy.HeaderName)
			if len(values) == 0 {
				// Extra metadata (e.g. the "content-type" header) takes
				// precedence over the user's metadata.
				values = md.Get(policy.HeaderName)
				if len(values) == 0 {
					// If the header isn't present at all, this policy is a no-op.
					continue
				}
			}
			joinedValues := strings.Join(values, ",")
			if policy.Regex != nil {
				joinedValues = policy.Regex.ReplaceAllString(joinedValues, policy.RegexSubstitution)
			}
			policyHash = xxhash.Sum64String(joinedValues)
			generatedHash = true
			generatedPolicyHash = true
		case xdsresource.HashPolicyTypeChannelID:
			// Use the static channel ID as the hash for this policy.
			policyHash = cs.r.channelID
			generatedHash = true
			generatedPolicyHash = true
		}

		// Deterministically combine the hash policies. Rotating prevents
		// duplicate hash policies from cancelling each other out and preserves
		// the 64 bits of entropy.
		if generatedPolicyHash {
			hash = bits.RotateLeft64(hash, 1)
			hash = hash ^ policyHash
		}

		// If terminal policy and a hash has already been generated, ignore the
		// rest of the policies and use that hash already generated.
		if policy.Terminal && generatedHash {
			break
		}
	}

	if generatedHash {
		return hash
	}
	// If no generated hash return a random long. In the grand scheme of things
	// this logically will map to choosing a random backend to route request to.
	return rand.Uint64()
}

func (cs *configSelector) newInterceptor(rt *route, cluster *routeCluster) (iresolver.ClientInterceptor, error) {
	if len(cs.httpFilterConfig) == 0 {
		return nil, nil
	}
	interceptors := make([]iresolver.ClientInterceptor, 0, len(cs.httpFilterConfig))
	for _, filter := range cs.httpFilterConfig {
		override := cluster.httpFilterConfigOverride[filter.Name] // cluster is highest priority
		if override == nil {
			override = rt.httpFilterConfigOverride[filter.Name] // route is second priority
		}
		if override == nil {
			override = cs.virtualHost.httpFilterConfigOverride[filter.Name] // VH is third & lowest priority
		}
		ib, ok := filter.Filter.(httpfilter.ClientInterceptorBuilder)
		if !ok {
			// Should not happen if it passed xdsClient validation.
			return nil, fmt.Errorf("filter does not support use in client")
		}
		i, err := ib.BuildClientInterceptor(filter.Config, override)
		if err != nil {
			return nil, fmt.Errorf("error constructing filter: %v", err)
		}
		if i != nil {
			interceptors = append(interceptors, i)
		}
	}
	return &interceptorList{interceptors: interceptors}, nil
}

// stop decrements refs of all clusters referenced by this config selector.
func (cs *configSelector) stop() {
	// The resolver's old configSelector may be nil.  Handle that here.
	if cs == nil {
		return
	}
	// If any refs drop to zero, we'll need a service config update to delete
	// the cluster.
	needUpdate := false
	// Loops over cs.clusters, but these are pointers to entries in
	// activeClusters.
	for _, ci := range cs.clusters {
		if v := atomic.AddInt32(&ci.refCount, -1); v == 0 {
			needUpdate = true
		}
	}
	// We stop the old config selector immediately after sending a new config
	// selector; we need another update to delete clusters from the config (if
	// we don't have another update pending already).
	if needUpdate {
		cs.r.serializer.TrySchedule(func(context.Context) {
			cs.r.onClusterRefDownToZero()
		})
	}
}

type interceptorList struct {
	interceptors []iresolver.ClientInterceptor
}

func (il *interceptorList) NewStream(ctx context.Context, ri iresolver.RPCInfo, _ func(), newStream func(ctx context.Context, _ func()) (iresolver.ClientStream, error)) (iresolver.ClientStream, error) {
	for i := len(il.interceptors) - 1; i >= 0; i-- {
		ns := newStream
		interceptor := il.interceptors[i]
		newStream = func(ctx context.Context, done func()) (iresolver.ClientStream, error) {
			return interceptor.NewStream(ctx, ri, done, ns)
		}
	}
	return newStream(ctx, func() {})
}
