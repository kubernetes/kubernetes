/*
 *
 * Copyright 2018 gRPC authors.
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

// Package envconfig contains grpc settings configured by environment variables.
package envconfig

import (
	"os"
	"strconv"
	"strings"
)

var (
	// EnableTXTServiceConfig is set if the DNS resolver should perform TXT
	// lookups for service config ("GRPC_ENABLE_TXT_SERVICE_CONFIG" is not
	// "false").
	EnableTXTServiceConfig = boolFromEnv("GRPC_ENABLE_TXT_SERVICE_CONFIG", true)

	// TXTErrIgnore is set if TXT errors should be ignored
	// ("GRPC_GO_IGNORE_TXT_ERRORS" is not "false").
	TXTErrIgnore = boolFromEnv("GRPC_GO_IGNORE_TXT_ERRORS", true)

	// RingHashCap indicates the maximum ring size which defaults to 4096
	// entries but may be overridden by setting the environment variable
	// "GRPC_RING_HASH_CAP".  This does not override the default bounds
	// checking which NACKs configs specifying ring sizes > 8*1024*1024 (~8M).
	RingHashCap = uint64FromEnv("GRPC_RING_HASH_CAP", 4096, 1, 8*1024*1024)

	// ALTSMaxConcurrentHandshakes is the maximum number of concurrent ALTS
	// handshakes that can be performed.
	ALTSMaxConcurrentHandshakes = uint64FromEnv("GRPC_ALTS_MAX_CONCURRENT_HANDSHAKES", 100, 1, 100)

	// EnforceALPNEnabled is set if TLS connections to servers with ALPN disabled
	// should be rejected. The HTTP/2 protocol requires ALPN to be enabled, this
	// option is present for backward compatibility. This option may be overridden
	// by setting the environment variable "GRPC_ENFORCE_ALPN_ENABLED" to "true"
	// or "false".
	EnforceALPNEnabled = boolFromEnv("GRPC_ENFORCE_ALPN_ENABLED", true)

	// XDSEndpointHashKeyBackwardCompat controls the parsing of the endpoint hash
	// key from EDS LbEndpoint metadata. Endpoint hash keys can be disabled by
	// setting "GRPC_XDS_ENDPOINT_HASH_KEY_BACKWARD_COMPAT" to "true". A future
	// release will remove this environment variable, enabling the new behavior
	// unconditionally.
	XDSEndpointHashKeyBackwardCompat = boolFromEnv("GRPC_XDS_ENDPOINT_HASH_KEY_BACKWARD_COMPAT", false)

	// LabelServerGoroutines controls setting [runtime/pprof.Labels] on the
	// goroutines spawned by [grpc.Server] type.
	// For now, this is limited to the goroutines spawned to handle incoming
	// requests on the server.
	// Set "GRPC_GO_SERVER_GOROUTINE_LABELS" to "grpc.method=true" to
	// enable this grpc.method label, or "all" to enable all valid labels.
	// This variable is a bit-field.
	LabelServerGoroutines = goroutineLabelsFromEnv("GRPC_GO_SERVER_GOROUTINE_LABELS", 0)

	// RingHashSetRequestHashKey is set if the ring hash balancer can get the
	// request hash header by setting the "requestHashHeader" field, according
	// to gRFC A76. It can be disabled by setting the environment variable
	// "GRPC_EXPERIMENTAL_RING_HASH_SET_REQUEST_HASH_KEY" to "false".
	RingHashSetRequestHashKey = boolFromEnv("GRPC_EXPERIMENTAL_RING_HASH_SET_REQUEST_HASH_KEY", true)

	// ALTSHandshakerKeepaliveParams is set if we should add the
	// KeepaliveParams when dial the ALTS handshaker service.
	ALTSHandshakerKeepaliveParams = boolFromEnv("GRPC_EXPERIMENTAL_ALTS_HANDSHAKER_KEEPALIVE_PARAMS", false)

	// EnableDefaultPortForProxyTarget controls whether the resolver adds a default port 443
	// to a target address that lacks one. This flag only has an effect when all of
	// the following conditions are met:
	//   - A connect proxy is being used.
	//   - Target resolution is disabled.
	//   - The DNS resolver is being used.
	EnableDefaultPortForProxyTarget = boolFromEnv("GRPC_EXPERIMENTAL_ENABLE_DEFAULT_PORT_FOR_PROXY_TARGET", true)

	// CaseSensitiveBalancerRegistries is set if the balancer registry should be
	// case-sensitive. This is enabled by default, but can be disabled by setting
	// the env variable "GRPC_GO_EXPERIMENTAL_CASE_SENSITIVE_BALANCER_REGISTRIES"
	// to "false".
	//
	// This env varible will be removed in release v1.82.0.
	CaseSensitiveBalancerRegistries = boolFromEnv("GRPC_GO_EXPERIMENTAL_CASE_SENSITIVE_BALANCER_REGISTRIES", true)

	// XDSAuthorityRewrite indicates whether xDS authority rewriting is enabled.
	// This feature is defined in gRFC A81 and is enabled by setting the
	// environment variable GRPC_EXPERIMENTAL_XDS_AUTHORITY_REWRITE to "true".
	XDSAuthorityRewrite = boolFromEnv("GRPC_EXPERIMENTAL_XDS_AUTHORITY_REWRITE", false)

	// PickFirstWeightedShuffling indicates whether weighted endpoint shuffling
	// is enabled in the pick_first LB policy, as defined in gRFC A113. This
	// feature can be disabled by setting the environment variable
	// GRPC_EXPERIMENTAL_PF_WEIGHTED_SHUFFLING to "false".
	PickFirstWeightedShuffling = boolFromEnv("GRPC_EXPERIMENTAL_PF_WEIGHTED_SHUFFLING", true)

	// XDSRecoverPanicInResourceParsing indicates whether the xdsclient should
	// recover from panics while parsing xDS resources.
	//
	// This feature can be disabled (e.g. for fuzz testing) by setting the
	// environment variable "GRPC_GO_EXPERIMENTAL_XDS_RESOURCE_PANIC_RECOVERY"
	// to "false".
	XDSRecoverPanicInResourceParsing = boolFromEnv("GRPC_GO_EXPERIMENTAL_XDS_RESOURCE_PANIC_RECOVERY", true)

	// EnablePriorityLBChildPolicyCache controls whether the priority balancer
	// should cache child balancers that are removed from the LB policy config,
	// for a period of 15 minutes. This is disabled by default, but can be
	// enabled by setting the env variable
	// GRPC_EXPERIMENTAL_ENABLE_PRIORITY_LB_CHILD_POLICY_CACHE to true.
	EnablePriorityLBChildPolicyCache = boolFromEnv("GRPC_EXPERIMENTAL_ENABLE_PRIORITY_LB_CHILD_POLICY_CACHE", false)

	// Enable8KBDefaultHeaderListSize indicates that default maximum header list
	// size is restricted to 8KB. This is disabled by default, but can be enabled
	// by setting the environment variable
	// "GRPC_GO_EXPERIMENTAL_ENABLE_8KB_DEFAULT_HEADER_LIST_SIZE" to "true".
	// When disabled, the default maximum header list size of 16MB is used.
	//
	// When enabled, RPCs with a total size of headers exceeding 8KB will fail
	// unless explicitly configured otherwise by the user.
	//
	// TODO: In release v1.82.0, env var will be enabled by default.
	Enable8KBDefaultHeaderListSize = boolFromEnv("GRPC_GO_EXPERIMENTAL_ENABLE_8KB_DEFAULT_HEADER_LIST_SIZE", false)

	// EnableHTTPFramerReadBufferPooling enables the use of the
	// readyreader.Reader interface to perform non-memory-pinning reads,
	// provided the underlying net.Conn supports it. This reduces memory usage
	// when subchannels are idle.
	//
	// This environment variable serves as an escape hatch to disable the
	// feature if unforeseen issues arise, and it will be removed in a future
	// release.
	EnableHTTPFramerReadBufferPooling = boolFromEnv("GRPC_GO_EXPERIMENTAL_HTTP_FRAMER_READ_BUFFER_POOLING", true)

	// ControlBufferThrottleLimit is the maximum number of control frames that can
	// be queued in the control buffer before throttling is applied. The value
	// must be between 1 and 10,000, and is set to 100 by default.
	//
	// This environment variable serves as an escape hatch to increase the
	// throttling limit if unforeseen issues arise, and it will be removed in a
	// future release.
	//
	// TODO: Remove this env var once v1.83.0 is release.
	ControlBufferThrottleLimit = uint64FromEnv("GRPC_GO_EXPERIMENTAL_CONTROL_BUFFER_THROTTLE_LIMIT", 100, 1, 10000)
)

func boolFromEnv(envVar string, def bool) bool {
	if def {
		// The default is true; return true unless the variable is "false".
		return !strings.EqualFold(os.Getenv(envVar), "false")
	}
	// The default is false; return false unless the variable is "true".
	return strings.EqualFold(os.Getenv(envVar), "true")
}

func uint64FromEnv(envVar string, def, min, max uint64) uint64 {
	v, err := strconv.ParseUint(os.Getenv(envVar), 10, 64)
	if err != nil {
		return def
	}
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// GoroutineLabels is a bitfield indicating which goroutine labels are enabled.
type GoroutineLabels uint16

func goroutineLabelsFromEnv(envVar string, def GoroutineLabels) GoroutineLabels {
	val := def
	v := os.Getenv(envVar)
	if strings.EqualFold(v, "all") {
		return AllGoroutineLabels
	} else if strings.EqualFold(v, "none") {
		return 0
	}
	for s := range strings.SplitSeq(v, ",") {
		s = strings.TrimSpace(s)
		if len(s) == 0 {
			continue
		}
		pre, post, ok := strings.Cut(s, "=")
		if !ok {
			// no equals sign
			continue
		}
		post = strings.TrimSpace(post)
		pre = strings.TrimSpace(pre)
		bitDesignator := GoroutineLabels(0)
		switch {
		case strings.EqualFold(pre, "grpc.method"):
			bitDesignator = GoroutineLabelServerMethod
		default:
			continue
		}
		if strings.EqualFold(post, "true") {
			val |= bitDesignator
		} else if strings.EqualFold(post, "false") {
			val &^= bitDesignator
		}
	}
	return val
}

const (
	// GoroutineLabelServerMethod sets the grpc.method label on new
	// server-side gRPC streams.
	GoroutineLabelServerMethod GoroutineLabels = 1 << iota
)

// AllGoroutineLabels is an or'd together bitfield of all valid GoroutineLabels
// constant values (above).
const AllGoroutineLabels = GoroutineLabelServerMethod
