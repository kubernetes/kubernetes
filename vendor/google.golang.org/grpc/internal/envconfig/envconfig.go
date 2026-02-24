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
	// setting "GRPC_XDS_ENDPOINT_HASH_KEY_BACKWARD_COMPAT" to "true". When the
	// implementation of A76 is stable, we will flip the default value to false
	// in a subsequent release. A final release will remove this environment
	// variable, enabling the new behavior unconditionally.
	XDSEndpointHashKeyBackwardCompat = boolFromEnv("GRPC_XDS_ENDPOINT_HASH_KEY_BACKWARD_COMPAT", true)

	// RingHashSetRequestHashKey is set if the ring hash balancer can get the
	// request hash header by setting the "requestHashHeader" field, according
	// to gRFC A76. It can be enabled by setting the environment variable
	// "GRPC_EXPERIMENTAL_RING_HASH_SET_REQUEST_HASH_KEY" to "true".
	RingHashSetRequestHashKey = boolFromEnv("GRPC_EXPERIMENTAL_RING_HASH_SET_REQUEST_HASH_KEY", false)

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

	// XDSAuthorityRewrite indicates whether xDS authority rewriting is enabled.
	// This feature is defined in gRFC A81 and is enabled by setting the
	// environment variable GRPC_EXPERIMENTAL_XDS_AUTHORITY_REWRITE to "true".
	XDSAuthorityRewrite = boolFromEnv("GRPC_EXPERIMENTAL_XDS_AUTHORITY_REWRITE", false)

	// PickFirstWeightedShuffling indicates whether weighted endpoint shuffling
	// is enabled in the pick_first LB policy, as defined in gRFC A113. This
	// feature can be disabled by setting the environment variable
	// GRPC_EXPERIMENTAL_PF_WEIGHTED_SHUFFLING to "false".
	PickFirstWeightedShuffling = boolFromEnv("GRPC_EXPERIMENTAL_PF_WEIGHTED_SHUFFLING", true)
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
