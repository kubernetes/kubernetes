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

// Package version defines constants to distinguish between supported xDS API
// versions.
package version

// Resource URLs. We need to be able to accept either version of the resource
// regardless of the version of the transport protocol in use.
const (
	googleapiPrefix = "type.googleapis.com/"

	V3ListenerType    = "envoy.config.listener.v3.Listener"
	V3RouteConfigType = "envoy.config.route.v3.RouteConfiguration"
	V3ClusterType     = "envoy.config.cluster.v3.Cluster"
	V3EndpointsType   = "envoy.config.endpoint.v3.ClusterLoadAssignment"

	V3ResourceWrapperURL      = googleapiPrefix + "envoy.service.discovery.v3.Resource"
	V3ListenerURL             = googleapiPrefix + V3ListenerType
	V3RouteConfigURL          = googleapiPrefix + V3RouteConfigType
	V3ClusterURL              = googleapiPrefix + V3ClusterType
	V3EndpointsURL            = googleapiPrefix + V3EndpointsType
	V3HTTPConnManagerURL      = googleapiPrefix + "envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager"
	V3UpstreamTLSContextURL   = googleapiPrefix + "envoy.extensions.transport_sockets.tls.v3.UpstreamTlsContext"
	V3DownstreamTLSContextURL = googleapiPrefix + "envoy.extensions.transport_sockets.tls.v3.DownstreamTlsContext"
)
