/*
 *
 * Copyright 2025 gRPC authors.
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

// Package xdsresource defines constants to distinguish between supported xDS
// API versions.
package xdsresource

// Resource URLs. We need to be able to accept either version of the resource
// regardless of the version of the transport protocol in use.
const (
	googleapiPrefix = "type.googleapis.com/"

	V3ListenerURL        = googleapiPrefix + "envoy.config.listener.v3.Listener"
	V3HTTPConnManagerURL = googleapiPrefix + "envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager"
	V3ResourceWrapperURL = googleapiPrefix + "envoy.service.discovery.v3.Resource"
)
