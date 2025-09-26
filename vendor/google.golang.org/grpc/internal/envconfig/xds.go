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

package envconfig

import (
	"os"
)

const (
	// XDSBootstrapFileNameEnv is the env variable to set bootstrap file name.
	// Do not use this and read from env directly. Its value is read and kept in
	// variable XDSBootstrapFileName.
	//
	// When both bootstrap FileName and FileContent are set, FileName is used.
	XDSBootstrapFileNameEnv = "GRPC_XDS_BOOTSTRAP"
	// XDSBootstrapFileContentEnv is the env variable to set bootstrap file
	// content. Do not use this and read from env directly. Its value is read
	// and kept in variable XDSBootstrapFileContent.
	//
	// When both bootstrap FileName and FileContent are set, FileName is used.
	XDSBootstrapFileContentEnv = "GRPC_XDS_BOOTSTRAP_CONFIG"
)

var (
	// XDSBootstrapFileName holds the name of the file which contains xDS
	// bootstrap configuration. Users can specify the location of the bootstrap
	// file by setting the environment variable "GRPC_XDS_BOOTSTRAP".
	//
	// When both bootstrap FileName and FileContent are set, FileName is used.
	XDSBootstrapFileName = os.Getenv(XDSBootstrapFileNameEnv)
	// XDSBootstrapFileContent holds the content of the xDS bootstrap
	// configuration. Users can specify the bootstrap config by setting the
	// environment variable "GRPC_XDS_BOOTSTRAP_CONFIG".
	//
	// When both bootstrap FileName and FileContent are set, FileName is used.
	XDSBootstrapFileContent = os.Getenv(XDSBootstrapFileContentEnv)

	// C2PResolverTestOnlyTrafficDirectorURI is the TD URI for testing.
	C2PResolverTestOnlyTrafficDirectorURI = os.Getenv("GRPC_TEST_ONLY_GOOGLE_C2P_RESOLVER_TRAFFIC_DIRECTOR_URI")

	// XDSDualstackEndpointsEnabled is true if gRPC should read the
	// "additional addresses" in the xDS endpoint resource.
	XDSDualstackEndpointsEnabled = boolFromEnv("GRPC_EXPERIMENTAL_XDS_DUALSTACK_ENDPOINTS", true)

	// XDSSystemRootCertsEnabled is true when xDS enabled gRPC clients can use
	// the system's default root certificates for TLS certificate validation.
	// For more details, see:
	// https://github.com/grpc/proposal/blob/master/A82-xds-system-root-certs.md.
	XDSSystemRootCertsEnabled = boolFromEnv("GRPC_EXPERIMENTAL_XDS_SYSTEM_ROOT_CERTS", false)
)
