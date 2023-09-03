/*
 *
 * Copyright 2022 gRPC authors.
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

import "os"

const (
	envObservabilityConfig     = "GRPC_GCP_OBSERVABILITY_CONFIG"
	envObservabilityConfigFile = "GRPC_GCP_OBSERVABILITY_CONFIG_FILE"
)

var (
	// ObservabilityConfig is the json configuration for the gcp/observability
	// package specified directly in the envObservabilityConfig env var.
	ObservabilityConfig = os.Getenv(envObservabilityConfig)
	// ObservabilityConfigFile is the json configuration for the
	// gcp/observability specified in a file with the location specified in
	// envObservabilityConfigFile env var.
	ObservabilityConfigFile = os.Getenv(envObservabilityConfigFile)
)
