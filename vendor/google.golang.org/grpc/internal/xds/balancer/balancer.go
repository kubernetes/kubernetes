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

// Package balancer installs all the xds balancers.
package balancer

import (
	_ "google.golang.org/grpc/balancer/leastrequest"                  // Register the least_request_experimental balancer
	_ "google.golang.org/grpc/balancer/weightedtarget"                // Register the weighted_target balancer
	_ "google.golang.org/grpc/internal/xds/balancer/cdsbalancer"      // Register the CDS balancer
	_ "google.golang.org/grpc/internal/xds/balancer/clusterimpl"      // Register the xds_cluster_impl balancer
	_ "google.golang.org/grpc/internal/xds/balancer/clustermanager"   // Register the xds_cluster_manager balancer
	_ "google.golang.org/grpc/internal/xds/balancer/clusterresolver"  // Register the xds_cluster_resolver balancer
	_ "google.golang.org/grpc/internal/xds/balancer/outlierdetection" // Register the outlier_detection balancer
	_ "google.golang.org/grpc/internal/xds/balancer/priority"         // Register the priority balancer
)
