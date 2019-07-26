/*
 *
 * Copyright 2019 gRPC authors.
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

// Package serviceconfig defines types and methods for operating on gRPC
// service configs.
//
// This package is EXPERIMENTAL.
package serviceconfig

import (
	"google.golang.org/grpc/internal"
)

// Config represents an opaque data structure holding a service config.
type Config interface {
	isConfig()
}

// LoadBalancingConfig represents an opaque data structure holding a load
// balancer config.
type LoadBalancingConfig interface {
	isLoadBalancingConfig()
}

// Parse parses the JSON service config provided into an internal form or
// returns an error if the config is invalid.
func Parse(ServiceConfigJSON string) (Config, error) {
	c, err := internal.ParseServiceConfig(ServiceConfigJSON)
	if err != nil {
		return nil, err
	}
	return c.(Config), err
}
