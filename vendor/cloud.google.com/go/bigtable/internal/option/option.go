/*
Copyright 2015 Google Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package option contains common code for dealing with client options.
package option

import (
	"fmt"
	"os"

	"google.golang.org/api/option"
	"google.golang.org/grpc"
)

// DefaultClientOptions returns the default client options to use for the
// client's gRPC connection.
func DefaultClientOptions(endpoint, scope, userAgent string) ([]option.ClientOption, error) {
	var o []option.ClientOption
	// Check the environment variables for the bigtable emulator.
	// Dial it directly and don't pass any credentials.
	if addr := os.Getenv("BIGTABLE_EMULATOR_HOST"); addr != "" {
		conn, err := grpc.Dial(addr, grpc.WithInsecure())
		if err != nil {
			return nil, fmt.Errorf("emulator grpc.Dial: %v", err)
		}
		o = []option.ClientOption{option.WithGRPCConn(conn)}
	} else {
		o = []option.ClientOption{
			option.WithEndpoint(endpoint),
			option.WithScopes(scope),
			option.WithUserAgent(userAgent),
		}
	}
	return o, nil
}
