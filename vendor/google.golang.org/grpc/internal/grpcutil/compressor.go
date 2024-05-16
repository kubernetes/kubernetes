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

package grpcutil

import (
	"strings"

	"google.golang.org/grpc/internal/envconfig"
)

// RegisteredCompressorNames holds names of the registered compressors.
var RegisteredCompressorNames []string

// IsCompressorNameRegistered returns true when name is available in registry.
func IsCompressorNameRegistered(name string) bool {
	for _, compressor := range RegisteredCompressorNames {
		if compressor == name {
			return true
		}
	}
	return false
}

// RegisteredCompressors returns a string of registered compressor names
// separated by comma.
func RegisteredCompressors() string {
	if !envconfig.AdvertiseCompressors {
		return ""
	}
	return strings.Join(RegisteredCompressorNames, ",")
}
