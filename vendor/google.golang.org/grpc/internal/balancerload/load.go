/*
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
 */

// Package balancerload defines APIs to parse server loads in trailers. The
// parsed loads are sent to balancers in DoneInfo.
package balancerload

import (
	"google.golang.org/grpc/metadata"
)

// Parser converts loads from metadata into a concrete type.
type Parser interface {
	// Parse parses loads from metadata.
	Parse(md metadata.MD) any
}

var parser Parser

// SetParser sets the load parser.
//
// Not mutex-protected, should be called before any gRPC functions.
func SetParser(lr Parser) {
	parser = lr
}

// Parse calls parser.Read().
func Parse(md metadata.MD) any {
	if parser == nil {
		return nil
	}
	return parser.Parse(md)
}
