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

// Package channelz exports internals of the channelz implementation as required
// by other gRPC packages.
//
// The implementation of the channelz spec as defined in
// https://github.com/grpc/proposal/blob/master/A14-channelz.md, is provided by
// the `internal/channelz` package.
//
// # Experimental
//
// Notice: All APIs in this package are experimental and may be removed in a
// later release.
package channelz

import "google.golang.org/grpc/internal/channelz"

// Identifier is an opaque identifier which uniquely identifies an entity in the
// channelz database.
type Identifier = channelz.Identifier
