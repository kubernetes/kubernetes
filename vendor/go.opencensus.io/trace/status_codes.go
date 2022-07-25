// Copyright 2018, OpenCensus Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package trace

// Status codes for use with Span.SetStatus. These correspond to the status
// codes used by gRPC defined here: https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto
const (
	StatusCodeOK                 = 0
	StatusCodeCancelled          = 1
	StatusCodeUnknown            = 2
	StatusCodeInvalidArgument    = 3
	StatusCodeDeadlineExceeded   = 4
	StatusCodeNotFound           = 5
	StatusCodeAlreadyExists      = 6
	StatusCodePermissionDenied   = 7
	StatusCodeResourceExhausted  = 8
	StatusCodeFailedPrecondition = 9
	StatusCodeAborted            = 10
	StatusCodeOutOfRange         = 11
	StatusCodeUnimplemented      = 12
	StatusCodeInternal           = 13
	StatusCodeUnavailable        = 14
	StatusCodeDataLoss           = 15
	StatusCodeUnauthenticated    = 16
)
