// Copyright 2020, Google LLC
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

package longrunning

import (
	"context"

	"google.golang.org/api/option"
	"google.golang.org/grpc"
)

// InternalFromConn is for use by the Google Cloud Libraries only.
//
// Deprecated. Use `NewOperationsClient(ctx, option.WithGRPCConn(conn))` instead.
func InternalFromConn(conn *grpc.ClientConn) *OperationsClient {
	c, _ := NewOperationsClient(context.Background(), option.WithGRPCConn(conn))
	return c
}
