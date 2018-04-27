// Copyright 2017 The etcd Authors
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

package clientv3

import "golang.org/x/net/context"

// TODO: remove this when "FailFast=false" is fixed.
// See https://github.com/grpc/grpc-go/issues/1532.
func readyWait(rpcCtx, clientCtx context.Context, ready <-chan struct{}) error {
	select {
	case <-ready:
		return nil
	case <-rpcCtx.Done():
		return rpcCtx.Err()
	case <-clientCtx.Done():
		return clientCtx.Err()
	}
}
