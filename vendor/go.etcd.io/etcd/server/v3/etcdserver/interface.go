// Copyright 2026 The etcd Authors
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

package etcdserver

import (
	"context"
	"time"
)

func (s *EtcdServer) ApplyWait(deadline uint64) <-chan struct{} {
	return s.applyWait.Wait(deadline)
}

func (s *EtcdServer) Done() <-chan struct{} {
	return s.done
}

func (s *EtcdServer) LeaderChanged() <-chan struct{} {
	return s.leaderChanged.Receive()
}

func (s *EtcdServer) NextRequestID() uint64 {
	return s.reqIDGen.Next()
}

func (s *EtcdServer) RequestTimeout() time.Duration {
	return s.Cfg.ReqTimeout()
}

func (s *EtcdServer) Stopping() <-chan struct{} {
	return s.stopping
}

func (s *EtcdServer) LinearizableReadNotify(ctx context.Context) error {
	return s.read.LinearizableReadNotify(ctx)
}
