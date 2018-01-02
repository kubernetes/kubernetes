// Copyright 2015 The etcd Authors
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

package mockwait

import (
	"github.com/coreos/etcd/pkg/testutil"
	"github.com/coreos/etcd/pkg/wait"
)

type WaitRecorder struct {
	wait.Wait
	testutil.Recorder
}

type waitRecorder struct {
	testutil.RecorderBuffered
}

func NewRecorder() *WaitRecorder {
	wr := &waitRecorder{}
	return &WaitRecorder{Wait: wr, Recorder: wr}
}
func NewNop() wait.Wait { return NewRecorder() }

func (w *waitRecorder) Register(id uint64) <-chan interface{} {
	w.Record(testutil.Action{Name: "Register"})
	return nil
}
func (w *waitRecorder) Trigger(id uint64, x interface{}) {
	w.Record(testutil.Action{Name: "Trigger"})
}

func (w *waitRecorder) IsRegistered(id uint64) bool {
	panic("waitRecorder.IsRegistered() shouldn't be called")
}
