// Copyright 2015 CoreOS, Inc.
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

package mockstorage

import (
	"fmt"

	"github.com/coreos/etcd/pkg/testutil"
	"github.com/coreos/etcd/raft"
	"github.com/coreos/etcd/raft/raftpb"
)

type storageRecorder struct {
	testutil.Recorder
	dbPath string // must have '/' suffix if set
}

func NewStorageRecorder(db string) *storageRecorder {
	return &storageRecorder{&testutil.RecorderBuffered{}, db}
}

func NewStorageRecorderStream(db string) *storageRecorder {
	return &storageRecorder{testutil.NewRecorderStream(), db}
}

func (p *storageRecorder) Save(st raftpb.HardState, ents []raftpb.Entry) error {
	p.Record(testutil.Action{Name: "Save"})
	return nil
}

func (p *storageRecorder) SaveSnap(st raftpb.Snapshot) error {
	if !raft.IsEmptySnap(st) {
		p.Record(testutil.Action{Name: "SaveSnap"})
	}
	return nil
}

func (p *storageRecorder) DBFilePath(id uint64) (string, error) {
	p.Record(testutil.Action{Name: "DBFilePath"})
	path := p.dbPath
	if path != "" {
		path = path + "/"
	}
	return fmt.Sprintf("%s%016x.snap.db", path, id), nil
}

func (p *storageRecorder) Close() error { return nil }
