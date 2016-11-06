/*
Copyright 2016 The Kubernetes Authors.

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

package native

import (
	"fmt"
	"github.com/gogo/protobuf/proto"
	"github.com/golang/glog"
	"github.com/hashicorp/raft"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/storage"
	"strings"
	"time"
)

const (
	DefaultRaftTimeout = 10 * time.Second
)

type RaftBackend struct {
	//log           *changeLog
	//expiryManager *expiryManager

	//mutex         sync.RWMutex
	//root          *bucket
	//lastLSN LSN

	fsm  *FSM
	raft *raft.Raft

	readableLog ReadableLog

	raftTimeout time.Duration
}

func NewRaftBackend(raft *raft.Raft, fsm *FSM) *RaftBackend {
	b := &RaftBackend{
		raftTimeout: DefaultRaftTimeout,
		fsm:         fsm,
		raft:        raft,
		readableLog: fsm.ReadableLog(),
	}

	//// Create the log store and stable store.
	//logStore, err := raftboltdb.NewBoltStore(filepath.Join(s.RaftDir, "raft.db"))
	//if err != nil {
	//	return fmt.Errorf("new bolt store: %s", err)
	//}

	go b.runExpirations()

	return b
}

var _ StorageServiceServer = &RaftBackend{}

func (s *RaftBackend) DoOperation(ctx context.Context, op *StorageOperation) (result *StorageOperationResult, err error) {
	glog.V(2).Infof("request %s %s", op.OpType, op.Path)
	switch op.OpType {
	case StorageOperationType_DELETE:
		result, err = s.opDelete(ctx, op)

	case StorageOperationType_GET:
		result, err = s.opGet(ctx, op)

	case StorageOperationType_LIST:
		result, err = s.opList(ctx, op)

	default:
		result, err = s.forwardOrApplyOperation(ctx, op)
	}

	if err != nil {
		glog.Warningf("returning error from operation: %v", err)
	}
	return result, err
}

func (s *RaftBackend) Watch(request *WatchRequest, sink StorageService_WatchServer) error {
	glog.V(2).Infof("Watch on %s", request.Path)
	path := request.Path
	recursive := request.Recursive
	if recursive {
		if !strings.HasSuffix(path, "/") {
			path += "/"
		}
	}

	position := LSN(request.StartPosition)

	for {
		entry := &RaftLogEntry{}
		wasApplied, err := s.readableLog.WaitLog(position, entry)

		if sink.Context().Err() != nil {
			glog.V(2).Infof("Watch closed: %v", sink.Context().Err())
			// TODO: I don't think we return error here
			return nil
		}

		if err != nil {
			glog.Warningf("out of range read; will return error to watch: %v", err)
			return sink.Send(&WatchEvent{
				Error: &ErrorInfo{
					Message: err.Error(),
				},
			})
		}

		if wasApplied {
			matches := true

			if request.Recursive {
				if !strings.HasPrefix(entry.Op.Path, path) {
					matches = false
				}
			} else {
				if entry.Op.Path != path {
					matches = false
				}
			}

			if matches {
				glog.V(2).Infof("Watch on %s found event: %s %s", request.Path, entry.Op.OpType, entry.Op.Path)

				err := sink.Send(&WatchEvent{
					Op: entry.Op,
				})
				if err != nil {
					return err
				}
			}
		}
		position++
	}
}

func (s *RaftBackend) forwardOrApplyOperation(ctx context.Context, op *StorageOperation) (*StorageOperationResult, error) {
	if s.raft.State() != raft.Leader {
		// TODO: proxy to leader
		return nil, fmt.Errorf("not leader")
	}

	var entry RaftLogEntry
	entry.Op = op

	b, err := proto.Marshal(&entry)
	if err != nil {
		return nil, err
	}

	f := s.raft.Apply(b, s.raftTimeout)
	err = f.Error()
	if err != nil {
		// TODO: Handle not-leader
		return nil, err
	}

	futResponse := f.Response()
	if futResponse == nil {
		panic("nil response from raft apply")
	}
	response := futResponse.(*RaftLogEntryResult)

	opResult := response.Result
	return opResult, nil
}

func (s *RaftBackend) opGet(ctx context.Context, op *StorageOperation) (*StorageOperationResult, error) {
	return s.fsm.opGet(ctx, op)
}

func (s *RaftBackend) opList(ctx context.Context, op *StorageOperation) (*StorageOperationResult, error) {
	return s.fsm.opList(ctx, op)
}

func (s *RaftBackend) opDelete(ctx context.Context, op *StorageOperation) (*StorageOperationResult, error) {
	// We rewrite this as a read-and-delete-conditional so that the log will contain the pre-image
	// This enables our log to work for watches without needing replays
	attempt := 0
	for {
		getOp := &StorageOperation{
			OpType: StorageOperationType_GET,
			Path:   op.Path,
		}
		getResult, err := s.opGet(ctx, getOp)
		if err != nil {
			glog.Warningf("unexpected error fetching pre-image for delete: %v", err)
			return nil, err
		}

		if getResult.ErrorCode != 0 {
			switch getResult.ErrorCode {
			case ErrorCode_NOT_FOUND:
				// I think it is safe to return immediately here; a concurrent create isn't
				// yet visible, which means we can't have ACKed it.
				return &StorageOperationResult{
					ErrorCode:  getResult.ErrorCode,
					CurrentLsn: getResult.CurrentLsn,
				}, nil

			default:
				return nil, fmt.Errorf("unexpected error code: %v", getResult.ErrorCode)
			}
		}

		if op.PreconditionUid != "" {
			if op.PreconditionUid != getResult.ItemData.Uid {
				errMsg := fmt.Sprintf("Precondition failed: UID in precondition: %v, UID in object meta: %v", op.PreconditionUid, getResult.ItemData.Uid)
				return nil, storage.NewInvalidObjError(op.Path, errMsg)
			}
		}

		if op.PreconditionLsn != 0 {
			return nil, fmt.Errorf("PreconditionLsn not supported")
		}

		conditionalDelete := &StorageOperation{
			OpType:          StorageOperationType_DELETE,
			Path:            op.Path,
			PreconditionLsn: getResult.ItemData.Lsn,
		}

		deleteResult, err := s.forwardOrApplyOperation(ctx, conditionalDelete)
		if err != nil {
			return nil, err
		}

		if deleteResult.ErrorCode != 0 {
			switch deleteResult.ErrorCode {
			case ErrorCode_PRECONDITION_NOT_MET_LSN:
				// When we read the value we had the wrong value
				glog.Warningf("Delete on out-of-date value, will retry")
				// TODO: sleep?

				// Prevent infinite recursion
				if attempt > 100 {
					glog.Warningf("Too many deletion attempts when deleting value; bailing out: %v", err)
					return nil, err
				}
				attempt++
				continue

			default:
				return nil, fmt.Errorf("unexpected error code: %v", getResult.ErrorCode)
			}
		}

		return deleteResult, err
	}
}

func (s *RaftBackend) runExpirations() {
	for {
		time.Sleep(1 * time.Second)

		if s.raft.State() != raft.Leader {
			s.fsm.enableExpiryManager(s, false)
			continue
		}

		// We could wait until the first expiry, but this feels cheap enough (?)

		expiryManager := s.fsm.enableExpiryManager(s, true)
		expiryManager.runOnce()
	}
}

func (s *RaftBackend) checkExpiration(now uint64, candidates []expiringItem) error {
	if s.raft.State() != raft.Leader {
		return fmt.Errorf("not leader")
	}

	if len(candidates) == 0 {
		return nil
	}

	var entries []*RaftLogEntry

	var itemData itemData

	for i := range candidates {
		candidate := &candidates[i]

		found := s.fsm.rawGet(candidate.path, &itemData)
		if !found {
			continue
		}

		if itemData.expiry > now {
			continue
		}

		op := &StorageOperation{
			OpType:          StorageOperationType_DELETE,
			Path:            candidate.path,
			PreconditionLsn: uint64(itemData.lsn),
		}
		entry := &RaftLogEntry{
			Op: op,
		}
		entries = append(entries, entry)
	}

	for _, entry := range entries {
		// We only ever want to expire on the leader, so we don't forward
		entryBytes, err := proto.Marshal(entry)
		if err != nil {
			return err
		}

		f := s.raft.Apply(entryBytes, s.raftTimeout)
		err = f.Error()
		if err != nil {
			return err
		}

		// TODO: Do we care if delete failed (e.g. if LSN is out of date)?  Probably not...
	}

	return nil
}
