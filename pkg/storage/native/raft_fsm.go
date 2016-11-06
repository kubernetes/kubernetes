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
	pbio "github.com/gogo/protobuf/io"
	"github.com/gogo/protobuf/proto"
	"github.com/golang/glog"
	"github.com/hashicorp/raft"
	"golang.org/x/net/context"
	"io"
	"k8s.io/kubernetes/pkg/types"
	"sort"
	"sync"
	"time"
)

const maxRecordSize = 128 * 1024 * 1024

type FSM struct {
	mutex sync.RWMutex

	lastLSN LSN

	root          *bucket
	expiryManager *expiryManager

	readableLog simpleReadableLog
}

var _ raft.FSM = &FSM{}

func (f *FSM) ReadableLog() ReadableLog {
	return &f.readableLog
}

func NewRaftFSM(log raft.LogStore) *FSM {
	f := &FSM{}
	f.root = newBucket(nil, "")
	f.readableLog.init(log)
	return f
}

type simpleReadableLog struct {
	mutex sync.Mutex
	cond  *sync.Cond

	maxLSN LSN

	applied BitSet

	log raft.LogStore
}

var _ ReadableLog = &simpleReadableLog{}

func (l *simpleReadableLog) init(log raft.LogStore) {
	l.cond = sync.NewCond(&l.mutex)
	l.log = log
}

func (l *simpleReadableLog) publishApplied(lsn LSN, wasApplied bool) {
	l.mutex.Lock()
	defer l.mutex.Unlock()

	if lsn <= l.maxLSN {
		panic("out of order LSN appends")
	}
	l.maxLSN = lsn
	l.applied.Put(uint64(lsn), wasApplied)

	l.cond.Broadcast()
}

func (l *simpleReadableLog) WaitLog(lsn LSN, dest *RaftLogEntry) (bool, error) {
	l.mutex.Lock()
	for lsn > l.maxLSN {
		l.cond.Wait()
	}
	l.mutex.Unlock()

	wasApplied := l.applied.Get(uint64(lsn))
	if !wasApplied {
		return wasApplied, nil
	}

	var log raft.Log
	err := l.log.GetLog(uint64(lsn), &log)
	if err != nil {
		return wasApplied, err
	}

	if log.Data == nil {
		panic("empty log data")
	}

	// TODO: Cache/reuse RaftLogEntry
	err = proto.Unmarshal(log.Data, dest)
	if err != nil {
		return wasApplied, err
	}

	if dest.Op == nil {
		glog.Warning("log entry had no op @%x: %v", lsn, dest.Op)
	} else if dest.Op.ItemData == nil {
		glog.Warning("log entry had no ItemData @%x: %v", lsn, dest.Op)
	} else {
		dest.Op.ItemData.Lsn = uint64(lsn)
	}

	return wasApplied, nil
}

// Apply applies a Raft log entry to the key-value store.
func (s *FSM) Apply(l *raft.Log) interface{} {
	var c RaftLogEntry
	if err := proto.Unmarshal(l.Data, &c); err != nil {
		panic(fmt.Sprintf("failed to unmarshal command: %s", err.Error()))
	}

	s.mutex.Lock()
	defer s.mutex.Unlock()

	lsn := LSN(l.Index)
	s.lastLSN = lsn

	glog.V(4).Infof("Apply %x %s %s", lsn, c.Op.OpType, c.Op.Path)

	var result *StorageOperationResult
	var wasApplied bool

	switch c.Op.OpType {
	case StorageOperationType_CREATE:
		result, wasApplied = s.opCreate(lsn, c.Op)
	case StorageOperationType_DELETE:
		result, wasApplied = s.opDelete(lsn, c.Op)
	case StorageOperationType_UPDATE:
		result, wasApplied = s.opUpdate(lsn, c.Op)
	default:
		panic(fmt.Sprintf("unrecognized command op @%x: %s", lsn, c.Op))
	}

	response := &RaftLogEntryResult{}
	response.Result = result

	// Note we send the notifications before returning ... should be interesting :-)
	s.readableLog.publishApplied(lsn, wasApplied)

	return response
}

func (s *FSM) opCreate(lsn LSN, op *StorageOperation) (*StorageOperationResult, bool) {
	bucket, k := splitPath(op.Path)

	b := s.root.resolveBucket(bucket, true)
	existing, found := b.items[k]
	if found {
		glog.V(4).Infof("response %s %s: ALREADY_EXISTS", op.OpType, op.Path)

		return &StorageOperationResult{
			ItemData:  toProto(existing),
			ErrorCode: ErrorCode_ALREADY_EXISTS,
		}, false
	}

	expiry := uint64(0)
	if op.ItemData.Ttl != 0 {
		now := uint64(time.Now().Unix())
		expiry = now + op.ItemData.Ttl
	}

	b.items[k] = itemData{
		uid:    types.UID(op.ItemData.Uid),
		data:   op.ItemData.Data,
		lsn:    lsn,
		expiry: expiry,
	}

	if expiry != 0 && s.expiryManager != nil {
		s.expiryManager.add(op.Path, expiry)
	}

	glog.V(4).Infof("response %s %s: OK", op.OpType, op.Path)

	return &StorageOperationResult{
		ItemData: &ItemData{
			Uid:  op.ItemData.Uid,
			Data: op.ItemData.Data,
			Ttl:  op.ItemData.Ttl,
			Lsn:  uint64(lsn),
		},
		CurrentLsn: uint64(lsn),
	}, true

}

func (s *FSM) opDelete(lsn LSN, op *StorageOperation) (*StorageOperationResult, bool) {
	bucket, k := splitPath(op.Path)

	b := s.root.resolveBucket(bucket, false)
	if b == nil {
		glog.V(4).Infof("response %s %s: NOT_FOUND", op.OpType, op.Path)
		return &StorageOperationResult{ErrorCode: ErrorCode_NOT_FOUND}, false
	}
	oldItem, found := b.items[k]
	if !found {
		glog.V(4).Infof("response %s %s: NOT_FOUND", op.OpType, op.Path)
		return &StorageOperationResult{ErrorCode: ErrorCode_NOT_FOUND}, false
	}

	if op.PreconditionUid != "" && types.UID(op.PreconditionUid) != oldItem.uid {
		glog.V(4).Infof("response %s %s: PRECONDITION_NOT_MET_UID", op.OpType, op.Path)
		return &StorageOperationResult{ItemData: toProto(oldItem), ErrorCode: ErrorCode_PRECONDITION_NOT_MET_UID}, false
	}

	if op.PreconditionLsn != 0 && LSN(op.PreconditionLsn) != oldItem.lsn {
		glog.V(4).Infof("response %s %s: PRECONDITION_NOT_MET_LSN", op.OpType, op.Path)
		return &StorageOperationResult{ItemData: toProto(oldItem), ErrorCode: ErrorCode_PRECONDITION_NOT_MET_LSN}, false
	}

	delete(b.items, k)

	glog.V(4).Infof("response %s %s: OK", op.OpType, op.Path)
	return &StorageOperationResult{
		ItemData:   toProto(oldItem),
		CurrentLsn: uint64(lsn),
	}, true
}

func toProto(i itemData) *ItemData {
	pb := &ItemData{
		Uid:  string(i.uid),
		Data: i.data,
		Lsn:  uint64(i.lsn),
	}
	if i.expiry != 0 {
		now := uint64(time.Now().Unix())
		var ttl uint64
		if i.expiry <= now {
			// Don't hide the TTL, even if it has expired
			ttl = 1
		} else {
			ttl = i.expiry - now
		}
		pb.Ttl = ttl
	}
	return pb
}

// response will be the new item if we swapped,
// or the existing item if err==errorLSNMismatch
func (s *FSM) opUpdate(lsn LSN, op *StorageOperation) (*StorageOperationResult, bool) {
	bucket, k := splitPath(op.Path)

	b := s.root.resolveBucket(bucket, true)
	oldItem, found := b.items[k]
	if !found {
		glog.V(4).Infof("response %s %s: NOT_FOUND", op.OpType, op.Path)
		return &StorageOperationResult{ErrorCode: ErrorCode_NOT_FOUND}, false
	}
	if op.PreconditionLsn != 0 && LSN(op.PreconditionLsn) != oldItem.lsn {
		glog.V(4).Infof("response %s %s: PRECONDITION_NOT_MET_LSN", op.OpType, op.Path)
		return &StorageOperationResult{ItemData: toProto(oldItem), ErrorCode: ErrorCode_PRECONDITION_NOT_MET_LSN}, false
	}

	if op.PreconditionUid != "" && types.UID(op.PreconditionUid) != oldItem.uid {
		glog.V(4).Infof("response %s %s: PRECONDITION_NOT_MET_UID", op.OpType, op.Path)
		return &StorageOperationResult{ItemData: toProto(oldItem), ErrorCode: ErrorCode_PRECONDITION_NOT_MET_UID}, false
	}

	expiry := uint64(0)
	if op.ItemData.Ttl != 0 {
		now := uint64(time.Now().Unix())
		expiry = now + op.ItemData.Ttl
	}

	b.items[k] = itemData{
		uid:    types.UID(op.ItemData.Uid),
		data:   op.ItemData.Data,
		lsn:    lsn,
		expiry: expiry,
	}

	if expiry != 0 && s.expiryManager != nil {
		s.expiryManager.add(op.Path, expiry)
	}

	glog.V(4).Infof("response %s %s: OK", op.OpType, op.Path)
	return &StorageOperationResult{
		ItemData: &ItemData{
			Uid:  op.ItemData.Uid,
			Data: op.ItemData.Data,
			Ttl:  op.ItemData.Ttl,
			Lsn:  uint64(lsn),
		},
		CurrentLsn: uint64(lsn),
	}, true
}

func (s *FSM) opGet(ctx context.Context, op *StorageOperation) (*StorageOperationResult, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	bucket, k := splitPath(op.Path)

	b := s.root.resolveBucket(bucket, false)
	if b == nil {
		glog.V(4).Infof("response %s %s: NOT_FOUND (bucket)", op.OpType, op.Path)

		return &StorageOperationResult{
			ErrorCode:  ErrorCode_NOT_FOUND,
			CurrentLsn: uint64(s.lastLSN),
		}, nil
	}
	item, found := b.items[k]
	if !found {
		glog.V(4).Infof("response %s %s: NOT_FOUND", op.OpType, op.Path)

		return &StorageOperationResult{
			ErrorCode:  ErrorCode_NOT_FOUND,
			CurrentLsn: uint64(s.lastLSN),
		}, nil
	}

	resultItemData := &ItemData{
		Data: item.data,
		Uid:  string(item.uid),
		Lsn:  uint64(item.lsn),
	}
	if item.expiry != 0 {
		now := uint64(time.Now().Unix())
		var ttl uint64
		if item.expiry <= now {
			// Don't hide the TTL, even if it has expired
			ttl = 1
		} else {
			ttl = item.expiry - now
		}
		resultItemData.Ttl = ttl
	}

	glog.V(4).Infof("response %s %s (@%x): OK", op.OpType, op.Path, resultItemData.Lsn)

	return &StorageOperationResult{
		ItemData:   resultItemData,
		CurrentLsn: uint64(s.lastLSN),
	}, nil
}

func (s *FSM) rawGet(path string, dest *itemData) bool {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	bucket, k := splitPath(path)

	b := s.root.resolveBucket(bucket, false)
	if b == nil {
		return false
	}
	var found bool
	*dest, found = b.items[k]

	glog.V(4).Infof("rawGet %s: found=%v", path, found)

	return found
}

func (s *FSM) opList(ctx context.Context, op *StorageOperation) (*StorageOperationResult, error) {
	s.mutex.RLock()
	defer s.mutex.RUnlock()

	b := s.root.resolveBucket(op.Path, false)
	if b == nil {
		glog.V(4).Infof("response %s %s: NOT_FOUND", op.OpType, op.Path)
		return &StorageOperationResult{
			ErrorCode:  ErrorCode_NOT_FOUND,
			CurrentLsn: uint64(s.lastLSN),
		}, nil
	}

	result := &StorageOperationResult{
		CurrentLsn: uint64(s.lastLSN),
	}

	now := uint64(time.Now().Unix())

	// Note we always do a fully recursive list
	var countBucket func(b *bucket)
	count := 0
	countBucket = func(b *bucket) {
		count += len(b.items)
		for _, childBucket := range b.children {
			countBucket(childBucket)
		}
	}
	countBucket(b)

	result.ItemList = make([]*ItemData, 0, count)

	var processBucket func(b *bucket)
	processBucket = func(b *bucket) {
		for p, item := range b.items {
			resultItemData := &ItemData{
				Data: item.data,
				Uid:  string(item.uid),
				Lsn:  uint64(item.lsn),
				Path: p,
			}
			if item.expiry != 0 {
				var ttl uint64
				if item.expiry <= now {
					// Don't hide the TTL, even if it has expired
					ttl = 1
				} else {
					ttl = item.expiry - now
				}
				resultItemData.Ttl = ttl
			}

			result.ItemList = append(result.ItemList, resultItemData)
		}
		for _, childBucket := range b.children {
			processBucket(childBucket)
		}
	}

	processBucket(b)

	// For compatibility with etcd, we sort by path
	sort.Sort(ByPath(result.ItemList))

	// But we don't need to return the paths
	// TODO: This is annoying, particularly as we hold the lock
	for _, item := range result.ItemList {
		item.Path = ""
	}

	glog.V(4).Infof("response %s %s: %d items", op.OpType, op.Path, len(result.ItemList))

	return result, nil
}

func (s *FSM) enableExpiryManager(backend *RaftBackend, enable bool) *expiryManager {
	s.mutex.Lock()
	defer s.mutex.Unlock()

	if !enable {
		if s.expiryManager != nil {
			glog.V(4).Infof("enableExpiryManager enabled=%v", enable)
			s.expiryManager = nil
		}
	} else if s.expiryManager == nil {
		glog.V(4).Infof("enableExpiryManager enabled=%v", enable)
		var expiringItems []expiringItem

		var addFn func(b *bucket)
		addFn = func(b *bucket) {
			for k, item := range b.items {
				if item.expiry != 0 {
					path := b.path + "/" + k
					expiringItems = append(expiringItems, expiringItem{path: path, expiry: item.expiry})
				}
			}
			for _, c := range b.children {
				addFn(c)
			}
		}
		addFn(s.root)

		s.expiryManager = newExpiryManager(backend, expiringItems)
	}

	return s.expiryManager
}

func (s *FSM) Snapshot() (raft.FSMSnapshot, error) {
	glog.V(4).Infof("Snapshot")

	// Snapshot is used to support log compaction. This call should
	// return an FSMSnapshot which can be used to save a point-in-time
	// snapshot of the FSM. Apply and Snapshot are not called in multiple
	// threads, but Apply will be called concurrently with Persist. This means
	// the FSM should be implemented in a fashion that allows for concurrent
	// updates while a snapshot is happening.
	//Snapshot()(FSMSnapshot, error)

	s.mutex.RLock()
	defer s.mutex.RUnlock()

	// TODO: A more efficient bucket snapshot (like bitset?)

	count := 0
	var countFn func(b *bucket)
	countFn = func(b *bucket) {
		count += len(b.items)
		for _, c := range b.children {
			countFn(c)
		}
	}
	countFn(s.root)

	items := make([]ItemData, 0, count)
	var addFn func(b *bucket)
	addFn = func(b *bucket) {
		for k, item := range b.items {
			path := b.path + "/" + k

			itemData := ItemData{
				Path: path,
				Data: item.data,
				Uid:  string(item.uid),
				Lsn:  uint64(item.lsn),
				// TODO: create expiry field or (ab)use the ttl field
				Ttl: item.expiry,
			}
			items = append(items, itemData)
		}
		for _, c := range b.children {
			addFn(c)
		}
	}
	addFn(s.root)

	appliedSnapshot := s.readableLog.applied.Snapshot()

	return &fsmSnapshot{items: items, applied: appliedSnapshot}, nil
}

func (s *FSM) Restore(in io.ReadCloser) error {
	glog.V(4).Infof("Restore")
	//// Restore is used to restore an FSM from a snapshot. It is not called
	//// concurrently with any other command. The FSM must discard all previous
	//// state.

	s.mutex.Lock()
	defer s.mutex.Unlock()

	r := pbio.NewDelimitedReader(in, maxRecordSize)

	snapshotInfo := &SnapshotInfo{}
	err := r.ReadMsg(snapshotInfo)
	if err != nil {
		return err
	}

	var pbItem ItemData
	for i := uint64(0); i < snapshotInfo.ItemCount; i++ {
		err := r.ReadMsg(&pbItem)
		if err != nil {
			return err
		}

		bucket, k := splitPath(pbItem.Path)

		b := s.root.resolveBucket(bucket, true)

		expiry := pbItem.Ttl
		// TODO: Warn if it already exists?

		b.items[k] = itemData{
			uid:    types.UID(pbItem.Uid),
			data:   pbItem.Data,
			lsn:    LSN(pbItem.Lsn),
			expiry: expiry,
		}

		if expiry != 0 && s.expiryManager != nil {
			s.expiryManager.add(pbItem.Path, expiry)
		}
	}

	if err := s.readableLog.applied.Unmarshal(r); err != nil {
		return err
	}

	return nil
}

type fsmSnapshot struct {
	items   []ItemData
	applied *BitSetSnapshot
}

var _ raft.FSMSnapshot = &fsmSnapshot{}

func (s *fsmSnapshot) Persist(sink raft.SnapshotSink) error {
	// Persist should dump all necessary state to the WriteCloser 'sink',
	// and call sink.Close() when finished or call sink.Cancel() on error.

	err := func() error {
		w := pbio.NewDelimitedWriter(sink)
		info := &SnapshotInfo{
			ItemCount: uint64(len(s.items)),
		}
		if err := w.WriteMsg(info); err != nil {
			return err
		}

		itemCount := uint64(0)
		for i := range s.items {
			if err := w.WriteMsg(&s.items[i]); err != nil {
				return err
			}
			itemCount++
		}
		if itemCount != info.ItemCount {
			return fmt.Errorf("concurrent modification detected")
		}

		if err := s.applied.Marshal(w); err != nil {
			return err
		}

		return nil
	}()

	if err != nil {
		err2 := sink.Cancel()
		if err2 != nil {
			glog.Warningf("Error from raft snapshot cancel: %v", err2)
		}
		return err
	}

	if err := sink.Close(); err != nil {
		return err
	}

	return nil
}

func (s *fsmSnapshot) Release() {
	// Release is invoked when we are finished with the snapshot.
	s.applied.Release()
}
