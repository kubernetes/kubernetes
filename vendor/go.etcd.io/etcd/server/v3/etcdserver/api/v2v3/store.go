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

package v2v3

import (
	"context"
	"fmt"
	"path"
	"sort"
	"strings"
	"time"

	"go.etcd.io/etcd/api/v3/mvccpb"
	"go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/concurrency"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2error"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v2store"
)

// store implements the Store interface for V2 using
// a v3 client.
type v2v3Store struct {
	c *clientv3.Client
	// pfx is the v3 prefix where keys should be stored.
	pfx string
	ctx context.Context
}

const maxPathDepth = 63

var errUnsupported = fmt.Errorf("TTLs are unsupported")

func NewStore(c *clientv3.Client, pfx string) v2store.Store { return newStore(c, pfx) }

func newStore(c *clientv3.Client, pfx string) *v2v3Store { return &v2v3Store{c, pfx, c.Ctx()} }

func (s *v2v3Store) Index() uint64 { panic("STUB") }

func (s *v2v3Store) Get(nodePath string, recursive, sorted bool) (*v2store.Event, error) {
	key := s.mkPath(nodePath)
	resp, err := s.c.Txn(s.ctx).Then(
		clientv3.OpGet(key+"/"),
		clientv3.OpGet(key),
	).Commit()
	if err != nil {
		return nil, err
	}

	if kvs := resp.Responses[0].GetResponseRange().Kvs; len(kvs) != 0 || isRoot(nodePath) {
		nodes, err := s.getDir(nodePath, recursive, sorted, resp.Header.Revision)
		if err != nil {
			return nil, err
		}
		cidx, midx := uint64(0), uint64(0)
		if len(kvs) > 0 {
			cidx, midx = mkV2Rev(kvs[0].CreateRevision), mkV2Rev(kvs[0].ModRevision)
		}
		return &v2store.Event{
			Action: v2store.Get,
			Node: &v2store.NodeExtern{
				Key:           nodePath,
				Dir:           true,
				Nodes:         nodes,
				CreatedIndex:  cidx,
				ModifiedIndex: midx,
			},
			EtcdIndex: mkV2Rev(resp.Header.Revision),
		}, nil
	}

	kvs := resp.Responses[1].GetResponseRange().Kvs
	if len(kvs) == 0 {
		return nil, v2error.NewError(v2error.EcodeKeyNotFound, nodePath, mkV2Rev(resp.Header.Revision))
	}

	return &v2store.Event{
		Action:    v2store.Get,
		Node:      s.mkV2Node(kvs[0]),
		EtcdIndex: mkV2Rev(resp.Header.Revision),
	}, nil
}

func (s *v2v3Store) getDir(nodePath string, recursive, sorted bool, rev int64) ([]*v2store.NodeExtern, error) {
	rootNodes, err := s.getDirDepth(nodePath, 1, rev)
	if err != nil || !recursive {
		if sorted {
			sort.Sort(v2store.NodeExterns(rootNodes))
		}
		return rootNodes, err
	}
	nextNodes := rootNodes
	nodes := make(map[string]*v2store.NodeExtern)
	// Breadth walk the subdirectories
	for i := 2; len(nextNodes) > 0; i++ {
		for _, n := range nextNodes {
			nodes[n.Key] = n
			if parent := nodes[path.Dir(n.Key)]; parent != nil {
				parent.Nodes = append(parent.Nodes, n)
			}
		}
		if nextNodes, err = s.getDirDepth(nodePath, i, rev); err != nil {
			return nil, err
		}
	}

	if sorted {
		sort.Sort(v2store.NodeExterns(rootNodes))
	}
	return rootNodes, nil
}

func (s *v2v3Store) getDirDepth(nodePath string, depth int, rev int64) ([]*v2store.NodeExtern, error) {
	pd := s.mkPathDepth(nodePath, depth)
	resp, err := s.c.Get(s.ctx, pd, clientv3.WithPrefix(), clientv3.WithRev(rev))
	if err != nil {
		return nil, err
	}

	nodes := make([]*v2store.NodeExtern, len(resp.Kvs))
	for i, kv := range resp.Kvs {
		nodes[i] = s.mkV2Node(kv)
	}
	return nodes, nil
}

func (s *v2v3Store) Set(
	nodePath string,
	dir bool,
	value string,
	expireOpts v2store.TTLOptionSet,
) (*v2store.Event, error) {
	if expireOpts.Refresh || !expireOpts.ExpireTime.IsZero() {
		return nil, errUnsupported
	}

	if isRoot(nodePath) {
		return nil, v2error.NewError(v2error.EcodeRootROnly, nodePath, 0)
	}

	ecode := 0
	applyf := func(stm concurrency.STM) error {
		// build path if any directories in path do not exist
		dirs := []string{}
		for p := path.Dir(nodePath); !isRoot(p); p = path.Dir(p) {
			pp := s.mkPath(p)
			if stm.Rev(pp) > 0 {
				ecode = v2error.EcodeNotDir
				return nil
			}
			if stm.Rev(pp+"/") == 0 {
				dirs = append(dirs, pp+"/")
			}
		}
		for _, d := range dirs {
			stm.Put(d, "")
		}

		key := s.mkPath(nodePath)
		if dir {
			if stm.Rev(key) != 0 {
				// exists as non-dir
				ecode = v2error.EcodeNotDir
				return nil
			}
			key = key + "/"
		} else if stm.Rev(key+"/") != 0 {
			ecode = v2error.EcodeNotFile
			return nil
		}
		stm.Put(key, value, clientv3.WithPrevKV())
		stm.Put(s.mkActionKey(), v2store.Set)
		return nil
	}

	resp, err := s.newSTM(applyf)
	if err != nil {
		return nil, err
	}
	if ecode != 0 {
		return nil, v2error.NewError(ecode, nodePath, mkV2Rev(resp.Header.Revision))
	}

	createRev := resp.Header.Revision
	var pn *v2store.NodeExtern
	if pkv := prevKeyFromPuts(resp); pkv != nil {
		pn = s.mkV2Node(pkv)
		createRev = pkv.CreateRevision
	}

	vp := &value
	if dir {
		vp = nil
	}
	return &v2store.Event{
		Action: v2store.Set,
		Node: &v2store.NodeExtern{
			Key:           nodePath,
			Value:         vp,
			Dir:           dir,
			ModifiedIndex: mkV2Rev(resp.Header.Revision),
			CreatedIndex:  mkV2Rev(createRev),
		},
		PrevNode:  pn,
		EtcdIndex: mkV2Rev(resp.Header.Revision),
	}, nil
}

func (s *v2v3Store) Update(nodePath, newValue string, expireOpts v2store.TTLOptionSet) (*v2store.Event, error) {
	if isRoot(nodePath) {
		return nil, v2error.NewError(v2error.EcodeRootROnly, nodePath, 0)
	}

	if expireOpts.Refresh || !expireOpts.ExpireTime.IsZero() {
		return nil, errUnsupported
	}

	key := s.mkPath(nodePath)
	ecode := 0
	applyf := func(stm concurrency.STM) error {
		if rev := stm.Rev(key + "/"); rev != 0 {
			ecode = v2error.EcodeNotFile
			return nil
		}
		if rev := stm.Rev(key); rev == 0 {
			ecode = v2error.EcodeKeyNotFound
			return nil
		}
		stm.Put(key, newValue, clientv3.WithPrevKV())
		stm.Put(s.mkActionKey(), v2store.Update)
		return nil
	}

	resp, err := s.newSTM(applyf)
	if err != nil {
		return nil, err
	}
	if ecode != 0 {
		return nil, v2error.NewError(v2error.EcodeNotFile, nodePath, mkV2Rev(resp.Header.Revision))
	}

	pkv := prevKeyFromPuts(resp)
	return &v2store.Event{
		Action: v2store.Update,
		Node: &v2store.NodeExtern{
			Key:           nodePath,
			Value:         &newValue,
			ModifiedIndex: mkV2Rev(resp.Header.Revision),
			CreatedIndex:  mkV2Rev(pkv.CreateRevision),
		},
		PrevNode:  s.mkV2Node(pkv),
		EtcdIndex: mkV2Rev(resp.Header.Revision),
	}, nil
}

func (s *v2v3Store) Create(
	nodePath string,
	dir bool,
	value string,
	unique bool,
	expireOpts v2store.TTLOptionSet,
) (*v2store.Event, error) {
	if isRoot(nodePath) {
		return nil, v2error.NewError(v2error.EcodeRootROnly, nodePath, 0)
	}
	if expireOpts.Refresh || !expireOpts.ExpireTime.IsZero() {
		return nil, errUnsupported
	}
	ecode := 0
	applyf := func(stm concurrency.STM) error {
		ecode = 0
		key := s.mkPath(nodePath)
		if unique {
			// append unique item under the node path
			for {
				key = nodePath + "/" + fmt.Sprintf("%020s", time.Now())
				key = path.Clean(path.Join("/", key))
				key = s.mkPath(key)
				if stm.Rev(key) == 0 {
					break
				}
			}
		}
		if stm.Rev(key) > 0 || stm.Rev(key+"/") > 0 {
			ecode = v2error.EcodeNodeExist
			return nil
		}
		// build path if any directories in path do not exist
		dirs := []string{}
		for p := path.Dir(nodePath); !isRoot(p); p = path.Dir(p) {
			pp := s.mkPath(p)
			if stm.Rev(pp) > 0 {
				ecode = v2error.EcodeNotDir
				return nil
			}
			if stm.Rev(pp+"/") == 0 {
				dirs = append(dirs, pp+"/")
			}
		}
		for _, d := range dirs {
			stm.Put(d, "")
		}

		if dir {
			// directories marked with extra slash in key name
			key += "/"
		}
		stm.Put(key, value)
		stm.Put(s.mkActionKey(), v2store.Create)
		return nil
	}

	resp, err := s.newSTM(applyf)
	if err != nil {
		return nil, err
	}
	if ecode != 0 {
		return nil, v2error.NewError(ecode, nodePath, mkV2Rev(resp.Header.Revision))
	}

	var v *string
	if !dir {
		v = &value
	}

	return &v2store.Event{
		Action: v2store.Create,
		Node: &v2store.NodeExtern{
			Key:           nodePath,
			Value:         v,
			Dir:           dir,
			ModifiedIndex: mkV2Rev(resp.Header.Revision),
			CreatedIndex:  mkV2Rev(resp.Header.Revision),
		},
		EtcdIndex: mkV2Rev(resp.Header.Revision),
	}, nil
}

func (s *v2v3Store) CompareAndSwap(
	nodePath string,
	prevValue string,
	prevIndex uint64,
	value string,
	expireOpts v2store.TTLOptionSet,
) (*v2store.Event, error) {
	if isRoot(nodePath) {
		return nil, v2error.NewError(v2error.EcodeRootROnly, nodePath, 0)
	}
	if expireOpts.Refresh || !expireOpts.ExpireTime.IsZero() {
		return nil, errUnsupported
	}

	key := s.mkPath(nodePath)
	resp, err := s.c.Txn(s.ctx).If(
		s.mkCompare(nodePath, prevValue, prevIndex)...,
	).Then(
		clientv3.OpPut(key, value, clientv3.WithPrevKV()),
		clientv3.OpPut(s.mkActionKey(), v2store.CompareAndSwap),
	).Else(
		clientv3.OpGet(key),
		clientv3.OpGet(key+"/"),
	).Commit()

	if err != nil {
		return nil, err
	}
	if !resp.Succeeded {
		return nil, compareFail(nodePath, prevValue, prevIndex, resp)
	}

	pkv := resp.Responses[0].GetResponsePut().PrevKv
	return &v2store.Event{
		Action: v2store.CompareAndSwap,
		Node: &v2store.NodeExtern{
			Key:           nodePath,
			Value:         &value,
			CreatedIndex:  mkV2Rev(pkv.CreateRevision),
			ModifiedIndex: mkV2Rev(resp.Header.Revision),
		},
		PrevNode:  s.mkV2Node(pkv),
		EtcdIndex: mkV2Rev(resp.Header.Revision),
	}, nil
}

func (s *v2v3Store) Delete(nodePath string, dir, recursive bool) (*v2store.Event, error) {
	if isRoot(nodePath) {
		return nil, v2error.NewError(v2error.EcodeRootROnly, nodePath, 0)
	}
	if !dir && !recursive {
		return s.deleteNode(nodePath)
	}
	if !recursive {
		return s.deleteEmptyDir(nodePath)
	}

	dels := make([]clientv3.Op, maxPathDepth+1)
	dels[0] = clientv3.OpDelete(s.mkPath(nodePath)+"/", clientv3.WithPrevKV())
	for i := 1; i < maxPathDepth; i++ {
		dels[i] = clientv3.OpDelete(s.mkPathDepth(nodePath, i), clientv3.WithPrefix())
	}
	dels[maxPathDepth] = clientv3.OpPut(s.mkActionKey(), v2store.Delete)

	resp, err := s.c.Txn(s.ctx).If(
		clientv3.Compare(clientv3.Version(s.mkPath(nodePath)+"/"), ">", 0),
		clientv3.Compare(clientv3.Version(s.mkPathDepth(nodePath, maxPathDepth)+"/"), "=", 0),
	).Then(
		dels...,
	).Commit()
	if err != nil {
		return nil, err
	}
	if !resp.Succeeded {
		return nil, v2error.NewError(v2error.EcodeNodeExist, nodePath, mkV2Rev(resp.Header.Revision))
	}
	dresp := resp.Responses[0].GetResponseDeleteRange()
	return &v2store.Event{
		Action:    v2store.Delete,
		PrevNode:  s.mkV2Node(dresp.PrevKvs[0]),
		EtcdIndex: mkV2Rev(resp.Header.Revision),
	}, nil
}

func (s *v2v3Store) deleteEmptyDir(nodePath string) (*v2store.Event, error) {
	resp, err := s.c.Txn(s.ctx).If(
		clientv3.Compare(clientv3.Version(s.mkPathDepth(nodePath, 1)), "=", 0).WithPrefix(),
	).Then(
		clientv3.OpDelete(s.mkPath(nodePath)+"/", clientv3.WithPrevKV()),
		clientv3.OpPut(s.mkActionKey(), v2store.Delete),
	).Commit()
	if err != nil {
		return nil, err
	}
	if !resp.Succeeded {
		return nil, v2error.NewError(v2error.EcodeDirNotEmpty, nodePath, mkV2Rev(resp.Header.Revision))
	}
	dresp := resp.Responses[0].GetResponseDeleteRange()
	if len(dresp.PrevKvs) == 0 {
		return nil, v2error.NewError(v2error.EcodeNodeExist, nodePath, mkV2Rev(resp.Header.Revision))
	}
	return &v2store.Event{
		Action:    v2store.Delete,
		PrevNode:  s.mkV2Node(dresp.PrevKvs[0]),
		EtcdIndex: mkV2Rev(resp.Header.Revision),
	}, nil
}

func (s *v2v3Store) deleteNode(nodePath string) (*v2store.Event, error) {
	resp, err := s.c.Txn(s.ctx).If(
		clientv3.Compare(clientv3.Version(s.mkPath(nodePath)+"/"), "=", 0),
	).Then(
		clientv3.OpDelete(s.mkPath(nodePath), clientv3.WithPrevKV()),
		clientv3.OpPut(s.mkActionKey(), v2store.Delete),
	).Commit()
	if err != nil {
		return nil, err
	}
	if !resp.Succeeded {
		return nil, v2error.NewError(v2error.EcodeNotFile, nodePath, mkV2Rev(resp.Header.Revision))
	}
	pkvs := resp.Responses[0].GetResponseDeleteRange().PrevKvs
	if len(pkvs) == 0 {
		return nil, v2error.NewError(v2error.EcodeKeyNotFound, nodePath, mkV2Rev(resp.Header.Revision))
	}
	pkv := pkvs[0]
	return &v2store.Event{
		Action: v2store.Delete,
		Node: &v2store.NodeExtern{
			Key:           nodePath,
			CreatedIndex:  mkV2Rev(pkv.CreateRevision),
			ModifiedIndex: mkV2Rev(resp.Header.Revision),
		},
		PrevNode:  s.mkV2Node(pkv),
		EtcdIndex: mkV2Rev(resp.Header.Revision),
	}, nil
}

func (s *v2v3Store) CompareAndDelete(nodePath, prevValue string, prevIndex uint64) (*v2store.Event, error) {
	if isRoot(nodePath) {
		return nil, v2error.NewError(v2error.EcodeRootROnly, nodePath, 0)
	}

	key := s.mkPath(nodePath)
	resp, err := s.c.Txn(s.ctx).If(
		s.mkCompare(nodePath, prevValue, prevIndex)...,
	).Then(
		clientv3.OpDelete(key, clientv3.WithPrevKV()),
		clientv3.OpPut(s.mkActionKey(), v2store.CompareAndDelete),
	).Else(
		clientv3.OpGet(key),
		clientv3.OpGet(key+"/"),
	).Commit()

	if err != nil {
		return nil, err
	}
	if !resp.Succeeded {
		return nil, compareFail(nodePath, prevValue, prevIndex, resp)
	}

	// len(pkvs) > 1 since txn only succeeds when key exists
	pkv := resp.Responses[0].GetResponseDeleteRange().PrevKvs[0]
	return &v2store.Event{
		Action: v2store.CompareAndDelete,
		Node: &v2store.NodeExtern{
			Key:           nodePath,
			CreatedIndex:  mkV2Rev(pkv.CreateRevision),
			ModifiedIndex: mkV2Rev(resp.Header.Revision),
		},
		PrevNode:  s.mkV2Node(pkv),
		EtcdIndex: mkV2Rev(resp.Header.Revision),
	}, nil
}

func compareFail(nodePath, prevValue string, prevIndex uint64, resp *clientv3.TxnResponse) error {
	if dkvs := resp.Responses[1].GetResponseRange().Kvs; len(dkvs) > 0 {
		return v2error.NewError(v2error.EcodeNotFile, nodePath, mkV2Rev(resp.Header.Revision))
	}
	kvs := resp.Responses[0].GetResponseRange().Kvs
	if len(kvs) == 0 {
		return v2error.NewError(v2error.EcodeKeyNotFound, nodePath, mkV2Rev(resp.Header.Revision))
	}
	kv := kvs[0]
	indexMatch := prevIndex == 0 || kv.ModRevision == int64(prevIndex)
	valueMatch := prevValue == "" || string(kv.Value) == prevValue
	var cause string
	switch {
	case indexMatch && !valueMatch:
		cause = fmt.Sprintf("[%v != %v]", prevValue, string(kv.Value))
	case valueMatch && !indexMatch:
		cause = fmt.Sprintf("[%v != %v]", prevIndex, kv.ModRevision)
	default:
		cause = fmt.Sprintf("[%v != %v] [%v != %v]", prevValue, string(kv.Value), prevIndex, kv.ModRevision)
	}
	return v2error.NewError(v2error.EcodeTestFailed, cause, mkV2Rev(resp.Header.Revision))
}

func (s *v2v3Store) mkCompare(nodePath, prevValue string, prevIndex uint64) []clientv3.Cmp {
	key := s.mkPath(nodePath)
	cmps := []clientv3.Cmp{clientv3.Compare(clientv3.Version(key), ">", 0)}
	if prevIndex != 0 {
		cmps = append(cmps, clientv3.Compare(clientv3.ModRevision(key), "=", mkV3Rev(prevIndex)))
	}
	if prevValue != "" {
		cmps = append(cmps, clientv3.Compare(clientv3.Value(key), "=", prevValue))
	}
	return cmps
}

func (s *v2v3Store) JsonStats() []byte                  { panic("STUB") }
func (s *v2v3Store) DeleteExpiredKeys(cutoff time.Time) { panic("STUB") }

func (s *v2v3Store) Version() int { return 2 }

// TODO: move this out of the Store interface?

func (s *v2v3Store) Save() ([]byte, error)       { panic("STUB") }
func (s *v2v3Store) Recovery(state []byte) error { panic("STUB") }
func (s *v2v3Store) Clone() v2store.Store        { panic("STUB") }
func (s *v2v3Store) SaveNoCopy() ([]byte, error) { panic("STUB") }
func (s *v2v3Store) HasTTLKeys() bool            { panic("STUB") }

func (s *v2v3Store) mkPath(nodePath string) string { return s.mkPathDepth(nodePath, 0) }

func (s *v2v3Store) mkNodePath(p string) string {
	return path.Clean(p[len(s.pfx)+len("/k/000/"):])
}

// mkPathDepth makes a path to a key that encodes its directory depth
// for fast directory listing. If a depth is provided, it is added
// to the computed depth.
func (s *v2v3Store) mkPathDepth(nodePath string, depth int) string {
	normalForm := path.Clean(path.Join("/", nodePath))
	n := strings.Count(normalForm, "/") + depth
	return fmt.Sprintf("%s/%03d/k/%s", s.pfx, n, normalForm)
}

func (s *v2v3Store) mkActionKey() string { return s.pfx + "/act" }

func isRoot(s string) bool { return len(s) == 0 || s == "/" || s == "/0" || s == "/1" }

func mkV2Rev(v3Rev int64) uint64 {
	if v3Rev == 0 {
		return 0
	}
	return uint64(v3Rev - 1)
}

func mkV3Rev(v2Rev uint64) int64 {
	if v2Rev == 0 {
		return 0
	}
	return int64(v2Rev + 1)
}

// mkV2Node creates a V2 NodeExtern from a V3 KeyValue
func (s *v2v3Store) mkV2Node(kv *mvccpb.KeyValue) *v2store.NodeExtern {
	if kv == nil {
		return nil
	}
	n := &v2store.NodeExtern{
		Key:           s.mkNodePath(string(kv.Key)),
		Dir:           kv.Key[len(kv.Key)-1] == '/',
		CreatedIndex:  mkV2Rev(kv.CreateRevision),
		ModifiedIndex: mkV2Rev(kv.ModRevision),
	}
	if !n.Dir {
		v := string(kv.Value)
		n.Value = &v
	}
	return n
}

// prevKeyFromPuts gets the prev key that is being put; ignores
// the put action response.
func prevKeyFromPuts(resp *clientv3.TxnResponse) *mvccpb.KeyValue {
	for _, r := range resp.Responses {
		pkv := r.GetResponsePut().PrevKv
		if pkv != nil && pkv.CreateRevision > 0 {
			return pkv
		}
	}
	return nil
}

func (s *v2v3Store) newSTM(applyf func(concurrency.STM) error) (*clientv3.TxnResponse, error) {
	return concurrency.NewSTM(s.c, applyf, concurrency.WithIsolation(concurrency.Serializable))
}
