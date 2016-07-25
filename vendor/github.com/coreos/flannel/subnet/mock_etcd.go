// Copyright 2015 flannel authors
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

package subnet

import (
	"fmt"
	"strings"
	"sync"
	"time"

	etcd "github.com/coreos/etcd/client"
	"golang.org/x/net/context"
)

const DEFAULT_TTL time.Duration = 8760 * time.Hour // one year

type mockEtcd struct {
	mux      sync.Mutex
	nodes    map[string]*etcd.Node
	watchers map[*watcher]struct{}
	// A given number of past events must be available for watchers, because
	// flannel always uses a new watcher instead of re-using old ones, and
	// the new watcher's index may be slightly in the past
	events []*etcd.Response
	index  uint64
}

func newMockEtcd() *mockEtcd {
	me := &mockEtcd{
		index:    1000,
		nodes:    make(map[string]*etcd.Node),
		watchers: make(map[*watcher]struct{}),
		events:   make([]*etcd.Response, 0, 50),
	}
	me.nodes["/"] = me.newNode("/", "", true)

	return me
}

func (me *mockEtcd) newNode(key, value string, dir bool) *etcd.Node {
	exp := time.Now().Add(DEFAULT_TTL)
	if dir {
		value = ""
	}
	return &etcd.Node{
		Key:           key,
		Value:         value,
		CreatedIndex:  me.index,
		ModifiedIndex: me.index,
		Dir:           dir,
		Expiration:    &exp,
		Nodes:         make([]*etcd.Node, 0, 20)}
}

func (me *mockEtcd) newError(code int, format string, args ...interface{}) etcd.Error {
	msg := fmt.Sprintf(format, args...)
	return etcd.Error{
		Code:    code,
		Message: msg,
		Cause:   "",
		Index:   me.index,
	}
}

func (me *mockEtcd) getKeyPath(key string) ([]string, error) {
	if !strings.HasPrefix(key, "/") {
		return []string{}, me.newError(etcd.ErrorCodeKeyNotFound, "Invalid key %s", key)
	}

	// Build up a list of each intermediate key's path
	path := []string{""}
	for i, p := range strings.Split(strings.Trim(key, "/"), "/") {
		if p == "" {
			return []string{}, me.newError(etcd.ErrorCodeKeyNotFound, "Invalid key %s", key)
		}
		path = append(path, fmt.Sprintf("%s/%s", path[i], p))
	}

	return path[1:], nil
}

// Returns the node and its parent respectively.  Returns a nil node (but not
// an error) if the requested node doest not exist.
func (me *mockEtcd) findNode(key string) (*etcd.Node, *etcd.Node, error) {
	if key == "/" {
		return me.nodes["/"], nil, nil
	}

	path, err := me.getKeyPath(key)
	if err != nil {
		return nil, nil, err
	}

	var node *etcd.Node
	var parent *etcd.Node
	var ok bool

	for i, part := range path {
		parent = node
		node, ok = me.nodes[part]
		if !ok {
			return nil, nil, nil
		}

		// intermediates must be directories
		if i < len(path)-1 && node.Dir != true {
			return nil, nil, me.newError(etcd.ErrorCodeNotDir, "Intermediate node %s not a directory", part)
		}
	}

	return node, parent, nil
}

// Returns whether @child is a child of @node, and whether it is an immediate child respsectively
func isChild(node *etcd.Node, child *etcd.Node) (bool, bool) {
	if !strings.HasPrefix(child.Key, fmt.Sprintf("%s/", node.Key)) {
		return false, false
	}

	nodeParts := strings.Split(node.Key, "/")
	childParts := strings.Split(child.Key, "/")
	return true, len(childParts) == len(nodeParts)+1
}

func (me *mockEtcd) copyNode(node *etcd.Node, recursive bool) *etcd.Node {
	n := *node
	n.Nodes = make([]*etcd.Node, 0)
	if recursive {
		for _, child := range me.nodes {
			if _, directChild := isChild(node, child); directChild {
				n.Nodes = append(n.Nodes, me.copyNode(child, true))
			}
		}
	}
	return &n
}

func (me *mockEtcd) Get(ctx context.Context, key string, opts *etcd.GetOptions) (*etcd.Response, error) {
	me.mux.Lock()
	defer me.mux.Unlock()

	node, _, err := me.findNode(key)
	if err != nil {
		return nil, err
	}
	if node == nil {
		return nil, me.newError(etcd.ErrorCodeKeyNotFound, "Key %s not found", key)
	}

	if opts == nil {
		opts = &etcd.GetOptions{}
	}

	return &etcd.Response{
		Action: "get",
		Node:   me.copyNode(node, opts.Recursive),
		Index:  me.index,
	}, nil
}

func (me *mockEtcd) sendEvent(resp *etcd.Response) {
	// Add to history log
	if len(me.events) == cap(me.events) {
		me.events = me.events[1:]
	}
	me.events = append(me.events, resp)

	// and notify watchers
	for w, _ := range me.watchers {
		w.notifyEvent(resp)
	}
}

// Returns the node created and its creation response
// Don't need to check for intermediate directories here as that was already done
// by the thing calling makeNode()
func (me *mockEtcd) makeNode(path []string, value string, isDir bool, ttl time.Duration) (*etcd.Node, *etcd.Response, error) {
	var child *etcd.Node
	var resp *etcd.Response
	var ok bool

	node := me.nodes["/"]
	for i, part := range path {
		node, ok = me.nodes[part]
		if !ok {
			me.index += 1
			if i < len(path)-1 {
				// intermediate node
				child = me.newNode(part, "", true)
			} else {
				// Final node
				exp := time.Now().Add(ttl)
				child = me.newNode(part, value, isDir)
				child.Expiration = &exp

				resp = &etcd.Response{
					Action: "create",
					Node:   me.copyNode(child, false),
					Index:  child.CreatedIndex,
				}
				me.sendEvent(resp)
			}
			me.nodes[child.Key] = child
			node = child
		}
	}

	return node, resp, nil
}

func (me *mockEtcd) set(ctx context.Context, key, value string, opts *etcd.SetOptions, action string) (*etcd.Response, error) {
	node, _, err := me.findNode(key)
	if err != nil {
		return nil, err
	}
	if opts.PrevExist == etcd.PrevExist && node == nil {
		return nil, me.newError(etcd.ErrorCodeKeyNotFound, "Key %s not found", key)
	} else if opts.PrevExist == etcd.PrevNoExist && node != nil {
		return nil, me.newError(etcd.ErrorCodeNodeExist, "Key %s already exists", key)
	}

	if opts.Dir {
		value = ""
	}

	var resp *etcd.Response

	if node != nil {
		if opts.PrevIndex > 0 && opts.PrevIndex < node.ModifiedIndex {
			return nil, me.newError(etcd.ErrorCodeTestFailed, "Key %s PrevIndex %s less than node ModifiedIndex %d", key, opts.PrevIndex, node.ModifiedIndex)
		}

		if opts.Dir != node.Dir {
			if opts.Dir == true {
				return nil, me.newError(etcd.ErrorCodeNotDir, "Key %s is not a directory", key)
			} else {
				return nil, me.newError(etcd.ErrorCodeNotFile, "Key %s is not a file", key)
			}
		}

		if opts.PrevValue != "" && opts.PrevValue != node.Value {
			return nil, me.newError(etcd.ErrorCodeTestFailed, "Key %s PrevValue did not match", key)
		}

		prevNode := me.copyNode(node, false)

		node.Value = value

		me.index += 1
		node.ModifiedIndex = me.index

		if opts.TTL > 0 {
			exp := time.Now().Add(opts.TTL)
			node.Expiration = &exp
		}

		resp = &etcd.Response{
			Action:   action,
			Node:     me.copyNode(node, false),
			PrevNode: prevNode,
			Index:    me.index,
		}
		me.sendEvent(resp)
	} else {
		// Create the node and its parents
		path, err := me.getKeyPath(key)
		if err != nil {
			return nil, err
		}

		_, resp, err = me.makeNode(path, value, opts.Dir, opts.TTL)
		if err != nil {
			return nil, err
		}
	}

	return resp, nil
}

func (me *mockEtcd) Set(ctx context.Context, key, value string, opts *etcd.SetOptions) (*etcd.Response, error) {
	me.mux.Lock()
	defer me.mux.Unlock()

	return me.set(ctx, key, value, opts, "set")
}

// Removes a node and all children
func (me *mockEtcd) deleteNode(node *etcd.Node, parent *etcd.Node, recursive bool) (*etcd.Response, error) {
	for _, child := range me.nodes {
		if isChild, directChild := isChild(node, child); isChild {
			if recursive == false {
				return nil, me.newError(etcd.ErrorCodeDirNotEmpty, "Key %s not empty", node.Key)
			}

			if directChild {
				me.deleteNode(child, node, true)
				me.index += 1
				node.ModifiedIndex = me.index
			}
		}
	}

	me.index += 1
	resp := &etcd.Response{
		Action: "delete",
		Node:   me.copyNode(node, false),
		Index:  me.index,
	}
	me.sendEvent(resp)

	delete(me.nodes, node.Key)

	return resp, nil
}

func (me *mockEtcd) Delete(ctx context.Context, key string, opts *etcd.DeleteOptions) (*etcd.Response, error) {
	me.mux.Lock()
	defer me.mux.Unlock()

	node, parent, err := me.findNode(key)
	if err != nil {
		return nil, err
	}
	if node == nil {
		return nil, me.newError(etcd.ErrorCodeKeyNotFound, "Key %s not found", key)
	}

	if opts == nil {
		opts = &etcd.DeleteOptions{}
	}

	if opts.PrevIndex > 0 && opts.PrevIndex < node.ModifiedIndex {
		return nil, me.newError(etcd.ErrorCodeTestFailed, "Key %s PrevIndex %s less than node ModifiedIndex %d", key, opts.PrevIndex, node.ModifiedIndex)
	}

	if opts.PrevValue != "" && opts.PrevValue != node.Value {
		return nil, me.newError(etcd.ErrorCodeTestFailed, "Key %s PrevValue did not match", key)
	}

	if opts.Dir != node.Dir {
		if opts.Dir == true {
			return nil, me.newError(etcd.ErrorCodeNotDir, "Key %s is not a directory", key)
		} else {
			return nil, me.newError(etcd.ErrorCodeNotFile, "Key %s is not a file", key)
		}
	}

	return me.deleteNode(node, parent, opts.Recursive)
}

func (me *mockEtcd) Create(ctx context.Context, key, value string) (*etcd.Response, error) {
	me.mux.Lock()
	defer me.mux.Unlock()

	return me.set(ctx, key, value, &etcd.SetOptions{PrevExist: etcd.PrevNoExist}, "create")
}

func (me *mockEtcd) CreateInOrder(ctx context.Context, dir, value string, opts *etcd.CreateInOrderOptions) (*etcd.Response, error) {
	panic(fmt.Errorf("Not implemented!"))
}

func (me *mockEtcd) Update(ctx context.Context, key, value string) (*etcd.Response, error) {
	me.mux.Lock()
	defer me.mux.Unlock()

	return me.set(ctx, key, value, &etcd.SetOptions{PrevExist: etcd.PrevExist}, "update")
}

type watcher struct {
	parent     *mockEtcd
	key        string
	childMatch string
	events     chan *etcd.Response
	after      uint64
	recursive  bool
}

func (me *mockEtcd) Watcher(key string, opts *etcd.WatcherOptions) etcd.Watcher {
	watcher := &watcher{
		parent:     me,
		key:        key,
		childMatch: fmt.Sprintf("%s/", key),
		events:     make(chan *etcd.Response, 25),
		recursive:  opts.Recursive,
	}
	if opts.AfterIndex > 0 {
		watcher.after = opts.AfterIndex
	}
	return watcher
}

func (w *watcher) shouldGrabEvent(resp *etcd.Response) bool {
	return (resp.Index > w.after) && ((resp.Node.Key == w.key) || (w.recursive && strings.HasPrefix(resp.Node.Key, w.childMatch)))
}

func (w *watcher) notifyEvent(resp *etcd.Response) {
	if w.shouldGrabEvent(resp) {
		w.events <- resp
	}
}

func (w *watcher) Next(ctx context.Context) (*etcd.Response, error) {
	w.parent.mux.Lock()

	// If the event is already in the history log return it from there

	for _, e := range w.parent.events {
		if e.Index > w.after && w.shouldGrabEvent(e) {
			w.after = e.Index
			w.parent.mux.Unlock()
			return e, nil
		}
	}

	// Watch must handle adding and removing itself from the parent when
	// it's done to ensure it can be garbage collected correctly
	w.parent.watchers[w] = struct{}{}

	w.parent.mux.Unlock()

	// Otherwise wait for new events
	for {
		select {
		case e := <-w.events:
			// Might have already been grabbed through the history log
			if e.Index <= w.after {
				continue
			}
			w.after = e.Index

			w.parent.mux.Lock()
			delete(w.parent.watchers, w)
			w.parent.mux.Unlock()

			return e, nil
		case <-ctx.Done():
			w.parent.mux.Lock()
			delete(w.parent.watchers, w)
			w.parent.mux.Unlock()

			return nil, context.Canceled
		}
	}
}
