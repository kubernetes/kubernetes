/*
 *
 * Copyright 2014, Google Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *
 *     * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above
 * copyright notice, this list of conditions and the following disclaimer
 * in the documentation and/or other materials provided with the
 * distribution.
 *     * Neither the name of Google Inc. nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

package etcd

import (
	etcdcl "github.com/coreos/etcd/client"
	"golang.org/x/net/context"
	"google.golang.org/grpc/naming"
)
// update defines an etcd key-value update.
type update struct {
	key, val string
}

// getNode reports the set of changes starting from node recursively.
func getNode(node *etcdcl.Node) (updates []*update) {
	for _, v := range node.Nodes {
		updates = append(updates, getNode(v)...)
	}
	if !node.Dir {
		u := &update{
			key: node.Key,
			val: node.Value,
		}
		updates = []*update{u}
	}
	return
}

type watcher struct {
	wr     etcdcl.Watcher
	ctx    context.Context
	cancel context.CancelFunc
	kv     map[string]string
}

func (w *watcher) Next() (nu []*naming.Update, _ error) {
	for {
		resp, err := w.wr.Next(w.ctx)
		if err != nil {
			return nil, err
		}
		updates := getNode(resp.Node)
		for _, u := range updates {
			switch resp.Action {
			case "set":
				if resp.PrevNode == nil {
					w.kv[u.key] = u.val
					nu = append(nu, &naming.Update{
						Op:   naming.Add,
						Addr: u.val,
					})
				} else {
					nu = append(nu, &naming.Update{
						Op:   naming.Delete,
						Addr: w.kv[u.key],
					})
					nu = append(nu, &naming.Update{
						Op:   naming.Add,
						Addr: u.val,
					})
					w.kv[u.key] = u.val
				}
			case "delete":
				nu = append(nu, &naming.Update{
					Op:   naming.Delete,
					Addr: w.kv[u.key],
				})
				delete(w.kv, u.key)
			}
		}
		if len(nu) > 0 {
			break
		}
	}
	return nu, nil
}

func (w *watcher) Stop() {
	w.cancel()
}

type resolver struct {
	kapi etcdcl.KeysAPI
	kv   map[string]string
}

func (r *resolver) NewWatcher(target string) naming.Watcher {
	ctx, cancel := context.WithCancel(context.Background())
	w := &watcher{
		wr:     r.kapi.Watcher(target, &etcdcl.WatcherOptions{Recursive: true}),
		ctx:    ctx,
		cancel: cancel,
	}
	for k, v := range r.kv {
		w.kv[k] = v
	}
	return w
}

func (r *resolver) Resolve(target string) (nu []*naming.Update, _ error) {
	resp, err := r.kapi.Get(context.Background(), target, &etcdcl.GetOptions{Recursive: true})
	if err != nil {
		return nil, err
	}
	updates := getNode(resp.Node)
	for _, u := range updates {
		r.kv[u.key] = u.val
		nu = append(nu, &naming.Update{
			Op:   naming.Add,
			Addr: u.val,
		})
	}
	return nu, nil
}

// NewResolver creates an etcd-based naming.Resolver.
func NewResolver(cfg etcdcl.Config) (naming.Resolver, error) {
	c, err := etcdcl.New(cfg)
	if err != nil {
		return nil, err
	}
	return &resolver{
		kapi: etcdcl.NewKeysAPI(c),
		kv:   make(map[string]string),
	}, nil
}
