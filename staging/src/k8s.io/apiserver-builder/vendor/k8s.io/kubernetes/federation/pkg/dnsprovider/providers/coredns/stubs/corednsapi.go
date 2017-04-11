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

// Package stubs implements a stub for the EtcdKeysAPI, used primarily for unit testing purposes
package stubs

import (
	"strings"

	etcd "github.com/coreos/etcd/client"
	"golang.org/x/net/context"
)

// Compile time check for interface conformance
var _ EtcdKeysAPI = &EtcdKeysAPIStub{}

type EtcdKeysAPI interface {
	Set(context context.Context, key, value string, options *etcd.SetOptions) (*etcd.Response, error)
	Get(context context.Context, key string, options *etcd.GetOptions) (*etcd.Response, error)
	Delete(context context.Context, key string, options *etcd.DeleteOptions) (*etcd.Response, error)
}

type EtcdKeysAPIStub struct {
	writes map[string]string
}

// NewEtcdKeysAPIStub returns an initialized EtcdKeysAPIStub
func NewEtcdKeysAPIStub() *EtcdKeysAPIStub {
	return &EtcdKeysAPIStub{make(map[string]string)}
}

func (ec *EtcdKeysAPIStub) Set(context context.Context, key, value string, options *etcd.SetOptions) (*etcd.Response, error) {
	ec.writes[key] = value
	return nil, nil
}

func (ec *EtcdKeysAPIStub) Delete(context context.Context, key string, options *etcd.DeleteOptions) (*etcd.Response, error) {
	for p := range ec.writes {
		if (options.Recursive && strings.HasPrefix(p, key)) || (!options.Recursive && p == key) {
			delete(ec.writes, p)
		}
	}
	return nil, nil
}

func (ec *EtcdKeysAPIStub) Get(context context.Context, key string, options *etcd.GetOptions) (*etcd.Response, error) {
	nodes := ec.GetAll(key)
	if len(nodes) == 0 {
		return nil, nil
	}
	if len(nodes) == 1 && nodes[key] != "" {
		return &etcd.Response{Node: &etcd.Node{Key: key, Value: nodes[key], Dir: false}}, nil
	}

	node := &etcd.Node{Key: key, Dir: true, Nodes: etcd.Nodes{}}
	for k, v := range nodes {
		n := &etcd.Node{Key: k, Value: v}
		node.Nodes = append(node.Nodes, n)
	}
	return &etcd.Response{Node: node}, nil
}

func (ec *EtcdKeysAPIStub) GetAll(key string) map[string]string {
	nodes := make(map[string]string)
	key = strings.ToLower(key)
	for path := range ec.writes {
		if strings.HasPrefix(path, key) {
			nodes[path] = ec.writes[path]
		}
	}
	return nodes
}
