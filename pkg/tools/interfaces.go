/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package tools

import (
	"github.com/coreos/go-etcd/etcd"
)

const (
	EtcdErrorCodeNotFound      = 100
	EtcdErrorCodeTestFailed    = 101
	EtcdErrorCodeNodeExist     = 105
	EtcdErrorCodeValueRequired = 200
)

var (
	EtcdErrorNotFound      = &etcd.EtcdError{ErrorCode: EtcdErrorCodeNotFound}
	EtcdErrorTestFailed    = &etcd.EtcdError{ErrorCode: EtcdErrorCodeTestFailed}
	EtcdErrorNodeExist     = &etcd.EtcdError{ErrorCode: EtcdErrorCodeNodeExist}
	EtcdErrorValueRequired = &etcd.EtcdError{ErrorCode: EtcdErrorCodeValueRequired}
)

// EtcdClient is an injectable interface for testing.
type EtcdClient interface {
	GetCluster() []string
	Get(key string, sort, recursive bool) (*etcd.Response, error)
	Set(key, value string, ttl uint64) (*etcd.Response, error)
	Create(key, value string, ttl uint64) (*etcd.Response, error)
	CompareAndSwap(key, value string, ttl uint64, prevValue string, prevIndex uint64) (*etcd.Response, error)
	Delete(key string, recursive bool) (*etcd.Response, error)
	// I'd like to use directional channels here (e.g. <-chan) but this interface mimics
	// the etcd client interface which doesn't, and it doesn't seem worth it to wrap the api.
	Watch(prefix string, waitIndex uint64, recursive bool, receiver chan *etcd.Response, stop chan bool) (*etcd.Response, error)
}
