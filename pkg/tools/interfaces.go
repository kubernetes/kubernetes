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
	"errors"
	etcd "github.com/coreos/etcd/client"
	"golang.org/x/net/context"
)

const (
	EtcdErrorCodeNotFound      = 100
	EtcdErrorCodeTestFailed    = 101
	EtcdErrorCodeNodeExist     = 105
	EtcdErrorCodeValueRequired = 200
)

var (
	EtcdErrorNotFound         = &etcd.Error{Code: EtcdErrorCodeNotFound}
	EtcdErrorTestFailed       = &etcd.Error{Code: EtcdErrorCodeTestFailed}
	EtcdErrorNodeExist        = &etcd.Error{Code: EtcdErrorCodeNodeExist}
	EtcdErrorValueRequired    = &etcd.Error{Code: EtcdErrorCodeValueRequired}
	EtcdErrWatchStoppedByUser = errors.New("Watch stopped by the user via stop channel")
)

//TODO: Eliminate this entirely through data hiding ~= pImpl idiom on storage layer.
//      At this point it's only used for the mock testing environment, and is eliminated
//      in the mainline code.  However, it needs to be cleaned in contrib modules.
type EtcdWatcher interface {
	Next(context.Context) (*etcd.Response, error)
}

type EtcdClient interface {
	Get(ctx context.Context, key string, opts *etcd.GetOptions) (*etcd.Response, error)
	Set(ctx context.Context, key, value string, opts *etcd.SetOptions) (*etcd.Response, error)
	Create(ctx context.Context, key, value string) (*etcd.Response, error)
	CreateInOrder(ctx context.Context, key, value string, opts *etcd.CreateInOrderOptions) (*etcd.Response, error)
	Update(ctx context.Context, key, value string) (*etcd.Response, error)
	Delete(ctx context.Context, key string, opts *etcd.DeleteOptions) (*etcd.Response, error)
	Watcher(key string, opts *etcd.WatcherOptions) etcd.Watcher
	// Watch(prefix string, waitIndex uint64, recursive bool, receiver chan *etcd.Response, stop chan bool) (*etcd.Response, error)
}
