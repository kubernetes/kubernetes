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

// Package v3client provides clientv3 interfaces from an etcdserver.
//
// Use v3client by creating an EtcdServer instance, then wrapping it with v3client.New:
//
//	import (
//		"context"
//
//		"go.etcd.io/etcd/server/v3/embed"
//		"go.etcd.io/etcd/server/v3/etcdserver/api/v3client"
//	)
//
//	...
//
//	// create an embedded EtcdServer from the default configuration
//	cfg := embed.NewConfig()
//	cfg.Dir = "default.etcd"
//	e, err := embed.StartEtcd(cfg)
//	if err != nil {
//		// handle error!
//	}
//
//	// wrap the EtcdServer with v3client
//	cli := v3client.New(e.Server)
//
//	// use like an ordinary clientv3
//	resp, err := cli.Put(context.TODO(), "some-key", "it works!")
//	if err != nil {
//		// handle error!
//	}
//
package v3client
