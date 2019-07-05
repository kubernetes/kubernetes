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

// Package namespace is a clientv3 wrapper that translates all keys to begin
// with a given prefix.
//
// First, create a client:
//
//	cli, err := clientv3.New(clientv3.Config{Endpoints: []string{"localhost:2379"}})
//	if err != nil {
//		// handle error!
//	}
//
// Next, override the client interfaces:
//
//	unprefixedKV := cli.KV
//	cli.KV = namespace.NewKV(cli.KV, "my-prefix/")
//	cli.Watcher = namespace.NewWatcher(cli.Watcher, "my-prefix/")
//	cli.Lease = namespace.NewLease(cli.Lease, "my-prefix/")
//
// Now calls using 'cli' will namespace / prefix all keys with "my-prefix/":
//
//	cli.Put(context.TODO(), "abc", "123")
//	resp, _ := unprefixedKV.Get(context.TODO(), "my-prefix/abc")
//	fmt.Printf("%s\n", resp.Kvs[0].Value)
//	// Output: 123
//	unprefixedKV.Put(context.TODO(), "my-prefix/abc", "456")
//	resp, _ = cli.Get("abc")
//	fmt.Printf("%s\n", resp.Kvs[0].Value)
//	// Output: 456
//
package namespace
