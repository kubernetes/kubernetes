// Copyright 2016 CoreOS, Inc.
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

// clientv3 is the official Go etcd client for v3.
//
// Create client using `clientv3.New`:
//
//	cli, err := clientv3.New(clientv3.Config{
//		Endpoints:   []string{"localhost:12378", "localhost:22378", "localhost:32378"},
//		DialTimeout: 5 * time.Second,
//	})
//	if err != nil {
//		// handle error!
//	}
//	defer cli.Close()
//
// Make sure to close the client after using it. If the client is not closed, the
// connection will have leaky goroutines.
//
// To specify client request timeout, pass context.WithTimeout to APIs:
//
//	ctx, cancel := context.WithTimeout(context.Background(), timeout)
//	resp, err := kvc.Put(ctx, "sample_key", "sample_value")
//	cancel()
//	if err != nil {
//	    // handle error!
//	}
//	// use the response
//
// etcd client returns 2 types of errors:
//
//	1. context error: canceled or deadline exceeded.
//	2. gRPC error: see https://github.com/coreos/etcd/blob/master/etcdserver/api/v3rpc/error.go.
//
// Here is the example code to handle client errors:
//
//	resp, err := kvc.Put(ctx, "", "")
//	if err != nil {
//		if err == context.Canceled {
//			// ctx is canceled by another routine
//		} else if err == context.DeadlineExceeded {
//			// ctx is attached with a deadline and it exceeded
//		} else if verr, ok := err.(*v3rpc.ErrEmptyKey); ok {
//			// process (verr.Errors)
//		} else {
//			// bad cluster endpoints, which are not etcd servers
//		}
//	}
//
package clientv3
