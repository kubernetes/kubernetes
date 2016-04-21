/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package etcd3

import (
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/golang/glog"
	"golang.org/x/net/context"
)

// compactor periodically compacts historical versions of keys in etcd.
// After compaction, old versions of keys set before given interval will be gone.
// Any API call for the old versions of keys will return error.
// interval: the interval between each compaction. The first compaction happens after "interval".
func compactor(ctx context.Context, client *clientv3.Client, interval time.Duration) {
	var curRev int64
	var err error
	for {
		select {
		case <-time.After(interval):
		case <-ctx.Done():
			return
		}

		curRev, err = compact(ctx, client, curRev)
		if err != nil {
			glog.Error(err)
			continue
		}
		glog.Infof("compactor: Compacted rev %d", curRev)
	}
}

// compact compacts etcd store and returns current rev.
// If it couldn't get current revision, the old rev will be returned.
func compact(ctx context.Context, client *clientv3.Client, oldRev int64) (int64, error) {
	resp, err := client.Get(ctx, "/")
	if err != nil {
		return oldRev, err
	}
	curRev := resp.Header.Revision
	if oldRev == 0 {
		return curRev, nil
	}
	err = client.Compact(ctx, oldRev)
	if err != nil {
		return curRev, err
	}
	return curRev, nil
}
