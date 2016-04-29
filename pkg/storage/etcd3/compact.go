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
	"sync"
	"time"

	"github.com/coreos/etcd/clientv3"
	"github.com/golang/glog"
	"golang.org/x/net/context"
)

const compactInterval = 10 * time.Minute

var (
	endpointsMapMu sync.Mutex
	endpointsMap   map[string]struct{}
)

func init() {
	endpointsMap = make(map[string]struct{})
}

// StartCompactor starts a compactor in the background in order to compact keys
// older than fixed time.
// We need to compact keys because we can't let on disk data grow forever.
// We save the most recent 10 minutes data. It should be enough for slow watchers and to tolerate burst.
// TODO: We might keep a longer history (12h) in the future once storage API can take
//       advantage of multi-version key.
func StartCompactor(ctx context.Context, client *clientv3.Client) {
	endpointsMapMu.Lock()
	defer endpointsMapMu.Unlock()

	// We can't have multiple compaction jobs for the same cluster.
	// Currently we rely on endpoints to differentiate clusters.
	var emptyStruct struct{}
	for _, ep := range client.Endpoints() {
		if _, ok := endpointsMap[ep]; ok {
			glog.V(4).Infof("compactor already exists for endpoints %v")
			return
		}
	}
	for _, ep := range client.Endpoints() {
		endpointsMap[ep] = emptyStruct
	}

	go compactor(ctx, client, compactInterval)
}

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
	glog.Infof("etcd: Compacted rev %d, endpoints %v", oldRev, client.Endpoints())
	return curRev, nil
}
