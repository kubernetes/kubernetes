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

package factory

import (
	"context"
	"strconv"
	"sync"
	"time"

	clientv3 "go.etcd.io/etcd/client/v3"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/klog/v2"
)

const (
	compactRevKey = "compact_rev_key"
)

func runETCD3Compaction(cfg storagebackend.Config, stopCh <-chan struct{}) error {
	client, err := newETCD3Client(cfg.Transport)
	if err != nil {
		return err
	}
	c := &etcd3Compactor{
		interval: cfg.CompactionInterval,
		client:   client,
	}
	c.wg.StartWithChannel(stopCh, c.run)
	return nil
}

// etcd3Compactor runs compactor in the background to compact old version of keys that's not needed.
// By default, we save the most recent 5 minutes data and compact versions > 5minutes ago.
// It should be enough for slow watchers and to tolerate burst.
// TODO: We might keep a longer history (12h) in the future once storage API can take advantage of past version of keys.
type etcd3Compactor struct {
	interval time.Duration
	client   *clientv3.Client

	wg       wait.Group
	stop     context.CancelFunc
	stopLock sync.Mutex
	stopped  bool
}

func (c *etcd3Compactor) Stop() {
	c.stopLock.Lock()
	if c.stopped {
		// avoid stopping twice (note: cachers are shared with subresources)
		c.stopLock.Unlock()
		return
	}
	c.stopped = true
	c.stop()
	c.wg.Wait()
	c.client.Close()
	c.stopLock.Unlock()
}

// RunCompaction periodically compacts historical versions of keys in etcd.
// It will compact keys with versions older than given interval.
// In other words, after compaction, it will only contain keys set during last interval.
// Any API call for the older versions of keys will return error.
// Interval is the time interval between each compaction. The first compaction happens after "interval".
func (c *etcd3Compactor) run(stopCh <-chan struct{}) {
	// Technical definitions:
	// We have a special key in etcd defined as *compactRevKey*.
	// compactRevKey's value will be set to the string of last compacted revision.
	// compactRevKey's version will be used as logical time for comparison. THe version is referred as compact time.
	// Initially, because the key doesn't exist, the compact time (version) is 0.
	//
	// Algorithm:
	// - Compare to see if (local compact_time) = (remote compact_time).
	// - If yes, increment both local and remote compact_time, and do a compaction.
	// - If not, set local to remote compact_time.
	//
	// Technical details/insights:
	//
	// The protocol here is lease based. If one compactor CAS successfully, the others would know it when they fail in
	// CAS later and would try again in 5 minutes. If an APIServer crashed, another one would "take over" the lease.
	//
	// For example, in the following diagram, we have a compactor C1 doing compaction in t1, t2. Another compactor C2
	// at t1' (t1 < t1' < t2) would CAS fail, set its known oldRev to rev at t1', and try again in t2' (t2' > t2).
	// If C1 crashed and wouldn't compact at t2, C2 would CAS successfully at t2'.
	//
	//                 oldRev(t2)     curRev(t2)
	//                                  +
	//   oldRev        curRev           |
	//     +             +              |
	//     |             |              |
	//     |             |    t1'       |     t2'
	// +---v-------------v----^---------v------^---->
	//     t0           t1             t2
	//
	// We have the guarantees:
	// - in normal cases, the interval is 5 minutes.
	// - in failover, the interval is >5m and <10m
	//
	// FAQ:
	// - What if time is not accurate? We don't care as long as someone did the compaction. Atomicity is ensured using
	//   etcd API.
	// - What happened under heavy load scenarios? Initially, each apiserver will do only one compaction
	//   every 5 minutes. This is very unlikely affecting or affected w.r.t. server load.

	var compactTime int64
	var rev int64
	var err error
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		select {
		case <-stopCh:
			cancel()
		}
	}()

	for {
		select {
		case <-time.After(c.interval):
		case <-ctx.Done():
			return
		}

		compactTime, rev, err = c.compact(ctx, compactTime, rev)
		if err != nil {
			klog.Errorf("etcd: compact failed: %v", err)
			continue
		}
	}
}

// compact compacts etcd store and returns current rev.
// It will return the current compact time and global revision if no error occurred.
// Note that CAS fail will not incur any error.
func (c *etcd3Compactor) compact(ctx context.Context, t, rev int64) (int64, int64, error) {
	resp, err := c.client.KV.Txn(ctx).If(
		clientv3.Compare(clientv3.Version(compactRevKey), "=", t),
	).Then(
		clientv3.OpPut(compactRevKey, strconv.FormatInt(rev, 10)), // Expect side effect: increment Version
	).Else(
		clientv3.OpGet(compactRevKey),
	).Commit()
	if err != nil {
		return t, rev, err
	}

	curRev := resp.Header.Revision

	if !resp.Succeeded {
		curTime := resp.Responses[0].GetResponseRange().Kvs[0].Version
		return curTime, curRev, nil
	}
	curTime := t + 1

	if rev == 0 {
		// We don't compact on bootstrap.
		return curTime, curRev, nil
	}
	if _, err = c.client.Compact(ctx, rev); err != nil {
		return curTime, curRev, err
	}
	klog.V(4).Infof("etcd: compacted rev (%d)", rev)
	return curTime, curRev, nil
}
