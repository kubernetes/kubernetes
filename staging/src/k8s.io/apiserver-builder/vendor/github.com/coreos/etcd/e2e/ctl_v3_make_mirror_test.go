// Copyright 2016 The etcd Authors
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

package e2e

import (
	"fmt"
	"testing"
	"time"
)

func TestCtlV3MakeMirror(t *testing.T) { testCtl(t, makeMirrorTest) }

func makeMirrorTest(cx ctlCtx) {
	// set up another cluster to mirror with
	cfg := configAutoTLS
	cfg.clusterSize = 1
	cfg.basePort = 10000
	cx2 := ctlCtx{
		t:           cx.t,
		cfg:         cfg,
		dialTimeout: 7 * time.Second,
	}

	epc, err := newEtcdProcessCluster(&cx2.cfg)
	if err != nil {
		cx.t.Fatalf("could not start etcd process cluster (%v)", err)
	}
	cx2.epc = epc

	defer func() {
		if err = cx2.epc.Close(); err != nil {
			cx.t.Fatalf("error closing etcd processes (%v)", err)
		}
	}()

	cmdArgs := append(cx.PrefixArgs(), "make-mirror", fmt.Sprintf("localhost:%d", cfg.basePort))
	proc, err := spawnCmd(cmdArgs)
	if err != nil {
		cx.t.Fatal(err)
	}
	defer func() {
		err = proc.Stop()
		if err != nil {
			cx.t.Fatal(err)
		}
	}()

	var kvs = []kv{{"key1", "val1"}, {"key2", "val2"}, {"key3", "val3"}}
	for i := range kvs {
		if err = ctlV3Put(cx, kvs[i].key, kvs[i].val, ""); err != nil {
			cx.t.Fatal(err)
		}
	}
	if err = ctlV3Get(cx, []string{"key", "--prefix"}, kvs...); err != nil {
		cx.t.Fatal(err)
	}
	if err = ctlV3Watch(cx2, []string{"key", "--rev", "1", "--prefix"}, kvs...); err != nil {
		cx.t.Fatal(err)
	}
}
