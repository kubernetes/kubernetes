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

func TestCtlV3MakeMirror(t *testing.T)                 { testCtl(t, makeMirrorTest) }
func TestCtlV3MakeMirrorModifyDestPrefix(t *testing.T) { testCtl(t, makeMirrorModifyDestPrefixTest) }
func TestCtlV3MakeMirrorNoDestPrefix(t *testing.T)     { testCtl(t, makeMirrorNoDestPrefixTest) }

func makeMirrorTest(cx ctlCtx) {
	var (
		flags  = []string{}
		kvs    = []kv{{"key1", "val1"}, {"key2", "val2"}, {"key3", "val3"}}
		prefix = "key"
	)
	testMirrorCommand(cx, flags, kvs, kvs, prefix, prefix)
}

func makeMirrorModifyDestPrefixTest(cx ctlCtx) {
	var (
		flags      = []string{"--prefix", "o_", "--dest-prefix", "d_"}
		kvs        = []kv{{"o_key1", "val1"}, {"o_key2", "val2"}, {"o_key3", "val3"}}
		kvs2       = []kv{{"d_key1", "val1"}, {"d_key2", "val2"}, {"d_key3", "val3"}}
		srcprefix  = "o_"
		destprefix = "d_"
	)
	testMirrorCommand(cx, flags, kvs, kvs2, srcprefix, destprefix)
}

func makeMirrorNoDestPrefixTest(cx ctlCtx) {
	var (
		flags      = []string{"--prefix", "o_", "--no-dest-prefix"}
		kvs        = []kv{{"o_key1", "val1"}, {"o_key2", "val2"}, {"o_key3", "val3"}}
		kvs2       = []kv{{"key1", "val1"}, {"key2", "val2"}, {"key3", "val3"}}
		srcprefix  = "o_"
		destprefix = "key"
	)

	testMirrorCommand(cx, flags, kvs, kvs2, srcprefix, destprefix)
}

func testMirrorCommand(cx ctlCtx, flags []string, sourcekvs, destkvs []kv, srcprefix, destprefix string) {
	// set up another cluster to mirror with
	mirrorcfg := configAutoTLS
	mirrorcfg.clusterSize = 1
	mirrorcfg.basePort = 10000
	mirrorctx := ctlCtx{
		t:           cx.t,
		cfg:         mirrorcfg,
		dialTimeout: 7 * time.Second,
	}

	mirrorepc, err := newEtcdProcessCluster(&mirrorctx.cfg)
	if err != nil {
		cx.t.Fatalf("could not start etcd process cluster (%v)", err)
	}
	mirrorctx.epc = mirrorepc

	defer func() {
		if err = mirrorctx.epc.Close(); err != nil {
			cx.t.Fatalf("error closing etcd processes (%v)", err)
		}
	}()

	cmdArgs := append(cx.PrefixArgs(), "make-mirror")
	cmdArgs = append(cmdArgs, flags...)
	cmdArgs = append(cmdArgs, fmt.Sprintf("localhost:%d", mirrorcfg.basePort))
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

	for i := range sourcekvs {
		if err = ctlV3Put(cx, sourcekvs[i].key, sourcekvs[i].val, ""); err != nil {
			cx.t.Fatal(err)
		}
	}
	if err = ctlV3Get(cx, []string{srcprefix, "--prefix"}, sourcekvs...); err != nil {
		cx.t.Fatal(err)
	}

	if err = ctlV3Watch(mirrorctx, []string{destprefix, "--rev", "1", "--prefix"}, destkvs...); err != nil {
		cx.t.Fatal(err)
	}
}
