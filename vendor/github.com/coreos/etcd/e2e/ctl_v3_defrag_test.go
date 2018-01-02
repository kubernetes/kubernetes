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

import "testing"

func TestCtlV3Defrag(t *testing.T)         { testCtl(t, defragTest) }
func TestCtlV3DefragWithAuth(t *testing.T) { testCtl(t, defragTestWithAuth) }

func maintenanceInitKeys(cx ctlCtx) {
	var kvs = []kv{{"key", "val1"}, {"key", "val2"}, {"key", "val3"}}
	for i := range kvs {
		if err := ctlV3Put(cx, kvs[i].key, kvs[i].val, ""); err != nil {
			cx.t.Fatal(err)
		}
	}
}

func defragTest(cx ctlCtx) {
	maintenanceInitKeys(cx)

	if err := ctlV3Compact(cx, 4, cx.compactPhysical); err != nil {
		cx.t.Fatal(err)
	}

	if err := ctlV3Defrag(cx); err != nil {
		cx.t.Fatalf("defragTest ctlV3Defrag error (%v)", err)
	}
}

func defragTestWithAuth(cx ctlCtx) {
	maintenanceInitKeys(cx)

	if err := authEnable(cx); err != nil {
		cx.t.Fatal(err)
	}

	cx.user, cx.pass = "root", "root"
	authSetupTestUser(cx)

	// ordinal user cannot defrag
	cx.user, cx.pass = "test-user", "pass"
	if err := ctlV3Defrag(cx); err == nil {
		cx.t.Fatal("ordinal user should not be able to issue a defrag request")
	}

	// root can defrag
	cx.user, cx.pass = "root", "root"
	if err := ctlV3Defrag(cx); err != nil {
		cx.t.Fatal(err)
	}
}

func ctlV3Defrag(cx ctlCtx) error {
	cmdArgs := append(cx.PrefixArgs(), "defrag")
	lines := make([]string, cx.epc.cfg.clusterSize)
	for i := range lines {
		lines[i] = "Finished defragmenting etcd member"
	}
	return spawnWithExpects(cmdArgs, lines...)
}
