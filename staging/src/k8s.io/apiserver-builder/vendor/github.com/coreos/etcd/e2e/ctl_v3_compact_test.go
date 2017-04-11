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
	"strconv"
	"strings"
	"testing"
)

func TestCtlV3Compact(t *testing.T)         { testCtl(t, compactTest) }
func TestCtlV3CompactPhysical(t *testing.T) { testCtl(t, compactTest, withCompactPhysical()) }

func compactTest(cx ctlCtx) {
	compactPhysical := cx.compactPhysical
	if err := ctlV3Compact(cx, 2, compactPhysical); err != nil {
		if !strings.Contains(err.Error(), "required revision is a future revision") {
			cx.t.Fatal(err)
		}
	} else {
		cx.t.Fatalf("expected '...future revision' error, got <nil>")
	}

	var kvs = []kv{{"key", "val1"}, {"key", "val2"}, {"key", "val3"}}
	for i := range kvs {
		if err := ctlV3Put(cx, kvs[i].key, kvs[i].val, ""); err != nil {
			cx.t.Fatalf("compactTest #%d: ctlV3Put error (%v)", i, err)
		}
	}

	if err := ctlV3Get(cx, []string{"key", "--rev", "3"}, kvs[1:2]...); err != nil {
		cx.t.Errorf("compactTest: ctlV3Get error (%v)", err)
	}

	if err := ctlV3Compact(cx, 4, compactPhysical); err != nil {
		cx.t.Fatal(err)
	}

	if err := ctlV3Get(cx, []string{"key", "--rev", "3"}, kvs[1:2]...); err != nil {
		if !strings.Contains(err.Error(), "required revision has been compacted") {
			cx.t.Errorf("compactTest: ctlV3Get error (%v)", err)
		}
	} else {
		cx.t.Fatalf("expected '...has been compacted' error, got <nil>")
	}

	if err := ctlV3Compact(cx, 2, compactPhysical); err != nil {
		if !strings.Contains(err.Error(), "required revision has been compacted") {
			cx.t.Fatal(err)
		}
	} else {
		cx.t.Fatalf("expected '...has been compacted' error, got <nil>")
	}
}

func ctlV3Compact(cx ctlCtx, rev int64, physical bool) error {
	rs := strconv.FormatInt(rev, 10)
	cmdArgs := append(cx.PrefixArgs(), "compact", rs)
	if physical {
		cmdArgs = append(cmdArgs, "--physical")
	}
	return spawnWithExpect(cmdArgs, "compacted revision "+rs)
}
