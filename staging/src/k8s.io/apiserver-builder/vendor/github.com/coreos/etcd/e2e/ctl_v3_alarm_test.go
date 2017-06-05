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
	"strings"
	"testing"
)

func TestCtlV3Alarm(t *testing.T) { testCtl(t, alarmTest, withQuota(64*1024)) }

func alarmTest(cx ctlCtx) {
	// test small put still works
	smallbuf := strings.Repeat("a", int(cx.quotaBackendBytes/100))
	if err := ctlV3Put(cx, "abc", smallbuf, ""); err != nil {
		cx.t.Fatal(err)
	}

	// test big put (to be rejected, and trigger quota alarm)
	bigbuf := strings.Repeat("a", int(cx.quotaBackendBytes))
	if err := ctlV3Put(cx, "abc", bigbuf, ""); err != nil {
		if !strings.Contains(err.Error(), "etcdserver: mvcc: database space exceeded") {
			cx.t.Fatal(err)
		}
	}
	if err := ctlV3Alarm(cx, "list", "alarm:NOSPACE"); err != nil {
		cx.t.Fatal(err)
	}

	// alarm is on rejecting Puts and Txns
	if err := ctlV3Put(cx, "def", smallbuf, ""); err != nil {
		if !strings.Contains(err.Error(), "etcdserver: mvcc: database space exceeded") {
			cx.t.Fatal(err)
		}
	}

	// turn off alarm
	if err := ctlV3Alarm(cx, "disarm", "alarm:NOSPACE"); err != nil {
		cx.t.Fatal(err)
	}

	// put one more key below quota
	if err := ctlV3Put(cx, "ghi", smallbuf, ""); err != nil {
		cx.t.Fatal(err)
	}
}

func ctlV3Alarm(cx ctlCtx, cmd string, as ...string) error {
	cmdArgs := append(cx.PrefixArgs(), "alarm", cmd)
	return spawnWithExpects(cmdArgs, as...)
}
