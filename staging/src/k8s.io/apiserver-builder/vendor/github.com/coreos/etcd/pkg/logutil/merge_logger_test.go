// Copyright 2015 The etcd Authors
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

package logutil

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/coreos/pkg/capnslog"
)

var (
	testLogger = capnslog.NewPackageLogger("github.com/coreos/etcd/pkg", "logutil")
)

func TestMergeLogger(t *testing.T) {
	var (
		txt      = "hello"
		repeatN  = 6
		duration = 2049843762 * time.Nanosecond
		mg       = NewMergeLogger(testLogger)
	)
	// overwrite this for testing
	defaultMergePeriod = time.Minute

	for i := 0; i < repeatN; i++ {
		mg.MergeError(txt)
		if i == 0 {
			time.Sleep(duration)
		}
	}

	if len(mg.statusm) != 1 {
		t.Errorf("got = %d, want = %d", len(mg.statusm), 1)
	}

	var l line
	for k := range mg.statusm {
		l = k
		break
	}

	if l.level != capnslog.ERROR {
		t.Errorf("got = %v, want = %v", l.level, capnslog.DEBUG)
	}
	if l.str != txt {
		t.Errorf("got = %s, want = %s", l.str, txt)
	}
	if mg.statusm[l].count != repeatN-1 {
		t.Errorf("got = %d, want = %d", mg.statusm[l].count, repeatN-1)
	}
	sum := mg.statusm[l].summary(time.Now())
	pre := fmt.Sprintf("[merged %d repeated lines in ", repeatN-1)
	if !strings.HasPrefix(sum, pre) {
		t.Errorf("got = %s, want = %s...", sum, pre)
	}
}
