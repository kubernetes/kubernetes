// Copyright 2015 CoreOS, Inc.
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

package integration

import (
	"os/exec"
	"testing"

	"github.com/coreos/etcd/pkg/testutil"
)

func TestUpgradeMember(t *testing.T) {
	defer testutil.AfterTest(t)
	m := mustNewMember(t, "integration046", nil, nil)
	cmd := exec.Command("cp", "-r", "testdata/integration046_data/conf", "testdata/integration046_data/log", "testdata/integration046_data/snapshot", m.DataDir)
	err := cmd.Run()
	if err != nil {
		t.Fatal(err)
	}
	if err := m.Launch(); err != nil {
		t.Fatal(err)
	}
	defer m.Terminate(t)
	m.WaitOK(t)

	clusterMustProgress(t, []*member{m})
}
