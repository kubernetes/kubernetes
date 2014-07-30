// Copyright 2014 Google Inc. All Rights Reserved.
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

package procfs

import (
	"fmt"
	"reflect"
	"testing"

	"code.google.com/p/gomock/gomock"

	"github.com/google/cadvisor/utils/fs"
	"github.com/google/cadvisor/utils/fs/mockfs"
)

func TestReadProcessSchedStat(t *testing.T) {

	mockCtrl := gomock.NewController(t)
	defer mockCtrl.Finish()

	mfs := mockfs.NewMockFileSystem(mockCtrl)

	pid := 10

	stat := &ProcessSchedStat{
		NumProcesses:  1,
		Running:       100,
		RunWait:       120,
		NumTimeSlices: 130,
	}

	path := fmt.Sprintf("/proc/%v/schedstat", pid)
	content := fmt.Sprintf("%v %v %v\n", stat.Running, stat.RunWait, stat.NumTimeSlices)
	mockfs.AddTextFile(mfs, path, content)
	fs.ChangeFileSystem(mfs)

	receivedStat := &ProcessSchedStat{}
	err := receivedStat.Add(pid)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(receivedStat, stat) {
		t.Errorf("Received wrong schedstat: %+v", receivedStat)
	}
}
