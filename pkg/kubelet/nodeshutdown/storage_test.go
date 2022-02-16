/*
Copyright 2022 The Kubernetes Authors.

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

package nodeshutdown

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLocalStorage(t *testing.T) {
	var localStorageStateFileName = "graceful_node_shutdown_state"
	tempdir := os.TempDir()
	path := filepath.Join(tempdir, localStorageStateFileName)
	l := localStorage{
		Path: path,
	}
	now := time.Now()
	want := state{
		StartTime: now,
		EndTime:   now,
	}
	err := l.Store(want)
	if err != nil {
		t.Error(err)
		return
	}

	got := state{}
	err = l.Load(&got)
	if err != nil {
		t.Error(err)
		return
	}

	if !want.StartTime.Equal(got.StartTime) || !want.EndTime.Equal(got.EndTime) {
		t.Errorf("got %+v, want %+v", got, want)
		return
	}

	raw, err := os.ReadFile(path)
	if err != nil {
		t.Error(err)
		return
	}
	nowStr := now.Format(time.RFC3339Nano)
	wantRaw := fmt.Sprintf(`{"startTime":"` + nowStr + `","endTime":"` + nowStr + `"}`)
	if string(raw) != wantRaw {
		t.Errorf("got %s, want %s", string(raw), wantRaw)
		return
	}

}
