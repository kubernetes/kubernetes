// Copyright 2016 The rkt Authors
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

// +build host coreos src

package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/coreos/rkt/tests/testutils"
)

func getPodCgroups(shortUUID string) ([]string, error) {
	var podCgroups []string
	walkCgroups := func(path string, info os.FileInfo, err error) error {
		// Maybe the file doesn't exist anymore. We've seen this during tests,
		// not sure about the reason though.
		if os.IsNotExist(err) {
			return nil
		}
		if err != nil {
			return err
		}
		if info.Mode().IsDir() && strings.Contains(path, shortUUID) {
			podCgroups = append(podCgroups, path)
		}
		return nil
	}
	if err := filepath.Walk("/sys/fs/cgroup", walkCgroups); err != nil {
		return nil, err
	}
	return podCgroups, nil
}

func TestGCCgroup(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	imagePath := getInspectImagePath()
	cmd := fmt.Sprintf("%s --insecure-options=image prepare %s", ctx.Cmd(), imagePath)
	uuid := runRktAndGetUUID(t, cmd)
	shortUUID := strings.Split(uuid, "-")[0]

	cmd = fmt.Sprintf("%s run-prepared %s", ctx.Cmd(), shortUUID)
	runRktAndCheckOutput(t, cmd, "", false)

	cgs, err := getPodCgroups(shortUUID)
	if err != nil {
		t.Fatalf("error getting pod cgroups: %v", err)
	}
	if len(cgs) == 0 {
		t.Fatalf("expected pod cgroup directories after run, but found none")
	}

	runGC(t, ctx)

	cgs, err = getPodCgroups(shortUUID)
	if err != nil {
		t.Fatalf("error getting pod cgroups: %v", err)
	}
	if len(cgs) > 0 {
		t.Fatalf(fmt.Sprintf("expected no pod cgroup directories after GC, but found %d: %v", len(cgs), cgs))
	}
}
