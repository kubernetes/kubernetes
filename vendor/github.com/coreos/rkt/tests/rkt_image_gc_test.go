// Copyright 2015 The rkt Authors
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

// +build host coreos src kvm

package main

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/coreos/gexpect"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/tests/testutils"
)

func TestImageGCTreeStore(t *testing.T) {
	ctx := testutils.NewRktRunCtx()
	defer ctx.Cleanup()

	expectedTreeStores := 2
	// If overlayfs is not supported only the stage1 image is rendered in the treeStore
	if common.SupportsOverlay() != nil {
		expectedTreeStores = 1
	}

	// at this point we know that RKT_INSPECT_IMAGE env var is not empty
	referencedACI := os.Getenv("RKT_INSPECT_IMAGE")
	cmd := fmt.Sprintf("%s --insecure-options=image run --mds-register=false %s", ctx.Cmd(), referencedACI)
	t.Logf("Running %s: %v", referencedACI, cmd)
	child, err := gexpect.Spawn(cmd)
	if err != nil {
		t.Fatalf("Cannot exec: %v", err)
	}
	waitOrFail(t, child, 0)

	treeStoreIDs, err := getTreeStoreIDs(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// We expect 2 treeStoreIDs for stage1 and app (only 1 if overlay is not supported/enabled)
	if len(treeStoreIDs) != expectedTreeStores {
		t.Fatalf("expected %d entries in the treestore but found %d entries", expectedTreeStores, len(treeStoreIDs))
	}

	runImageGC(t, ctx)

	treeStoreIDs, err = getTreeStoreIDs(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// We expect 1/2 treeStoreIDs again as no pod gc has been executed
	if len(treeStoreIDs) != expectedTreeStores {
		t.Fatalf("expected %d entries in the treestore but found %d entries", expectedTreeStores, len(treeStoreIDs))
	}

	runGC(t, ctx)
	runImageGC(t, ctx)

	treeStoreIDs, err = getTreeStoreIDs(ctx)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(treeStoreIDs) != 0 {
		t.Fatalf("expected empty treestore but found %d entries", len(treeStoreIDs))
	}
}

func getTreeStoreIDs(ctx *testutils.RktRunCtx) (map[string]struct{}, error) {
	treeStoreIDs := map[string]struct{}{}
	ls, err := ioutil.ReadDir(filepath.Join(ctx.DataDir(), "cas", "tree"))
	if err != nil {
		if os.IsNotExist(err) {
			return treeStoreIDs, nil
		}
		return nil, fmt.Errorf("cannot read treestore directory: %v", err)
	}

	for _, p := range ls {
		if p.IsDir() {
			id := filepath.Base(p.Name())
			treeStoreIDs[id] = struct{}{}
		}
	}
	return treeStoreIDs, nil
}
