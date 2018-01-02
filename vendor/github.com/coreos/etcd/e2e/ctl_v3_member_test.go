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
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/coreos/etcd/etcdserver/etcdserverpb"
)

func TestCtlV3MemberList(t *testing.T) { testCtl(t, memberListTest) }
func TestCtlV3MemberRemove(t *testing.T) {
	testCtl(t, memberRemoveTest, withQuorum(), withNoStrictReconfig())
}
func TestCtlV3MemberAdd(t *testing.T)    { testCtl(t, memberAddTest) }
func TestCtlV3MemberUpdate(t *testing.T) { testCtl(t, memberUpdateTest) }

func memberListTest(cx ctlCtx) {
	if err := ctlV3MemberList(cx); err != nil {
		cx.t.Fatalf("memberListTest ctlV3MemberList error (%v)", err)
	}
}

func ctlV3MemberList(cx ctlCtx) error {
	cmdArgs := append(cx.PrefixArgs(), "member", "list")
	lines := make([]string, cx.cfg.clusterSize)
	for i := range lines {
		lines[i] = "started"
	}
	return spawnWithExpects(cmdArgs, lines...)
}

func getMemberList(cx ctlCtx) (etcdserverpb.MemberListResponse, error) {
	cmdArgs := append(cx.PrefixArgs(), "--write-out", "json", "member", "list")

	proc, err := spawnCmd(cmdArgs)
	if err != nil {
		return etcdserverpb.MemberListResponse{}, err
	}
	var txt string
	txt, err = proc.Expect("members")
	if err != nil {
		return etcdserverpb.MemberListResponse{}, err
	}
	if err = proc.Close(); err != nil {
		return etcdserverpb.MemberListResponse{}, err
	}

	resp := etcdserverpb.MemberListResponse{}
	dec := json.NewDecoder(strings.NewReader(txt))
	if err := dec.Decode(&resp); err == io.EOF {
		return etcdserverpb.MemberListResponse{}, err
	}
	return resp, nil
}

func memberRemoveTest(cx ctlCtx) {
	n1 := cx.cfg.clusterSize
	if n1 < 2 {
		cx.t.Fatalf("%d-node is too small to test 'member remove'", n1)
	}
	resp, err := getMemberList(cx)
	if err != nil {
		cx.t.Fatal(err)
	}
	if n1 != len(resp.Members) {
		cx.t.Fatalf("expected %d, got %d", n1, len(resp.Members))
	}

	var (
		memIDToRemove = fmt.Sprintf("%x", resp.Header.MemberId)
		cluserID      = fmt.Sprintf("%x", resp.Header.ClusterId)
	)
	if err = ctlV3MemberRemove(cx, memIDToRemove, cluserID); err != nil {
		cx.t.Fatal(err)
	}
}

func ctlV3MemberRemove(cx ctlCtx, memberID, clusterID string) error {
	cmdArgs := append(cx.PrefixArgs(), "member", "remove", memberID)
	return spawnWithExpect(cmdArgs, fmt.Sprintf("%s removed from cluster %s", memberID, clusterID))
}

func memberAddTest(cx ctlCtx) {
	if err := ctlV3MemberAdd(cx, fmt.Sprintf("http://localhost:%d", etcdProcessBasePort+11)); err != nil {
		cx.t.Fatal(err)
	}
}

func ctlV3MemberAdd(cx ctlCtx, peerURL string) error {
	cmdArgs := append(cx.PrefixArgs(), "member", "add", "newmember", fmt.Sprintf("--peer-urls=%s", peerURL))
	return spawnWithExpect(cmdArgs, " added to cluster ")
}

func memberUpdateTest(cx ctlCtx) {
	mr, err := getMemberList(cx)
	if err != nil {
		cx.t.Fatal(err)
	}

	peerURL := fmt.Sprintf("http://localhost:%d", etcdProcessBasePort+11)
	memberID := fmt.Sprintf("%x", mr.Members[0].ID)
	if err = ctlV3MemberUpdate(cx, memberID, peerURL); err != nil {
		cx.t.Fatal(err)
	}
}

func ctlV3MemberUpdate(cx ctlCtx, memberID, peerURL string) error {
	cmdArgs := append(cx.PrefixArgs(), "member", "update", memberID, fmt.Sprintf("--peer-urls=%s", peerURL))
	return spawnWithExpect(cmdArgs, " updated in cluster ")
}
