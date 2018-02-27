/*
Copyright 2017 The Kubernetes Authors.

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

package csi

import (
	"errors"
	"testing"

	grpctx "golang.org/x/net/context"
	"google.golang.org/grpc"
	api "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/volume/csi/fake"
)

func setupClient(t *testing.T, stageUnstageSet bool) *csiDriverClient {
	client := newCsiDriverClient("unix", "/tmp/test.sock")
	client.conn = new(grpc.ClientConn) //avoids creating conn object

	// setup mock grpc clients
	client.idClient = fake.NewIdentityClient()
	client.nodeClient = fake.NewNodeClient(stageUnstageSet)
	client.ctrlClient = fake.NewControllerClient()

	return client
}

func TestClientNodePublishVolume(t *testing.T) {
	testCases := []struct {
		name       string
		volID      string
		targetPath string
		fsType     string
		mustFail   bool
		err        error
	}{
		{name: "test ok", volID: "vol-test", targetPath: "/test/path"},
		{name: "missing volID", targetPath: "/test/path", mustFail: true},
		{name: "missing target path", volID: "vol-test", mustFail: true},
		{name: "bad fs", volID: "vol-test", targetPath: "/test/path", fsType: "badfs", mustFail: true},
		{name: "grpc error", volID: "vol-test", targetPath: "/test/path", mustFail: true, err: errors.New("grpc error")},
	}

	client := setupClient(t, false)

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		client.nodeClient.(*fake.NodeClient).SetNextError(tc.err)
		err := client.NodePublishVolume(
			grpctx.Background(),
			tc.volID,
			false,
			"",
			tc.targetPath,
			api.ReadWriteOnce,
			map[string]string{"device": "/dev/null"},
			map[string]string{"attr0": "val0"},
			map[string]string{},
			tc.fsType,
		)

		if tc.mustFail && err == nil {
			t.Error("test must fail, but err is nil")
		}
	}
}

func TestClientNodeUnpublishVolume(t *testing.T) {
	testCases := []struct {
		name       string
		volID      string
		targetPath string
		mustFail   bool
		err        error
	}{
		{name: "test ok", volID: "vol-test", targetPath: "/test/path"},
		{name: "missing volID", targetPath: "/test/path", mustFail: true},
		{name: "missing target path", volID: "vol-test", mustFail: true},
		{name: "grpc error", volID: "vol-test", targetPath: "/test/path", mustFail: true, err: errors.New("grpc error")},
	}

	client := setupClient(t, false)

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		client.nodeClient.(*fake.NodeClient).SetNextError(tc.err)
		err := client.NodeUnpublishVolume(grpctx.Background(), tc.volID, tc.targetPath)
		if tc.mustFail && err == nil {
			t.Error("test must fail, but err is nil")
		}
	}
}

func TestClientNodeStageVolume(t *testing.T) {
	testCases := []struct {
		name              string
		volID             string
		stagingTargetPath string
		fsType            string
		secret            map[string]string
		mustFail          bool
		err               error
	}{
		{name: "test ok", volID: "vol-test", stagingTargetPath: "/test/path", fsType: "ext4"},
		{name: "missing volID", stagingTargetPath: "/test/path", mustFail: true},
		{name: "missing target path", volID: "vol-test", mustFail: true},
		{name: "bad fs", volID: "vol-test", stagingTargetPath: "/test/path", fsType: "badfs", mustFail: true},
		{name: "grpc error", volID: "vol-test", stagingTargetPath: "/test/path", mustFail: true, err: errors.New("grpc error")},
	}

	client := setupClient(t, false)

	for _, tc := range testCases {
		t.Logf("Running test case: %s", tc.name)
		client.nodeClient.(*fake.NodeClient).SetNextError(tc.err)
		err := client.NodeStageVolume(
			grpctx.Background(),
			tc.volID,
			map[string]string{"device": "/dev/null"},
			tc.stagingTargetPath,
			tc.fsType,
			api.ReadWriteOnce,
			tc.secret,
			map[string]string{"attr0": "val0"},
		)

		if tc.mustFail && err == nil {
			t.Error("test must fail, but err is nil")
		}
	}
}

func TestClientNodeUnstageVolume(t *testing.T) {
	testCases := []struct {
		name              string
		volID             string
		stagingTargetPath string
		mustFail          bool
		err               error
	}{
		{name: "test ok", volID: "vol-test", stagingTargetPath: "/test/path"},
		{name: "missing volID", stagingTargetPath: "/test/path", mustFail: true},
		{name: "missing target path", volID: "vol-test", mustFail: true},
		{name: "grpc error", volID: "vol-test", stagingTargetPath: "/test/path", mustFail: true, err: errors.New("grpc error")},
	}

	client := setupClient(t, false)

	for _, tc := range testCases {
		t.Logf("Running test case: %s", tc.name)
		client.nodeClient.(*fake.NodeClient).SetNextError(tc.err)
		err := client.NodeUnstageVolume(
			grpctx.Background(),
			tc.volID, tc.stagingTargetPath,
		)
		if tc.mustFail && err == nil {
			t.Error("test must fail, but err is nil")
		}
	}
}
