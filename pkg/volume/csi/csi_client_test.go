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

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	grpctx "golang.org/x/net/context"
	"google.golang.org/grpc"
	api "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/volume/csi/fake"
)

func setupClient(t *testing.T) *csiDriverClient {
	client := newCsiDriverClient("unix", "/tmp/test.sock")
	client.conn = new(grpc.ClientConn) //avoids creating conn object

	// setup mock grpc clients
	client.idClient = fake.NewIdentityClient()
	client.nodeClient = fake.NewNodeClient()
	client.ctrlClient = fake.NewControllerClient()

	return client
}

func TestClientAssertSupportedVersion(t *testing.T) {
	testCases := []struct {
		testName string
		ver      *csipb.Version
		mustFail bool
		err      error
	}{
		{testName: "supported version", ver: &csipb.Version{Major: 0, Minor: 0, Patch: 0}},
		{testName: "supported version", ver: &csipb.Version{Major: 0, Minor: 1, Patch: 0}},
		{testName: "supported version", ver: &csipb.Version{Major: 0, Minor: 1, Patch: 10}},
		{testName: "supported version", ver: &csipb.Version{Major: 1, Minor: 1, Patch: 0}},
		{testName: "supported version", ver: &csipb.Version{Major: 1, Minor: 0, Patch: 10}},
		{testName: "unsupported version", ver: &csipb.Version{Major: 10, Minor: 0, Patch: 0}, mustFail: true},
		{testName: "unsupported version", ver: &csipb.Version{Major: 0, Minor: 10, Patch: 0}, mustFail: true},
		{testName: "grpc error", ver: &csipb.Version{Major: 0, Minor: 1, Patch: 0}, mustFail: true, err: errors.New("grpc error")},
	}

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.testName)
		client := setupClient(t)
		client.idClient.(*fake.IdentityClient).SetNextError(tc.err)
		err := client.AssertSupportedVersion(grpctx.Background(), tc.ver)
		if tc.mustFail && err == nil {
			t.Error("test must fail, but err = nil")
		}
	}
}

func TestClientNodeProbe(t *testing.T) {
	testCases := []struct {
		testName string
		ver      *csipb.Version
		mustFail bool
		err      error
	}{
		{testName: "supported version", ver: &csipb.Version{Major: 0, Minor: 1, Patch: 0}},
		{testName: "grpc error", ver: &csipb.Version{Major: 0, Minor: 1, Patch: 0}, mustFail: true, err: errors.New("grpc error")},
	}

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.testName)
		client := setupClient(t)
		client.nodeClient.(*fake.NodeClient).SetNextError(tc.err)
		err := client.NodeProbe(grpctx.Background(), tc.ver)
		if tc.mustFail && err == nil {
			t.Error("test must fail, but err = nil")
		}
	}
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

	client := setupClient(t)

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		client.nodeClient.(*fake.NodeClient).SetNextError(tc.err)
		err := client.NodePublishVolume(
			grpctx.Background(),
			tc.volID,
			false,
			tc.targetPath,
			api.ReadWriteOnce,
			map[string]string{"device": "/dev/null"},
			map[string]string{"attr0": "val0"},
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

	client := setupClient(t)

	for _, tc := range testCases {
		t.Logf("test case: %s", tc.name)
		client.nodeClient.(*fake.NodeClient).SetNextError(tc.err)
		err := client.NodeUnpublishVolume(grpctx.Background(), tc.volID, tc.targetPath)
		if tc.mustFail && err == nil {
			t.Error("test must fail, but err is nil")
		}
	}
}
