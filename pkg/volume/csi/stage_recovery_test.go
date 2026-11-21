/*
Copyright 2026 The Kubernetes Authors.

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
	"context"
	"io"
	"os"
	"reflect"
	"testing"
	"time"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	fakecsi "k8s.io/kubernetes/pkg/volume/csi/fake"
)

type recoveryTestNode struct {
	csipb.NodeClient
	calls                           []string
	health                          *csipb.VolumeHealth
	healthErr, publishErr, stageErr error
	onHealth                        func()
	healthRequest                   *csipb.NodeGetVolumeHealthRequest
	stageRequest                    *csipb.NodeStageVolumeRequest
	stageSupported, healthSupported bool
}

func (n *recoveryTestNode) NodeGetCapabilities(context.Context, *csipb.NodeGetCapabilitiesRequest, ...grpc.CallOption) (*csipb.NodeGetCapabilitiesResponse, error) {
	resp := &csipb.NodeGetCapabilitiesResponse{}
	for capability, supported := range map[csipb.NodeServiceCapability_RPC_Type]bool{
		csipb.NodeServiceCapability_RPC_STAGE_UNSTAGE_VOLUME: n.stageSupported,
		csipb.NodeServiceCapability_RPC_GET_VOLUME_HEALTH:    n.healthSupported,
	} {
		if supported {
			resp.Capabilities = append(resp.Capabilities, &csipb.NodeServiceCapability{Type: &csipb.NodeServiceCapability_Rpc{Rpc: &csipb.NodeServiceCapability_RPC{Type: capability}}})
		}
	}
	return resp, nil
}

func (n *recoveryTestNode) NodeStageVolume(_ context.Context, req *csipb.NodeStageVolumeRequest, _ ...grpc.CallOption) (*csipb.NodeStageVolumeResponse, error) {
	n.calls = append(n.calls, "stage")
	n.stageRequest = req
	return &csipb.NodeStageVolumeResponse{}, n.stageErr
}

func (n *recoveryTestNode) NodePublishVolume(_ context.Context, _ *csipb.NodePublishVolumeRequest, _ ...grpc.CallOption) (*csipb.NodePublishVolumeResponse, error) {
	n.calls = append(n.calls, "publish")
	return &csipb.NodePublishVolumeResponse{}, n.publishErr
}

func (n *recoveryTestNode) NodeGetVolumeHealth(_ context.Context, req *csipb.NodeGetVolumeHealthRequest, _ ...grpc.CallOption) (*csipb.NodeGetVolumeHealthResponse, error) {
	n.calls = append(n.calls, "health")
	n.healthRequest = req
	if n.onHealth != nil {
		n.onHealth()
	}
	return &csipb.NodeGetVolumeHealthResponse{VolumeHealth: n.health}, n.healthErr
}

func newRecoveryTestVolume(t *testing.T, enabled, optedIn bool) (*csiAttacher, *csiMountMgr, *recoveryTestNode, string) {
	t.Helper()
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, enabled)
	driver := getTestCSIDriver(testDriver, nil, nil, []storagev1.VolumeLifecycleMode{storagev1.VolumeLifecyclePersistent})
	if optedIn {
		driver.Annotations = map[string]string{"csi.storage.k8s.io/stage-recovery-poc": "true"}
	}
	plug, dir := newTestPlugin(t, fakeclient.NewSimpleClientset(driver))
	t.Cleanup(func() { os.RemoveAll(dir) })
	registerFakePlugin(testDriver, "endpoint", []string{"1.0.0"}, t)
	pv := makeTestPV("recovery-pv", 10, testDriver, "recovery-volume")
	pv.UID = "recovery-pv-uid"
	pv.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteMany}
	spec := volume.NewSpecFromPersistentVolume(pv, false)
	mounter, err := plug.NewMounter(spec, &v1.Pod{ObjectMeta: metav1.ObjectMeta{UID: testPodUID, Namespace: testns}})
	if err != nil {
		t.Fatal(err)
	}
	m := mounter.(*csiMountMgr)
	attachID := getAttachmentName(m.volumeID, string(m.driverName), string(plug.host.GetNodeName()))
	if _, err := m.k8s.StorageV1().VolumeAttachments().Create(context.Background(), makeTestAttachment(attachID, "test-node", pv.Name), metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	node := &recoveryTestNode{
		NodeClient:     fakecsi.NewNodeClient(true),
		stageSupported: true, healthSupported: true,
		publishErr: status.Error(codes.Internal, "staging is lost"),
		health: &csipb.VolumeHealth{VolumeId: m.volumeID, HealthStatuses: []*csipb.VolumeHealth_VolumeHealthEntry{
			{Status: csipb.VolumeHealthErrorType_INACCESSIBLE, Reason: "KubeletStageRecoveryRequired"},
		}},
	}
	client := &csiDriverClient{driverName: m.driverName, nodeV1ClientCreator: func(csiAddr, *MetricsManager) (csipb.NodeClient, io.Closer, error) {
		return node, fakecsi.NewCloser(t), nil
	}}
	m.csiClient = client
	attacher := &csiAttacher{plugin: plug, k8s: m.k8s, csiClient: client, watchTimeout: time.Second}
	path, err := attacher.GetDeviceMountPath(spec)
	if err != nil {
		t.Fatal(err)
	}
	if err := attacher.MountDevice(spec, "", path, volume.DeviceMounterArgs{}); err != nil {
		t.Fatal(err)
	}
	return attacher, m, node, path
}

func TestStageRecoveryProbesFailedPublish(t *testing.T) {
	_, m, node, path := newRecoveryTestVolume(t, true, true)
	if err := m.SetUp(volume.MounterArgs{}); err == nil {
		t.Fatal("Publish must fail")
	}
	if !reflect.DeepEqual(node.calls, []string{"stage", "publish", "health"}) {
		t.Fatalf("RPCs = %v, want Stage, failed Publish, then a health probe", node.calls)
	}
	req := node.healthRequest
	if req.GetVolumeId() != m.volumeID || req.GetStagingTargetPath() != path || req.GetVolumePublishPath() != m.GetPath() {
		t.Fatalf("health probe did not identify the failed publication: %v", req)
	}
}
