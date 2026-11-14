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
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

func TestStageRecoverySignal(t *testing.T) {
	tests := []struct {
		name                    string
		disabled, notOptedIn    bool
		change                  func(*csiMountMgr, *recoveryTestNode)
		args                    volume.MounterArgs
		wantProbe, wantRecovery bool
	}{
		{name: "explicit signal", wantProbe: true, wantRecovery: true},
		{name: "failed precondition", change: func(_ *csiMountMgr, n *recoveryTestNode) {
			n.publishErr = status.Error(codes.FailedPrecondition, "staging lost")
		}, wantProbe: true, wantRecovery: true},
		{name: "gate off", disabled: true},
		{name: "driver not opted in", notOptedIn: true},
		{name: "no staging capability", change: func(_ *csiMountMgr, n *recoveryTestNode) { n.stageSupported = false }},
		{name: "no health capability", change: func(_ *csiMountMgr, n *recoveryTestNode) { n.healthSupported = false }},
		{name: "remount", args: volume.MounterArgs{IsRemount: true}},
		{name: "reconstruction", args: volume.MounterArgs{ReconstructedVolume: true}},
		{name: "publish may still run", change: func(_ *csiMountMgr, n *recoveryTestNode) {
			n.publishErr = status.Error(codes.DeadlineExceeded, "timeout")
		}},
		{name: "invalid arguments", change: func(_ *csiMountMgr, n *recoveryTestNode) {
			n.publishErr = status.Error(codes.InvalidArgument, "invalid")
		}},
		{name: "healthy", change: func(_ *csiMountMgr, n *recoveryTestNode) { n.health.HealthStatuses = nil }, wantProbe: true},
		{name: "nil health", change: func(_ *csiMountMgr, n *recoveryTestNode) { n.health = nil }, wantProbe: true},
		{name: "nil condition", change: func(_ *csiMountMgr, n *recoveryTestNode) { n.health.HealthStatuses[0] = nil }, wantProbe: true},
		{name: "wrong volume", change: func(_ *csiMountMgr, n *recoveryTestNode) { n.health.VolumeId = "another-volume" }, wantProbe: true},
		{name: "no volume ID", change: func(_ *csiMountMgr, n *recoveryTestNode) { n.health.VolumeId = "" }, wantProbe: true},
		{name: "generic inaccessible", change: func(_ *csiMountMgr, n *recoveryTestNode) { n.health.HealthStatuses[0].Reason = "NetworkUnavailable" }, wantProbe: true},
		{name: "conflict", change: func(_ *csiMountMgr, n *recoveryTestNode) { n.health.HealthStatuses[0].Reason = "StagingPathConflict" }, wantProbe: true},
		{name: "wrong status", change: func(_ *csiMountMgr, n *recoveryTestNode) {
			n.health.HealthStatuses[0].Status = csipb.VolumeHealthErrorType_DATA_LOSS
		}, wantProbe: true},
		{name: "message is not signal", change: func(_ *csiMountMgr, n *recoveryTestNode) {
			n.health.HealthStatuses[0].Reason = "Unknown"
			n.health.HealthStatuses[0].Message = stageRecoveryReason
		}, wantProbe: true},
		{name: "additional condition", change: func(_ *csiMountMgr, n *recoveryTestNode) {
			n.health.HealthStatuses = append(n.health.HealthStatuses, &csipb.VolumeHealth_VolumeHealthEntry{Status: csipb.VolumeHealthErrorType_DATA_LOSS, Reason: "Corruption"})
		}, wantProbe: true},
		{name: "duplicate signal", change: func(_ *csiMountMgr, n *recoveryTestNode) {
			n.health.HealthStatuses = append(n.health.HealthStatuses, n.health.HealthStatuses[0])
		}, wantProbe: true},
		{name: "probe error", change: func(_ *csiMountMgr, n *recoveryTestNode) { n.healthErr = status.Error(codes.Unavailable, "offline") }, wantProbe: true},
		{name: "partial target", change: func(m *csiMountMgr, _ *recoveryTestNode) {
			if err := os.MkdirAll(m.GetPath(), 0750); err != nil {
				panic(err)
			}
			if err := os.WriteFile(filepath.Join(m.GetPath(), "keep"), []byte("partially published"), 0600); err != nil {
				panic(err)
			}
		}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			_, m, n, path := newRecoveryTestVolume(t, !tc.disabled, !tc.notOptedIn)
			if tc.change != nil {
				tc.change(m, n)
			}
			err := m.SetUp(tc.args)
			if err == nil {
				t.Fatal("failed Publish must remain an error")
			}
			if got := volumetypes.IsStageRecoveryRequiredError(err); got != tc.wantRecovery {
				t.Fatalf("recovery=%v, want %v, err=%v", got, tc.wantRecovery, err)
			}
			gotProbe := n.healthRequest != nil
			if gotProbe != tc.wantProbe {
				t.Fatalf("health probe=%v, want %v; RPCs=%v", gotProbe, tc.wantProbe, n.calls)
			}
			if tc.wantRecovery && status.Code(err) != status.Code(n.publishErr) {
				t.Fatalf("lost original Publish error code: %v", err)
			}
			_, state, loadErr := loadStageRecovery(path)
			if loadErr != nil {
				t.Fatal(loadErr)
			}
			if pending := state != nil && state.Pending; pending != tc.wantRecovery {
				t.Fatalf("pending=%v, want %v", pending, tc.wantRecovery)
			}
		})
	}
}

func TestStageRecoveryRetriesWithPreservedMetadata(t *testing.T) {
	a, m, n, path := newRecoveryTestVolume(t, true, true)
	if err := m.SetUp(volume.MounterArgs{}); !volumetypes.IsStageRecoveryRequiredError(err) {
		t.Fatal(err)
	}
	before, err := os.ReadFile(filepath.Join(filepath.Dir(path), volDataFileName))
	if err != nil {
		t.Fatal(err)
	}
	calls := len(n.calls)
	if err := m.SetUp(volume.MounterArgs{}); !volumetypes.IsStageRecoveryRequiredError(err) {
		t.Fatal(err)
	}
	if len(n.calls) != calls {
		t.Fatal("pending recovery must block another Publish")
	}

	// A new attacher resumes the persisted request, rather than depending on
	// the mounter which observed the health signal.
	resumed := *a
	n.stageErr = status.Error(codes.Internal, "restage failed")
	err = resumed.MountDevice(m.spec, "", path, volume.DeviceMounterArgs{})
	if !volumetypes.IsUncertainProgressError(err) {
		t.Fatalf("failed restage must remain uncertain: %v", err)
	}
	after, err := os.ReadFile(filepath.Join(filepath.Dir(path), volDataFileName))
	if err != nil || !reflect.DeepEqual(before, after) {
		t.Fatalf("failed restage changed global metadata: %v", err)
	}

	n.stageErr = nil
	if err := resumed.MountDevice(m.spec, "", path, volume.DeviceMounterArgs{}); err != nil {
		t.Fatal(err)
	}
	n.publishErr = nil
	if err := m.SetUp(volume.MounterArgs{}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(n.calls, []string{"stage", "publish", "health", "stage", "stage", "publish"}) {
		t.Fatalf("unexpected recovery order: %v", n.calls)
	}
	_, state, err := loadStageRecovery(path)
	if err != nil || state == nil || state.Pending {
		t.Fatalf("recovery did not complete: state=%+v err=%v", state, err)
	}
}

func TestStageRecoveryRechecksPrerequisites(t *testing.T) {
	for _, change := range []string{"gate", "stage capability", "health capability", "identity", "mount options", "access mode"} {
		t.Run(change, func(t *testing.T) {
			a, m, n, path := newRecoveryTestVolume(t, true, true)
			if err := m.SetUp(volume.MounterArgs{}); !volumetypes.IsStageRecoveryRequiredError(err) {
				t.Fatal(err)
			}
			switch change {
			case "gate":
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CSIVolumeHealth, false)
			case "stage capability":
				n.stageSupported = false
			case "health capability":
				n.healthSupported = false
			case "identity":
				m.spec.PersistentVolume.UID = "replaced-pv"
			case "mount options":
				m.spec.PersistentVolume.Spec.MountOptions = []string{"different"}
			case "access mode":
				m.spec.PersistentVolume.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce}
			}
			calls := len(n.calls)
			err := a.MountDevice(m.spec, "", path, volume.DeviceMounterArgs{})
			if !volumetypes.IsUncertainProgressError(err) {
				t.Fatalf("unsafe restage was not blocked: %v", err)
			}
			if len(n.calls) != calls {
				t.Fatalf("unexpected recovery RPC: %v", n.calls)
			}
			_, state, err := loadStageRecovery(path)
			if err != nil || state == nil || !state.Pending {
				t.Fatalf("pending state was lost: %v", err)
			}
		})
	}
}

func TestStageRecoveryRefreshesSecrets(t *testing.T) {
	a, m, n, path := newRecoveryTestVolume(t, true, true)
	if err := m.SetUp(volume.MounterArgs{}); !volumetypes.IsStageRecoveryRequiredError(err) {
		t.Fatal(err)
	}
	ref := &v1.SecretReference{Name: "stage-secret", Namespace: "default"}
	m.spec.PersistentVolume.Spec.CSI.NodeStageSecretRef = ref
	secret := &v1.Secret{ObjectMeta: metav1.ObjectMeta{Name: ref.Name, Namespace: ref.Namespace}, Data: map[string][]byte{"key": []byte("rotated-test-value")}}
	if _, err := m.k8s.CoreV1().Secrets(ref.Namespace).Create(t.Context(), secret, metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}
	if err := a.MountDevice(m.spec, "", path, volume.DeviceMounterArgs{}); err != nil {
		t.Fatal(err)
	}
	if n.stageRequest.GetSecrets()["key"] != "rotated-test-value" {
		t.Fatal("Stage did not fetch current credentials")
	}
	data, err := os.ReadFile(filepath.Join(filepath.Dir(path), volDataFileName))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "rotated-test-value") {
		t.Fatal("credentials persisted in global metadata")
	}
}

func TestStageRecoveryRequiresBaseline(t *testing.T) {
	_, m, n, path := newRecoveryTestVolume(t, true, true)
	// Neither a failed probe nor missing Stage history authorizes recovery.
	n.healthErr = errors.New("probe unavailable")
	if err := m.SetUp(volume.MounterArgs{}); volumetypes.IsStageRecoveryRequiredError(err) {
		t.Fatal("probe failure requested recovery")
	}
	if err := os.Remove(filepath.Join(filepath.Dir(path), volDataFileName)); err != nil {
		t.Fatal(err)
	}
	n.healthErr = nil
	if err := m.SetUp(volume.MounterArgs{}); volumetypes.IsStageRecoveryRequiredError(err) {
		t.Fatal("missing baseline requested recovery")
	}
}

func TestStageRecoveryPersistenceFailure(t *testing.T) {
	_, m, n, path := newRecoveryTestVolume(t, true, true)
	dir := filepath.Dir(path)
	backup := dir + ".saved"
	n.onHealth = func() {
		if err := os.Rename(dir, backup); err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if _, err := os.Stat(backup); err == nil {
				if err := os.Remove(dir); err != nil {
					t.Error(err)
				}
				if err := os.Rename(backup, dir); err != nil {
					t.Error(err)
				}
			}
		})
		if err := os.WriteFile(dir, []byte("block recovery write"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	err := m.SetUp(volume.MounterArgs{})
	if err == nil || volumetypes.IsStageRecoveryRequiredError(err) {
		t.Fatalf("persistence failure must not invalidate staging: %v", err)
	}
	if status.Code(err) != codes.Internal {
		t.Fatalf("original Publish error changed: %v", err)
	}
	if err := os.Remove(dir); err != nil {
		t.Fatal(err)
	}
	if err := os.Rename(backup, dir); err != nil {
		t.Fatal(err)
	}
	_, state, err := loadStageRecovery(path)
	if err != nil || state == nil || state.Pending {
		t.Fatalf("failed write changed prior metadata: %+v %v", state, err)
	}
}

func TestStageRecoveryDoesNotProbeSuccessfulPublish(t *testing.T) {
	_, m, n, _ := newRecoveryTestVolume(t, true, true)
	n.publishErr = nil
	if err := m.SetUp(volume.MounterArgs{}); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(n.calls, []string{"stage", "publish"}) {
		t.Fatalf("unexpected RPCs: %v", n.calls)
	}
}

func TestStageRecoveryUnstageCleansInterruptedWrite(t *testing.T) {
	a, m, _, path := newRecoveryTestVolume(t, true, true)
	if err := m.SetUp(volume.MounterArgs{}); !volumetypes.IsStageRecoveryRequiredError(err) {
		t.Fatal(err)
	}
	temporaryPath := filepath.Join(filepath.Dir(path), stageRecoveryTemporaryFile)
	if err := os.WriteFile(temporaryPath, []byte("incomplete metadata"), 0600); err != nil {
		t.Fatal(err)
	}
	if err := a.UnmountDevice(path); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Dir(path)); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("global metadata not cleaned: %v", err)
	}
}
