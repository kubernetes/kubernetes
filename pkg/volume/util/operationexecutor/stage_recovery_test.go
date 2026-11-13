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

package operationexecutor

import (
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

func TestStageRecoveryPreservesExistingConsumer(t *testing.T) {
	for _, confirmedLost := range []bool{false, true} {
		for _, code := range []codes.Code{codes.Internal, codes.FailedPrecondition} {
			name := code.String()
			if confirmedLost {
				name += "/confirmed-mount-loss"
			}
			t.Run(name, func(t *testing.T) {
				plugin := volumetesting.ProbeVolumePlugins(volume.VolumeConfig{})[0].(*volumetesting.FakeVolumePlugin)
				testPlugin := &stagingMountTestPlugin{FakeVolumePlugin: plugin, path: "/test/globalmount"}
				host := volumetesting.NewFakeKubeletVolumeHost(t, t.TempDir(), nil, []volume.VolumePlugin{testPlugin})
				manager := host.GetPluginMgr()
				plugin.DisableNodeExpansion = true
				generator := getTestOperationGenerator(manager)
				state := &stagingMountTestASW{mountedPods: map[volumetypes.UniquePodName]bool{}}
				pv := getTestPV("stage-recovery", "1Gi")
				pv.Spec.AccessModes = []v1.PersistentVolumeAccessMode{v1.ReadWriteMany}
				pod := getTestPod("old-consumer", "claim01")
				request := VolumeToMount{VolumeName: v1.UniqueVolumeName(pv.Name), PodName: volumetypes.UniquePodName(pod.UID), VolumeSpec: volume.NewSpecFromPersistentVolume(pv, false), Pod: pod}
				lost := false
				plugin.SetUpHook = func(_ volume.VolumePlugin, _ volume.MounterArgs) error {
					if lost && plugin.Attachers[0].GetMountDeviceCallCount() == 1 {
						err := status.Error(code, "publish failed")
						if confirmedLost {
							return volumetypes.NewStageRecoveryRequiredError(err)
						}
						return err
					}
					return nil
				}
				run := func() error {
					operation := generator.GenerateMountVolumeFunc(time.Second, request, state, false)
					_, err := operation.Run()
					return err
				}
				if err := run(); err != nil {
					t.Fatal(err)
				}
				oldPod := request.PodName
				lost = true
				request.Pod = getTestPod("new-consumer", "claim01")
				request.Pod.UID = "new-consumer"
				request.PodName = volumetypes.UniquePodName(request.Pod.UID)
				if err := run(); err == nil {
					t.Fatal("Publish must fail on lost stage")
				}
				if confirmedLost {
					if state.deviceMountState != DeviceMountUncertain {
						t.Fatalf("device state=%v, want uncertain", state.deviceMountState)
					}
					if state.deviceMountPath == "" || state.uncertainMountPath != state.deviceMountPath {
						t.Fatalf("uncertain path=%q, initial Stage path=%q", state.uncertainMountPath, state.deviceMountPath)
					}
					if err := run(); err != nil {
						t.Fatal(err)
					}
					if got := plugin.Attachers[0].GetMountDeviceCallCount(); got != 2 {
						t.Fatalf("Stage calls=%d, want 2", got)
					}
					if !state.mountedPods[request.PodName] {
						t.Fatal("new consumer was not mounted")
					}
				} else {
					if err := run(); err == nil {
						t.Fatal("generic publish error must remain an error")
					}
					if got := plugin.Attachers[0].GetMountDeviceCallCount(); got != 1 {
						t.Fatalf("generic error triggered Stage: calls=%d", got)
					}
				}
				if !state.mountedPods[oldPod] {
					t.Fatal("old consumer mount state changed")
				}
			})
		}
	}
}

type stagingMountTestASW struct {
	attemptTrackingTestASW
	mountedPods                         map[volumetypes.UniquePodName]bool
	deviceMountPath, uncertainMountPath string
}

func (a *stagingMountTestASW) MarkDeviceAsMounted(_ v1.UniqueVolumeName, _ string, path string, _ string) error {
	a.deviceMountPath = path
	a.deviceMountState = DeviceGloballyMounted
	return nil
}
func (a *stagingMountTestASW) MarkDeviceAsUncertain(_ v1.UniqueVolumeName, _ string, path string, _ string) error {
	a.uncertainMountPath = path
	a.deviceMountState = DeviceMountUncertain
	return nil
}
func (a *stagingMountTestASW) GetDeviceMountState(_ v1.UniqueVolumeName) DeviceMountState {
	return a.deviceMountState
}
func (a *stagingMountTestASW) GetVolumeMountState(_ v1.UniqueVolumeName, pod volumetypes.UniquePodName) VolumeMountState {
	if a.mountedPods[pod] {
		return VolumeMounted
	}
	return VolumeNotMounted
}
func (a *stagingMountTestASW) IsVolumeMountedElsewhere(_ v1.UniqueVolumeName, pod volumetypes.UniquePodName) bool {
	for other, mounted := range a.mountedPods {
		if other != pod && mounted {
			return true
		}
	}
	return false
}
func (a *stagingMountTestASW) IsVolumeReconstructed(_ v1.UniqueVolumeName, _ volumetypes.UniquePodName) bool {
	return false
}
func (a *stagingMountTestASW) MarkVolumeMountAsUncertain(_ MarkVolumeOpts) error { return nil }
func (a *stagingMountTestASW) MarkVolumeAsMounted(opts MarkVolumeOpts) error {
	a.mountedPods[opts.PodName] = true
	return nil
}

// The generic fake returns an empty staging path; use a real path to verify
// that invalidation preserves the path supplied to the successful Stage.
type stagingMountTestPlugin struct {
	*volumetesting.FakeVolumePlugin
	path string
}

func (p *stagingMountTestPlugin) NewDeviceMounter() (volume.DeviceMounter, error) {
	mounter, err := p.FakeVolumePlugin.NewDeviceMounter()
	return &stagingMountTestDeviceMounter{DeviceMounter: mounter, path: p.path}, err
}

type stagingMountTestDeviceMounter struct {
	volume.DeviceMounter
	path string
}

func (m *stagingMountTestDeviceMounter) GetDeviceMountPath(*volume.Spec) (string, error) {
	return m.path, nil
}
