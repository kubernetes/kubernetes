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
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"

	csipb "github.com/container-storage-interface/spec/lib/go/csi"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	v1 "k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// These are deliberately non-standard PoC opt-ins, not CSI health semantics.
// The driver must guarantee that the failed Publish has ended without a target
// requiring cleanup, and that repeating Stage is safe for existing consumers.
// Stage must revalidate the volume identity and reject any conflicting resource.
const (
	stageRecoveryAnnotation    = "csi.storage.k8s.io/stage-recovery-poc"
	stageRecoveryReason        = "KubeletStageRecoveryRequired"
	stageRecoveryDataKey       = "stageRecoveryPoC"
	stageRecoveryTemporaryFile = ".stage-recovery.tmp"
)

type stageRecoveryState struct {
	Version                  int
	PVUID, Path, RequestHash string
	Pending                  bool
}

func (p *csiPlugin) stageRecoveryEnabled(driver string) bool {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIVolumeHealth) || p.csiDriverLister == nil {
		return false
	}
	obj, err := p.csiDriverLister.Get(driver)
	return err == nil && obj.Annotations[stageRecoveryAnnotation] == "true"
}

// Do not use the API-facing health mapper here: it truncates and deduplicates
// conditions for reporting. Recovery requires the complete, unambiguous reply.
func stageRecoveryRequested(health *csipb.VolumeHealth, volumeID string) bool {
	if volumeID == "" || health.GetVolumeId() != volumeID || len(health.GetHealthStatuses()) != 1 {
		return false
	}
	entry := health.GetHealthStatuses()[0]
	return entry.GetStatus() == csipb.VolumeHealthErrorType_INACCESSIBLE && entry.GetReason() == stageRecoveryReason
}

func loadStageRecovery(path string) (map[string]string, *stageRecoveryState, error) {
	data, err := loadVolumeData(filepath.Dir(path), volDataFileName)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil, nil
	}
	if err != nil {
		return nil, nil, err
	}
	encoded, found := data[stageRecoveryDataKey]
	if !found {
		return data, nil, nil
	}
	var state stageRecoveryState
	if err := json.Unmarshal([]byte(encoded), &state); err != nil {
		return data, nil, fmt.Errorf("invalid CSI stage recovery record: %w", err)
	}
	if state.Version != 1 || state.PVUID == "" || state.Path != path || len(state.RequestHash) != sha256.Size*2 {
		return data, nil, fmt.Errorf("invalid CSI stage recovery identity or version")
	}
	return data, &state, nil
}

func matchingStageRecovery(data map[string]string, state *stageRecoveryState, spec *volume.Spec) bool {
	if state == nil || spec == nil || spec.PersistentVolume == nil || spec.PersistentVolume.Spec.CSI == nil {
		return false
	}
	pv := spec.PersistentVolume
	return state.PVUID == string(pv.UID) &&
		data[volDataKey.driverName] == pv.Spec.CSI.Driver &&
		data[volDataKey.volHandle] == pv.Spec.CSI.VolumeHandle &&
		data[volDataKey.specVolID] == spec.Name()
}

func stageRecoveryRequestHash(spec *volume.Spec, path string, accessMode v1.PersistentVolumeAccessMode, options []string, group *int64, publishContext map[string]string) string {
	pv := spec.PersistentVolume
	request := struct {
		Version                               int
		PVUID, Driver, VolumeID, Path, FSType string
		AccessMode                            v1.PersistentVolumeAccessMode
		Options                               []string
		Group                                 *int64
		Attributes, PublishContext            map[string]string
	}{
		Version: 1, PVUID: string(pv.UID), Driver: pv.Spec.CSI.Driver,
		VolumeID: pv.Spec.CSI.VolumeHandle, Path: path, FSType: pv.Spec.CSI.FSType,
		AccessMode: accessMode, Options: options, Group: group,
		Attributes: pv.Spec.CSI.VolumeAttributes, PublishContext: publishContext,
	}
	// Credentials are fetched afresh for each Stage and are never persisted.
	encoded, _ := json.Marshal(request)
	return fmt.Sprintf("%x", sha256.Sum256(encoded))
}

func saveStageRecovery(path string, data map[string]string, state *stageRecoveryState) error {
	encoded, err := json.Marshal(state)
	if err != nil {
		return err
	}
	data[stageRecoveryDataKey] = string(encoded)
	dir := filepath.Dir(path)
	temporaryPath := filepath.Join(dir, stageRecoveryTemporaryFile)
	if err := os.Remove(temporaryPath); err != nil && !errors.Is(err, os.ErrNotExist) {
		return err
	}
	f, err := os.OpenFile(temporaryPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0600)
	if err != nil {
		return err
	}
	defer os.Remove(f.Name())
	defer f.Close()
	if err := json.NewEncoder(f).Encode(data); err != nil {
		return err
	}
	if err := f.Sync(); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return err
	}
	return os.Rename(f.Name(), filepath.Join(dir, volDataFileName))
}

func (c *csiMountMgr) requestStageRecovery(ctx context.Context, client csiClient, path, target string, publishErr error) error {
	if path == "" || !c.plugin.stageRecoveryEnabled(string(c.driverName)) {
		return publishErr
	}
	// removeMountDir also succeeds when a target is still mounted. Such a
	// partial publication must not authorize a device-level recovery.
	if _, err := os.Lstat(target); !errors.Is(err, os.ErrNotExist) {
		return publishErr
	}
	// A transport failure may leave Publish in flight. Parameter and permission
	// errors do not become recoverable merely because health is also abnormal.
	if code := status.Code(publishErr); code != codes.Internal && code != codes.FailedPrecondition {
		return publishErr
	}
	data, state, err := loadStageRecovery(path)
	if err != nil || !matchingStageRecovery(data, state, c.spec) {
		return publishErr
	}
	supported, err := client.NodeSupportsVolumeHealth(ctx)
	if err != nil || !supported {
		return publishErr
	}
	required, err := client.NodeStageRecoveryRequired(ctx, c.volumeID, path, target)
	if err != nil {
		klog.V(4).InfoS("CSI stage recovery health probe failed", "volumeID", c.volumeID, "err", err)
		return publishErr
	}
	if !required {
		return publishErr
	}
	state.Pending = true
	if err := saveStageRecovery(path, data, state); err != nil {
		klog.ErrorS(err, "Failed to persist CSI stage recovery", "volumeID", c.volumeID)
		return publishErr
	}
	return volumetypes.NewStageRecoveryRequiredError(publishErr)
}
