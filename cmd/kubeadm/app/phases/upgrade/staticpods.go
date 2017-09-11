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

package upgrade

import (
	"fmt"
	"os"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// StaticPodPathManager is responsible for tracking the directories used in the static pod upgrade transition
type StaticPodPathManager interface {
	// MoveFile should move a file from oldPath to newPath
	MoveFile(oldPath, newPath string) error
	// RealManifestPath gets the file path for the component in the "real" static pod manifest directory used by the kubelet
	RealManifestPath(component string) string
	// RealManifestDir should point to the static pod manifest directory used by the kubelet
	RealManifestDir() string
	// TempManifestPath gets the file path for the component in the temporary directory created for generating new manifests for the upgrade
	TempManifestPath(component string) string
	// TempManifestDir should point to the temporary directory created for generating new manifests for the upgrade
	TempManifestDir() string
	// BackupManifestPath gets the file path for the component in the backup directory used for backuping manifests during the transition
	BackupManifestPath(component string) string
	// BackupManifestDir should point to the backup directory used for backuping manifests during the transition
	BackupManifestDir() string
}

// KubeStaticPodPathManager is a real implementation of StaticPodPathManager that is used when upgrading a static pod cluster
type KubeStaticPodPathManager struct {
	realManifestDir   string
	tempManifestDir   string
	backupManifestDir string
}

// NewKubeStaticPodPathManager creates a new instance of KubeStaticPodPathManager
func NewKubeStaticPodPathManager(realDir, tempDir, backupDir string) StaticPodPathManager {
	return &KubeStaticPodPathManager{
		realManifestDir:   realDir,
		tempManifestDir:   tempDir,
		backupManifestDir: backupDir,
	}
}

// NewKubeStaticPodPathManagerUsingTempDirs creates a new instance of KubeStaticPodPathManager with temporary directories backing it
func NewKubeStaticPodPathManagerUsingTempDirs(realManifestDir string) (StaticPodPathManager, error) {
	upgradedManifestsDir, err := constants.CreateTempDirForKubeadm("kubeadm-upgraded-manifests")
	if err != nil {
		return nil, err
	}
	backupManifestsDir, err := constants.CreateTempDirForKubeadm("kubeadm-backup-manifests")
	if err != nil {
		return nil, err
	}

	return NewKubeStaticPodPathManager(realManifestDir, upgradedManifestsDir, backupManifestsDir), nil
}

// MoveFile should move a file from oldPath to newPath
func (spm *KubeStaticPodPathManager) MoveFile(oldPath, newPath string) error {
	return os.Rename(oldPath, newPath)
}

// RealManifestPath gets the file path for the component in the "real" static pod manifest directory used by the kubelet
func (spm *KubeStaticPodPathManager) RealManifestPath(component string) string {
	return constants.GetStaticPodFilepath(component, spm.realManifestDir)
}

// RealManifestDir should point to the static pod manifest directory used by the kubelet
func (spm *KubeStaticPodPathManager) RealManifestDir() string {
	return spm.realManifestDir
}

// TempManifestPath gets the file path for the component in the temporary directory created for generating new manifests for the upgrade
func (spm *KubeStaticPodPathManager) TempManifestPath(component string) string {
	return constants.GetStaticPodFilepath(component, spm.tempManifestDir)
}

// TempManifestDir should point to the temporary directory created for generating new manifests for the upgrade
func (spm *KubeStaticPodPathManager) TempManifestDir() string {
	return spm.tempManifestDir
}

// BackupManifestPath gets the file path for the component in the backup directory used for backuping manifests during the transition
func (spm *KubeStaticPodPathManager) BackupManifestPath(component string) string {
	return constants.GetStaticPodFilepath(component, spm.backupManifestDir)
}

// BackupManifestDir should point to the backup directory used for backuping manifests during the transition
func (spm *KubeStaticPodPathManager) BackupManifestDir() string {
	return spm.backupManifestDir
}

// StaticPodControlPlane upgrades a static pod-hosted control plane
func StaticPodControlPlane(waiter apiclient.Waiter, pathMgr StaticPodPathManager, cfg *kubeadmapi.MasterConfiguration) error {

	// This string-string map stores the component name and backup filepath (if a rollback is needed).
	// If a rollback is needed,
	recoverManifests := map[string]string{}

	beforePodHashMap, err := waiter.WaitForStaticPodControlPlaneHashes(cfg.NodeName)
	if err != nil {
		return err
	}

	// Write the updated static Pod manifests into the temporary directory
	fmt.Printf("[upgrade/staticpods] Writing upgraded Static Pod manifests to %q\n", pathMgr.TempManifestDir())
	err = controlplane.CreateInitStaticPodManifestFiles(pathMgr.TempManifestDir(), cfg)

	for _, component := range constants.MasterComponents {
		// The old manifest is here; in the /etc/kubernetes/manifests/
		currentManifestPath := pathMgr.RealManifestPath(component)
		// The new, upgraded manifest will be written here
		newManifestPath := pathMgr.TempManifestPath(component)
		// The old manifest will be moved here; into a subfolder of the temporary directory
		// If a rollback is needed, these manifests will be put back to where they where initially
		backupManifestPath := pathMgr.BackupManifestPath(component)

		// Store the backup path in the recover list. If something goes wrong now, this component will be rolled back.
		recoverManifests[component] = backupManifestPath

		// Move the old manifest into the old-manifests directory
		if err := pathMgr.MoveFile(currentManifestPath, backupManifestPath); err != nil {
			return rollbackOldManifests(recoverManifests, err, pathMgr)
		}

		// Move the new manifest into the manifests directory
		if err := pathMgr.MoveFile(newManifestPath, currentManifestPath); err != nil {
			return rollbackOldManifests(recoverManifests, err, pathMgr)
		}

		fmt.Printf("[upgrade/staticpods] Moved upgraded manifest to %q and backed up old manifest to %q\n", currentManifestPath, backupManifestPath)
		fmt.Println("[upgrade/staticpods] Waiting for the kubelet to restart the component")

		// Wait for the mirror Pod hash to change; otherwise we'll run into race conditions here when the kubelet hasn't had time to
		// notice the removal of the Static Pod, leading to a false positive below where we check that the API endpoint is healthy
		// If we don't do this, there is a case where we remove the Static Pod manifest, kubelet is slow to react, kubeadm checks the
		// API endpoint below of the OLD Static Pod component and proceeds quickly enough, which might lead to unexpected results.
		if err := waiter.WaitForStaticPodControlPlaneHashChange(cfg.NodeName, component, beforePodHashMap[component]); err != nil {
			return rollbackOldManifests(recoverManifests, err, pathMgr)
		}

		// Wait for the static pod component to come up and register itself as a mirror pod
		if err := waiter.WaitForPodsWithLabel("component=" + component); err != nil {
			return rollbackOldManifests(recoverManifests, err, pathMgr)
		}

		fmt.Printf("[upgrade/staticpods] Component %q upgraded successfully!\n", component)
	}
	// Remove the temporary directories used on a best-effort (don't fail if the calls error out)
	// The calls are set here by design; we should _not_ use "defer" above as that would remove the directories
	// even in the "fail and rollback" case, where we want the directories preserved for the user.
	os.RemoveAll(pathMgr.TempManifestDir())
	os.RemoveAll(pathMgr.BackupManifestDir())

	return nil
}

// rollbackOldManifests rolls back the backuped manifests if something went wrong
func rollbackOldManifests(oldManifests map[string]string, origErr error, pathMgr StaticPodPathManager) error {
	errs := []error{origErr}
	for component, backupPath := range oldManifests {
		// Where we should put back the backed up manifest
		realManifestPath := pathMgr.RealManifestPath(component)

		// Move the backup manifest back into the manifests directory
		err := pathMgr.MoveFile(backupPath, realManifestPath)
		if err != nil {
			errs = append(errs, err)
		}
	}
	// Let the user know there we're problems, but we tried to reçover
	return fmt.Errorf("couldn't upgrade control plane. kubeadm has tried to recover everything into the earlier state. Errors faced: %v", errs)
}
