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
	"strings"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	etcdphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd"
	"k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	"k8s.io/kubernetes/pkg/util/version"
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
	// BackupEtcdDir should point to the backup directory used for backuping manifests during the transition
	BackupEtcdDir() string
}

// KubeStaticPodPathManager is a real implementation of StaticPodPathManager that is used when upgrading a static pod cluster
type KubeStaticPodPathManager struct {
	realManifestDir   string
	tempManifestDir   string
	backupManifestDir string
	backupEtcdDir     string
}

// NewKubeStaticPodPathManager creates a new instance of KubeStaticPodPathManager
func NewKubeStaticPodPathManager(realDir, tempDir, backupDir, backupEtcdDir string) StaticPodPathManager {
	return &KubeStaticPodPathManager{
		realManifestDir:   realDir,
		tempManifestDir:   tempDir,
		backupManifestDir: backupDir,
		backupEtcdDir:     backupEtcdDir,
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
	backupEtcdDir, err := constants.CreateTempDirForKubeadm("kubeadm-backup-etcd")
	if err != nil {
		return nil, err
	}

	return NewKubeStaticPodPathManager(realManifestDir, upgradedManifestsDir, backupManifestsDir, backupEtcdDir), nil
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

// BackupEtcdDir should point to the backup directory used for backuping manifests during the transition
func (spm *KubeStaticPodPathManager) BackupEtcdDir() string {
	return spm.backupEtcdDir
}

func upgradeComponent(component string, waiter apiclient.Waiter, pathMgr StaticPodPathManager, cfg *kubeadmapi.MasterConfiguration, beforePodHash string, recoverManifests map[string]string) error {
	// Special treatment is required for etcd case, when rollbackOldManifests should roll back etcd
	// manifests only for the case when component is Etcd
	recoverEtcd := false
	if component == constants.Etcd {
		recoverEtcd = true
	}
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
		return rollbackOldManifests(recoverManifests, err, pathMgr, recoverEtcd)
	}

	// Move the new manifest into the manifests directory
	if err := pathMgr.MoveFile(newManifestPath, currentManifestPath); err != nil {
		return rollbackOldManifests(recoverManifests, err, pathMgr, recoverEtcd)
	}

	fmt.Printf("[upgrade/staticpods] Moved new manifest to %q and backed up old manifest to %q\n", currentManifestPath, backupManifestPath)
	fmt.Println("[upgrade/staticpods] Waiting for the kubelet to restart the component")

	// Wait for the mirror Pod hash to change; otherwise we'll run into race conditions here when the kubelet hasn't had time to
	// notice the removal of the Static Pod, leading to a false positive below where we check that the API endpoint is healthy
	// If we don't do this, there is a case where we remove the Static Pod manifest, kubelet is slow to react, kubeadm checks the
	// API endpoint below of the OLD Static Pod component and proceeds quickly enough, which might lead to unexpected results.
	if err := waiter.WaitForStaticPodControlPlaneHashChange(cfg.NodeName, component, beforePodHash); err != nil {
		return rollbackOldManifests(recoverManifests, err, pathMgr, recoverEtcd)
	}

	// Wait for the static pod component to come up and register itself as a mirror pod
	if err := waiter.WaitForPodsWithLabel("component=" + component); err != nil {
		return rollbackOldManifests(recoverManifests, err, pathMgr, recoverEtcd)
	}

	fmt.Printf("[upgrade/staticpods] Component %q upgraded successfully!\n", component)
	return nil
}

// performEtcdStaticPodUpgrade performs upgrade of etcd, it returns bool which indicates fatal error or not and the actual error.
func performEtcdStaticPodUpgrade(waiter apiclient.Waiter, pathMgr StaticPodPathManager, cfg *kubeadmapi.MasterConfiguration, recoverManifests map[string]string) (bool, error) {
	// Add etcd static pod spec only if external etcd is not configured
	if len(cfg.Etcd.Endpoints) != 0 {
		return false, fmt.Errorf("external etcd detected, won't try to change any etcd state")
	}
	// Checking health state of etcd before proceeding with the upgrtade
	etcdCluster := util.LocalEtcdCluster{}
	etcdStatus, err := etcdCluster.GetEtcdClusterStatus()
	if err != nil {
		return true, fmt.Errorf("etcd cluster is not healthy: %v", err)
	}

	// Backing up etcd data store
	backupEtcdDir := pathMgr.BackupEtcdDir()
	runningEtcdDir := cfg.Etcd.DataDir
	if err := util.CopyDir(runningEtcdDir, backupEtcdDir); err != nil {
		return true, fmt.Errorf("fail to back up etcd data: %v", err)
	}

	// Need to check currently used version and version from constants, if differs then upgrade
	desiredEtcdVersion, err := constants.EtcdSupportedVersion(cfg.KubernetesVersion)
	if err != nil {
		return true, fmt.Errorf("failed to parse the desired etcd version(%s): %v", desiredEtcdVersion.String(), err)
	}
	currentEtcdVersion, err := version.ParseSemantic(etcdStatus.Version)
	if err != nil {
		return true, fmt.Errorf("failed to parse the current etcd version(%s): %v", currentEtcdVersion.String(), err)
	}

	// Comparing current etcd version with desired to catch the same version or downgrade condition and fail on them.
	if desiredEtcdVersion.LessThan(currentEtcdVersion) {
		return false, fmt.Errorf("the desired etcd version for this Kubernetes version %q is %q, but the current etcd version is %q. Won't downgrade etcd, instead just continue", cfg.KubernetesVersion, desiredEtcdVersion.String(), currentEtcdVersion.String())
	}
	// For the case when desired etcd version is the same as current etcd version
	if strings.Compare(desiredEtcdVersion.String(), currentEtcdVersion.String()) == 0 {
		return false, nil
	}

	beforeEtcdPodHash, err := waiter.WaitForStaticPodSingleHash(cfg.NodeName, constants.Etcd)
	if err != nil {
		return true, fmt.Errorf("fail to get etcd pod's hash: %v", err)
	}

	// Write the updated etcd static Pod manifest into the temporary directory, at this point no etcd change
	// has occured in any aspects.
	if err := etcdphase.CreateLocalEtcdStaticPodManifestFile(pathMgr.TempManifestDir(), cfg); err != nil {
		return true, fmt.Errorf("error creating local etcd static pod manifest file: %v", err)
	}

	// Perform etcd upgrade using common to all control plane components function
	if err := upgradeComponent(constants.Etcd, waiter, pathMgr, cfg, beforeEtcdPodHash, recoverManifests); err != nil {
		// Since etcd upgrade component failed, the old manifest has been restored
		// now we need to check the heatlth of etcd cluster if it came back up with old manifest
		if _, err := etcdCluster.GetEtcdClusterStatus(); err != nil {
			// At this point we know that etcd cluster is dead and it is safe to copy backup datastore and to rollback old etcd manifest
			if err := rollbackEtcdData(cfg, fmt.Errorf("etcd cluster is not healthy after upgrade: %v rolling back", err), pathMgr); err != nil {
				// Even copying back datastore failed, no options for recovery left, bailing out
				return true, fmt.Errorf("fatal error upgrading local etcd cluster: %v, the backup of etcd database is stored here:(%s)", err, backupEtcdDir)
			}
			// Old datastore has been copied, rolling back old manifests
			if err := rollbackOldManifests(recoverManifests, err, pathMgr, true); err != nil {
				// Rolling back to old manifests failed, no options for recovery left, bailing out
				return true, fmt.Errorf("fatal error upgrading local etcd cluster: %v, the backup of etcd database is stored here:(%s)", err, backupEtcdDir)
			}
			// Since rollback of the old etcd manifest was successful, checking again the status of etcd cluster
			if _, err := etcdCluster.GetEtcdClusterStatus(); err != nil {
				// Nothing else left to try to recover etcd cluster
				return true, fmt.Errorf("fatal error upgrading local etcd cluster: %v, the backup of etcd database is stored here:(%s)", err, backupEtcdDir)
			}

			return true, fmt.Errorf("fatal error upgrading local etcd cluster: %v, rolled the state back to pre-upgrade state", err)
		}
		// Since etcd cluster came back up with the old manifest
		return true, fmt.Errorf("fatal error when trying to upgrade the etcd cluster: %v, rolled the state back to pre-upgrade state", err)
	}

	// Checking health state of etcd after the upgrade
	if _, err = etcdCluster.GetEtcdClusterStatus(); err != nil {
		// Despite the fact that upgradeComponent was sucessfull, there is something wrong with etcd cluster
		// First step is to restore back up of datastore
		if err := rollbackEtcdData(cfg, fmt.Errorf("etcd cluster is not healthy after upgrade: %v rolling back", err), pathMgr); err != nil {
			// Even copying back datastore failed, no options for recovery left, bailing out
			return true, fmt.Errorf("fatal error upgrading local etcd cluster: %v, the backup of etcd database is stored here:(%s)", err, backupEtcdDir)
		}
		// Old datastore has been copied, rolling back old manifests
		if err := rollbackOldManifests(recoverManifests, err, pathMgr, true); err != nil {
			// Rolling back to old manifests failed, no options for recovery left, bailing out
			return true, fmt.Errorf("fatal error upgrading local etcd cluster: %v, the backup of etcd database is stored here:(%s)", err, backupEtcdDir)
		}
		// Since rollback of the old etcd manifest was successful, checking again the status of etcd cluster
		if _, err := etcdCluster.GetEtcdClusterStatus(); err != nil {
			// Nothing else left to try to recover etcd cluster
			return true, fmt.Errorf("fatal error upgrading local etcd cluster: %v, the backup of etcd database is stored here:(%s)", err, backupEtcdDir)
		}

		return true, fmt.Errorf("fatal error upgrading local etcd cluster: %v, rolled the state back to pre-upgrade state", err)
	}

	return false, nil
}

// StaticPodControlPlane upgrades a static pod-hosted control plane
func StaticPodControlPlane(waiter apiclient.Waiter, pathMgr StaticPodPathManager, cfg *kubeadmapi.MasterConfiguration, etcdUpgrade bool) error {
	recoverManifests := map[string]string{}

	// etcd upgrade is done prior to other control plane components
	if etcdUpgrade {
		// Perform etcd upgrade using common to all control plane components function
		fatal, err := performEtcdStaticPodUpgrade(waiter, pathMgr, cfg, recoverManifests)
		if err != nil {
			if fatal {
				return err
			}
			fmt.Printf("[upgrade/etcd] non fatal issue encountered during upgrade: %v\n", err)
		}
	}

	beforePodHashMap, err := waiter.WaitForStaticPodControlPlaneHashes(cfg.NodeName)
	if err != nil {
		return err
	}

	// Write the updated static Pod manifests into the temporary directory
	fmt.Printf("[upgrade/staticpods] Writing new Static Pod manifests to %q\n", pathMgr.TempManifestDir())
	err = controlplane.CreateInitStaticPodManifestFiles(pathMgr.TempManifestDir(), cfg)
	if err != nil {
		return fmt.Errorf("error creating init static pod manifest files: %v", err)
	}

	for _, component := range constants.MasterComponents {
		if err = upgradeComponent(component, waiter, pathMgr, cfg, beforePodHashMap[component], recoverManifests); err != nil {
			return err
		}
	}

	// Remove the temporary directories used on a best-effort (don't fail if the calls error out)
	// The calls are set here by design; we should _not_ use "defer" above as that would remove the directories
	// even in the "fail and rollback" case, where we want the directories preserved for the user.
	os.RemoveAll(pathMgr.TempManifestDir())
	os.RemoveAll(pathMgr.BackupManifestDir())
	os.RemoveAll(pathMgr.BackupEtcdDir())

	return nil
}

// rollbackOldManifests rolls back the backuped manifests if something went wrong
func rollbackOldManifests(oldManifests map[string]string, origErr error, pathMgr StaticPodPathManager, restoreEtcd bool) error {
	errs := []error{origErr}
	for component, backupPath := range oldManifests {
		// Will restore etcd manifest only if it was explicitely requested by setting restoreEtcd to True
		if component == constants.Etcd && !restoreEtcd {
			continue
		}
		// Where we should put back the backed up manifest
		realManifestPath := pathMgr.RealManifestPath(component)

		// Move the backup manifest back into the manifests directory
		err := pathMgr.MoveFile(backupPath, realManifestPath)
		if err != nil {
			errs = append(errs, err)
		}
	}
	// Let the user know there were problems, but we tried to recover
	return fmt.Errorf("couldn't upgrade control plane. kubeadm has tried to recover everything into the earlier state. Errors faced: %v", errs)
}

// rollbackEtcdData rolls back the the content of etcd folder if something went wrong
func rollbackEtcdData(cfg *kubeadmapi.MasterConfiguration, origErr error, pathMgr StaticPodPathManager) error {
	errs := []error{origErr}
	backupEtcdDir := pathMgr.BackupEtcdDir()
	runningEtcdDir := cfg.Etcd.DataDir
	err := util.CopyDir(backupEtcdDir, runningEtcdDir)

	if err != nil {
		errs = append(errs, err)
	}

	// Let the user know there we're problems, but we tried to reçover
	return fmt.Errorf("couldn't recover etcd database with error: %v, the location of etcd backup: %s ", errs, backupEtcdDir)
}
