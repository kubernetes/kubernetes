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
	"path/filepath"
	"strings"
	"time"

	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/renewal"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	etcdphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	etcdutil "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/image"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

// StaticPodPathManager is responsible for tracking the directories used in the static pod upgrade transition
type StaticPodPathManager interface {
	// MoveFile should move a file from oldPath to newPath
	MoveFile(oldPath, newPath string) error
	// KubernetesDir is the directory Kubernetes owns for storing various configuration files
	KubernetesDir() string
	// PatchesDir should point to the folder where patches for components are stored
	PatchesDir() string
	// RealManifestPath gets the file path for the component in the "real" static pod manifest directory used by the kubelet
	RealManifestPath(component string) string
	// RealManifestDir should point to the static pod manifest directory used by the kubelet
	RealManifestDir() string
	// TempManifestPath gets the file path for the component in the temporary directory created for generating new manifests for the upgrade
	TempManifestPath(component string) string
	// TempManifestDir should point to the temporary directory created for generating new manifests for the upgrade
	TempManifestDir() string
	// BackupManifestPath gets the file path for the component in the backup directory used for backing up manifests during the transition
	BackupManifestPath(component string) string
	// BackupManifestDir should point to the backup directory used for backing up manifests during the transition
	BackupManifestDir() string
	// BackupEtcdDir should point to the backup directory used for backing up manifests during the transition
	BackupEtcdDir() string
	// CleanupDirs cleans up all temporary directories
	CleanupDirs() error
}

// KubeStaticPodPathManager is a real implementation of StaticPodPathManager that is used when upgrading a static pod cluster
type KubeStaticPodPathManager struct {
	kubernetesDir     string
	patchesDir        string
	realManifestDir   string
	tempManifestDir   string
	backupManifestDir string
	backupEtcdDir     string

	keepManifestDir bool
	keepEtcdDir     bool
}

// NewKubeStaticPodPathManager creates a new instance of KubeStaticPodPathManager
func NewKubeStaticPodPathManager(kubernetesDir, patchesDir, tempDir, backupDir, backupEtcdDir string, keepManifestDir, keepEtcdDir bool) StaticPodPathManager {
	return &KubeStaticPodPathManager{
		kubernetesDir:     kubernetesDir,
		patchesDir:        patchesDir,
		realManifestDir:   filepath.Join(kubernetesDir, constants.ManifestsSubDirName),
		tempManifestDir:   tempDir,
		backupManifestDir: backupDir,
		backupEtcdDir:     backupEtcdDir,
		keepManifestDir:   keepManifestDir,
		keepEtcdDir:       keepEtcdDir,
	}
}

// NewKubeStaticPodPathManagerUsingTempDirs creates a new instance of KubeStaticPodPathManager with temporary directories backing it
func NewKubeStaticPodPathManagerUsingTempDirs(kubernetesDir, patchesDir string, saveManifestsDir, saveEtcdDir bool) (StaticPodPathManager, error) {
	upgradedManifestsDir, err := constants.CreateTempDir(kubernetesDir, "kubeadm-upgraded-manifests")
	if err != nil {
		return nil, err
	}
	backupManifestsDir, err := constants.CreateTimestampDir(kubernetesDir, "kubeadm-backup-manifests")
	if err != nil {
		return nil, err
	}
	backupEtcdDir, err := constants.CreateTimestampDir(kubernetesDir, "kubeadm-backup-etcd")
	if err != nil {
		return nil, err
	}

	return NewKubeStaticPodPathManager(kubernetesDir, patchesDir, upgradedManifestsDir, backupManifestsDir, backupEtcdDir, saveManifestsDir, saveEtcdDir), nil
}

// MoveFile should move a file from oldPath to newPath
func (spm *KubeStaticPodPathManager) MoveFile(oldPath, newPath string) error {
	return kubeadmutil.MoveFile(oldPath, newPath)
}

// KubernetesDir should point to the directory Kubernetes owns for storing various configuration files
func (spm *KubeStaticPodPathManager) KubernetesDir() string {
	return spm.kubernetesDir
}

// PatchesDir should point to the folder where patches for components are stored
func (spm *KubeStaticPodPathManager) PatchesDir() string {
	return spm.patchesDir
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

// BackupManifestPath gets the file path for the component in the backup directory used for backing up manifests during the transition
func (spm *KubeStaticPodPathManager) BackupManifestPath(component string) string {
	return constants.GetStaticPodFilepath(component, spm.backupManifestDir)
}

// BackupManifestDir should point to the backup directory used for backing up manifests during the transition
func (spm *KubeStaticPodPathManager) BackupManifestDir() string {
	return spm.backupManifestDir
}

// BackupEtcdDir should point to the backup directory used for backing up manifests during the transition
func (spm *KubeStaticPodPathManager) BackupEtcdDir() string {
	return spm.backupEtcdDir
}

// CleanupDirs cleans up all temporary directories except those the user has requested to keep around
func (spm *KubeStaticPodPathManager) CleanupDirs() error {
	var errlist []error
	if err := os.RemoveAll(spm.TempManifestDir()); err != nil {
		errlist = append(errlist, err)
	}
	if !spm.keepManifestDir {
		if err := os.RemoveAll(spm.BackupManifestDir()); err != nil {
			errlist = append(errlist, err)
		}
	}

	if !spm.keepEtcdDir {
		if err := os.RemoveAll(spm.BackupEtcdDir()); err != nil {
			errlist = append(errlist, err)
		}
	}

	return utilerrors.NewAggregate(errlist)
}

func upgradeComponent(component string, certsRenewMgr *renewal.Manager, waiter apiclient.Waiter, pathMgr StaticPodPathManager, cfg *kubeadmapi.InitConfiguration, beforePodHash string, recoverManifests map[string]string) error {
	// Special treatment is required for etcd case, when rollbackOldManifests should roll back etcd
	// manifests only for the case when component is Etcd. Also check if the kube-apiserver needs an upgrade
	// by comparing its manifests and later decide if etcd certs can be renewed and etcd can be restarted.
	recoverEtcd := false
	apiServerUpgrade := false
	if component == constants.Etcd {
		recoverEtcd = true
		equal, _, err := staticpod.ManifestFilesAreEqual(pathMgr.RealManifestPath(constants.KubeAPIServer),
			pathMgr.TempManifestPath(constants.KubeAPIServer))
		if err != nil {
			return err
		}
		apiServerUpgrade = !equal
	}

	fmt.Printf("[upgrade/staticpods] Preparing for %q upgrade\n", component)

	// The old manifest is here; in the /etc/kubernetes/manifests/
	currentManifestPath := pathMgr.RealManifestPath(component)
	// The new, upgraded manifest will be written here
	newManifestPath := pathMgr.TempManifestPath(component)
	// The old manifest will be moved here; into a subfolder of the temporary directory
	// If a rollback is needed, these manifests will be put back to where they where initially
	backupManifestPath := pathMgr.BackupManifestPath(component)

	// Store the backup path in the recover list. If something goes wrong now, this component will be rolled back.
	recoverManifests[component] = backupManifestPath

	// Skip upgrade if current and new manifests are equal
	equal, diff, err := staticpod.ManifestFilesAreEqual(currentManifestPath, newManifestPath)
	if err != nil {
		return err
	}
	if equal {
		if !apiServerUpgrade {
			fmt.Printf("[upgrade/staticpods] Current and new manifests of %s are equal, skipping upgrade\n", component)
			return nil
		}
		klog.V(4).Infof("[upgrade/staticpods] Current and new manifests of %s are equal, but %s will be upgraded\n",
			component, constants.KubeAPIServer)
	} else {
		klog.V(4).Infof("Pod manifest files diff:\n%s\n", diff)
	}

	// if certificate renewal should be performed
	if certsRenewMgr != nil {
		// renew all the certificates used by the current component
		if err := renewCertsByComponent(cfg, component, certsRenewMgr); err != nil {
			return rollbackOldManifests(recoverManifests, errors.Wrapf(err, "failed to renew certificates for component %q", component), pathMgr, recoverEtcd)
		}
	}

	// This can only happen if the component is etcd, cert renewal is enabled and kube-apiserver needs an upgrade
	if equal && certsRenewMgr != nil {
		fmt.Printf("[upgrade/staticpods] Restarting the %s static pod and backing up its manifest to %q\n",
			component, backupManifestPath)
	} else {
		fmt.Printf("[upgrade/staticpods] Moving new manifest to %q and backing up old manifest to %q\n",
			currentManifestPath, backupManifestPath)
	}

	// Move the old manifest into the old-manifests directory
	if err := pathMgr.MoveFile(currentManifestPath, backupManifestPath); err != nil {
		return rollbackOldManifests(recoverManifests, err, pathMgr, recoverEtcd)
	}

	// Move the new manifest into the manifests directory
	if err := pathMgr.MoveFile(newManifestPath, currentManifestPath); err != nil {
		return rollbackOldManifests(recoverManifests, err, pathMgr, recoverEtcd)
	}

	fmt.Println("[upgrade/staticpods] Waiting for the kubelet to restart the component")
	fmt.Printf("[upgrade/staticpods] This can take up to %v\n", kubeadmapi.GetActiveTimeouts().UpgradeManifests.Duration)

	// Wait for the mirror Pod hash to change; otherwise we'll run into race conditions here when the kubelet hasn't had time to
	// notice the removal of the Static Pod, leading to a false positive below where we check that the API endpoint is healthy
	// If we don't do this, there is a case where we remove the Static Pod manifest, kubelet is slow to react, kubeadm checks the
	// API endpoint below of the OLD Static Pod component and proceeds quickly enough, which might lead to unexpected results.
	if !equal {
		if err := waiter.WaitForStaticPodHashChange(cfg.NodeRegistration.Name, component, beforePodHash); err != nil {
			return rollbackOldManifests(recoverManifests, err, pathMgr, recoverEtcd)
		}
	}

	// Wait for the static pod component to come up and register itself as a mirror pod
	if err := waiter.WaitForPodsWithLabel("component=" + component); err != nil {
		return rollbackOldManifests(recoverManifests, err, pathMgr, recoverEtcd)
	}

	fmt.Printf("[upgrade/staticpods] Component %q upgraded successfully!\n", component)

	return nil
}

// performEtcdStaticPodUpgrade performs upgrade of etcd, it returns bool which indicates fatal error or not and the actual error.
func performEtcdStaticPodUpgrade(certsRenewMgr *renewal.Manager, client clientset.Interface, waiter apiclient.Waiter, pathMgr StaticPodPathManager, cfg *kubeadmapi.InitConfiguration, recoverManifests map[string]string, oldEtcdClient, newEtcdClient etcdutil.ClusterInterrogator) (bool, error) {
	// Add etcd static pod spec only if external etcd is not configured
	if cfg.Etcd.External != nil {
		return false, errors.New("external etcd detected, won't try to change any etcd state")
	}

	// Checking health state of etcd before proceeding with the upgrade
	err := oldEtcdClient.CheckClusterHealth()
	if err != nil {
		return true, errors.Wrap(err, "etcd cluster is not healthy")
	}

	// Backing up etcd data store
	backupEtcdDir := pathMgr.BackupEtcdDir()
	runningEtcdDir := cfg.Etcd.Local.DataDir
	output, err := kubeadmutil.CopyDir(runningEtcdDir, backupEtcdDir)
	if err != nil {
		return true, errors.Wrapf(err, "failed to back up etcd data, output: %q", output)
	}

	// Get the desired etcd version. That's either the one specified by the user in cfg.Etcd.Local.ImageTag
	// or the kubeadm preferred one for the desired Kubernetes version
	var desiredEtcdVersion *version.Version
	if cfg.Etcd.Local.ImageTag != "" {
		desiredEtcdVersion, err = version.ParseSemantic(
			convertImageTagMetadataToSemver(cfg.Etcd.Local.ImageTag))
		if err != nil {
			return true, errors.Wrapf(err, "failed to parse tag %q as a semantic version", cfg.Etcd.Local.ImageTag)
		}
	} else {
		// Need to check currently used version and version from constants, if differs then upgrade
		var warning error
		desiredEtcdVersion, warning, err = constants.EtcdSupportedVersion(constants.SupportedEtcdVersion, cfg.KubernetesVersion)
		if err != nil {
			return true, errors.Wrap(err, "failed to retrieve an etcd version for the target Kubernetes version")
		}
		if warning != nil {
			klog.V(1).Infof("[upgrade/etcd] WARNING: %v", warning)
		}
	}

	// Get the etcd version of the local/stacked etcd member running on the current machine
	currentEtcdVersionStr, err := GetEtcdImageTagFromStaticPod(pathMgr.RealManifestDir())
	if err != nil {
		return true, errors.Wrap(err, "failed to retrieve the current etcd version")
	}

	cmpResult, err := desiredEtcdVersion.Compare(currentEtcdVersionStr)
	if err != nil {
		return true, errors.Wrapf(err, "failed comparing the current etcd version %q to the desired one %q", currentEtcdVersionStr, desiredEtcdVersion)
	}
	if cmpResult < 0 {
		return false, errors.Errorf("the desired etcd version %q is older than the currently installed %q. Skipping etcd upgrade", desiredEtcdVersion, currentEtcdVersionStr)
	}

	beforeEtcdPodHash, err := waiter.WaitForStaticPodSingleHash(cfg.NodeRegistration.Name, constants.Etcd)
	if err != nil {
		return true, err
	}

	// Write the updated etcd static Pod manifest into the temporary directory, at this point no etcd change
	// has occurred in any aspects.
	if err := etcdphase.CreateLocalEtcdStaticPodManifestFile(pathMgr.TempManifestDir(), pathMgr.PatchesDir(), cfg.NodeRegistration.Name, &cfg.ClusterConfiguration, &cfg.LocalAPIEndpoint, false /* isDryRun */); err != nil {
		return true, errors.Wrap(err, "error creating local etcd static pod manifest file")
	}

	retries := 10
	retryInterval := 15 * time.Second

	// Perform etcd upgrade using common to all control plane components function
	if err := upgradeComponent(constants.Etcd, certsRenewMgr, waiter, pathMgr, cfg, beforeEtcdPodHash, recoverManifests); err != nil {
		fmt.Printf("[upgrade/etcd] Failed to upgrade etcd: %v\n", err)
		// Since upgrade component failed, the old etcd manifest has either been restored or was never touched
		// Now we need to check the health of etcd cluster if it is up with old manifest
		fmt.Println("[upgrade/etcd] Waiting for previous etcd to become available")
		if _, err := oldEtcdClient.WaitForClusterAvailable(retries, retryInterval); err != nil {
			fmt.Printf("[upgrade/etcd] Failed to healthcheck previous etcd: %v\n", err)

			// At this point we know that etcd cluster is dead and it is safe to copy backup datastore and to rollback old etcd manifest
			fmt.Println("[upgrade/etcd] Rolling back etcd data")
			if err := rollbackEtcdData(cfg, pathMgr); err != nil {
				// Even copying back datastore failed, no options for recovery left, bailing out
				return true, errors.Errorf("fatal error rolling back local etcd cluster datadir: %v, the backup of etcd database is stored here:(%s)", err, backupEtcdDir)
			}
			fmt.Println("[upgrade/etcd] Etcd data rollback successful")

			// Now that we've rolled back the data, let's check if the cluster comes up
			fmt.Println("[upgrade/etcd] Waiting for previous etcd to become available")
			if _, err := oldEtcdClient.WaitForClusterAvailable(retries, retryInterval); err != nil {
				fmt.Printf("[upgrade/etcd] Failed to healthcheck previous etcd: %v\n", err)
				// Nothing else left to try to recover etcd cluster
				return true, errors.Wrapf(err, "fatal error rolling back local etcd cluster manifest, the backup of etcd database is stored here:(%s)", backupEtcdDir)
			}

			// We've recovered to the previous etcd from this case
		}
		fmt.Println("[upgrade/etcd] Etcd was rolled back and is now available")

		// Since etcd cluster came back up with the old manifest
		return true, errors.Wrap(err, "fatal error when trying to upgrade the etcd cluster, rolled the state back to pre-upgrade state")
	}

	// Initialize the new etcd client if it wasn't pre-initialized
	if newEtcdClient == nil {
		etcdClient, err := etcdutil.NewFromCluster(client, cfg.CertificatesDir)
		if err != nil {
			return true, errors.Wrap(err, "fatal error creating etcd client")
		}
		newEtcdClient = etcdClient
	}

	// Checking health state of etcd after the upgrade
	fmt.Println("[upgrade/etcd] Waiting for etcd to become available")
	if _, err = newEtcdClient.WaitForClusterAvailable(retries, retryInterval); err != nil {
		fmt.Printf("[upgrade/etcd] Failed to healthcheck etcd: %v\n", err)
		// Despite the fact that upgradeComponent was successful, there is something wrong with the etcd cluster
		// First step is to restore back up of datastore
		fmt.Println("[upgrade/etcd] Rolling back etcd data")
		if err := rollbackEtcdData(cfg, pathMgr); err != nil {
			// Even copying back datastore failed, no options for recovery left, bailing out
			return true, errors.Wrapf(err, "fatal error rolling back local etcd cluster datadir, the backup of etcd database is stored here:(%s)", backupEtcdDir)
		}
		fmt.Println("[upgrade/etcd] Etcd data rollback successful")

		// Old datastore has been copied, rolling back old manifests
		fmt.Println("[upgrade/etcd] Rolling back etcd manifest")
		rollbackOldManifests(recoverManifests, err, pathMgr, true)
		// rollbackOldManifests() always returns an error -- ignore it and continue

		// Assuming rollback of the old etcd manifest was successful, check the status of etcd cluster again
		fmt.Println("[upgrade/etcd] Waiting for previous etcd to become available")
		if _, err := oldEtcdClient.WaitForClusterAvailable(retries, retryInterval); err != nil {
			fmt.Printf("[upgrade/etcd] Failed to healthcheck previous etcd: %v\n", err)
			// Nothing else left to try to recover etcd cluster
			return true, errors.Wrapf(err, "fatal error rolling back local etcd cluster manifest, the backup of etcd database is stored here:(%s)", backupEtcdDir)
		}
		fmt.Println("[upgrade/etcd] Etcd was rolled back and is now available")

		// We've successfully rolled back etcd, and now return an error describing that the upgrade failed
		return true, errors.Wrap(err, "fatal error upgrading local etcd cluster, rolled the state back to pre-upgrade state")
	}

	return false, nil
}

// StaticPodControlPlane upgrades a static pod-hosted control plane
func StaticPodControlPlane(client clientset.Interface, waiter apiclient.Waiter, pathMgr StaticPodPathManager, cfg *kubeadmapi.InitConfiguration, etcdUpgrade, renewCerts bool, oldEtcdClient, newEtcdClient etcdutil.ClusterInterrogator) error {
	recoverManifests := map[string]string{}
	var isExternalEtcd bool

	beforePodHashMap, err := waiter.WaitForStaticPodControlPlaneHashes(cfg.NodeRegistration.Name)
	if err != nil {
		return err
	}

	if oldEtcdClient == nil {
		if cfg.Etcd.External != nil {
			// External etcd
			isExternalEtcd = true
			etcdClient, err := etcdutil.New(
				cfg.Etcd.External.Endpoints,
				cfg.Etcd.External.CAFile,
				cfg.Etcd.External.CertFile,
				cfg.Etcd.External.KeyFile,
			)
			if err != nil {
				return errors.Wrap(err, "failed to create etcd client for external etcd")
			}
			oldEtcdClient = etcdClient
			// Since etcd is managed externally, the new etcd client will be the same as the old client
			if newEtcdClient == nil {
				newEtcdClient = etcdClient
			}
		} else {
			// etcd Static Pod
			etcdClient, err := etcdutil.NewFromCluster(client, cfg.CertificatesDir)
			if err != nil {
				return errors.Wrap(err, "failed to create etcd client")
			}
			oldEtcdClient = etcdClient
		}
	}

	var certsRenewMgr *renewal.Manager
	if renewCerts {
		certsRenewMgr, err = renewal.NewManager(&cfg.ClusterConfiguration, pathMgr.KubernetesDir())
		if err != nil {
			return errors.Wrap(err, "failed to create the certificate renewal manager")
		}
	}

	// Write the updated static Pod manifests into the temporary directory.
	// As part of the design, this must be called before the below performEtcdStaticPodUpgrade() call.
	fmt.Printf("[upgrade/staticpods] Writing new Static Pod manifests to %q\n", pathMgr.TempManifestDir())
	err = controlplane.CreateInitStaticPodManifestFiles(pathMgr.TempManifestDir(), pathMgr.PatchesDir(), cfg, false /* isDryRun */)
	if err != nil {
		return errors.Wrap(err, "error creating init static pod manifest files")
	}

	// etcd upgrade is done prior to other control plane components
	if !isExternalEtcd && etcdUpgrade {
		// Perform etcd upgrade using common to all control plane components function
		fatal, err := performEtcdStaticPodUpgrade(certsRenewMgr, client, waiter, pathMgr, cfg, recoverManifests, oldEtcdClient, newEtcdClient)
		if err != nil {
			if fatal {
				return err
			}
			fmt.Printf("[upgrade/etcd] Non fatal issue encountered during upgrade: %v\n", err)
		}
	}

	// Upgrade the rest of the control plane components
	for _, component := range constants.ControlPlaneComponents {
		if err = upgradeComponent(component, certsRenewMgr, waiter, pathMgr, cfg, beforePodHashMap[component], recoverManifests); err != nil {
			return err
		}
	}

	if renewCerts {
		// renew the certificate embedded in the admin.conf file
		renewed, err := certsRenewMgr.RenewUsingLocalCA(constants.AdminKubeConfigFileName)
		if err != nil {
			return rollbackOldManifests(recoverManifests, errors.Wrapf(err, "failed to upgrade the %s certificates", constants.AdminKubeConfigFileName), pathMgr, false)
		}

		if !renewed {
			// if not error, but not renewed because of external CA detected, inform the user
			fmt.Printf("[upgrade/staticpods] External CA detected, %s certificate can't be renewed\n", constants.AdminKubeConfigFileName)
		}

		// Do the same for super-admin.conf, but only if it exists
		if _, err := os.Stat(filepath.Join(pathMgr.KubernetesDir(), constants.SuperAdminKubeConfigFileName)); err == nil {
			// renew the certificate embedded in the super-admin.conf file
			renewed, err := certsRenewMgr.RenewUsingLocalCA(constants.SuperAdminKubeConfigFileName)
			if err != nil {
				return rollbackOldManifests(recoverManifests, errors.Wrapf(err, "failed to upgrade the %s certificates", constants.SuperAdminKubeConfigFileName), pathMgr, false)
			}

			if !renewed {
				// if not error, but not renewed because of external CA detected, inform the user
				fmt.Printf("[upgrade/staticpods] External CA detected, %s certificate can't be renewed\n", constants.SuperAdminKubeConfigFileName)
			}
		}
	}

	// Remove the temporary directories used on a best-effort (don't fail if the calls error out)
	// The calls are set here by design; we should _not_ use "defer" above as that would remove the directories
	// even in the "fail and rollback" case, where we want the directories preserved for the user.
	return pathMgr.CleanupDirs()
}

// rollbackOldManifests rolls back the backed-up manifests if something went wrong.
// It always returns an error to the caller.
func rollbackOldManifests(oldManifests map[string]string, origErr error, pathMgr StaticPodPathManager, restoreEtcd bool) error {
	errs := []error{origErr}
	for component, backupPath := range oldManifests {
		// Will restore etcd manifest only if it was explicitly requested by setting restoreEtcd to True
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
	return errors.Wrap(utilerrors.NewAggregate(errs),
		"couldn't upgrade control plane. kubeadm has tried to recover everything into the earlier state. Errors faced")
}

// rollbackEtcdData rolls back the content of etcd folder if something went wrong.
// When the folder contents are successfully rolled back, nil is returned, otherwise an error is returned.
func rollbackEtcdData(cfg *kubeadmapi.InitConfiguration, pathMgr StaticPodPathManager) error {
	backupEtcdDir := pathMgr.BackupEtcdDir()
	runningEtcdDir := cfg.Etcd.Local.DataDir

	output, err := kubeadmutil.CopyDir(backupEtcdDir, runningEtcdDir)
	if err != nil {
		// Let the user know there we're problems, but we tried to reçover
		return errors.Wrapf(err, "couldn't recover etcd database with error, the location of etcd backup: %s, output: %q", backupEtcdDir, output)
	}

	return nil
}

// renewCertsByComponent takes charge of renewing certificates used by a specific component before
// the static pod of the component is upgraded
func renewCertsByComponent(cfg *kubeadmapi.InitConfiguration, component string, certsRenewMgr *renewal.Manager) error {
	var certificates []string

	// if etcd, only in case of local etcd, renew server, peer and health check certificate
	if component == constants.Etcd {
		if cfg.Etcd.Local != nil {
			certificates = []string{
				certsphase.KubeadmCertEtcdServer().Name,
				certsphase.KubeadmCertEtcdPeer().Name,
				certsphase.KubeadmCertEtcdHealthcheck().Name,
			}
		}
	}

	// if apiserver, renew apiserver serving certificate, kubelet and front-proxy client certificate.
	//if local etcd, renew also the etcd client certificate
	if component == constants.KubeAPIServer {
		certificates = []string{
			certsphase.KubeadmCertAPIServer().Name,
			certsphase.KubeadmCertKubeletClient().Name,
			certsphase.KubeadmCertFrontProxyClient().Name,
		}
		if cfg.Etcd.Local != nil {
			certificates = append(certificates, certsphase.KubeadmCertEtcdAPIClient().Name)
		}
	}

	// if controller-manager, renew the certificate embedded in the controller-manager kubeConfig file
	if component == constants.KubeControllerManager {
		certificates = []string{
			constants.ControllerManagerKubeConfigFileName,
		}
	}

	// if scheduler, renew the certificate embedded in the scheduler kubeConfig file
	if component == constants.KubeScheduler {
		certificates = []string{
			constants.SchedulerKubeConfigFileName,
		}
	}

	// renew the selected components
	for _, cert := range certificates {
		fmt.Printf("[upgrade/staticpods] Renewing %s certificate\n", cert)
		renewed, err := certsRenewMgr.RenewUsingLocalCA(cert)
		if err != nil {
			return err
		}
		if !renewed {
			// if not error, but not renewed because of external CA detected, inform the user
			fmt.Printf("[upgrade/staticpods] External CA detected, %s certificate can't be renewed\n", cert)
		}
	}

	return nil
}

// GetPathManagerForUpgrade returns a path manager properly configured for the given InitConfiguration.
func GetPathManagerForUpgrade(kubernetesDir, patchesDir string, internalcfg *kubeadmapi.InitConfiguration, etcdUpgrade bool) (StaticPodPathManager, error) {
	isExternalEtcd := internalcfg.Etcd.External != nil
	return NewKubeStaticPodPathManagerUsingTempDirs(kubernetesDir, patchesDir, true, etcdUpgrade && !isExternalEtcd)
}

// PerformStaticPodUpgrade performs the upgrade of the control plane components for a static pod hosted cluster
func PerformStaticPodUpgrade(client clientset.Interface, waiter apiclient.Waiter, internalcfg *kubeadmapi.InitConfiguration, etcdUpgrade, renewCerts bool, patchesDir string) error {
	pathManager, err := GetPathManagerForUpgrade(constants.KubernetesDir, patchesDir, internalcfg, etcdUpgrade)
	if err != nil {
		return err
	}

	// The arguments oldEtcdClient and newEtdClient, are uninitialized because passing in the clients allow for mocking the client during testing
	return StaticPodControlPlane(client, waiter, pathManager, internalcfg, etcdUpgrade, renewCerts, nil, nil)
}

// DryRunStaticPodUpgrade fakes an upgrade of the control plane
func DryRunStaticPodUpgrade(patchesDir string, internalcfg *kubeadmapi.InitConfiguration) error {
	dryRunManifestDir, err := constants.GetDryRunDir(constants.EnvVarUpgradeDryRunDir, "kubeadm-upgrade-dryrun", klog.Warningf)
	if err != nil {
		return err
	}

	defer os.RemoveAll(dryRunManifestDir)
	if err := controlplane.CreateInitStaticPodManifestFiles(dryRunManifestDir, patchesDir, internalcfg, true /* isDryRun */); err != nil {
		return err
	}

	// Print the contents of the upgraded manifests and pretend like they were in /etc/kubernetes/manifests
	files := []dryrunutil.FileToPrint{}
	for _, component := range constants.ControlPlaneComponents {
		realPath := constants.GetStaticPodFilepath(component, dryRunManifestDir)
		outputPath := constants.GetStaticPodFilepath(component, constants.GetStaticPodDirectory())
		files = append(files, dryrunutil.NewFileToPrint(realPath, outputPath))
	}

	return dryrunutil.PrintDryRunFiles(files, os.Stdout)
}

// GetEtcdImageTagFromStaticPod returns the image tag of the local etcd static pod
func GetEtcdImageTagFromStaticPod(manifestDir string) (string, error) {
	realPath := constants.GetStaticPodFilepath(constants.Etcd, manifestDir)
	pod, err := staticpod.ReadStaticPodFromDisk(realPath)
	if err != nil {
		return "", err
	}

	return convertImageTagMetadataToSemver(image.TagFromImage(pod.Spec.Containers[0].Image)), nil
}

// convertImageTagMetadataToSemver converts imagetag in the format of semver_metadata to semver+metadata
func convertImageTagMetadataToSemver(tag string) string {
	// Container registries do not support `+` characters in tag names. This prevents imagetags from
	// correctly representing semantic versions which use the plus symbol to delimit build metadata.
	// Kubernetes uses the convention of using an underscore in image registries to preserve
	// build metadata information in imagetags.
	return strings.Replace(tag, "_", "+", 1)
}
