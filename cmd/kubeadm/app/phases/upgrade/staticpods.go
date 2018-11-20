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
	"time"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/renewal"
	controlplanephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	etcdphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd"
	"k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	etcdutil "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

const (
	// UpgradeManifestTimeout is timeout of upgrading the static pod manifest
	UpgradeManifestTimeout = 5 * time.Minute
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
	// CleanupDirs cleans up all temporary directories
	CleanupDirs() error
}

// KubeStaticPodPathManager is a real implementation of StaticPodPathManager that is used when upgrading a static pod cluster
type KubeStaticPodPathManager struct {
	realManifestDir   string
	tempManifestDir   string
	backupManifestDir string
	backupEtcdDir     string

	keepManifestDir bool
	keepEtcdDir     bool
}

// NewKubeStaticPodPathManager creates a new instance of KubeStaticPodPathManager
func NewKubeStaticPodPathManager(realDir, tempDir, backupDir, backupEtcdDir string, keepManifestDir, keepEtcdDir bool) StaticPodPathManager {
	return &KubeStaticPodPathManager{
		realManifestDir:   realDir,
		tempManifestDir:   tempDir,
		backupManifestDir: backupDir,
		backupEtcdDir:     backupEtcdDir,
		keepManifestDir:   keepManifestDir,
		keepEtcdDir:       keepEtcdDir,
	}
}

// NewKubeStaticPodPathManagerUsingTempDirs creates a new instance of KubeStaticPodPathManager with temporary directories backing it
func NewKubeStaticPodPathManagerUsingTempDirs(realManifestDir string, saveManifestsDir, saveEtcdDir bool) (StaticPodPathManager, error) {
	upgradedManifestsDir, err := constants.CreateTempDirForKubeadm("kubeadm-upgraded-manifests")
	if err != nil {
		return nil, err
	}
	backupManifestsDir, err := constants.CreateTimestampDirForKubeadm("kubeadm-backup-manifests")
	if err != nil {
		return nil, err
	}
	backupEtcdDir, err := constants.CreateTimestampDirForKubeadm("kubeadm-backup-etcd")
	if err != nil {
		return nil, err
	}

	return NewKubeStaticPodPathManager(realManifestDir, upgradedManifestsDir, backupManifestsDir, backupEtcdDir, saveManifestsDir, saveEtcdDir), nil
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

// CleanupDirs cleans up all temporary directories except those the user has requested to keep around
func (spm *KubeStaticPodPathManager) CleanupDirs() error {
	if err := os.RemoveAll(spm.TempManifestDir()); err != nil {
		return err
	}
	if !spm.keepManifestDir {
		if err := os.RemoveAll(spm.BackupManifestDir()); err != nil {
			return err
		}
	}

	if !spm.keepEtcdDir {
		if err := os.RemoveAll(spm.BackupEtcdDir()); err != nil {
			return err
		}
	}

	return nil
}

func upgradeComponent(component string, waiter apiclient.Waiter, pathMgr StaticPodPathManager, cfg *kubeadmapi.InitConfiguration, beforePodHash string, recoverManifests map[string]string, isTLSUpgrade bool) error {
	// Special treatment is required for etcd case, when rollbackOldManifests should roll back etcd
	// manifests only for the case when component is Etcd
	recoverEtcd := false
	waitForComponentRestart := true
	if component == constants.Etcd {
		recoverEtcd = true
	}
	if isTLSUpgrade {
		// We currently depend on getting the Etcd mirror Pod hash from the KubeAPIServer;
		// Upgrading the Etcd protocol takes down the apiserver, so we can't verify component restarts if we restart Etcd independently.
		// Skip waiting for Etcd to restart and immediately move on to updating the apiserver.
		if component == constants.Etcd {
			waitForComponentRestart = false
		}
		// Normally, if an Etcd upgrade is successful, but the apiserver upgrade fails, Etcd is not rolled back.
		// In the case of a TLS upgrade, the old KubeAPIServer config is incompatible with the new Etcd confg, so we rollback Etcd
		// if the APIServer upgrade fails.
		if component == constants.KubeAPIServer {
			recoverEtcd = true
			fmt.Printf("[upgrade/staticpods] The %s manifest will be restored if component %q fails to upgrade\n", constants.Etcd, component)
		}
	}

	if err := renewCerts(cfg, component); err != nil {
		return errors.Wrapf(err, "failed to renew certificates for component %q", component)
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

	// Skip upgrade if current and new manifests are equal
	equal, err := staticpod.ManifestFilesAreEqual(currentManifestPath, newManifestPath)
	if err != nil {
		return err
	}
	if equal {
		fmt.Printf("[upgrade/staticpods] current and new manifests of %s are equal, skipping upgrade\n", component)
		return nil
	}

	// Move the old manifest into the old-manifests directory
	if err := pathMgr.MoveFile(currentManifestPath, backupManifestPath); err != nil {
		return rollbackOldManifests(recoverManifests, err, pathMgr, recoverEtcd)
	}

	// Move the new manifest into the manifests directory
	if err := pathMgr.MoveFile(newManifestPath, currentManifestPath); err != nil {
		return rollbackOldManifests(recoverManifests, err, pathMgr, recoverEtcd)
	}

	fmt.Printf("[upgrade/staticpods] Moved new manifest to %q and backed up old manifest to %q\n", currentManifestPath, backupManifestPath)

	if waitForComponentRestart {
		fmt.Println("[upgrade/staticpods] Waiting for the kubelet to restart the component")
		fmt.Printf("[upgrade/staticpods] This might take a minute or longer depending on the component/version gap (timeout %v)\n", UpgradeManifestTimeout)

		// Wait for the mirror Pod hash to change; otherwise we'll run into race conditions here when the kubelet hasn't had time to
		// notice the removal of the Static Pod, leading to a false positive below where we check that the API endpoint is healthy
		// If we don't do this, there is a case where we remove the Static Pod manifest, kubelet is slow to react, kubeadm checks the
		// API endpoint below of the OLD Static Pod component and proceeds quickly enough, which might lead to unexpected results.
		if err := waiter.WaitForStaticPodHashChange(cfg.NodeRegistration.Name, component, beforePodHash); err != nil {
			return rollbackOldManifests(recoverManifests, err, pathMgr, recoverEtcd)
		}

		// Wait for the static pod component to come up and register itself as a mirror pod
		if err := waiter.WaitForPodsWithLabel("component=" + component); err != nil {
			return rollbackOldManifests(recoverManifests, err, pathMgr, recoverEtcd)
		}

		fmt.Printf("[upgrade/staticpods] Component %q upgraded successfully!\n", component)
	} else {
		fmt.Printf("[upgrade/staticpods] Not waiting for pod-hash change for component %q\n", component)
	}

	return nil
}

// performEtcdStaticPodUpgrade performs upgrade of etcd, it returns bool which indicates fatal error or not and the actual error.
func performEtcdStaticPodUpgrade(client clientset.Interface, waiter apiclient.Waiter, pathMgr StaticPodPathManager, cfg *kubeadmapi.InitConfiguration, recoverManifests map[string]string, isTLSUpgrade bool, oldEtcdClient, newEtcdClient etcdutil.ClusterInterrogator) (bool, error) {
	// Add etcd static pod spec only if external etcd is not configured
	if cfg.Etcd.External != nil {
		return false, errors.New("external etcd detected, won't try to change any etcd state")
	}

	// Checking health state of etcd before proceeding with the upgrade
	_, err := oldEtcdClient.GetClusterStatus()
	if err != nil {
		return true, errors.Wrap(err, "etcd cluster is not healthy")
	}

	// Backing up etcd data store
	backupEtcdDir := pathMgr.BackupEtcdDir()
	runningEtcdDir := cfg.Etcd.Local.DataDir
	if err := util.CopyDir(runningEtcdDir, backupEtcdDir); err != nil {
		return true, errors.Wrap(err, "failed to back up etcd data")
	}

	// Need to check currently used version and version from constants, if differs then upgrade
	desiredEtcdVersion, err := constants.EtcdSupportedVersion(cfg.KubernetesVersion)
	if err != nil {
		return true, errors.Wrap(err, "failed to retrieve an etcd version for the target Kubernetes version")
	}

	// gets the etcd version of the local/stacked etcd member running on the current machine
	currentEtcdVersions, err := oldEtcdClient.GetClusterVersions()
	if err != nil {
		return true, errors.Wrap(err, "failed to retrieve the current etcd version")
	}
	currentEtcdVersionStr, ok := currentEtcdVersions[etcdutil.GetClientURL(cfg)]
	if !ok {
		fmt.Println(currentEtcdVersions)
		return true, errors.Wrap(err, "failed to retrieve the current etcd version")
	}

	currentEtcdVersion, err := version.ParseSemantic(currentEtcdVersionStr)
	if err != nil {
		return true, errors.Wrapf(err, "failed to parse the current etcd version(%s)", currentEtcdVersionStr)
	}

	// Comparing current etcd version with desired to catch the same version or downgrade condition and fail on them.
	if desiredEtcdVersion.LessThan(currentEtcdVersion) {
		return false, errors.Errorf("the desired etcd version for this Kubernetes version %q is %q, but the current etcd version is %q. Won't downgrade etcd, instead just continue", cfg.KubernetesVersion, desiredEtcdVersion.String(), currentEtcdVersion.String())
	}
	// For the case when desired etcd version is the same as current etcd version
	if strings.Compare(desiredEtcdVersion.String(), currentEtcdVersion.String()) == 0 {
		return false, nil
	}

	beforeEtcdPodHash, err := waiter.WaitForStaticPodSingleHash(cfg.NodeRegistration.Name, constants.Etcd)
	if err != nil {
		return true, errors.Wrap(err, "failed to get etcd pod's hash")
	}

	// Write the updated etcd static Pod manifest into the temporary directory, at this point no etcd change
	// has occurred in any aspects.
	if err := etcdphase.CreateLocalEtcdStaticPodManifestFile(pathMgr.TempManifestDir(), cfg); err != nil {
		return true, errors.Wrap(err, "error creating local etcd static pod manifest file")
	}

	// Waiter configurations for checking etcd status
	noDelay := 0 * time.Second
	podRestartDelay := noDelay
	if isTLSUpgrade {
		// If we are upgrading TLS we need to wait for old static pod to be removed.
		// This is needed because we are not able to currently verify that the static pod
		// has been updated through the apiserver across an etcd TLS upgrade.
		// This value is arbitrary but seems to be long enough in manual testing.
		podRestartDelay = 30 * time.Second
	}
	retries := 10
	retryInterval := 15 * time.Second

	// Perform etcd upgrade using common to all control plane components function
	if err := upgradeComponent(constants.Etcd, waiter, pathMgr, cfg, beforeEtcdPodHash, recoverManifests, isTLSUpgrade); err != nil {
		fmt.Printf("[upgrade/etcd] Failed to upgrade etcd: %v\n", err)
		// Since upgrade component failed, the old etcd manifest has either been restored or was never touched
		// Now we need to check the health of etcd cluster if it is up with old manifest
		fmt.Println("[upgrade/etcd] Waiting for previous etcd to become available")
		if _, err := oldEtcdClient.WaitForClusterAvailable(noDelay, retries, retryInterval); err != nil {
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
			if _, err := oldEtcdClient.WaitForClusterAvailable(noDelay, retries, retryInterval); err != nil {
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
	if _, err = newEtcdClient.WaitForClusterAvailable(podRestartDelay, retries, retryInterval); err != nil {
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
		if _, err := oldEtcdClient.WaitForClusterAvailable(noDelay, retries, retryInterval); err != nil {
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
func StaticPodControlPlane(client clientset.Interface, waiter apiclient.Waiter, pathMgr StaticPodPathManager, cfg *kubeadmapi.InitConfiguration, etcdUpgrade bool, oldEtcdClient, newEtcdClient etcdutil.ClusterInterrogator) error {
	recoverManifests := map[string]string{}
	var isTLSUpgrade bool
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

	// etcd upgrade is done prior to other control plane components
	if !isExternalEtcd && etcdUpgrade {
		previousEtcdHasTLS := oldEtcdClient.HasTLS()

		// set the TLS upgrade flag for all components
		isTLSUpgrade = !previousEtcdHasTLS
		if isTLSUpgrade {
			fmt.Printf("[upgrade/etcd] Upgrading to TLS for %s\n", constants.Etcd)
		}

		// Perform etcd upgrade using common to all control plane components function
		fatal, err := performEtcdStaticPodUpgrade(client, waiter, pathMgr, cfg, recoverManifests, isTLSUpgrade, oldEtcdClient, newEtcdClient)
		if err != nil {
			if fatal {
				return err
			}
			fmt.Printf("[upgrade/etcd] non fatal issue encountered during upgrade: %v\n", err)
		}
	}

	// Write the updated static Pod manifests into the temporary directory
	fmt.Printf("[upgrade/staticpods] Writing new Static Pod manifests to %q\n", pathMgr.TempManifestDir())
	err = controlplanephase.CreateInitStaticPodManifestFiles(pathMgr.TempManifestDir(), cfg)
	if err != nil {
		return errors.Wrap(err, "error creating init static pod manifest files")
	}

	for _, component := range constants.MasterComponents {
		if err = upgradeComponent(component, waiter, pathMgr, cfg, beforePodHashMap[component], recoverManifests, isTLSUpgrade); err != nil {
			return err
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
	return errors.New("couldn't upgrade control plane. kubeadm has tried to recover everything into the earlier state. Errors faced")
}

// rollbackEtcdData rolls back the content of etcd folder if something went wrong.
// When the folder contents are successfully rolled back, nil is returned, otherwise an error is returned.
func rollbackEtcdData(cfg *kubeadmapi.InitConfiguration, pathMgr StaticPodPathManager) error {
	backupEtcdDir := pathMgr.BackupEtcdDir()
	runningEtcdDir := cfg.Etcd.Local.DataDir

	if err := util.CopyDir(backupEtcdDir, runningEtcdDir); err != nil {
		// Let the user know there we're problems, but we tried to reçover
		return errors.Wrapf(err, "couldn't recover etcd database with error, the location of etcd backup: %s ", backupEtcdDir)
	}

	return nil
}

func renewCerts(cfg *kubeadmapi.InitConfiguration, component string) error {
	if cfg.Etcd.Local != nil {
		// ensure etcd certs are loaded for etcd and kube-apiserver
		if component == constants.Etcd || component == constants.KubeAPIServer {
			caCert, caKey, err := certsphase.LoadCertificateAuthority(cfg.CertificatesDir, certsphase.KubeadmCertEtcdCA.BaseName)
			if err != nil {
				return errors.Wrapf(err, "failed to upgrade the %s CA certificate and key", constants.Etcd)
			}
			renewer := renewal.NewFileRenewal(caCert, caKey)

			if component == constants.Etcd {
				for _, cert := range []*certsphase.KubeadmCert{
					&certsphase.KubeadmCertEtcdServer,
					&certsphase.KubeadmCertEtcdPeer,
					&certsphase.KubeadmCertEtcdHealthcheck,
				} {
					if err := renewal.RenewExistingCert(cfg.CertificatesDir, cert.BaseName, renewer); err != nil {
						return errors.Wrapf(err, "failed to renew %s certificate and key", cert.Name)
					}
				}
			}
			if component == constants.KubeAPIServer {
				cert := certsphase.KubeadmCertEtcdAPIClient
				if err := renewal.RenewExistingCert(cfg.CertificatesDir, cert.BaseName, renewer); err != nil {
					return errors.Wrapf(err, "failed to renew %s certificate and key", cert.Name)
				}
			}
		}
	}
	return nil
}
