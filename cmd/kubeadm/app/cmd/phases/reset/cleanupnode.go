/*
Copyright 2019 The Kubernetes Authors.

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

package phases

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/pkg/errors"

	"k8s.io/klog/v2"
	utilsexec "k8s.io/utils/exec"

	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/initsystem"
	utilruntime "k8s.io/kubernetes/cmd/kubeadm/app/util/runtime"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/users"
)

// NewCleanupNodePhase creates a kubeadm workflow phase that cleanup the node
func NewCleanupNodePhase() workflow.Phase {
	return workflow.Phase{
		Name:    "cleanup-node",
		Aliases: []string{"cleanupnode"},
		Short:   "Run cleanup node.",
		Run:     runCleanupNode,
		InheritFlags: []string{
			options.CertificatesDir,
			options.NodeCRISocket,
		},
	}
}

func runCleanupNode(c workflow.RunData) error {
	dirsToClean := []string{filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ManifestsSubDirName)}
	r, ok := c.(resetData)
	if !ok {
		return errors.New("cleanup-node phase invoked with an invalid data struct")
	}
	certsDir := r.CertificatesDir()

	// Try to stop the kubelet service
	klog.V(1).Infoln("[reset] Getting init system")
	initSystem, err := initsystem.GetInitSystem()
	if err != nil {
		klog.Warningln("[reset] The kubelet service could not be stopped by kubeadm. Unable to detect a supported init system!")
		klog.Warningln("[reset] Please ensure kubelet is stopped manually")
	} else {
		if !r.DryRun() {
			fmt.Println("[reset] Stopping the kubelet service")
			if err := initSystem.ServiceStop("kubelet"); err != nil {
				klog.Warningf("[reset] The kubelet service could not be stopped by kubeadm: [%v]\n", err)
				klog.Warningln("[reset] Please ensure kubelet is stopped manually")
			}
		} else {
			fmt.Println("[reset] Would stop the kubelet service")
		}
	}

	if !r.DryRun() {
		// Try to unmount mounted directories under kubeadmconstants.KubeletRunDirectory in order to be able to remove the kubeadmconstants.KubeletRunDirectory directory later
		fmt.Printf("[reset] Unmounting mounted directories in %q\n", kubeadmconstants.KubeletRunDirectory)
		// In case KubeletRunDirectory holds a symbolic link, evaluate it
		kubeletRunDir, err := absoluteKubeletRunDirectory()
		if err == nil {
			// Only clean absoluteKubeletRunDirectory if umountDirsCmd passed without error
			dirsToClean = append(dirsToClean, kubeletRunDir)
		}
	} else {
		fmt.Printf("[reset] Would unmount mounted directories in %q\n", kubeadmconstants.KubeletRunDirectory)
	}

	if !r.DryRun() {
		klog.V(1).Info("[reset] Removing Kubernetes-managed containers")
		if err := removeContainers(utilsexec.New(), r.CRISocketPath()); err != nil {
			klog.Warningf("[reset] Failed to remove containers: %v\n", err)
		}
	} else {
		fmt.Println("[reset] Would remove Kubernetes-managed containers")
	}

	// Remove contents from the config and pki directories
	if certsDir != kubeadmapiv1.DefaultCertificatesDir {
		klog.Warningf("[reset] WARNING: Cleaning a non-default certificates directory: %q\n", certsDir)
	}
	dirsToClean = append(dirsToClean, certsDir)
	resetConfigDir(kubeadmconstants.KubernetesDir, dirsToClean, r.DryRun())

	if r.Cfg() != nil && features.Enabled(r.Cfg().FeatureGates, features.RootlessControlPlane) {
		if !r.DryRun() {
			klog.V(1).Infoln("[reset] Removing users and groups created for rootless control-plane")
			if err := users.RemoveUsersAndGroups(); err != nil {
				klog.Warningf("[reset] Failed to remove users and groups: %v\n", err)
			}
		} else {
			fmt.Println("[reset] Would remove users and groups created for rootless control-plane")
		}
	}

	return nil
}

func absoluteKubeletRunDirectory() (string, error) {
	absoluteKubeletRunDirectory, err := filepath.EvalSymlinks(kubeadmconstants.KubeletRunDirectory)
	if err != nil {
		klog.Warningf("[reset] Failed to evaluate the %q directory. Skipping its unmount and cleanup: %v\n", kubeadmconstants.KubeletRunDirectory, err)
		return "", err
	}
	err = unmountKubeletDirectory(absoluteKubeletRunDirectory)
	if err != nil {
		klog.Warningf("[reset] Failed to unmount mounted directories in %s \n", kubeadmconstants.KubeletRunDirectory)
		return "", err
	}
	return absoluteKubeletRunDirectory, nil
}

func removeContainers(execer utilsexec.Interface, criSocketPath string) error {
	containerRuntime, err := utilruntime.NewContainerRuntime(execer, criSocketPath)
	if err != nil {
		return err
	}
	containers, err := containerRuntime.ListKubeContainers()
	if err != nil {
		return err
	}
	return containerRuntime.RemoveContainers(containers)
}

// resetConfigDir is used to cleanup the files in the folder defined in dirsToClean.
func resetConfigDir(configPathDir string, dirsToClean []string, isDryRun bool) {
	if !isDryRun {
		fmt.Printf("[reset] Deleting contents of directories: %v\n", dirsToClean)
		for _, dir := range dirsToClean {
			if err := CleanDir(dir); err != nil {
				klog.Warningf("[reset] Failed to delete contents of %q directory: %v", dir, err)
			}
		}
	} else {
		fmt.Printf("[reset] Would delete contents of directories: %v\n", dirsToClean)
	}

	filesToClean := []string{
		filepath.Join(configPathDir, kubeadmconstants.AdminKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.KubeletKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.KubeletBootstrapKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.ControllerManagerKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.SchedulerKubeConfigFileName),
	}

	if !isDryRun {
		fmt.Printf("[reset] Deleting files: %v\n", filesToClean)
		for _, path := range filesToClean {
			if err := os.RemoveAll(path); err != nil {
				klog.Warningf("[reset] Failed to remove file: %q [%v]\n", path, err)
			}
		}
	} else {
		fmt.Printf("[reset] Would delete files: %v\n", filesToClean)
	}
}

// CleanDir removes everything in a directory, but not the directory itself
func CleanDir(filePath string) error {
	// If the directory doesn't even exist there's nothing to do, and we do
	// not consider this an error
	if _, err := os.Stat(filePath); os.IsNotExist(err) {
		return nil
	}

	d, err := os.Open(filePath)
	if err != nil {
		return err
	}
	defer d.Close()
	names, err := d.Readdirnames(-1)
	if err != nil {
		return err
	}
	for _, name := range names {
		if err = os.RemoveAll(filepath.Join(filePath, name)); err != nil {
			return err
		}
	}
	return nil
}

func IsDirEmpty(dir string) (bool, error) {
	d, err := os.Open(dir)
	if err != nil {
		return false, err
	}
	defer d.Close()
	_, err = d.Readdirnames(1)
	if err == io.EOF {
		return true, nil
	}
	return false, nil
}
