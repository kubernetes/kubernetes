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
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"

	"k8s.io/klog"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/initsystem"
	utilruntime "k8s.io/kubernetes/cmd/kubeadm/app/util/runtime"
	utilsexec "k8s.io/utils/exec"
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
		fmt.Println("[reset] Stopping the kubelet service")
		if err := initSystem.ServiceStop("kubelet"); err != nil {
			klog.Warningf("[reset] The kubelet service could not be stopped by kubeadm: [%v]\n", err)
			klog.Warningln("[reset] Please ensure kubelet is stopped manually")
		}
	}

	// Try to unmount mounted directories under kubeadmconstants.KubeletRunDirectory in order to be able to remove the kubeadmconstants.KubeletRunDirectory directory later
	fmt.Printf("[reset] Unmounting mounted directories in %q\n", kubeadmconstants.KubeletRunDirectory)
	// In case KubeletRunDirectory holds a symbolic link, evaluate it
	kubeletRunDir, err := absoluteKubeletRunDirectory()
	if err == nil {
		// Only clean absoluteKubeletRunDirectory if umountDirsCmd passed without error
		r.AddDirsToClean(kubeletRunDir)
	}

	klog.V(1).Info("[reset] Removing Kubernetes-managed containers")
	if err := removeContainers(utilsexec.New(), r.CRISocketPath()); err != nil {
		klog.Warningf("[reset] Failed to remove containers: %v\n", err)
	}

	r.AddDirsToClean("/etc/cni/net.d", "/var/lib/dockershim", "/var/run/kubernetes", "/var/lib/cni")

	// Remove contents from the config and pki directories
	klog.V(1).Infoln("[reset] Removing contents from the config and pki directories")
	if certsDir != kubeadmapiv1beta2.DefaultCertificatesDir {
		klog.Warningf("[reset] WARNING: Cleaning a non-default certificates directory: %q\n", certsDir)
	}
	resetConfigDir(kubeadmconstants.KubernetesDir, certsDir)

	return nil
}

func absoluteKubeletRunDirectory() (string, error) {
	absoluteKubeletRunDirectory, err := filepath.EvalSymlinks(kubeadmconstants.KubeletRunDirectory)
	if err != nil {
		klog.Warningf("[reset] Failed to evaluate the %q directory. Skipping its unmount and cleanup: %v\n", kubeadmconstants.KubeletRunDirectory, err)
		return "", err
	}

	// Only unmount mount points which start with "/var/lib/kubelet" or absolute path of symbolic link, and avoid using empty absoluteKubeletRunDirectory
	umountDirsCmd := fmt.Sprintf("awk '$2 ~ path {print $2}' path=%s/ /proc/mounts | xargs -r umount", absoluteKubeletRunDirectory)
	klog.V(1).Infof("[reset] Executing command %q", umountDirsCmd)
	umountOutputBytes, err := exec.Command("sh", "-c", umountDirsCmd).Output()
	if err != nil {
		klog.Warningf("[reset] Failed to unmount mounted directories in %s: %s\n", kubeadmconstants.KubeletRunDirectory, string(umountOutputBytes))
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

// resetConfigDir is used to cleanup the files kubeadm writes in /etc/kubernetes/.
func resetConfigDir(configPathDir, pkiPathDir string) {
	dirsToClean := []string{
		filepath.Join(configPathDir, kubeadmconstants.ManifestsSubDirName),
		pkiPathDir,
	}
	fmt.Printf("[reset] Deleting contents of config directories: %v\n", dirsToClean)
	for _, dir := range dirsToClean {
		if err := CleanDir(dir); err != nil {
			klog.Warningf("[reset] Failed to remove directory: %q [%v]\n", dir, err)
		}
	}

	filesToClean := []string{
		filepath.Join(configPathDir, kubeadmconstants.AdminKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.KubeletKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.KubeletBootstrapKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.ControllerManagerKubeConfigFileName),
		filepath.Join(configPathDir, kubeadmconstants.SchedulerKubeConfigFileName),
	}
	fmt.Printf("[reset] Deleting files: %v\n", filesToClean)
	for _, path := range filesToClean {
		if err := os.RemoveAll(path); err != nil {
			klog.Warningf("[reset] Failed to remove file: %q [%v]\n", path, err)
		}
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
