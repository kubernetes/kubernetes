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

package controlplane

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

const (
	k8sCertsVolumeName   = "k8s-certs"
	caCertsVolumeName    = "ca-certs"
	caCertsVolumePath    = "/etc/ssl/certs"
	caCertsPkiVolumeName = "ca-certs-etc-pki"
	kubeConfigVolumeName = "kubeconfig"
)

// caCertsPkiVolumePath specifies the path that can be conditionally mounted into the apiserver and controller-manager containers
// as /etc/ssl/certs might be a symlink to it. It's a variable since it may be changed in unit testing. This var MUST NOT be changed
// in normal codepaths during runtime.
var caCertsPkiVolumePath = "/etc/pki"

// getHostPathVolumesForTheControlPlane gets the required hostPath volumes and mounts for the control plane
func getHostPathVolumesForTheControlPlane(cfg *kubeadmapi.MasterConfiguration) controlPlaneHostPathMounts {
	mounts := newControlPlaneHostPathMounts()

	// HostPath volumes for the API Server
	// Read-only mount for the certificates directory
	// TODO: Always mount the K8s Certificates directory to a static path inside of the container
	mounts.NewHostPathMount(kubeadmconstants.KubeAPIServer, k8sCertsVolumeName, cfg.CertificatesDir, cfg.CertificatesDir, true)
	// Read-only mount for the ca certs (/etc/ssl/certs) directory
	mounts.NewHostPathMount(kubeadmconstants.KubeAPIServer, caCertsVolumeName, caCertsVolumePath, caCertsVolumePath, true)

	// If external etcd is specified, mount the directories needed for accessing the CA/serving certs and the private key
	if len(cfg.Etcd.Endpoints) != 0 {
		etcdVols, etcdVolMounts := getEtcdCertVolumes(cfg.Etcd)
		mounts.AddHostPathMounts(kubeadmconstants.KubeAPIServer, etcdVols, etcdVolMounts)
	}

	// HostPath volumes for the controller manager
	// Read-only mount for the certificates directory
	// TODO: Always mount the K8s Certificates directory to a static path inside of the container
	mounts.NewHostPathMount(kubeadmconstants.KubeControllerManager, k8sCertsVolumeName, cfg.CertificatesDir, cfg.CertificatesDir, true)
	// Read-only mount for the ca certs (/etc/ssl/certs) directory
	mounts.NewHostPathMount(kubeadmconstants.KubeControllerManager, caCertsVolumeName, caCertsVolumePath, caCertsVolumePath, true)
	// Read-only mount for the controller manager kubeconfig file
	controllerManagerKubeConfigFile := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ControllerManagerKubeConfigFileName)
	mounts.NewHostPathMount(kubeadmconstants.KubeControllerManager, kubeConfigVolumeName, controllerManagerKubeConfigFile, controllerManagerKubeConfigFile, true)

	// HostPath volumes for the scheduler
	// Read-only mount for the scheduler kubeconfig file
	schedulerKubeConfigFile := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.SchedulerKubeConfigFileName)
	mounts.NewHostPathMount(kubeadmconstants.KubeScheduler, kubeConfigVolumeName, schedulerKubeConfigFile, schedulerKubeConfigFile, true)

	// On some systems were we host-mount /etc/ssl/certs, it is also required to mount /etc/pki. This is needed
	// due to symlinks pointing from files in /etc/ssl/certs into /etc/pki/
	if isPkiVolumeMountNeeded() {
		mounts.NewHostPathMount(kubeadmconstants.KubeAPIServer, caCertsPkiVolumeName, caCertsPkiVolumePath, caCertsPkiVolumePath, true)
		mounts.NewHostPathMount(kubeadmconstants.KubeControllerManager, caCertsPkiVolumeName, caCertsPkiVolumePath, caCertsPkiVolumePath, true)
	}

	return mounts
}

// controlPlaneHostPathMounts is a helper struct for handling all the control plane's hostPath mounts in an easy way
type controlPlaneHostPathMounts struct {
	volumes      map[string][]v1.Volume
	volumeMounts map[string][]v1.VolumeMount
}

func newControlPlaneHostPathMounts() controlPlaneHostPathMounts {
	return controlPlaneHostPathMounts{
		volumes:      map[string][]v1.Volume{},
		volumeMounts: map[string][]v1.VolumeMount{},
	}
}

func (c *controlPlaneHostPathMounts) NewHostPathMount(component, mountName, hostPath, containerPath string, readOnly bool) {
	c.volumes[component] = append(c.volumes[component], staticpodutil.NewVolume(mountName, hostPath))
	c.volumeMounts[component] = append(c.volumeMounts[component], staticpodutil.NewVolumeMount(mountName, containerPath, readOnly))
}

func (c *controlPlaneHostPathMounts) AddHostPathMounts(component string, vols []v1.Volume, volMounts []v1.VolumeMount) {
	c.volumes[component] = append(c.volumes[component], vols...)
	c.volumeMounts[component] = append(c.volumeMounts[component], volMounts...)
}

func (c *controlPlaneHostPathMounts) GetVolumes(component string) []v1.Volume {
	return c.volumes[component]
}

func (c *controlPlaneHostPathMounts) GetVolumeMounts(component string) []v1.VolumeMount {
	return c.volumeMounts[component]
}

// getEtcdCertVolumes returns the volumes/volumemounts needed for talking to an external etcd cluster
func getEtcdCertVolumes(etcdCfg kubeadmapi.Etcd) ([]v1.Volume, []v1.VolumeMount) {
	certPaths := []string{etcdCfg.CAFile, etcdCfg.CertFile, etcdCfg.KeyFile}
	certDirs := sets.NewString()
	for _, certPath := range certPaths {
		certDir := filepath.Dir(certPath)
		// Ignore ".", which is the result of passing an empty path.
		// Also ignore the cert directories that already may be mounted; /etc/ssl/certs and /etc/pki. If the etcd certs are in there, it's okay, we don't have to do anything
		if certDir == "." || strings.HasPrefix(certDir, caCertsVolumePath) || strings.HasPrefix(certDir, caCertsPkiVolumePath) {
			continue
		}
		// Filter out any existing hostpath mounts in the list that contains a subset of the path
		alreadyExists := false
		for _, existingCertDir := range certDirs.List() {
			// If the current directory is a parent of an existing one, remove the already existing one
			if strings.HasPrefix(existingCertDir, certDir) {
				certDirs.Delete(existingCertDir)
			} else if strings.HasPrefix(certDir, existingCertDir) {
				// If an existing directory is a parent of the current one, don't add the current one
				alreadyExists = true
			}
		}
		if alreadyExists {
			continue
		}
		certDirs.Insert(certDir)
	}

	volumes := []v1.Volume{}
	volumeMounts := []v1.VolumeMount{}
	for i, certDir := range certDirs.List() {
		name := fmt.Sprintf("etcd-certs-%d", i)
		volumes = append(volumes, staticpodutil.NewVolume(name, certDir))
		volumeMounts = append(volumeMounts, staticpodutil.NewVolumeMount(name, certDir, true))
	}
	return volumes, volumeMounts
}

// isPkiVolumeMountNeeded specifies whether /etc/pki should be host-mounted into the containers
// On some systems were we host-mount /etc/ssl/certs, it is also required to mount /etc/pki. This is needed
// due to symlinks pointing from files in /etc/ssl/certs into /etc/pki/
func isPkiVolumeMountNeeded() bool {
	if _, err := os.Stat(caCertsPkiVolumePath); err == nil {
		return true
	}
	return false
}
