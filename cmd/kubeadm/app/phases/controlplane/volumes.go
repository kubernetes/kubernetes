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

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

const (
	caCertsVolumeName              = "ca-certs"
	caCertsVolumePath              = "/etc/ssl/certs"
	flexvolumeDirVolumeName        = "flexvolume-dir"
	defaultFlexvolumeDirVolumePath = "/usr/libexec/kubernetes/kubelet-plugins/volume/exec"
)

// caCertsExtraVolumePaths specifies the paths that can be conditionally mounted into the apiserver and controller-manager containers
// as /etc/ssl/certs might be or contain a symlink to them. It's a variable since it may be changed in unit testing. This var MUST
// NOT be changed in normal codepaths during runtime.
var caCertsExtraVolumePaths = []string{"/etc/pki/ca-trust", "/etc/pki/tls/certs", "/etc/ca-certificates", "/usr/share/ca-certificates", "/usr/local/share/ca-certificates"}

// getHostPathVolumesForTheControlPlane gets the required hostPath volumes and mounts for the control plane
func getHostPathVolumesForTheControlPlane(cfg *kubeadmapi.ClusterConfiguration) controlPlaneHostPathMounts {
	hostPathDirectoryOrCreate := v1.HostPathDirectoryOrCreate
	hostPathFileOrCreate := v1.HostPathFileOrCreate
	mounts := newControlPlaneHostPathMounts()

	// HostPath volumes for the API Server
	// Read-only mount for the certificates directory
	// TODO: Always mount the K8s Certificates directory to a static path inside of the container
	mounts.NewHostPathMount(kubeadmconstants.KubeAPIServer, kubeadmconstants.KubeCertificatesVolumeName, cfg.CertificatesDir, cfg.CertificatesDir, true, &hostPathDirectoryOrCreate)
	// Read-only mount for the ca certs (/etc/ssl/certs) directory
	mounts.NewHostPathMount(kubeadmconstants.KubeAPIServer, caCertsVolumeName, caCertsVolumePath, caCertsVolumePath, true, &hostPathDirectoryOrCreate)

	// If external etcd is specified, mount the directories needed for accessing the CA/serving certs and the private key
	if cfg.Etcd.External != nil {
		etcdVols, etcdVolMounts := getEtcdCertVolumes(cfg.Etcd.External, cfg.CertificatesDir)
		mounts.AddHostPathMounts(kubeadmconstants.KubeAPIServer, etcdVols, etcdVolMounts)
	}

	// HostPath volumes for the controller manager
	// Read-only mount for the certificates directory
	// TODO: Always mount the K8s Certificates directory to a static path inside of the container
	mounts.NewHostPathMount(kubeadmconstants.KubeControllerManager, kubeadmconstants.KubeCertificatesVolumeName, cfg.CertificatesDir, cfg.CertificatesDir, true, &hostPathDirectoryOrCreate)
	// Read-only mount for the ca certs (/etc/ssl/certs) directory
	mounts.NewHostPathMount(kubeadmconstants.KubeControllerManager, caCertsVolumeName, caCertsVolumePath, caCertsVolumePath, true, &hostPathDirectoryOrCreate)
	// Read-only mount for the controller manager kubeconfig file
	controllerManagerKubeConfigFile := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.ControllerManagerKubeConfigFileName)
	mounts.NewHostPathMount(kubeadmconstants.KubeControllerManager, kubeadmconstants.KubeConfigVolumeName, controllerManagerKubeConfigFile, controllerManagerKubeConfigFile, true, &hostPathFileOrCreate)
	// Mount for the flexvolume directory (/usr/libexec/kubernetes/kubelet-plugins/volume/exec by default)
	// Flexvolume dir must NOT be readonly as it is used for third-party plugins to integrate with their storage backends via unix domain socket.
	flexvolumeDirVolumePath, idx := kubeadmapi.GetArgValue(cfg.ControllerManager.ExtraArgs, "flex-volume-plugin-dir", -1)
	if idx == -1 {
		flexvolumeDirVolumePath = defaultFlexvolumeDirVolumePath
	}
	mounts.NewHostPathMount(kubeadmconstants.KubeControllerManager, flexvolumeDirVolumeName, flexvolumeDirVolumePath, flexvolumeDirVolumePath, false, &hostPathDirectoryOrCreate)

	// HostPath volumes for the scheduler
	// Read-only mount for the scheduler kubeconfig file
	schedulerKubeConfigFile := filepath.Join(kubeadmconstants.KubernetesDir, kubeadmconstants.SchedulerKubeConfigFileName)
	mounts.NewHostPathMount(kubeadmconstants.KubeScheduler, kubeadmconstants.KubeConfigVolumeName, schedulerKubeConfigFile, schedulerKubeConfigFile, true, &hostPathFileOrCreate)

	// On some systems where we host-mount /etc/ssl/certs, it is also required to mount additional directories.
	// This is needed due to symlinks pointing from files in /etc/ssl/certs to these directories.
	for _, caCertsExtraVolumePath := range caCertsExtraVolumePaths {
		if isExtraVolumeMountNeeded(caCertsExtraVolumePath) {
			caCertsExtraVolumeName := strings.Replace(caCertsExtraVolumePath, "/", "-", -1)[1:]
			mounts.NewHostPathMount(kubeadmconstants.KubeAPIServer, caCertsExtraVolumeName, caCertsExtraVolumePath, caCertsExtraVolumePath, true, &hostPathDirectoryOrCreate)
			mounts.NewHostPathMount(kubeadmconstants.KubeControllerManager, caCertsExtraVolumeName, caCertsExtraVolumePath, caCertsExtraVolumePath, true, &hostPathDirectoryOrCreate)
		}
	}

	// Merge user defined mounts and ensure unique volume and volume mount
	// names
	mounts.AddExtraHostPathMounts(kubeadmconstants.KubeAPIServer, cfg.APIServer.ExtraVolumes)
	mounts.AddExtraHostPathMounts(kubeadmconstants.KubeControllerManager, cfg.ControllerManager.ExtraVolumes)
	mounts.AddExtraHostPathMounts(kubeadmconstants.KubeScheduler, cfg.Scheduler.ExtraVolumes)

	return mounts
}

// controlPlaneHostPathMounts is a helper struct for handling all the control plane's hostPath mounts in an easy way
type controlPlaneHostPathMounts struct {
	// volumes is a nested map that forces a unique volumes. The outer map's
	// keys are a string that should specify the target component to add the
	// volume to. The values (inner map) of the outer map are maps with string
	// keys and v1.Volume values. The inner map's key should specify the volume
	// name.
	volumes map[string]map[string]v1.Volume
	// volumeMounts is a nested map that forces a unique volume mounts. The
	// outer map's keys are a string that should specify the target component
	// to add the volume mount to. The values (inner map) of the outer map are
	// maps with string keys and v1.VolumeMount values. The inner map's key
	// should specify the volume mount name.
	volumeMounts map[string]map[string]v1.VolumeMount
}

func newControlPlaneHostPathMounts() controlPlaneHostPathMounts {
	return controlPlaneHostPathMounts{
		volumes:      map[string]map[string]v1.Volume{},
		volumeMounts: map[string]map[string]v1.VolumeMount{},
	}
}

func (c *controlPlaneHostPathMounts) NewHostPathMount(component, mountName, hostPath, containerPath string, readOnly bool, hostPathType *v1.HostPathType) {
	vol := staticpodutil.NewVolume(mountName, hostPath, hostPathType)
	c.addComponentVolume(component, vol)
	volMount := staticpodutil.NewVolumeMount(mountName, containerPath, readOnly)
	c.addComponentVolumeMount(component, volMount)
}

func (c *controlPlaneHostPathMounts) AddHostPathMounts(component string, vols []v1.Volume, volMounts []v1.VolumeMount) {
	for _, v := range vols {
		c.addComponentVolume(component, v)
	}
	for _, v := range volMounts {
		c.addComponentVolumeMount(component, v)
	}
}

// AddExtraHostPathMounts adds host path mounts and overwrites the default
// paths in the case that a user specifies the same volume/volume mount name.
func (c *controlPlaneHostPathMounts) AddExtraHostPathMounts(component string, extraVols []kubeadmapi.HostPathMount) {
	for _, extraVol := range extraVols {
		hostPathType := extraVol.PathType
		c.NewHostPathMount(component, extraVol.Name, extraVol.HostPath, extraVol.MountPath, extraVol.ReadOnly, &hostPathType)
	}
}

func (c *controlPlaneHostPathMounts) GetVolumes(component string) map[string]v1.Volume {
	return c.volumes[component]
}

func (c *controlPlaneHostPathMounts) GetVolumeMounts(component string) map[string]v1.VolumeMount {
	return c.volumeMounts[component]
}

func (c *controlPlaneHostPathMounts) addComponentVolume(component string, vol v1.Volume) {
	if _, ok := c.volumes[component]; !ok {
		c.volumes[component] = map[string]v1.Volume{}
	}
	c.volumes[component][vol.Name] = vol
}

func (c *controlPlaneHostPathMounts) addComponentVolumeMount(component string, volMount v1.VolumeMount) {
	if _, ok := c.volumeMounts[component]; !ok {
		c.volumeMounts[component] = map[string]v1.VolumeMount{}
	}
	c.volumeMounts[component][volMount.Name] = volMount
}

// getEtcdCertVolumes returns the volumes/volumemounts needed for talking to an external etcd cluster
func getEtcdCertVolumes(etcdCfg *kubeadmapi.ExternalEtcd, k8sCertificatesDir string) ([]v1.Volume, []v1.VolumeMount) {
	certPaths := []string{etcdCfg.CAFile, etcdCfg.CertFile, etcdCfg.KeyFile}
	certDirs := sets.New[string]()
	for _, certPath := range certPaths {
		certDir := filepath.ToSlash(filepath.Dir(certPath))
		// Ignore ".", which is the result of passing an empty path.
		// Also ignore the cert directories that already may be mounted; /etc/ssl/certs, /etc/pki/ca-trust/ or Kubernetes CertificatesDir
		// If the etcd certs are in there, it's okay, we don't have to do anything
		extraVolumePath := false
		for _, caCertsExtraVolumePath := range caCertsExtraVolumePaths {
			if strings.HasPrefix(certDir, caCertsExtraVolumePath) {
				extraVolumePath = true
				break
			}
		}
		if certDir == "." || extraVolumePath || strings.HasPrefix(certDir, caCertsVolumePath) || strings.HasPrefix(certDir, k8sCertificatesDir) {
			continue
		}
		// Filter out any existing hostpath mounts in the list that contains a subset of the path
		alreadyExists := false
		for _, existingCertDir := range sets.List(certDirs) {
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
	pathType := v1.HostPathDirectoryOrCreate
	for i, certDir := range sets.List(certDirs) {
		name := fmt.Sprintf("etcd-certs-%d", i)
		volumes = append(volumes, staticpodutil.NewVolume(name, certDir, &pathType))
		volumeMounts = append(volumeMounts, staticpodutil.NewVolumeMount(name, certDir, true))
	}
	return volumes, volumeMounts
}

// isExtraVolumeMountNeeded specifies whether /etc/pki/ca-trust/ should be host-mounted into the containers
// On some systems were we host-mount /etc/ssl/certs, it is also required to mount /etc/pki/ca-trust/. This is needed
// due to symlinks pointing from files in /etc/ssl/certs into /etc/pki/ca-trust/
func isExtraVolumeMountNeeded(caCertsExtraVolumePath string) bool {
	if _, err := os.Stat(caCertsExtraVolumePath); err == nil {
		return true
	}
	return false
}
