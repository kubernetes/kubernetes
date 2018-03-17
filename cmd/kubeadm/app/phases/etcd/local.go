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

package etcd

import (
	"fmt"
	"path/filepath"

	"k8s.io/api/core/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

const (
	etcdVolumeName  = "etcd-data"
	certsVolumeName = "etcd-certs"
)

// CreateLocalEtcdStaticPodManifestFile will write local etcd static pod manifest file.
func CreateLocalEtcdStaticPodManifestFile(manifestDir string, cfg *kubeadmapi.MasterConfiguration) error {

	// gets etcd StaticPodSpec, actualized for the current MasterConfiguration
	spec := GetEtcdPodSpec(cfg)
	// writes etcd StaticPod to disk
	if err := staticpodutil.WriteStaticPodToDisk(kubeadmconstants.Etcd, manifestDir, spec); err != nil {
		return err
	}

	fmt.Printf("[etcd] Wrote Static Pod manifest for a local etcd instance to %q\n", kubeadmconstants.GetStaticPodFilepath(kubeadmconstants.Etcd, manifestDir))
	return nil
}

// GetEtcdPodSpec returns the etcd static Pod actualized to the context of the current MasterConfiguration
// NB. GetEtcdPodSpec methods holds the information about how kubeadm creates etcd static pod manifests.
func GetEtcdPodSpec(cfg *kubeadmapi.MasterConfiguration) v1.Pod {
	pathType := v1.HostPathDirectoryOrCreate
	etcdMounts := map[string]v1.Volume{
		etcdVolumeName:  staticpodutil.NewVolume(etcdVolumeName, cfg.Etcd.DataDir, &pathType),
		certsVolumeName: staticpodutil.NewVolume(certsVolumeName, cfg.CertificatesDir+"/etcd", &pathType),
	}
	return staticpodutil.ComponentPod(v1.Container{
		Name:            kubeadmconstants.Etcd,
		Command:         getEtcdCommand(cfg),
		Image:           images.GetCoreImage(kubeadmconstants.Etcd, cfg.ImageRepository, cfg.KubernetesVersion, cfg.Etcd.Image),
		ImagePullPolicy: cfg.ImagePullPolicy,
		// Mount the etcd datadir path read-write so etcd can store data in a more persistent manner
		VolumeMounts: []v1.VolumeMount{
			staticpodutil.NewVolumeMount(etcdVolumeName, cfg.Etcd.DataDir, false),
			staticpodutil.NewVolumeMount(certsVolumeName, cfg.CertificatesDir+"/etcd", false),
		},
		LivenessProbe: staticpodutil.EtcdProbe(
			cfg, kubeadmconstants.Etcd, 2379, cfg.CertificatesDir,
			kubeadmconstants.EtcdCACertName, kubeadmconstants.EtcdHealthcheckClientCertName, kubeadmconstants.EtcdHealthcheckClientKeyName,
		),
	}, etcdMounts)
}

// getEtcdCommand builds the right etcd command from the given config object
func getEtcdCommand(cfg *kubeadmapi.MasterConfiguration) []string {
	defaultArguments := map[string]string{
		"listen-client-urls":    "https://127.0.0.1:2379",
		"advertise-client-urls": "https://127.0.0.1:2379",
		"data-dir":              cfg.Etcd.DataDir,
		"cert-file":             filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdServerCertName),
		"key-file":              filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdServerKeyName),
		"trusted-ca-file":       filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdCACertName),
		"client-cert-auth":      "true",
		"peer-cert-file":        filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdPeerCertName),
		"peer-key-file":         filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdPeerKeyName),
		"peer-trusted-ca-file":  filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdCACertName),
		"peer-client-cert-auth": "true",
	}

	command := []string{"etcd"}
	command = append(command, kubeadmutil.BuildArgumentListFromMap(defaultArguments, cfg.Etcd.ExtraArgs)...)
	return command
}
