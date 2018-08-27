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

	"github.com/golang/glog"

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
func CreateLocalEtcdStaticPodManifestFile(manifestDir string, cfg *kubeadmapi.InitConfiguration) error {
	glog.V(1).Infoln("creating local etcd static pod manifest file")
	// gets etcd StaticPodSpec, actualized for the current InitConfiguration
	spec := GetEtcdPodSpec(cfg)
	// writes etcd StaticPod to disk
	if err := staticpodutil.WriteStaticPodToDisk(kubeadmconstants.Etcd, manifestDir, spec); err != nil {
		return err
	}

	fmt.Printf("[etcd] Wrote Static Pod manifest for a local etcd instance to %q\n", kubeadmconstants.GetStaticPodFilepath(kubeadmconstants.Etcd, manifestDir))
	return nil
}

// GetEtcdPodSpec returns the etcd static Pod actualized to the context of the current InitConfiguration
// NB. GetEtcdPodSpec methods holds the information about how kubeadm creates etcd static pod manifests.
func GetEtcdPodSpec(cfg *kubeadmapi.InitConfiguration) v1.Pod {
	pathType := v1.HostPathDirectoryOrCreate
	etcdMounts := map[string]v1.Volume{
		etcdVolumeName:  staticpodutil.NewVolume(etcdVolumeName, cfg.Etcd.Local.DataDir, &pathType),
		certsVolumeName: staticpodutil.NewVolume(certsVolumeName, cfg.CertificatesDir+"/etcd", &pathType),
	}
	return staticpodutil.ComponentPod(v1.Container{
		Name:            kubeadmconstants.Etcd,
		Command:         getEtcdCommand(cfg),
		Image:           images.GetEtcdImage(&cfg.ClusterConfiguration),
		ImagePullPolicy: v1.PullIfNotPresent,
		// Mount the etcd datadir path read-write so etcd can store data in a more persistent manner
		VolumeMounts: []v1.VolumeMount{
			staticpodutil.NewVolumeMount(etcdVolumeName, cfg.Etcd.Local.DataDir, false),
			staticpodutil.NewVolumeMount(certsVolumeName, cfg.CertificatesDir+"/etcd", false),
		},
		LivenessProbe: staticpodutil.EtcdProbe(
			cfg, kubeadmconstants.Etcd, 2379, cfg.CertificatesDir,
			kubeadmconstants.EtcdCACertName, kubeadmconstants.EtcdHealthcheckClientCertName, kubeadmconstants.EtcdHealthcheckClientKeyName,
		),
	}, etcdMounts)
}

// getEtcdCommand builds the right etcd command from the given config object
func getEtcdCommand(cfg *kubeadmapi.InitConfiguration) []string {
	defaultArguments := map[string]string{
		"name":                        cfg.GetNodeName(),
		"listen-client-urls":          "https://127.0.0.1:2379",
		"advertise-client-urls":       "https://127.0.0.1:2379",
		"listen-peer-urls":            "https://127.0.0.1:2380",
		"initial-advertise-peer-urls": "https://127.0.0.1:2380",
		"data-dir":                    cfg.Etcd.Local.DataDir,
		"cert-file":                   filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdServerCertName),
		"key-file":                    filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdServerKeyName),
		"trusted-ca-file":             filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdCACertName),
		"client-cert-auth":            "true",
		"peer-cert-file":              filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdPeerCertName),
		"peer-key-file":               filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdPeerKeyName),
		"peer-trusted-ca-file":        filepath.Join(cfg.CertificatesDir, kubeadmconstants.EtcdCACertName),
		"peer-client-cert-auth":       "true",
		"snapshot-count":              "10000",
		"initial-cluster":             fmt.Sprintf("%s=https://127.0.0.1:2380", cfg.GetNodeName()),
	}

	command := []string{"etcd"}
	command = append(command, kubeadmutil.BuildArgumentListFromMap(defaultArguments, cfg.Etcd.Local.ExtraArgs)...)
	return command
}
