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
	"k8s.io/api/core/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

const (
	etcdVolumeName = "etcd"
)

// CreateLocalEtcdStaticPodManifestFile will write local etcd static pod manifest file.
func CreateLocalEtcdStaticPodManifestFile(manifestDir string, cfg *kubeadmapi.MasterConfiguration) error {

	// gets etcd StaticPodSpec, actualized for the current MasterConfiguration
	spec := GetEtcdPodSpec(cfg)

	// writes etcd StaticPod to disk
	return staticpodutil.WriteStaticPodToDisk(kubeadmconstants.Etcd, manifestDir, spec)
}

// GetEtcdPodSpec returns the etcd static Pod actualized to the context of the current MasterConfiguration
// NB. GetEtcdPodSpec methods holds the information about how kubeadm creates etcd static pod mainfests.
func GetEtcdPodSpec(cfg *kubeadmapi.MasterConfiguration) v1.Pod {
	return staticpodutil.ComponentPod(v1.Container{
		Name:    kubeadmconstants.Etcd,
		Command: getEtcdCommand(cfg),
		Image:   images.GetCoreImage(kubeadmconstants.Etcd, cfg.ImageRepository, "", cfg.Etcd.Image),
		// Mount the etcd datadir path read-write so etcd can store data in a more persistent manner
		VolumeMounts:  []v1.VolumeMount{staticpodutil.NewVolumeMount(etcdVolumeName, cfg.Etcd.DataDir, false)},
		LivenessProbe: staticpodutil.ComponentProbe(2379, "/health", v1.URISchemeHTTP),
	}, []v1.Volume{staticpodutil.NewVolume(etcdVolumeName, cfg.Etcd.DataDir)})
}

// getEtcdCommand builds the right etcd command from the given config object
func getEtcdCommand(cfg *kubeadmapi.MasterConfiguration) []string {
	defaultArguments := map[string]string{
		"listen-client-urls":    "http://127.0.0.1:2379",
		"advertise-client-urls": "http://127.0.0.1:2379",
		"data-dir":              cfg.Etcd.DataDir,
	}

	command := []string{"etcd"}
	command = append(command, staticpodutil.GetExtraParameters(cfg.Etcd.ExtraArgs, defaultArguments)...)
	return command
}
