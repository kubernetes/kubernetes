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

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/net"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
	"k8s.io/kubernetes/pkg/util/node"
)

const (
	etcdVolumeName = "etcd"
)

// CreateLocalEtcdStaticPodManifestFile will write local etcd static pod manifest file.
func CreateLocalEtcdStaticPodManifestFile(manifestDir string, cfg *kubeadmapi.MasterConfiguration) error {

	// gets etcd StaticPodSpec, actualized for the current MasterConfiguration
	spec, err := GetEtcdPodSpec(cfg)
	if err != nil {
		return err
	}

	// writes etcd StaticPod to disk
	if err := staticpodutil.WriteStaticPodToDisk(kubeadmconstants.Etcd, manifestDir, spec); err != nil {
		return err
	}

	fmt.Printf("[etcd] Wrote Static Pod manifest for a local etcd instance to %q\n", kubeadmconstants.GetStaticPodFilepath(kubeadmconstants.Etcd, manifestDir))
	return nil
}

// GetEtcdPodSpec returns the etcd static Pod actualized to the context of the current MasterConfiguration
// NB. GetEtcdPodSpec methods holds the information about how kubeadm creates etcd static pod mainfests.
func GetEtcdPodSpec(cfg *kubeadmapi.MasterConfiguration) (v1.Pod, error) {
	pathType := v1.HostPathDirectoryOrCreate
	etcdCommand, err := getEtcdCommand(cfg)
	if err != nil {
		return v1.Pod{}, err
	}
	return staticpodutil.ComponentPod(v1.Container{
		Name:    kubeadmconstants.Etcd,
		Command: etcdCommand,
		Image:   images.GetCoreImage(kubeadmconstants.Etcd, cfg.ImageRepository, "", cfg.Etcd.Image),
		// Mount the etcd datadir path read-write so etcd can store data in a more persistent manner
		VolumeMounts:  []v1.VolumeMount{
			staticpodutil.NewVolumeMount(etcdVolumeName, cfg.Etcd.DataDir, false),
			certsVolumeMount(),
			k8sVolumeMount(),
		},
		LivenessProbe: staticpodutil.ComponentProbe(2379, "/health", v1.URISchemeHTTP),
	}, []v1.Volume{
		staticpodutil.NewVolume(etcdVolumeName, cfg.Etcd.DataDir, &pathType),
		certsVolume(cfg),
		k8sVolume(),
	}), nil
}

// getEtcdCommand builds the right etcd command from the given config object
func getEtcdCommand(cfg *kubeadmapi.MasterConfiguration) ([]string, error) {
	var defaultArguments map[string]string
	if len(cfg.Etcd.Discovery) > 1 {
		//Use etcd discovery for multi master
		name := node.GetHostname(cfg.NodeName)
		ip, err := net.ChooseHostInterface()
		if err != nil {
			return nil, fmt.Errorf("failed to get host interface address for etcd [%v]", err)
		}
		defaultArguments = map[string]string{
			"name": name,
			"initial-advertise-peer-urls": fmt.Sprintf("http://%v:2380", ip.String()),
			"listen-peer-urls":            fmt.Sprintf("http://%v:2380", ip.String()),
			"listen-client-urls":          fmt.Sprintf("http://%v:2379,http://127.0.0.1:2379", ip.String()),
			"advertise-client-urls":       fmt.Sprintf("http://%v:2379", ip.String()),
			"discovery":                   cfg.Etcd.Discovery,
			"data-dir":                    cfg.Etcd.DataDir,
		}
	} else {
		defaultArguments = map[string]string{
			"listen-client-urls":    "http://127.0.0.1:2379",
			"advertise-client-urls": "http://127.0.0.1:2379",
			"data-dir":              cfg.Etcd.DataDir,
		}
	}

	command := []string{"etcd"}
	command = append(command, kubeadmutil.BuildArgumentListFromMap(defaultArguments, cfg.Etcd.ExtraArgs)...)
	return command, nil
}

// certsVolume exposes host SSL certificates to pod containers.
// TODO(phase1+) make path configurable
func certsVolume(cfg *kubeadmapi.MasterConfiguration) v1.Volume {
	return v1.Volume{
		Name: "certs",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{Path: "/etc/ssl/certs"},
		},
	}
}

func certsVolumeMount() v1.VolumeMount {
	return v1.VolumeMount{
		Name:      "certs",
		MountPath: "/etc/ssl/certs",
	}
}

func k8sVolume() v1.Volume {
	return v1.Volume{
		Name: "k8s",
		VolumeSource: v1.VolumeSource{
			HostPath: &v1.HostPathVolumeSource{Path: kubeadmconstants.KubernetesDir},
		},
	}
}

func k8sVolumeMount() v1.VolumeMount {
	return v1.VolumeMount{
		Name:      "k8s",
		MountPath: kubeadmconstants.KubernetesDir,
		ReadOnly:  true,
	}
}
