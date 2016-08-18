/*
Copyright 2016 The Kubernetes Authors.

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
package kubemaster

import (
	"fmt"
	"path"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
)

func createKubeProxyPodSpec(params *kubeadmapi.BootstrapParams) api.PodSpec {
	privilegedTrue := true
	return api.PodSpec{
		SecurityContext: &api.PodSecurityContext{HostNetwork: true},
		Containers: []api.Container{{
			Name:  "kube-proxy",
			Image: params.EnvParams["hyperkube_image"],
			Command: []string{
				"/hyperkube",
				"proxy",
				"--kubeconfig=/run/kubeconfig",
				COMPONENT_LOGLEVEL,
			},
			SecurityContext: &api.SecurityContext{Privileged: &privilegedTrue},
			VolumeMounts: []api.VolumeMount{
				{
					Name:      "dbus",
					MountPath: "/var/run/dbus",
					ReadOnly:  false,
				},
				{
					// TODO there are handful of clever options to get around this, but it's
					// easier to just mount kubelet's config here; we should probably just
					// make sure that proxy reads the token and CA cert from /run/secrets
					// and accepts `--master` at the same time
					//
					// clever options include:
					//  - do CSR dance and create kubeconfig and mount it as secrete
					//  - create a service account with a second secret enconding kubeconfig
					//  - use init container to convert known information to kubeconfig
					//  - ...whatever
					Name:      "kubeconfig",
					MountPath: "/run/kubeconfig",
					ReadOnly:  false,
				},
			},
		}},
		Volumes: []api.Volume{
			{
				Name: "kubeconfig",
				VolumeSource: api.VolumeSource{
					HostPath: &api.HostPathVolumeSource{Path: path.Join(params.EnvParams["prefix"], "kubelet.conf")},
				},
			},
			{
				Name: "dbus",
				VolumeSource: api.VolumeSource{
					HostPath: &api.HostPathVolumeSource{Path: "/var/run/dbus"},
				},
			},
		},
	}
}

func CreateEssentialAddons(params *kubeadmapi.BootstrapParams, client *clientset.Clientset) error {
	kubeProxyDaemonSet := NewDaemonSet("kube-proxy", createKubeProxyPodSpec(params))

	if _, err := client.Extensions().DaemonSets(api.NamespaceSystem).Create(kubeProxyDaemonSet); err != nil {
		return fmt.Errorf("<master/addons> failed creating essential kube-proxy addon [%s]", err)
	}

	fmt.Println("<master/addons> created essential addon: kube-proxy")

	// TODO should we wait for it to become ready at least on the master?

	return nil
}
