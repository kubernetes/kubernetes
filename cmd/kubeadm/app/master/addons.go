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

package master

import (
	"fmt"
	"net"
	"path"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func createKubeProxyPodSpec(cfg *kubeadmapi.MasterConfiguration) v1.PodSpec {
	privilegedTrue := true
	return v1.PodSpec{
		HostNetwork:     true,
		SecurityContext: &v1.PodSecurityContext{},
		Containers: []v1.Container{{
			Name:            kubeProxy,
			Image:           images.GetCoreImage(images.KubeProxyImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
			Command:         append(getProxyCommand(cfg), "--kubeconfig=/run/kubeconfig"),
			SecurityContext: &v1.SecurityContext{Privileged: &privilegedTrue},
			VolumeMounts: []v1.VolumeMount{
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
					//  - do CSR dance and create kubeconfig and mount it as a secret
					//  - create a service account with a second secret encoding kubeconfig
					//  - use init container to convert known information to kubeconfig
					//  - ...whatever
					Name:      "kubeconfig",
					MountPath: "/run/kubeconfig",
					ReadOnly:  false,
				},
			},
		}},
		Volumes: []v1.Volume{
			{
				Name: "kubeconfig",
				VolumeSource: v1.VolumeSource{
					HostPath: &v1.HostPathVolumeSource{Path: path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, "kubelet.conf")},
				},
			},
			{
				Name: "dbus",
				VolumeSource: v1.VolumeSource{
					HostPath: &v1.HostPathVolumeSource{Path: "/var/run/dbus"},
				},
			},
		},
	}
}

func createKubeDNSPodSpec(cfg *kubeadmapi.MasterConfiguration) v1.PodSpec {
	kubeDNSPort := int32(10053)
	dnsmasqPort := int32(53)

	return v1.PodSpec{
		Containers: []v1.Container{
			// DNS server
			{
				Name:  "kubedns",
				Image: images.GetAddonImage(images.KubeDNSImage),
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceName(v1.ResourceMemory): resource.MustParse("170Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceName(v1.ResourceCPU):    resource.MustParse("100m"),
						v1.ResourceName(v1.ResourceMemory): resource.MustParse("70Mi"),
					},
				},
				LivenessProbe: &v1.Probe{
					Handler: v1.Handler{
						HTTPGet: &v1.HTTPGetAction{
							Path:   "/healthcheck/kubedns",
							Port:   intstr.FromInt(10054),
							Scheme: v1.URISchemeHTTP,
						},
					},
					InitialDelaySeconds: 60,
					TimeoutSeconds:      5,
					SuccessThreshold:    1,
					FailureThreshold:    5,
				},
				// # we poll on pod startup for the Kubernetes master service and
				// # only setup the /readiness HTTP server once that's available.
				ReadinessProbe: &v1.Probe{
					Handler: v1.Handler{
						HTTPGet: &v1.HTTPGetAction{
							Path:   "/readiness",
							Port:   intstr.FromInt(8081),
							Scheme: v1.URISchemeHTTP,
						},
					},
					InitialDelaySeconds: 3,
					TimeoutSeconds:      5,
				},
				Args: []string{
					fmt.Sprintf("--domain=%s", cfg.Networking.DNSDomain),
					fmt.Sprintf("--dns-port=%d", kubeDNSPort),
					"--config-map=kube-dns",
					"--v=2",
				},
				Env: []v1.EnvVar{
					{
						Name:  "PROMETHEUS_PORT",
						Value: "10055",
					},
				},
				Ports: []v1.ContainerPort{
					{
						ContainerPort: kubeDNSPort,
						Name:          "dns-local",
						Protocol:      v1.ProtocolUDP,
					},
					{
						ContainerPort: kubeDNSPort,
						Name:          "dns-tcp-local",
						Protocol:      v1.ProtocolTCP,
					},
					{
						ContainerPort: 10055,
						Name:          "metrics",
						Protocol:      v1.ProtocolTCP,
					},
				},
			},
			// dnsmasq
			{
				Name:  "dnsmasq",
				Image: images.GetAddonImage(images.KubeDNSmasqImage),
				LivenessProbe: &v1.Probe{
					Handler: v1.Handler{
						HTTPGet: &v1.HTTPGetAction{
							Path:   "/healthcheck/dnsmasq",
							Port:   intstr.FromInt(10054),
							Scheme: v1.URISchemeHTTP,
						},
					},
					InitialDelaySeconds: 60,
					TimeoutSeconds:      5,
					SuccessThreshold:    1,
					FailureThreshold:    5,
				},
				Args: []string{
					"--cache-size=1000",
					"--no-resolv",
					fmt.Sprintf("--server=127.0.0.1#%d", kubeDNSPort),
					"--log-facility=-",
				},
				Ports: []v1.ContainerPort{
					{
						ContainerPort: dnsmasqPort,
						Name:          "dns",
						Protocol:      v1.ProtocolUDP,
					},
					{
						ContainerPort: dnsmasqPort,
						Name:          "dns-tcp",
						Protocol:      v1.ProtocolTCP,
					},
				},
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceName(v1.ResourceCPU):    resource.MustParse("150m"),
						v1.ResourceName(v1.ResourceMemory): resource.MustParse("10Mi"),
					},
				},
			},
			{
				Name:  "sidecar",
				Image: images.GetAddonImage(images.KubeDNSSidecarImage),
				LivenessProbe: &v1.Probe{
					Handler: v1.Handler{
						HTTPGet: &v1.HTTPGetAction{
							Path:   "/metrics",
							Port:   intstr.FromInt(10054),
							Scheme: v1.URISchemeHTTP,
						},
					},
					InitialDelaySeconds: 60,
					TimeoutSeconds:      5,
					SuccessThreshold:    1,
					FailureThreshold:    5,
				},
				Args: []string{
					"--v=2",
					"--logtostderr",
					fmt.Sprintf("--probe=kubedns,127.0.0.1:10053,kubernetes.default.svc.%s,5,A", cfg.Networking.DNSDomain),
					fmt.Sprintf("--probe=dnsmasq,127.0.0.1:53,kubernetes.default.svc.%s,5,A", cfg.Networking.DNSDomain),
				},
				Ports: []v1.ContainerPort{
					{
						ContainerPort: 10054,
						Name:          "metrics",
						Protocol:      v1.ProtocolTCP,
					},
				},
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceName(v1.ResourceMemory): resource.MustParse("20Mi"),
						v1.ResourceName(v1.ResourceCPU):    resource.MustParse("10m"),
					},
				},
			},
		},
		DNSPolicy: v1.DNSDefault,
	}
}

func createKubeDNSServiceSpec(cfg *kubeadmapi.MasterConfiguration) (*v1.ServiceSpec, error) {
	_, n, err := net.ParseCIDR(cfg.Networking.ServiceSubnet)
	if err != nil {
		return nil, fmt.Errorf("could not parse %q: %v", cfg.Networking.ServiceSubnet, err)
	}
	ip, err := ipallocator.GetIndexedIP(n, 10)
	if err != nil {
		return nil, fmt.Errorf("unable to allocate IP address for kube-dns addon from the given CIDR %q: [%v]", cfg.Networking.ServiceSubnet, err)
	}

	return &v1.ServiceSpec{
		Selector: map[string]string{"name": "kube-dns"},
		Ports: []v1.ServicePort{
			{Name: "dns", Port: 53, Protocol: v1.ProtocolUDP},
			{Name: "dns-tcp", Port: 53, Protocol: v1.ProtocolTCP},
		},
		ClusterIP: ip.String(),
	}, nil
}

func CreateEssentialAddons(cfg *kubeadmapi.MasterConfiguration, client *clientset.Clientset) error {
	kubeProxyDaemonSet := NewDaemonSet(kubeProxy, createKubeProxyPodSpec(cfg))
	SetMasterTaintTolerations(&kubeProxyDaemonSet.Spec.Template.ObjectMeta)
	SetNodeAffinity(&kubeProxyDaemonSet.Spec.Template.ObjectMeta, NativeArchitectureNodeAffinity())

	if _, err := client.Extensions().DaemonSets(api.NamespaceSystem).Create(kubeProxyDaemonSet); err != nil {
		return fmt.Errorf("failed creating essential kube-proxy addon [%v]", err)
	}

	fmt.Println("[addons] Created essential addon: kube-proxy")

	kubeDNSDeployment := NewDeployment("kube-dns", 1, createKubeDNSPodSpec(cfg))
	SetMasterTaintTolerations(&kubeDNSDeployment.Spec.Template.ObjectMeta)
	SetNodeAffinity(&kubeDNSDeployment.Spec.Template.ObjectMeta, NativeArchitectureNodeAffinity())

	if _, err := client.Extensions().Deployments(api.NamespaceSystem).Create(kubeDNSDeployment); err != nil {
		return fmt.Errorf("failed creating essential kube-dns addon [%v]", err)
	}

	kubeDNSServiceSpec, err := createKubeDNSServiceSpec(cfg)
	if err != nil {
		return fmt.Errorf("failed creating essential kube-dns addon [%v]", err)
	}

	kubeDNSService := NewService("kube-dns", *kubeDNSServiceSpec)
	kubeDNSService.ObjectMeta.Labels["kubernetes.io/name"] = "KubeDNS"
	if _, err := client.Services(api.NamespaceSystem).Create(kubeDNSService); err != nil {
		return fmt.Errorf("failed creating essential kube-dns addon [%v]", err)
	}

	fmt.Println("[addons] Created essential addon: kube-dns")

	return nil
}
