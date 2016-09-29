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
	"runtime"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/api"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	ipallocator "k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/util/intstr"
)

// TODO(phase1+): kube-proxy should be a daemonset, three different daemonsets should not be here
func createKubeProxyPodSpec(s *kubeadmapi.MasterConfiguration, architecture string) api.PodSpec {
	envParams := kubeadmapi.GetEnvParams()
	privilegedTrue := true
	return api.PodSpec{
		SecurityContext: &api.PodSecurityContext{HostNetwork: true},
		NodeSelector: map[string]string{
			"beta.kubernetes.io/arch": architecture,
		},
		Containers: []api.Container{{
			Name:            kubeProxy,
			Image:           images.GetCoreImage(images.KubeProxyImage, s, envParams["hyperkube_image"]),
			Command:         append(getComponentCommand("proxy", s), "--kubeconfig=/run/kubeconfig"),
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
		Volumes: []api.Volume{
			{
				Name: "kubeconfig",
				VolumeSource: api.VolumeSource{
					HostPath: &api.HostPathVolumeSource{Path: path.Join(envParams["kubernetes_dir"], "kubelet.conf")},
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

func createKubeDNSPodSpec(s *kubeadmapi.MasterConfiguration) api.PodSpec {

	dnsPodResources := api.ResourceList{
		api.ResourceName(api.ResourceCPU):    resource.MustParse("100m"),
		api.ResourceName(api.ResourceMemory): resource.MustParse("170Mi"),
	}

	healthzPodResources := api.ResourceList{
		api.ResourceName(api.ResourceCPU):    resource.MustParse("10m"),
		api.ResourceName(api.ResourceMemory): resource.MustParse("50Mi"),
	}

	kubeDNSPort := int32(10053)
	dnsmasqPort := int32(53)

	nslookup := fmt.Sprintf("nslookup kubernetes.default.svc.%s 127.0.0.1", s.Networking.DNSDomain)

	nslookup = fmt.Sprintf("-cmd=%s:%d >/dev/null && %s:%d >/dev/null",
		nslookup, dnsmasqPort,
		nslookup, kubeDNSPort,
	)

	return api.PodSpec{
		NodeSelector: map[string]string{
			"beta.kubernetes.io/arch": runtime.GOARCH,
		},
		Containers: []api.Container{
			// DNS server
			{
				Name:  "kube-dns",
				Image: images.GetAddonImage(images.KubeDNSImage),
				Resources: api.ResourceRequirements{
					Limits:   dnsPodResources,
					Requests: dnsPodResources,
				},
				Args: []string{
					fmt.Sprintf("--domain=%s", s.Networking.DNSDomain),
					fmt.Sprintf("--dns-port=%d", kubeDNSPort),
					// TODO __PILLAR__FEDERATIONS__DOMAIN__MAP__
				},
				LivenessProbe: &api.Probe{
					Handler: api.Handler{
						HTTPGet: &api.HTTPGetAction{
							Path:   "/healthz",
							Port:   intstr.FromInt(8080),
							Scheme: api.URISchemeHTTP,
						},
					},
					InitialDelaySeconds: 60,
					TimeoutSeconds:      5,
					SuccessThreshold:    1,
					FailureThreshold:    1,
				},
				// # we poll on pod startup for the Kubernetes master service and
				// # only setup the /readiness HTTP server once that's available.
				ReadinessProbe: &api.Probe{
					Handler: api.Handler{
						HTTPGet: &api.HTTPGetAction{
							Path:   "/readiness",
							Port:   intstr.FromInt(8081),
							Scheme: api.URISchemeHTTP,
						},
					},
					InitialDelaySeconds: 30,
					TimeoutSeconds:      5,
				},
				Ports: []api.ContainerPort{
					{
						ContainerPort: kubeDNSPort,
						Name:          "dns-local",
						Protocol:      api.ProtocolUDP,
					},
					{
						ContainerPort: kubeDNSPort,
						Name:          "dns-tcp-local",
						Protocol:      api.ProtocolTCP,
					},
				},
			},
			// dnsmasq
			{
				Name:  "dnsmasq",
				Image: images.GetAddonImage(images.KubeDNSmasqImage),
				Resources: api.ResourceRequirements{
					Limits:   dnsPodResources,
					Requests: dnsPodResources,
				},
				Args: []string{
					"--cache-size=1000",
					"--no-resolv",
					fmt.Sprintf("--server=127.0.0.1#%d", kubeDNSPort),
				},
				Ports: []api.ContainerPort{
					{
						ContainerPort: dnsmasqPort,
						Name:          "dns",
						Protocol:      api.ProtocolUDP,
					},
					{
						ContainerPort: dnsmasqPort,
						Name:          "dns-tcp",
						Protocol:      api.ProtocolTCP,
					},
				},
			},
			// healthz
			{
				Name:  "healthz",
				Image: images.GetAddonImage(images.KubeExechealthzImage),
				Resources: api.ResourceRequirements{
					Limits:   healthzPodResources,
					Requests: healthzPodResources,
				},
				Args: []string{
					nslookup,
					"-port=8080",
					"-quiet",
				},
				Ports: []api.ContainerPort{{
					ContainerPort: 8080,
					Protocol:      api.ProtocolTCP,
				}},
			},
		},
		DNSPolicy: api.DNSDefault,
	}

}

func createKubeDNSServiceSpec(s *kubeadmapi.MasterConfiguration) (*api.ServiceSpec, error) {
	_, n, err := net.ParseCIDR(s.Networking.ServiceSubnet)
	if err != nil {
		return nil, fmt.Errorf("could not parse %q: %v", s.Networking.ServiceSubnet, err)
	}
	ip, err := ipallocator.GetIndexedIP(n, 10)
	if err != nil {
		return nil, fmt.Errorf("unable to allocate IP address for kube-dns addon from the given CIDR (%q) [%v]", s.Networking.ServiceSubnet, err)
	}

	svc := &api.ServiceSpec{
		Selector: map[string]string{"name": "kube-dns"},
		Ports: []api.ServicePort{
			{Name: "dns", Port: 53, Protocol: api.ProtocolUDP},
			{Name: "dns-tcp", Port: 53, Protocol: api.ProtocolTCP},
		},
		ClusterIP: ip.String(),
	}

	return svc, nil
}

func CreateEssentialAddons(s *kubeadmapi.MasterConfiguration, client *clientset.Clientset) error {
	arches := [3]string{"amd64", "arm", "arm64"}

	for _, arch := range arches {
		kubeProxyDaemonSet := NewDaemonSet(kubeProxy+"-"+arch, createKubeProxyPodSpec(s, arch))
		SetMasterTaintTolerations(&kubeProxyDaemonSet.Spec.Template.ObjectMeta)

		if _, err := client.Extensions().DaemonSets(api.NamespaceSystem).Create(kubeProxyDaemonSet); err != nil {
			return fmt.Errorf("<master/addons> failed creating essential kube-proxy addon [%v]", err)
		}
	}

	fmt.Println("<master/addons> created essential addon: kube-proxy")

	kubeDNSDeployment := NewDeployment("kube-dns", 1, createKubeDNSPodSpec(s))
	SetMasterTaintTolerations(&kubeDNSDeployment.Spec.Template.ObjectMeta)

	if _, err := client.Extensions().Deployments(api.NamespaceSystem).Create(kubeDNSDeployment); err != nil {
		return fmt.Errorf("<master/addons> failed creating essential kube-dns addon [%v]", err)
	}

	kubeDNSServiceSpec, err := createKubeDNSServiceSpec(s)
	if err != nil {
		return fmt.Errorf("<master/addons> failed creating essential kube-dns addon - %v", err)
	}

	kubeDNSService := NewService("kube-dns", *kubeDNSServiceSpec)
	if _, err := client.Services(api.NamespaceSystem).Create(kubeDNSService); err != nil {
		return fmt.Errorf("<master/addons> failed creating essential kube-dns addon [%v]", err)
	}

	fmt.Println("<master/addons> created essential addon: kube-dns")

	return nil
}
