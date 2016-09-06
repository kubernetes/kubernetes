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
	"runtime"
	"strconv"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
	"k8s.io/kubernetes/pkg/util/intstr"
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
				params.EnvParams["component_loglevel"],
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

func createKubeDNSPodSpec(params *kubeadmapi.BootstrapParams) api.PodSpec {

	dnsPodResources := api.ResourceList{
		api.ResourceName(api.ResourceCPU):    resource.MustParse("100m"),
		api.ResourceName(api.ResourceMemory): resource.MustParse("170Mi"),
	}

	healthzPodResources := api.ResourceList{
		api.ResourceName(api.ResourceCPU):    resource.MustParse("10m"),
		api.ResourceName(api.ResourceMemory): resource.MustParse("50Mi"),
	}

	return api.PodSpec{
		Containers: []api.Container{
			// DNS server
			{
				Name:  "kube-dns",
				Image: "gcr.io/google_containers/kubedns-" + runtime.GOARCH + ":1.7",
				Resources: api.ResourceRequirements{
					Limits:   dnsPodResources,
					Requests: dnsPodResources,
				},
				Args: []string{
					"--domain=" + params.EnvParams["dns_domain"],
					"--dns-port=10053",
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
						ContainerPort: 10053,
						Name:          "dns-local",
						Protocol:      api.ProtocolUDP,
					},
					{
						ContainerPort: 10053,
						Name:          "dns-tcp-local",
						Protocol:      api.ProtocolTCP,
					},
				},
			},
			// dnsmasq
			{
				Name:  "dnsmasq",
				Image: "gcr.io/google_containers/kube-dnsmasq-" + runtime.GOARCH + ":1.3",
				Resources: api.ResourceRequirements{
					Limits:   dnsPodResources,
					Requests: dnsPodResources,
				},
				Args: []string{
					"--cache-size=1000",
					"--no-resolv",
					"--server=127.0.0.1#10053",
				},
				Ports: []api.ContainerPort{
					{
						ContainerPort: 53,
						Name:          "dns",
						Protocol:      api.ProtocolUDP,
					},
					{
						ContainerPort: 53,
						Name:          "dns-tcp",
						Protocol:      api.ProtocolTCP,
					},
				},
			},
			// healthz
			{
				Name:  "healthz",
				Image: "gcr.io/google_containers/exechealthz-" + runtime.GOARCH + ":1.1",
				Resources: api.ResourceRequirements{
					Limits:   healthzPodResources,
					Requests: healthzPodResources,
				},
				Args: []string{
					"-cmd=nslookup kubernetes.default.svc." + params.EnvParams["dns_domain"] + " 127.0.0.1 >/dev/null && nslookup kubernetes.default.svc." + params.EnvParams["dns_domain"] + " 127.0.0.1:10053 >/dev/null",
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
func createKubeDNSServiceSpec(params *kubeadmapi.BootstrapParams) api.ServiceSpec {
	return api.ServiceSpec{
		Selector: map[string]string{"name": "kube-dns"},
		Ports: []api.ServicePort{
			{Name: "dns", Port: 53, Protocol: api.ProtocolUDP},
			{Name: "dns-tcp", Port: 53, Protocol: api.ProtocolTCP},
		},
		ClusterIP: "100.64.0.2",
	}
}

func CreateEssentialAddons(params *kubeadmapi.BootstrapParams, client *clientset.Clientset) error {
	kubeProxyDaemonSet := NewDaemonSet("kube-proxy", createKubeProxyPodSpec(params))
	SetMasterTaintTolerations(&kubeProxyDaemonSet.Spec.Template.ObjectMeta)

	if _, err := client.Extensions().DaemonSets(api.NamespaceSystem).Create(kubeProxyDaemonSet); err != nil {
		return fmt.Errorf("<master/addons> failed creating essential kube-proxy addon [%s]", err)
	}

	fmt.Println("<master/addons> created essential addon: kube-proxy")

	// TODO should we wait for it to become ready at least on the master?

	dnsReplicas, err := strconv.Atoi(params.EnvParams["dns_replicas"])
	if err != nil {
		dnsReplicas = 1
	}

	kubeDNSDeployment := NewDeployment("kube-dns", int32(dnsReplicas), createKubeDNSPodSpec(params))
	SetMasterTaintTolerations(&kubeDNSDeployment.Spec.Template.ObjectMeta)

	if _, err := client.Extensions().Deployments(api.NamespaceSystem).Create(kubeDNSDeployment); err != nil {
		return fmt.Errorf("<master/addons> failed creating essential kube-dns addon [%s]", err)
	}

	kubeDNSService := NewService("kube-dns", createKubeDNSServiceSpec(params))

	if _, err := client.Services(api.NamespaceSystem).Create(kubeDNSService); err != nil {
		return fmt.Errorf("<master/addons> failed creating essential kube-dns addon [%s]", err)
	}

	fmt.Println("<master/addons> created essential addon: kube-dns")

	return nil
}
