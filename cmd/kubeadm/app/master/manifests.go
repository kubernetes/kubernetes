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
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path"
	"strings"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/api"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	api "k8s.io/kubernetes/pkg/api/v1"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/intstr"
)

// Static pod definitions in golang form are included below so that `kubeadm init` can get going.

const (
	DefaultClusterName     = "kubernetes"
	DefaultCloudConfigPath = "/etc/kubernetes/cloud-config.json"

	etcd                  = "etcd"
	apiServer             = "apiserver"
	controllerManager     = "controller-manager"
	scheduler             = "scheduler"
	proxy                 = "proxy"
	kubeAPIServer         = "kube-apiserver"
	kubeControllerManager = "kube-controller-manager"
	kubeScheduler         = "kube-scheduler"
	kubeProxy             = "kube-proxy"
	pkiDir                = "/etc/kubernetes/pki"
)

// WriteStaticPodManifests builds manifest objects based on user provided configuration and then dumps it to disk
// where kubelet will pick and schedule them.
func WriteStaticPodManifests(s *kubeadmapi.MasterConfiguration) error {
	envParams := kubeadmapi.GetEnvParams()
	// Prepare static pod specs
	staticPodSpecs := map[string]api.Pod{
		kubeAPIServer: componentPod(api.Container{
			Name:          kubeAPIServer,
			Image:         images.GetCoreImage(images.KubeAPIServerImage, s, envParams["hyperkube_image"]),
			Command:       getComponentCommand(apiServer, s),
			VolumeMounts:  []api.VolumeMount{certsVolumeMount(), k8sVolumeMount()},
			LivenessProbe: componentProbe(8080, "/healthz"),
			Resources:     componentResources("250m"),
		}, certsVolume(s), k8sVolume(s)),
		kubeControllerManager: componentPod(api.Container{
			Name:          kubeControllerManager,
			Image:         images.GetCoreImage(images.KubeControllerManagerImage, s, envParams["hyperkube_image"]),
			Command:       getComponentCommand(controllerManager, s),
			VolumeMounts:  []api.VolumeMount{certsVolumeMount(), k8sVolumeMount()},
			LivenessProbe: componentProbe(10252, "/healthz"),
			Resources:     componentResources("200m"),
		}, certsVolume(s), k8sVolume(s)),
		kubeScheduler: componentPod(api.Container{
			Name:          kubeScheduler,
			Image:         images.GetCoreImage(images.KubeSchedulerImage, s, envParams["hyperkube_image"]),
			Command:       getComponentCommand(scheduler, s),
			LivenessProbe: componentProbe(10251, "/healthz"),
			Resources:     componentResources("100m"),
		}),
	}

	// Add etcd static pod spec only if external etcd is not configured
	if len(s.Etcd.Endpoints) == 0 {
		staticPodSpecs[etcd] = componentPod(api.Container{
			Name: etcd,
			Command: []string{
				"etcd",
				"--listen-client-urls=http://127.0.0.1:2379",
				"--advertise-client-urls=http://127.0.0.1:2379",
				"--data-dir=/var/etcd/data",
			},
			VolumeMounts:  []api.VolumeMount{certsVolumeMount(), etcdVolumeMount(), k8sVolumeMount()},
			Image:         images.GetCoreImage(images.KubeEtcdImage, s, envParams["etcd_image"]),
			LivenessProbe: componentProbe(2379, "/health"),
			Resources:     componentResources("200m"),
			SecurityContext: &api.SecurityContext{
				SELinuxOptions: &api.SELinuxOptions{
					// TODO: This implies our etcd container is not being restricted by
					// SELinux. This is not optimal and would be nice to adjust in future
					// so it can create and write /var/lib/etcd, but for now this avoids
					// recommending setenforce 0 system-wide.
					Type: "unconfined_t",
				},
			},
		}, certsVolume(s), etcdVolume(s), k8sVolume(s))
	}

	manifestsPath := path.Join(envParams["kubernetes_dir"], "manifests")
	if err := os.MkdirAll(manifestsPath, 0700); err != nil {
		return fmt.Errorf("<master/manifests> failed to create directory %q [%v]", manifestsPath, err)
	}
	for name, spec := range staticPodSpecs {
		filename := path.Join(manifestsPath, name+".json")
		serialized, err := json.MarshalIndent(spec, "", "  ")
		if err != nil {
			return fmt.Errorf("<master/manifests> failed to marshall manifest for %q to JSON [%v]", name, err)
		}
		if err := cmdutil.DumpReaderToFile(bytes.NewReader(serialized), filename); err != nil {
			return fmt.Errorf("<master/manifests> failed to create static pod manifest file for %q (%q) [%v]", name, filename, err)
		}
	}
	return nil
}

// etcdVolume exposes a path on the host in order to guarantee data survival during reboot.
func etcdVolume(s *kubeadmapi.MasterConfiguration) api.Volume {
	envParams := kubeadmapi.GetEnvParams()
	return api.Volume{
		Name: "etcd",
		VolumeSource: api.VolumeSource{
			HostPath: &api.HostPathVolumeSource{Path: envParams["host_etcd_path"]},
		},
	}
}

func etcdVolumeMount() api.VolumeMount {
	return api.VolumeMount{
		Name:      "etcd",
		MountPath: "/var/etcd",
	}
}

// certsVolume exposes host SSL certificates to pod containers.
func certsVolume(s *kubeadmapi.MasterConfiguration) api.Volume {
	return api.Volume{
		Name: "certs",
		VolumeSource: api.VolumeSource{
			// TODO(phase1+) make path configurable
			HostPath: &api.HostPathVolumeSource{Path: "/etc/ssl/certs"},
		},
	}
}

func certsVolumeMount() api.VolumeMount {
	return api.VolumeMount{
		Name:      "certs",
		MountPath: "/etc/ssl/certs",
	}
}

func k8sVolume(s *kubeadmapi.MasterConfiguration) api.Volume {
	envParams := kubeadmapi.GetEnvParams()
	return api.Volume{
		Name: "pki",
		VolumeSource: api.VolumeSource{
			HostPath: &api.HostPathVolumeSource{Path: envParams["kubernetes_dir"]},
		},
	}
}

func k8sVolumeMount() api.VolumeMount {
	return api.VolumeMount{
		Name:      "pki",
		MountPath: "/etc/kubernetes/",
		ReadOnly:  true,
	}
}

func componentResources(cpu string) api.ResourceRequirements {
	return api.ResourceRequirements{
		Requests: api.ResourceList{
			api.ResourceName(api.ResourceCPU): resource.MustParse(cpu),
		},
	}
}

func componentProbe(port int, path string) *api.Probe {
	return &api.Probe{
		Handler: api.Handler{
			HTTPGet: &api.HTTPGetAction{
				Host: "127.0.0.1",
				Path: path,
				Port: intstr.FromInt(port),
			},
		},
		InitialDelaySeconds: 15,
		TimeoutSeconds:      15,
		FailureThreshold:    8,
	}
}

func componentPod(container api.Container, volumes ...api.Volume) api.Pod {
	return api.Pod{
		TypeMeta: unversioned.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: api.ObjectMeta{
			Name:      container.Name,
			Namespace: "kube-system",
			Labels:    map[string]string{"component": container.Name, "tier": "control-plane"},
		},
		Spec: api.PodSpec{
			Containers:  []api.Container{container},
			HostNetwork: true,
			Volumes:     volumes,
		},
	}
}

func getComponentCommand(component string, s *kubeadmapi.MasterConfiguration) (command []string) {
	baseFlags := map[string][]string{
		apiServer: {
			"--insecure-bind-address=127.0.0.1",
			"--etcd-servers=http://127.0.0.1:2379",
			"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,ResourceQuota",
			"--service-cluster-ip-range=" + s.Networking.ServiceSubnet,
			"--service-account-key-file=" + pkiDir + "/apiserver-key.pem",
			"--client-ca-file=" + pkiDir + "/ca.pem",
			"--tls-cert-file=" + pkiDir + "/apiserver.pem",
			"--tls-private-key-file=" + pkiDir + "/apiserver-key.pem",
			"--token-auth-file=" + pkiDir + "/tokens.csv",
			"--secure-port=443",
			"--allow-privileged",
		},
		controllerManager: {
			"--address=127.0.0.1",
			"--leader-elect",
			"--master=127.0.0.1:8080",
			"--cluster-name=" + DefaultClusterName,
			"--root-ca-file=" + pkiDir + "/ca.pem",
			"--service-account-private-key-file=" + pkiDir + "/apiserver-key.pem",
			"--cluster-signing-cert-file=" + pkiDir + "/ca.pem",
			"--cluster-signing-key-file=" + pkiDir + "/ca-key.pem",
			"--insecure-experimental-approve-all-kubelet-csrs-for-group=system:kubelet-bootstrap",
		},
		scheduler: {
			"--address=127.0.0.1",
			"--leader-elect",
			"--master=127.0.0.1:8080",
		},
		proxy: {},
	}

	envParams := kubeadmapi.GetEnvParams()
	if envParams["hyperkube_image"] != "" {
		command = []string{"/hyperkube", component}
	} else {
		command = []string{"/usr/local/bin/kube-" + component}
	}

	command = append(command, envParams["component_loglevel"])
	command = append(command, baseFlags[component]...)

	if component == apiServer {
		// Check if the user decided to use an external etcd cluster
		if len(s.Etcd.Endpoints) > 0 {
			command = append(command, fmt.Sprintf("--etcd-servers=%s", strings.Join(s.Etcd.Endpoints, ",")))
		} else {
			command = append(command, "--etcd-servers=http://127.0.0.1:2379")
		}

		// Is etcd secured?
		if s.Etcd.CAFile != "" {
			command = append(command, fmt.Sprintf("--etcd-cafile=%s", s.Etcd.CAFile))
		}
		if s.Etcd.CertFile != "" && s.Etcd.KeyFile != "" {
			etcdClientFileArg := fmt.Sprintf("--etcd-certfile=%s", s.Etcd.CertFile)
			etcdKeyFileArg := fmt.Sprintf("--etcd-keyfile=%s", s.Etcd.KeyFile)
			command = append(command, etcdClientFileArg, etcdKeyFileArg)
		}
	}

	if component == controllerManager {
		if s.CloudProvider != "" {
			command = append(command, "--cloud-provider="+s.CloudProvider)

			// Only append the --cloud-config option if there's a such file
			// TODO(phase1+) this won't work unless it's in one of the few directories we bind-mount
			if _, err := os.Stat(DefaultCloudConfigPath); err == nil {
				command = append(command, "--cloud-config="+DefaultCloudConfigPath)
			}
		}
		// Let the controller-manager allocate Node CIDRs for the Pod network.
		// Each node will get a subspace of the address CIDR provided with --pod-network-cidr.
		command = append(command, "--allocate-node-cidrs=true", "--cluster-cidr="+s.Networking.PodSubnet)
	}

	return
}
