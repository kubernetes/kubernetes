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

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	api "k8s.io/kubernetes/pkg/api/v1"
	kubeadmapi "k8s.io/kubernetes/pkg/kubeadm/api"
	"k8s.io/kubernetes/pkg/kubeadm/images"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/intstr"
)

// Static pod definitions in golang form are included below so that `kubeadm
// init master` and `kubeadm manual bootstrap master` can get going.

const (
	DefaultClusterName = "--cluster-name=kubernetes"

	etcd                  = "etcd"
	apiServer             = "apiserver"
	controllerManager     = "controller-manager"
	scheduler             = "scheduler"
	proxy                 = "proxy"
	kubeAPIServer         = "kube-apiserver"
	kubeControllerManager = "kube-controller-manager"
	kubeScheduler         = "kube-scheduler"
	kubeProxy             = "kube-proxy"
)

// TODO look into what this really means, scheduler prints it for some reason
//
//E0817 17:53:22.242658       1 event.go:258] Could not construct reference to: '&api.Endpoints{TypeMeta:unversioned.TypeMeta{Kind:"", APIVersion:""}, ObjectMeta:api.ObjectMeta{Name:"kube-scheduler", GenerateName:"", Namespace:"kube-system", SelfLink:"", UID:"", ResourceVersion:"", Generation:0, CreationTimestamp:unversioned.Time{Time:time.Time{sec:0, nsec:0, loc:(*time.Location)(nil)}}, DeletionTimestamp:(*unversioned.Time)(nil), DeletionGracePeriodSeconds:(*int64)(nil), Labels:map[string]string(nil), Annotations:map[string]string(nil), OwnerReferences:[]api.OwnerReference(nil), Finalizers:[]string(nil)}, Subsets:[]api.EndpointSubset(nil)}' due to: 'selfLink was empty, can't make reference'. Will not report event: 'Normal' '%v became leader' 'moby'

func WriteStaticPodManifests(s *kubeadmapi.KubeadmConfig) error {
	staticPodSpecs := map[string]api.Pod{
		// TODO this needs a volume
		etcd: componentPod(api.Container{
			Command: []string{
				"/usr/local/bin/etcd",
				"--listen-client-urls=http://127.0.0.1:2379",
				"--advertise-client-urls=http://127.0.0.1:2379",
				"--data-dir=/var/etcd/data",
			},
			Image:         images.GetCoreImage(images.KubeEtcdImage, s.EnvParams["etcd_image"]),
			LivenessProbe: componentProbe(2379, "/health"),
			Name:          etcd,
			Resources:     componentResources("200m"),
		}),
		// TODO bind-mount certs in
		kubeAPIServer: componentPod(api.Container{
			Name:          kubeAPIServer,
			Image:         images.GetCoreImage(images.KubeApiServerImage, s.EnvParams["hyperkube_image"]),
			Command:       getComponentCommand(apiServer, s),
			VolumeMounts:  []api.VolumeMount{pkiVolumeMount()},
			LivenessProbe: componentProbe(8080, "/healthz"),
			Resources:     componentResources("250m"),
		}, pkiVolume(s)),
		kubeControllerManager: componentPod(api.Container{
			Name:          kubeControllerManager,
			Image:         images.GetCoreImage(images.KubeControllerManagerImage, s.EnvParams["hyperkube_image"]),
			Command:       getComponentCommand(controllerManager, s),
			VolumeMounts:  []api.VolumeMount{pkiVolumeMount()},
			LivenessProbe: componentProbe(10252, "/healthz"),
			Resources:     componentResources("200m"),
		}, pkiVolume(s)),
		kubeScheduler: componentPod(api.Container{
			Name:          kubeScheduler,
			Image:         images.GetCoreImage(images.KubeSchedulerImage, s.EnvParams["hyperkube_image"]),
			Command:       getComponentCommand(scheduler, s),
			LivenessProbe: componentProbe(10251, "/healthz"),
			Resources:     componentResources("100m"),
		}),
	}

	manifestsPath := path.Join(s.EnvParams["prefix"], "manifests")
	if err := os.MkdirAll(manifestsPath, 0700); err != nil {
		return fmt.Errorf("<master/manifests> failed to create directory %q [%s]", manifestsPath, err)
	}
	for name, spec := range staticPodSpecs {
		filename := path.Join(manifestsPath, name+".json")
		serialized, err := json.MarshalIndent(spec, "", "  ")
		if err != nil {
			return fmt.Errorf("<master/manifests> failed to marshall manifest for %q to JSON [%s]", name, err)
		}
		if err := cmdutil.DumpReaderToFile(bytes.NewReader(serialized), filename); err != nil {
			return fmt.Errorf("<master/manifests> failed to create static pod manifest file for %q (%q) [%s]", name, filename, err)
		}
	}
	return nil
}

func pkiVolume(s *kubeadmapi.KubeadmConfig) api.Volume {
	return api.Volume{
		Name: "pki",
		VolumeSource: api.VolumeSource{
			HostPath: &api.HostPathVolumeSource{Path: s.EnvParams["host_pki_path"]},
		},
	}
}

func pkiVolumeMount() api.VolumeMount {
	return api.VolumeMount{
		Name:      "pki",
		MountPath: "/etc/kubernetes/pki",
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

func getComponentCommand(component string, s *kubeadmapi.KubeadmConfig) (command []string) {
	baseFalgs := map[string][]string{
		apiServer: []string{
			"--address=127.0.0.1",
			"--etcd-servers=http://127.0.0.1:2379",
			"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,ResourceQuota",
			"--service-cluster-ip-range=" + s.InitFlags.Services.CIDR.String(),
			"--service-account-key-file=/etc/kubernetes/pki/apiserver-key.pem",
			"--client-ca-file=/etc/kubernetes/pki/ca.pem",
			"--tls-cert-file=/etc/kubernetes/pki/apiserver.pem",
			"--tls-private-key-file=/etc/kubernetes/pki/apiserver-key.pem",
			"--secure-port=443",
			"--allow-privileged",
			"--token-auth-file=/etc/kubernetes/pki/tokens.csv",
		},
		controllerManager: []string{
			"--leader-elect",
			"--master=127.0.0.1:8080",
			DefaultClusterName,
			"--root-ca-file=/etc/kubernetes/pki/ca.pem",
			"--service-account-private-key-file=/etc/kubernetes/pki/apiserver-key.pem",
			"--cluster-signing-cert-file=/etc/kubernetes/pki/ca.pem",
			"--cluster-signing-key-file=/etc/kubernetes/pki/ca-key.pem",
			"--insecure-experimental-approve-all-kubelet-csrs-for-group=system:kubelet-bootstrap",
			"--cluster-cidr=" + s.InitFlags.Services.CIDR.String(),
		},
		scheduler: []string{
			"--leader-elect",
			"--master=127.0.0.1:8080",
		},
		proxy: []string{},
	}

	if s.EnvParams["hyperkube_image"] != "" {
		command = []string{"/hyperkube", component}
	} else {
		command = []string{"/usr/local/bin/kube-" + component}
	}

	command = append(command, s.EnvParams["component_loglevel"])
	command = append(command, baseFalgs[component]...)

	if component == controllerManager && s.InitFlags.CloudProvider != "" {
		command = append(command, "--cloud-provider="+s.InitFlags.CloudProvider)
	}

	return
}
