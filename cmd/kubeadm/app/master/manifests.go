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

// Static pod definitions in golang form are included below so that `kubeadm
// init master` and `kubeadm manual bootstrap master` can get going.

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
)

// TODO look into what this really means, scheduler prints it for some reason
//
//E0817 17:53:22.242658       1 event.go:258] Could not construct reference to: '&api.Endpoints{TypeMeta:unversioned.TypeMeta{Kind:"", APIVersion:""}, ObjectMeta:api.ObjectMeta{Name:"kube-scheduler", GenerateName:"", Namespace:"kube-system", SelfLink:"", UID:"", ResourceVersion:"", Generation:0, CreationTimestamp:unversioned.Time{Time:time.Time{sec:0, nsec:0, loc:(*time.Location)(nil)}}, DeletionTimestamp:(*unversioned.Time)(nil), DeletionGracePeriodSeconds:(*int64)(nil), Labels:map[string]string(nil), Annotations:map[string]string(nil), OwnerReferences:[]api.OwnerReference(nil), Finalizers:[]string(nil)}, Subsets:[]api.EndpointSubset(nil)}' due to: 'selfLink was empty, can't make reference'. Will not report event: 'Normal' '%v became leader' 'moby'

// WriteStaticPodManifests builds manifest objects based on user provided configuration and then dumps it to disk
// where kubelet will pick and schedule them.
func WriteStaticPodManifests(s *kubeadmapi.KubeadmConfig) error {
	// Placeholder for kube-apiserver pod spec command
	apiServerCommand := getComponentCommand(apiServer, s)

	// Check if the user decided to use an external etcd cluster
	if len(s.InitFlags.API.Etcd.ExternalEndpoints) > 0 {
		arg := fmt.Sprintf("--etcd-servers=%s", strings.Join(s.InitFlags.API.Etcd.ExternalEndpoints, ","))
		apiServerCommand = append(apiServerCommand, arg)
	} else {
		apiServerCommand = append(apiServerCommand, "--etcd-servers=http://127.0.0.1:2379")
	}

	// Is etcd secured?
	if s.InitFlags.API.Etcd.ExternalCAFile != "" {
		etcdCAFileArg := fmt.Sprintf("--etcd-cafile=%s", s.InitFlags.API.Etcd.ExternalCAFile)
		apiServerCommand = append(apiServerCommand, etcdCAFileArg)
	}
	if s.InitFlags.API.Etcd.ExternalCertFile != "" && s.InitFlags.API.Etcd.ExternalKeyFile != "" {
		etcdClientFileArg := fmt.Sprintf("--etcd-certfile=%s", s.InitFlags.API.Etcd.ExternalCertFile)
		etcdKeyFileArg := fmt.Sprintf("--etcd-keyfile=%s", s.InitFlags.API.Etcd.ExternalKeyFile)
		apiServerCommand = append(apiServerCommand, etcdClientFileArg, etcdKeyFileArg)
	}

	// Prepare static pod specs
	staticPodSpecs := map[string]api.Pod{
		kubeAPIServer: componentPod(api.Container{
			Name:          kubeAPIServer,
			Image:         images.GetCoreImage(images.KubeAPIServerImage, s.EnvParams["hyperkube_image"]),
			Command:       apiServerCommand,
			VolumeMounts:  []api.VolumeMount{certsVolumeMount(), k8sVolumeMount()},
			LivenessProbe: componentProbe(8080, "/healthz"),
			Resources:     componentResources("250m"),
		}, certsVolume(s), k8sVolume(s)),
		kubeControllerManager: componentPod(api.Container{
			Name:          kubeControllerManager,
			Image:         images.GetCoreImage(images.KubeControllerManagerImage, s.EnvParams["hyperkube_image"]),
			Command:       getComponentCommand(controllerManager, s),
			VolumeMounts:  []api.VolumeMount{k8sVolumeMount()},
			LivenessProbe: componentProbe(10252, "/healthz"),
			Resources:     componentResources("200m"),
		}, k8sVolume(s)),
		kubeScheduler: componentPod(api.Container{
			Name:          kubeScheduler,
			Image:         images.GetCoreImage(images.KubeSchedulerImage, s.EnvParams["hyperkube_image"]),
			Command:       getComponentCommand(scheduler, s),
			LivenessProbe: componentProbe(10251, "/healthz"),
			Resources:     componentResources("100m"),
		}),
	}

	// Add etcd static pod spec only if external etcd is not configured
	if len(s.InitFlags.API.Etcd.ExternalEndpoints) == 0 {
		staticPodSpecs[etcd] = componentPod(api.Container{
			Name: etcd,
			Command: []string{
				"etcd",
				"--listen-client-urls=http://127.0.0.1:2379",
				"--advertise-client-urls=http://127.0.0.1:2379",
				"--data-dir=/var/etcd/data",
			},
			VolumeMounts:  []api.VolumeMount{certsVolumeMount(), etcdVolumeMount(), k8sVolumeMount()},
			Image:         images.GetCoreImage(images.KubeEtcdImage, s.EnvParams["etcd_image"]),
			LivenessProbe: componentProbe(2379, "/health"),
			Resources:     componentResources("200m"),
		}, certsVolume(s), etcdVolume(s), k8sVolume(s))
	}

	manifestsPath := path.Join(s.EnvParams["kubernetes_dir"], "manifests")
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

// etcdVolume exposes a path on the host in order to guarantee data survival during reboot.
func etcdVolume(s *kubeadmapi.KubeadmConfig) api.Volume {
	return api.Volume{
		Name: "etcd",
		VolumeSource: api.VolumeSource{
			HostPath: &api.HostPathVolumeSource{Path: s.EnvParams["host_etcd_path"]},
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
func certsVolume(s *kubeadmapi.KubeadmConfig) api.Volume {
	return api.Volume{
		Name: "certs",
		VolumeSource: api.VolumeSource{
			// TODO make path configurable
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

func k8sVolume(s *kubeadmapi.KubeadmConfig) api.Volume {
	return api.Volume{
		Name: "pki",
		VolumeSource: api.VolumeSource{
			HostPath: &api.HostPathVolumeSource{Path: s.EnvParams["kubernetes_dir"]},
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
	// TODO: make a global constant of this
	pkiDir := "/etc/kubernetes/pki"

	baseFlags := map[string][]string{
		apiServer: []string{
			"--address=127.0.0.1",
			"--etcd-servers=http://127.0.0.1:2379",
			"--admission-control=NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,ResourceQuota",
			"--service-cluster-ip-range=" + s.InitFlags.Services.CIDR.String(),
			"--service-account-key-file=" + pkiDir + "/apiserver-key.pem",
			"--client-ca-file=" + pkiDir + "/ca.pem",
			"--tls-cert-file=" + pkiDir + "/apiserver.pem",
			"--tls-private-key-file=" + pkiDir + "/apiserver-key.pem",
			"--token-auth-file=" + pkiDir + "/tokens.csv",
			"--secure-port=443",
			"--allow-privileged",
		},
		controllerManager: []string{
			// TODO: consider adding --address=127.0.0.1 in order to not expose the cm port to the rest of the world
			"--leader-elect",
			"--master=127.0.0.1:8080",
			"--cluster-name=" + DefaultClusterName,
			"--root-ca-file=" + pkiDir + "/ca.pem",
			"--service-account-private-key-file=" + pkiDir + "/apiserver-key.pem",
			"--cluster-signing-cert-file=" + pkiDir + "/ca.pem",
			"--cluster-signing-key-file=" + pkiDir + "/ca-key.pem",
			"--insecure-experimental-approve-all-kubelet-csrs-for-group=system:kubelet-bootstrap",
			"--cluster-cidr=" + s.InitFlags.Services.CIDR.String(),
		},
		scheduler: []string{
			// TODO: consider adding --address=127.0.0.1 in order to not expose the scheduler port to the rest of the world
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
	command = append(command, baseFlags[component]...)

	if component == controllerManager && s.InitFlags.CloudProvider != "" {
		command = append(command, "--cloud-provider="+s.InitFlags.CloudProvider)

		// Only append the --cloud-config option if there's a such file
		if _, err := os.Stat(DefaultCloudConfigPath); err == nil {
			command = append(command, "--cloud-config="+DefaultCloudConfigPath)
		}
	}

	return
}
