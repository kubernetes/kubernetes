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
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/ghodss/yaml"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/client-go/pkg/api/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/images"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
	authzmodes "k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/util/version"
)

// Static pod definitions in golang form are included below so that `kubeadm init` can get going.
const (
	DefaultCloudConfigPath = "/etc/kubernetes/cloud-config"

	defaultv16AdmissionControl = "NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,ResourceQuota"
	defaultv17AdmissionControl = "Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota"

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

// WriteStaticPodManifests builds manifest objects based on user provided configuration and then dumps it to disk
// where kubelet will pick and schedule them.
func WriteStaticPodManifests(cfg *kubeadmapi.MasterConfiguration) error {
	volumes := []api.Volume{k8sVolume()}
	volumeMounts := []api.VolumeMount{k8sVolumeMount()}

	if isCertsVolumeMountNeeded() {
		volumes = append(volumes, certsVolume(cfg))
		volumeMounts = append(volumeMounts, certsVolumeMount())
	}

	if isPkiVolumeMountNeeded() {
		volumes = append(volumes, pkiVolume())
		volumeMounts = append(volumeMounts, pkiVolumeMount())
	}

	if !strings.HasPrefix(cfg.CertificatesDir, kubeadmapiext.DefaultCertificatesDir) {
		volumes = append(volumes, newVolume("certdir", cfg.CertificatesDir))
		volumeMounts = append(volumeMounts, newVolumeMount("certdir", cfg.CertificatesDir))
	}

	k8sVersion, err := version.ParseSemantic(cfg.KubernetesVersion)
	if err != nil {
		return err
	}

	// Prepare static pod specs
	staticPodSpecs := map[string]api.Pod{
		kubeAPIServer: componentPod(api.Container{
			Name:          kubeAPIServer,
			Image:         images.GetCoreImage(images.KubeAPIServerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
			Command:       getAPIServerCommand(cfg, false, k8sVersion),
			VolumeMounts:  volumeMounts,
			LivenessProbe: componentProbe(int(cfg.API.BindPort), "/healthz", api.URISchemeHTTPS),
			Resources:     componentResources("250m"),
			Env:           getProxyEnvVars(),
		}, volumes...),
		kubeControllerManager: componentPod(api.Container{
			Name:          kubeControllerManager,
			Image:         images.GetCoreImage(images.KubeControllerManagerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
			Command:       getControllerManagerCommand(cfg, false, k8sVersion),
			VolumeMounts:  volumeMounts,
			LivenessProbe: componentProbe(10252, "/healthz", api.URISchemeHTTP),
			Resources:     componentResources("200m"),
			Env:           getProxyEnvVars(),
		}, volumes...),
		kubeScheduler: componentPod(api.Container{
			Name:          kubeScheduler,
			Image:         images.GetCoreImage(images.KubeSchedulerImage, cfg, kubeadmapi.GlobalEnvParams.HyperkubeImage),
			Command:       getSchedulerCommand(cfg, false),
			VolumeMounts:  []api.VolumeMount{k8sVolumeMount()},
			LivenessProbe: componentProbe(10251, "/healthz", api.URISchemeHTTP),
			Resources:     componentResources("100m"),
			Env:           getProxyEnvVars(),
		}, k8sVolume()),
	}

	// Add etcd static pod spec only if external etcd is not configured
	if len(cfg.Etcd.Endpoints) == 0 {
		etcdPod := componentPod(api.Container{
			Name:          etcd,
			Command:       getEtcdCommand(cfg),
			VolumeMounts:  []api.VolumeMount{certsVolumeMount(), etcdVolumeMount(cfg.Etcd.DataDir), k8sVolumeMount()},
			Image:         images.GetCoreImage(images.KubeEtcdImage, cfg, kubeadmapi.GlobalEnvParams.EtcdImage),
			LivenessProbe: componentProbe(2379, "/health", api.URISchemeHTTP),
		}, certsVolume(cfg), etcdVolume(cfg), k8sVolume())

		etcdPod.Spec.SecurityContext = &api.PodSecurityContext{
			SELinuxOptions: &api.SELinuxOptions{
				// Unconfine the etcd container so it can write to the data dir with SELinux enforcing:
				Type: "spc_t",
			},
		}

		staticPodSpecs[etcd] = etcdPod
	}

	manifestsPath := filepath.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.ManifestsSubDirName)
	if err := os.MkdirAll(manifestsPath, 0700); err != nil {
		return fmt.Errorf("failed to create directory %q [%v]", manifestsPath, err)
	}
	for name, spec := range staticPodSpecs {
		filename := filepath.Join(manifestsPath, name+".yaml")
		serialized, err := yaml.Marshal(spec)
		if err != nil {
			return fmt.Errorf("failed to marshal manifest for %q to YAML [%v]", name, err)
		}
		if err := cmdutil.DumpReaderToFile(bytes.NewReader(serialized), filename); err != nil {
			return fmt.Errorf("failed to create static pod manifest file for %q (%q) [%v]", name, filename, err)
		}
	}
	return nil
}

func newVolume(name, path string) api.Volume {
	return api.Volume{
		Name: name,
		VolumeSource: api.VolumeSource{
			HostPath: &api.HostPathVolumeSource{Path: path},
		},
	}
}

func newVolumeMount(name, path string) api.VolumeMount {
	return api.VolumeMount{
		Name:      name,
		MountPath: path,
	}
}

// etcdVolume exposes a path on the host in order to guarantee data survival during reboot.
func etcdVolume(cfg *kubeadmapi.MasterConfiguration) api.Volume {
	return api.Volume{
		Name: "etcd",
		VolumeSource: api.VolumeSource{
			HostPath: &api.HostPathVolumeSource{Path: cfg.Etcd.DataDir},
		},
	}
}

func etcdVolumeMount(dataDir string) api.VolumeMount {
	return api.VolumeMount{
		Name:      "etcd",
		MountPath: dataDir,
	}
}

func isCertsVolumeMountNeeded() bool {
	// Always return true for now. We may add conditional logic here for images which do not require host mounting /etc/ssl
	// hyperkube for example already has valid ca-certificates installed
	return true
}

// certsVolume exposes host SSL certificates to pod containers.
func certsVolume(cfg *kubeadmapi.MasterConfiguration) api.Volume {
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

func isPkiVolumeMountNeeded() bool {
	// On some systems were we host-mount /etc/ssl/certs, it is also required to mount /etc/pki. This is needed
	// due to symlinks pointing from files in /etc/ssl/certs into /etc/pki/
	if _, err := os.Stat("/etc/pki"); err == nil {
		return true
	}
	return false
}

func pkiVolume() api.Volume {
	return api.Volume{
		Name: "pki",
		VolumeSource: api.VolumeSource{
			// TODO(phase1+) make path configurable
			HostPath: &api.HostPathVolumeSource{Path: "/etc/pki"},
		},
	}
}

func pkiVolumeMount() api.VolumeMount {
	return api.VolumeMount{
		Name:      "pki",
		MountPath: "/etc/pki",
	}
}

func flockVolume() api.Volume {
	return api.Volume{
		Name: "var-lock",
		VolumeSource: api.VolumeSource{
			HostPath: &api.HostPathVolumeSource{Path: "/var/lock"},
		},
	}
}

func flockVolumeMount() api.VolumeMount {
	return api.VolumeMount{
		Name:      "var-lock",
		MountPath: "/var/lock",
		ReadOnly:  false,
	}
}

func k8sVolume() api.Volume {
	return api.Volume{
		Name: "k8s",
		VolumeSource: api.VolumeSource{
			HostPath: &api.HostPathVolumeSource{Path: kubeadmapi.GlobalEnvParams.KubernetesDir},
		},
	}
}

func k8sVolumeMount() api.VolumeMount {
	return api.VolumeMount{
		Name:      "k8s",
		MountPath: kubeadmapi.GlobalEnvParams.KubernetesDir,
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

func componentProbe(port int, path string, scheme api.URIScheme) *api.Probe {
	return &api.Probe{
		Handler: api.Handler{
			HTTPGet: &api.HTTPGetAction{
				Host:   "127.0.0.1",
				Path:   path,
				Port:   intstr.FromInt(port),
				Scheme: scheme,
			},
		},
		InitialDelaySeconds: 15,
		TimeoutSeconds:      15,
		FailureThreshold:    8,
	}
}

func componentPod(container api.Container, volumes ...api.Volume) api.Pod {
	return api.Pod{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "v1",
			Kind:       "Pod",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:        container.Name,
			Namespace:   "kube-system",
			Annotations: map[string]string{kubetypes.CriticalPodAnnotationKey: ""},
			Labels:      map[string]string{"component": container.Name, "tier": "control-plane"},
		},
		Spec: api.PodSpec{
			Containers:  []api.Container{container},
			HostNetwork: true,
			Volumes:     volumes,
		},
	}
}

func getComponentBaseCommand(component string) []string {
	if kubeadmapi.GlobalEnvParams.HyperkubeImage != "" {
		return []string{"/hyperkube", component}
	}

	return []string{"kube-" + component}
}

func getAPIServerCommand(cfg *kubeadmapi.MasterConfiguration, selfHosted bool, k8sVersion *version.Version) []string {
	var command []string

	// self-hosted apiserver needs to wait on a lock
	if selfHosted {
		command = []string{"/usr/bin/flock", "--exclusive", "--timeout=30", "/var/lock/api-server.lock"}
	}

	defaultArguments := map[string]string{
		"insecure-port":                     "0",
		"admission-control":                 defaultv16AdmissionControl,
		"service-cluster-ip-range":          cfg.Networking.ServiceSubnet,
		"service-account-key-file":          filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPublicKeyName),
		"client-ca-file":                    filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName),
		"tls-cert-file":                     filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerCertName),
		"tls-private-key-file":              filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKeyName),
		"kubelet-client-certificate":        filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientCertName),
		"kubelet-client-key":                filepath.Join(cfg.CertificatesDir, kubeadmconstants.APIServerKubeletClientKeyName),
		"secure-port":                       fmt.Sprintf("%d", cfg.API.BindPort),
		"allow-privileged":                  "true",
		"experimental-bootstrap-token-auth": "true",
		"kubelet-preferred-address-types":   "InternalIP,ExternalIP,Hostname",
		// add options to configure the front proxy.  Without the generated client cert, this will never be useable
		// so add it unconditionally with recommended values
		"requestheader-username-headers":     "X-Remote-User",
		"requestheader-group-headers":        "X-Remote-Group",
		"requestheader-extra-headers-prefix": "X-Remote-Extra-",
		"requestheader-client-ca-file":       filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyCACertName),
		"requestheader-allowed-names":        "front-proxy-client",
	}
	if k8sVersion.AtLeast(kubeadmconstants.MinimumAPIAggregationVersion) {
		// add options which allow the kube-apiserver to act as a front-proxy to aggregated API servers
		defaultArguments["proxy-client-cert-file"] = filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyClientCertName)
		defaultArguments["proxy-client-key-file"] = filepath.Join(cfg.CertificatesDir, kubeadmconstants.FrontProxyClientKeyName)
	}
	if kubeadmutil.IsNodeAuthorizerSupported(k8sVersion) {
		// enable the NodeRestriction admission plugin
		defaultArguments["admission-control"] = defaultv17AdmissionControl
	}

	command = getComponentBaseCommand(apiServer)
	command = append(command, getExtraParameters(cfg.APIServerExtraArgs, defaultArguments)...)
	command = append(command, getAuthzParameters(cfg.AuthorizationModes)...)

	if selfHosted {
		command = append(command, "--advertise-address=$(POD_IP)")
	} else {
		command = append(command, fmt.Sprintf("--advertise-address=%s", cfg.API.AdvertiseAddress))
	}

	// Check if the user decided to use an external etcd cluster
	if len(cfg.Etcd.Endpoints) > 0 {
		command = append(command, fmt.Sprintf("--etcd-servers=%s", strings.Join(cfg.Etcd.Endpoints, ",")))
	} else {
		command = append(command, "--etcd-servers=http://127.0.0.1:2379")
	}

	// Is etcd secured?
	if cfg.Etcd.CAFile != "" {
		command = append(command, fmt.Sprintf("--etcd-cafile=%s", cfg.Etcd.CAFile))
	}
	if cfg.Etcd.CertFile != "" && cfg.Etcd.KeyFile != "" {
		etcdClientFileArg := fmt.Sprintf("--etcd-certfile=%s", cfg.Etcd.CertFile)
		etcdKeyFileArg := fmt.Sprintf("--etcd-keyfile=%s", cfg.Etcd.KeyFile)
		command = append(command, etcdClientFileArg, etcdKeyFileArg)
	}

	if cfg.CloudProvider != "" {
		command = append(command, "--cloud-provider="+cfg.CloudProvider)

		// Only append the --cloud-config option if there's a such file
		if _, err := os.Stat(DefaultCloudConfigPath); err == nil {
			command = append(command, "--cloud-config="+DefaultCloudConfigPath)
		}
	}

	return command
}

func getEtcdCommand(cfg *kubeadmapi.MasterConfiguration) []string {
	var command []string

	defaultArguments := map[string]string{
		"listen-client-urls":    "http://127.0.0.1:2379",
		"advertise-client-urls": "http://127.0.0.1:2379",
		"data-dir":              cfg.Etcd.DataDir,
	}

	command = append(command, "etcd")
	command = append(command, getExtraParameters(cfg.Etcd.ExtraArgs, defaultArguments)...)

	return command
}

func getControllerManagerCommand(cfg *kubeadmapi.MasterConfiguration, selfHosted bool, k8sVersion *version.Version) []string {
	var command []string

	// self-hosted controller-manager needs to wait on a lock
	if selfHosted {
		command = []string{"/usr/bin/flock", "--exclusive", "--timeout=30", "/var/lock/controller-manager.lock"}
	}

	defaultArguments := map[string]string{
		"address":                          "127.0.0.1",
		"leader-elect":                     "true",
		"kubeconfig":                       filepath.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.ControllerManagerKubeConfigFileName),
		"root-ca-file":                     filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName),
		"service-account-private-key-file": filepath.Join(cfg.CertificatesDir, kubeadmconstants.ServiceAccountPrivateKeyName),
		"cluster-signing-cert-file":        filepath.Join(cfg.CertificatesDir, kubeadmconstants.CACertName),
		"cluster-signing-key-file":         filepath.Join(cfg.CertificatesDir, kubeadmconstants.CAKeyName),
		"use-service-account-credentials":  "true",
		"controllers":                      "*,bootstrapsigner,tokencleaner",
	}
	if k8sVersion.LessThan(kubeadmconstants.MinimumCSRSARApproverVersion) {
		// enable the former CSR group approver for v1.6 clusters.
		// TODO(luxas): Remove this once we're targeting v1.8 at HEAD
		defaultArguments["insecure-experimental-approve-all-kubelet-csrs-for-group"] = bootstrapapi.BootstrapGroup
	}

	command = getComponentBaseCommand(controllerManager)
	command = append(command, getExtraParameters(cfg.ControllerManagerExtraArgs, defaultArguments)...)

	if cfg.CloudProvider != "" {
		command = append(command, "--cloud-provider="+cfg.CloudProvider)

		// Only append the --cloud-config option if there's a such file
		if _, err := os.Stat(DefaultCloudConfigPath); err == nil {
			command = append(command, "--cloud-config="+DefaultCloudConfigPath)
		}
	}

	// Let the controller-manager allocate Node CIDRs for the Pod network.
	// Each node will get a subspace of the address CIDR provided with --pod-network-cidr.
	if cfg.Networking.PodSubnet != "" {
		command = append(command, "--allocate-node-cidrs=true", "--cluster-cidr="+cfg.Networking.PodSubnet)
	}

	return command
}

func getSchedulerCommand(cfg *kubeadmapi.MasterConfiguration, selfHosted bool) []string {
	var command []string

	// self-hosted apiserver needs to wait on a lock
	if selfHosted {
		command = []string{"/usr/bin/flock", "--exclusive", "--timeout=30", "/var/lock/api-server.lock"}
	}

	defaultArguments := map[string]string{
		"address":      "127.0.0.1",
		"leader-elect": "true",
		"kubeconfig":   filepath.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.SchedulerKubeConfigFileName),
	}

	command = getComponentBaseCommand(scheduler)
	command = append(command, getExtraParameters(cfg.SchedulerExtraArgs, defaultArguments)...)

	return command
}

func getProxyEnvVars() []api.EnvVar {
	envs := []api.EnvVar{}
	for _, env := range os.Environ() {
		pos := strings.Index(env, "=")
		if pos == -1 {
			// malformed environment variable, skip it.
			continue
		}
		name := env[:pos]
		value := env[pos+1:]
		if strings.HasSuffix(strings.ToLower(name), "_proxy") && value != "" {
			envVar := api.EnvVar{Name: name, Value: value}
			envs = append(envs, envVar)
		}
	}
	return envs
}

func getSelfHostedAPIServerEnv() []api.EnvVar {
	podIPEnvVar := api.EnvVar{
		Name: "POD_IP",
		ValueFrom: &api.EnvVarSource{
			FieldRef: &api.ObjectFieldSelector{
				FieldPath: "status.podIP",
			},
		},
	}

	return append(getProxyEnvVars(), podIPEnvVar)
}

func getAuthzParameters(modes []string) []string {
	command := []string{}
	strset := sets.NewString(modes...)

	// RBAC must always be set, prepend that if not present in the list
	// TODO(luxas): In v1.8, require the Node and RBAC authorizers to be present
	// In v1.8 we can utilize the API defaulting/validation to do enforce this,
	// currently this logic has to be split up into two places, which is sub-optimal.
	// Cleanup work is possible in and should be done in v1.8
	if !strset.Has(authzmodes.ModeRBAC) {
		modes = append([]string{authzmodes.ModeRBAC}, modes...)
	}

	if strset.Has(authzmodes.ModeABAC) {
		command = append(command, "--authorization-policy-file="+kubeadmconstants.AuthorizationPolicyPath)
	}
	if strset.Has(authzmodes.ModeWebhook) {
		command = append(command, "--authorization-webhook-config-file="+kubeadmconstants.AuthorizationWebhookConfigPath)
	}

	command = append(command, "--authorization-mode="+strings.Join(modes, ","))
	return command
}

func getExtraParameters(overrides map[string]string, defaults map[string]string) []string {
	var command []string
	for k, v := range overrides {
		if len(v) > 0 {
			command = append(command, fmt.Sprintf("--%s=%s", k, v))
		}
	}
	for k, v := range defaults {
		if _, overrideExists := overrides[k]; !overrideExists {
			command = append(command, fmt.Sprintf("--%s=%s", k, v))
		}
	}
	return command
}
