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

package kubelet

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilvalidation "k8s.io/apimachinery/pkg/util/validation"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/api/v1/resource"
	podshelper "k8s.io/kubernetes/pkg/apis/core/pods"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/fieldpath"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/cri/streaming/portforward"
	remotecommandserver "k8s.io/kubernetes/pkg/kubelet/cri/streaming/remotecommand"
	"k8s.io/kubernetes/pkg/kubelet/envvars"
	"k8s.io/kubernetes/pkg/kubelet/images"
	"k8s.io/kubernetes/pkg/kubelet/status"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/pkg/volume/util/hostutil"
	"k8s.io/kubernetes/pkg/volume/util/subpath"
	"k8s.io/kubernetes/pkg/volume/util/volumepathhandler"
	volumevalidation "k8s.io/kubernetes/pkg/volume/validation"
	"k8s.io/kubernetes/third_party/forked/golang/expansion"
	utilnet "k8s.io/utils/net"
)

const (
	managedHostsHeader                = "# Kubernetes-managed hosts file.\n"
	managedHostsHeaderWithHostNetwork = "# Kubernetes-managed hosts file (host network).\n"
)

// Container state reason list
const (
	PodInitializing   = "PodInitializing"
	ContainerCreating = "ContainerCreating"
)

// Get a list of pods that have data directories.
func (kl *Kubelet) listPodsFromDisk() ([]types.UID, error) {
	podInfos, err := ioutil.ReadDir(kl.getPodsDir())
	if err != nil {
		return nil, err
	}
	pods := []types.UID{}
	for i := range podInfos {
		if podInfos[i].IsDir() {
			pods = append(pods, types.UID(podInfos[i].Name()))
		}
	}
	return pods, nil
}

// GetActivePods returns pods that may have a running container (a
// terminated pod is one that is known to have no running containers and
// will not get any more).
func (kl *Kubelet) GetActivePods() []*v1.Pod {
	allPods := kl.podManager.GetPods()
	activePods := kl.filterOutTerminatedPods(allPods)
	return activePods
}

// makeBlockVolumes maps the raw block devices specified in the path of the container
// Experimental
func (kl *Kubelet) makeBlockVolumes(pod *v1.Pod, container *v1.Container, podVolumes kubecontainer.VolumeMap, blkutil volumepathhandler.BlockVolumePathHandler) ([]kubecontainer.DeviceInfo, error) {
	var devices []kubecontainer.DeviceInfo
	for _, device := range container.VolumeDevices {
		// check path is absolute
		if !filepath.IsAbs(device.DevicePath) {
			return nil, fmt.Errorf("error DevicePath `%s` must be an absolute path", device.DevicePath)
		}
		vol, ok := podVolumes[device.Name]
		if !ok || vol.BlockVolumeMapper == nil {
			klog.ErrorS(nil, "Block volume cannot be satisfied for container, because the volume is missing or the volume mapper is nil", "containerName", container.Name, "device", device)
			return nil, fmt.Errorf("cannot find volume %q to pass into container %q", device.Name, container.Name)
		}
		// Get a symbolic link associated to a block device under pod device path
		dirPath, volName := vol.BlockVolumeMapper.GetPodDeviceMapPath()
		symlinkPath := path.Join(dirPath, volName)
		if islinkExist, checkErr := blkutil.IsSymlinkExist(symlinkPath); checkErr != nil {
			return nil, checkErr
		} else if islinkExist {
			// Check readOnly in PVCVolumeSource and set read only permission if it's true.
			permission := "mrw"
			if vol.ReadOnly {
				permission = "r"
			}
			klog.V(4).InfoS("Device will be attached to container in the corresponding path on host", "containerName", container.Name, "path", symlinkPath)
			devices = append(devices, kubecontainer.DeviceInfo{PathOnHost: symlinkPath, PathInContainer: device.DevicePath, Permissions: permission})
		}
	}

	return devices, nil
}

// shouldMountHostsFile checks if the nodes /etc/hosts should be mounted
// Kubernetes only mounts on /etc/hosts if:
// - container is not an infrastructure (pause) container
// - container is not already mounting on /etc/hosts
// - if it is Windows and ContainerD is used.
// Kubernetes will not mount /etc/hosts if:
// - when the Pod sandbox is being created, its IP is still unknown. Hence, PodIP will not have been set.
// - Windows pod contains a hostProcess container
func shouldMountHostsFile(pod *v1.Pod, podIPs []string, supportsSingleFileMapping bool) bool {
	shouldMount := len(podIPs) > 0 && supportsSingleFileMapping
	if runtime.GOOS == "windows" && utilfeature.DefaultFeatureGate.Enabled(features.WindowsHostProcessContainers) {
		return shouldMount && !kubecontainer.HasWindowsHostProcessContainer(pod)
	}
	return shouldMount
}

// makeMounts determines the mount points for the given container.
func makeMounts(pod *v1.Pod, podDir string, container *v1.Container, hostName, hostDomain string, podIPs []string, podVolumes kubecontainer.VolumeMap, hu hostutil.HostUtils, subpather subpath.Interface, expandEnvs []kubecontainer.EnvVar, supportsSingleFileMapping bool) ([]kubecontainer.Mount, func(), error) {
	mountEtcHostsFile := shouldMountHostsFile(pod, podIPs, supportsSingleFileMapping)
	klog.V(3).InfoS("Creating hosts mount for container", "pod", klog.KObj(pod), "containerName", container.Name, "podIPs", podIPs, "path", mountEtcHostsFile)
	mounts := []kubecontainer.Mount{}
	var cleanupAction func()
	for i, mount := range container.VolumeMounts {
		// do not mount /etc/hosts if container is already mounting on the path
		mountEtcHostsFile = mountEtcHostsFile && (mount.MountPath != etcHostsPath)
		vol, ok := podVolumes[mount.Name]
		if !ok || vol.Mounter == nil {
			klog.ErrorS(nil, "Mount cannot be satisfied for the container, because the volume is missing or the volume mounter (vol.Mounter) is nil",
				"containerName", container.Name, "ok", ok, "volumeMounter", mount)
			return nil, cleanupAction, fmt.Errorf("cannot find volume %q to mount into container %q", mount.Name, container.Name)
		}

		relabelVolume := false
		// If the volume supports SELinux and it has not been
		// relabeled already and it is not a read-only volume,
		// relabel it and mark it as labeled
		if vol.Mounter.GetAttributes().Managed && vol.Mounter.GetAttributes().SupportsSELinux && !vol.SELinuxLabeled {
			vol.SELinuxLabeled = true
			relabelVolume = true
		}
		hostPath, err := volumeutil.GetPath(vol.Mounter)
		if err != nil {
			return nil, cleanupAction, err
		}

		subPath := mount.SubPath
		if mount.SubPathExpr != "" {
			if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeSubpath) {
				return nil, cleanupAction, fmt.Errorf("volume subpaths are disabled")
			}

			subPath, err = kubecontainer.ExpandContainerVolumeMounts(mount, expandEnvs)

			if err != nil {
				return nil, cleanupAction, err
			}
		}

		if subPath != "" {
			if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeSubpath) {
				return nil, cleanupAction, fmt.Errorf("volume subpaths are disabled")
			}

			if filepath.IsAbs(subPath) {
				return nil, cleanupAction, fmt.Errorf("error SubPath `%s` must not be an absolute path", subPath)
			}

			err = volumevalidation.ValidatePathNoBacksteps(subPath)
			if err != nil {
				return nil, cleanupAction, fmt.Errorf("unable to provision SubPath `%s`: %v", subPath, err)
			}

			volumePath := hostPath
			hostPath = filepath.Join(volumePath, subPath)

			if subPathExists, err := hu.PathExists(hostPath); err != nil {
				klog.ErrorS(nil, "Could not determine if subPath exists, will not attempt to change its permissions", "path", hostPath)
			} else if !subPathExists {
				// Create the sub path now because if it's auto-created later when referenced, it may have an
				// incorrect ownership and mode. For example, the sub path directory must have at least g+rwx
				// when the pod specifies an fsGroup, and if the directory is not created here, Docker will
				// later auto-create it with the incorrect mode 0750
				// Make extra care not to escape the volume!
				perm, err := hu.GetMode(volumePath)
				if err != nil {
					return nil, cleanupAction, err
				}
				if err := subpather.SafeMakeDir(subPath, volumePath, perm); err != nil {
					// Don't pass detailed error back to the user because it could give information about host filesystem
					klog.ErrorS(err, "Failed to create subPath directory for volumeMount of the container", "containerName", container.Name, "volumeMountName", mount.Name)
					return nil, cleanupAction, fmt.Errorf("failed to create subPath directory for volumeMount %q of container %q", mount.Name, container.Name)
				}
			}
			hostPath, cleanupAction, err = subpather.PrepareSafeSubpath(subpath.Subpath{
				VolumeMountIndex: i,
				Path:             hostPath,
				VolumeName:       vol.InnerVolumeSpecName,
				VolumePath:       volumePath,
				PodDir:           podDir,
				ContainerName:    container.Name,
			})
			if err != nil {
				// Don't pass detailed error back to the user because it could give information about host filesystem
				klog.ErrorS(err, "Failed to prepare subPath for volumeMount of the container", "containerName", container.Name, "volumeMountName", mount.Name)
				return nil, cleanupAction, fmt.Errorf("failed to prepare subPath for volumeMount %q of container %q", mount.Name, container.Name)
			}
		}

		// Docker Volume Mounts fail on Windows if it is not of the form C:/
		if volumeutil.IsWindowsLocalPath(runtime.GOOS, hostPath) {
			hostPath = volumeutil.MakeAbsolutePath(runtime.GOOS, hostPath)
		}

		containerPath := mount.MountPath
		// IsAbs returns false for UNC path/SMB shares/named pipes in Windows. So check for those specifically and skip MakeAbsolutePath
		if !volumeutil.IsWindowsUNCPath(runtime.GOOS, containerPath) && !filepath.IsAbs(containerPath) {
			containerPath = volumeutil.MakeAbsolutePath(runtime.GOOS, containerPath)
		}

		propagation, err := translateMountPropagation(mount.MountPropagation)
		if err != nil {
			return nil, cleanupAction, err
		}
		klog.V(5).InfoS("Mount has propagation", "pod", klog.KObj(pod), "containerName", container.Name, "volumeMountName", mount.Name, "propagation", propagation)
		mustMountRO := vol.Mounter.GetAttributes().ReadOnly

		mounts = append(mounts, kubecontainer.Mount{
			Name:           mount.Name,
			ContainerPath:  containerPath,
			HostPath:       hostPath,
			ReadOnly:       mount.ReadOnly || mustMountRO,
			SELinuxRelabel: relabelVolume,
			Propagation:    propagation,
		})
	}
	if mountEtcHostsFile {
		hostAliases := pod.Spec.HostAliases
		hostsMount, err := makeHostsMount(podDir, podIPs, hostName, hostDomain, hostAliases, pod.Spec.HostNetwork)
		if err != nil {
			return nil, cleanupAction, err
		}
		mounts = append(mounts, *hostsMount)
	}
	return mounts, cleanupAction, nil
}

// translateMountPropagation transforms v1.MountPropagationMode to
// runtimeapi.MountPropagation.
func translateMountPropagation(mountMode *v1.MountPropagationMode) (runtimeapi.MountPropagation, error) {
	if runtime.GOOS == "windows" {
		// Windows containers doesn't support mount propagation, use private for it.
		// Refer https://docs.docker.com/storage/bind-mounts/#configure-bind-propagation.
		return runtimeapi.MountPropagation_PROPAGATION_PRIVATE, nil
	}

	switch {
	case mountMode == nil:
		// PRIVATE is the default
		return runtimeapi.MountPropagation_PROPAGATION_PRIVATE, nil
	case *mountMode == v1.MountPropagationHostToContainer:
		return runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER, nil
	case *mountMode == v1.MountPropagationBidirectional:
		return runtimeapi.MountPropagation_PROPAGATION_BIDIRECTIONAL, nil
	case *mountMode == v1.MountPropagationNone:
		return runtimeapi.MountPropagation_PROPAGATION_PRIVATE, nil
	default:
		return 0, fmt.Errorf("invalid MountPropagation mode: %q", *mountMode)
	}
}

// getEtcHostsPath returns the full host-side path to a pod's generated /etc/hosts file
func getEtcHostsPath(podDir string) string {
	hostsFilePath := path.Join(podDir, "etc-hosts")
	// Volume Mounts fail on Windows if it is not of the form C:/
	return volumeutil.MakeAbsolutePath(runtime.GOOS, hostsFilePath)
}

// makeHostsMount makes the mountpoint for the hosts file that the containers
// in a pod are injected with. podIPs is provided instead of podIP as podIPs
// are present even if dual-stack feature flag is not enabled.
func makeHostsMount(podDir string, podIPs []string, hostName, hostDomainName string, hostAliases []v1.HostAlias, useHostNetwork bool) (*kubecontainer.Mount, error) {
	hostsFilePath := getEtcHostsPath(podDir)
	if err := ensureHostsFile(hostsFilePath, podIPs, hostName, hostDomainName, hostAliases, useHostNetwork); err != nil {
		return nil, err
	}
	return &kubecontainer.Mount{
		Name:           "k8s-managed-etc-hosts",
		ContainerPath:  etcHostsPath,
		HostPath:       hostsFilePath,
		ReadOnly:       false,
		SELinuxRelabel: true,
	}, nil
}

// ensureHostsFile ensures that the given host file has an up-to-date ip, host
// name, and domain name.
func ensureHostsFile(fileName string, hostIPs []string, hostName, hostDomainName string, hostAliases []v1.HostAlias, useHostNetwork bool) error {
	var hostsFileContent []byte
	var err error

	if useHostNetwork {
		// if Pod is using host network, read hosts file from the node's filesystem.
		// `etcHostsPath` references the location of the hosts file on the node.
		// `/etc/hosts` for *nix systems.
		hostsFileContent, err = nodeHostsFileContent(etcHostsPath, hostAliases)
		if err != nil {
			return err
		}
	} else {
		// if Pod is not using host network, create a managed hosts file with Pod IP and other information.
		hostsFileContent = managedHostsFileContent(hostIPs, hostName, hostDomainName, hostAliases)
	}

	return ioutil.WriteFile(fileName, hostsFileContent, 0644)
}

// nodeHostsFileContent reads the content of node's hosts file.
func nodeHostsFileContent(hostsFilePath string, hostAliases []v1.HostAlias) ([]byte, error) {
	hostsFileContent, err := ioutil.ReadFile(hostsFilePath)
	if err != nil {
		return nil, err
	}
	var buffer bytes.Buffer
	buffer.WriteString(managedHostsHeaderWithHostNetwork)
	buffer.Write(hostsFileContent)
	buffer.Write(hostsEntriesFromHostAliases(hostAliases))
	return buffer.Bytes(), nil
}

// managedHostsFileContent generates the content of the managed etc hosts based on Pod IPs and other
// information.
func managedHostsFileContent(hostIPs []string, hostName, hostDomainName string, hostAliases []v1.HostAlias) []byte {
	var buffer bytes.Buffer
	buffer.WriteString(managedHostsHeader)
	buffer.WriteString("127.0.0.1\tlocalhost\n")                      // ipv4 localhost
	buffer.WriteString("::1\tlocalhost ip6-localhost ip6-loopback\n") // ipv6 localhost
	buffer.WriteString("fe00::0\tip6-localnet\n")
	buffer.WriteString("fe00::0\tip6-mcastprefix\n")
	buffer.WriteString("fe00::1\tip6-allnodes\n")
	buffer.WriteString("fe00::2\tip6-allrouters\n")
	if len(hostDomainName) > 0 {
		// host entry generated for all IPs in podIPs
		// podIPs field is populated for clusters even
		// dual-stack feature flag is not enabled.
		for _, hostIP := range hostIPs {
			buffer.WriteString(fmt.Sprintf("%s\t%s.%s\t%s\n", hostIP, hostName, hostDomainName, hostName))
		}
	} else {
		for _, hostIP := range hostIPs {
			buffer.WriteString(fmt.Sprintf("%s\t%s\n", hostIP, hostName))
		}
	}
	buffer.Write(hostsEntriesFromHostAliases(hostAliases))
	return buffer.Bytes()
}

func hostsEntriesFromHostAliases(hostAliases []v1.HostAlias) []byte {
	if len(hostAliases) == 0 {
		return []byte{}
	}

	var buffer bytes.Buffer
	buffer.WriteString("\n")
	buffer.WriteString("# Entries added by HostAliases.\n")
	// for each IP, write all aliases onto single line in hosts file
	for _, hostAlias := range hostAliases {
		buffer.WriteString(fmt.Sprintf("%s\t%s\n", hostAlias.IP, strings.Join(hostAlias.Hostnames, "\t")))
	}
	return buffer.Bytes()
}

// truncatePodHostnameIfNeeded truncates the pod hostname if it's longer than 63 chars.
func truncatePodHostnameIfNeeded(podName, hostname string) (string, error) {
	// Cap hostname at 63 chars (specification is 64bytes which is 63 chars and the null terminating char).
	const hostnameMaxLen = 63
	if len(hostname) <= hostnameMaxLen {
		return hostname, nil
	}
	truncated := hostname[:hostnameMaxLen]
	klog.ErrorS(nil, "Hostname for pod was too long, truncated it", "podName", podName, "hostnameMaxLen", hostnameMaxLen, "truncatedHostname", truncated)
	// hostname should not end with '-' or '.'
	truncated = strings.TrimRight(truncated, "-.")
	if len(truncated) == 0 {
		// This should never happen.
		return "", fmt.Errorf("hostname for pod %q was invalid: %q", podName, hostname)
	}
	return truncated, nil
}

// GeneratePodHostNameAndDomain creates a hostname and domain name for a pod,
// given that pod's spec and annotations or returns an error.
func (kl *Kubelet) GeneratePodHostNameAndDomain(pod *v1.Pod) (string, string, error) {
	clusterDomain := kl.dnsConfigurer.ClusterDomain

	hostname := pod.Name
	if len(pod.Spec.Hostname) > 0 {
		if msgs := utilvalidation.IsDNS1123Label(pod.Spec.Hostname); len(msgs) != 0 {
			return "", "", fmt.Errorf("pod Hostname %q is not a valid DNS label: %s", pod.Spec.Hostname, strings.Join(msgs, ";"))
		}
		hostname = pod.Spec.Hostname
	}

	hostname, err := truncatePodHostnameIfNeeded(pod.Name, hostname)
	if err != nil {
		return "", "", err
	}

	hostDomain := ""
	if len(pod.Spec.Subdomain) > 0 {
		if msgs := utilvalidation.IsDNS1123Label(pod.Spec.Subdomain); len(msgs) != 0 {
			return "", "", fmt.Errorf("pod Subdomain %q is not a valid DNS label: %s", pod.Spec.Subdomain, strings.Join(msgs, ";"))
		}
		hostDomain = fmt.Sprintf("%s.%s.svc.%s", pod.Spec.Subdomain, pod.Namespace, clusterDomain)
	}

	return hostname, hostDomain, nil
}

// GetPodCgroupParent gets pod cgroup parent from container manager.
func (kl *Kubelet) GetPodCgroupParent(pod *v1.Pod) string {
	pcm := kl.containerManager.NewPodContainerManager()
	_, cgroupParent := pcm.GetPodContainerName(pod)
	return cgroupParent
}

// GenerateRunContainerOptions generates the RunContainerOptions, which can be used by
// the container runtime to set parameters for launching a container.
func (kl *Kubelet) GenerateRunContainerOptions(pod *v1.Pod, container *v1.Container, podIP string, podIPs []string) (*kubecontainer.RunContainerOptions, func(), error) {
	opts, err := kl.containerManager.GetResources(pod, container)
	if err != nil {
		return nil, nil, err
	}
	// The value of hostname is the short host name and it is sent to makeMounts to create /etc/hosts file.
	hostname, hostDomainName, err := kl.GeneratePodHostNameAndDomain(pod)
	if err != nil {
		return nil, nil, err
	}
	// nodename will be equals to hostname if SetHostnameAsFQDN is nil or false. If SetHostnameFQDN
	// is true and hostDomainName is defined, nodename will be the FQDN (hostname.hostDomainName)
	nodename, err := util.GetNodenameForKernel(hostname, hostDomainName, pod.Spec.SetHostnameAsFQDN)
	if err != nil {
		return nil, nil, err
	}
	opts.Hostname = nodename
	podName := volumeutil.GetUniquePodName(pod)
	volumes := kl.volumeManager.GetMountedVolumesForPod(podName)

	blkutil := volumepathhandler.NewBlockVolumePathHandler()
	blkVolumes, err := kl.makeBlockVolumes(pod, container, volumes, blkutil)
	if err != nil {
		return nil, nil, err
	}
	opts.Devices = append(opts.Devices, blkVolumes...)

	envs, err := kl.makeEnvironmentVariables(pod, container, podIP, podIPs)
	if err != nil {
		return nil, nil, err
	}
	opts.Envs = append(opts.Envs, envs...)

	// we can only mount individual files (e.g.: /etc/hosts, termination-log files) on Windows only if we're using Containerd.
	supportsSingleFileMapping := kl.containerRuntime.SupportsSingleFileMapping()
	// only podIPs is sent to makeMounts, as podIPs is populated even if dual-stack feature flag is not enabled.
	mounts, cleanupAction, err := makeMounts(pod, kl.getPodDir(pod.UID), container, hostname, hostDomainName, podIPs, volumes, kl.hostutil, kl.subpather, opts.Envs, supportsSingleFileMapping)
	if err != nil {
		return nil, cleanupAction, err
	}
	opts.Mounts = append(opts.Mounts, mounts...)

	// adding TerminationMessagePath on Windows is only allowed if ContainerD is used. Individual files cannot
	// be mounted as volumes using Docker for Windows.
	if len(container.TerminationMessagePath) != 0 && supportsSingleFileMapping {
		p := kl.getPodContainerDir(pod.UID, container.Name)
		if err := os.MkdirAll(p, 0750); err != nil {
			klog.ErrorS(err, "Error on creating dir", "path", p)
		} else {
			opts.PodContainerDir = p
		}
	}

	// only do this check if the experimental behavior is enabled, otherwise allow it to default to false
	if kl.experimentalHostUserNamespaceDefaulting {
		opts.EnableHostUserNamespace = kl.enableHostUserNamespace(pod)
	}

	return opts, cleanupAction, nil
}

var masterServices = sets.NewString("kubernetes")

// getServiceEnvVarMap makes a map[string]string of env vars for services a
// pod in namespace ns should see.
func (kl *Kubelet) getServiceEnvVarMap(ns string, enableServiceLinks bool) (map[string]string, error) {
	var (
		serviceMap = make(map[string]*v1.Service)
		m          = make(map[string]string)
	)

	// Get all service resources from the master (via a cache),
	// and populate them into service environment variables.
	if kl.serviceLister == nil {
		// Kubelets without masters (e.g. plain GCE ContainerVM) don't set env vars.
		return m, nil
	}
	services, err := kl.serviceLister.List(labels.Everything())
	if err != nil {
		return m, fmt.Errorf("failed to list services when setting up env vars")
	}

	// project the services in namespace ns onto the master services
	for i := range services {
		service := services[i]
		// ignore services where ClusterIP is "None" or empty
		if !v1helper.IsServiceIPSet(service) {
			continue
		}
		serviceName := service.Name

		// We always want to add environment variabled for master services
		// from the master service namespace, even if enableServiceLinks is false.
		// We also add environment variables for other services in the same
		// namespace, if enableServiceLinks is true.
		if service.Namespace == kl.masterServiceNamespace && masterServices.Has(serviceName) {
			if _, exists := serviceMap[serviceName]; !exists {
				serviceMap[serviceName] = service
			}
		} else if service.Namespace == ns && enableServiceLinks {
			serviceMap[serviceName] = service
		}
	}

	mappedServices := []*v1.Service{}
	for key := range serviceMap {
		mappedServices = append(mappedServices, serviceMap[key])
	}

	for _, e := range envvars.FromServices(mappedServices) {
		m[e.Name] = e.Value
	}
	return m, nil
}

// Make the environment variables for a pod in the given namespace.
func (kl *Kubelet) makeEnvironmentVariables(pod *v1.Pod, container *v1.Container, podIP string, podIPs []string) ([]kubecontainer.EnvVar, error) {
	if pod.Spec.EnableServiceLinks == nil {
		return nil, fmt.Errorf("nil pod.spec.enableServiceLinks encountered, cannot construct envvars")
	}

	// If the pod originates from the kube-api, when we know that the kube-apiserver is responding and the kubelet's credentials are valid.
	// Knowing this, it is reasonable to wait until the service lister has synchronized at least once before attempting to build
	// a service env var map.  This doesn't present the race below from happening entirely, but it does prevent the "obvious"
	// failure case of services simply not having completed a list operation that can reasonably be expected to succeed.
	// One common case this prevents is a kubelet restart reading pods before services and some pod not having the
	// KUBERNETES_SERVICE_HOST injected because we didn't wait a short time for services to sync before proceeding.
	// The KUBERNETES_SERVICE_HOST link is special because it is unconditionally injected into pods and is read by the
	// in-cluster-config for pod clients
	if !kubetypes.IsStaticPod(pod) && !kl.serviceHasSynced() {
		return nil, fmt.Errorf("services have not yet been read at least once, cannot construct envvars")
	}

	var result []kubecontainer.EnvVar
	// Note:  These are added to the docker Config, but are not included in the checksum computed
	// by kubecontainer.HashContainer(...).  That way, we can still determine whether an
	// v1.Container is already running by its hash. (We don't want to restart a container just
	// because some service changed.)
	//
	// Note that there is a race between Kubelet seeing the pod and kubelet seeing the service.
	// To avoid this users can: (1) wait between starting a service and starting; or (2) detect
	// missing service env var and exit and be restarted; or (3) use DNS instead of env vars
	// and keep trying to resolve the DNS name of the service (recommended).
	serviceEnv, err := kl.getServiceEnvVarMap(pod.Namespace, *pod.Spec.EnableServiceLinks)
	if err != nil {
		return result, err
	}

	var (
		configMaps = make(map[string]*v1.ConfigMap)
		secrets    = make(map[string]*v1.Secret)
		tmpEnv     = make(map[string]string)
	)

	// Env will override EnvFrom variables.
	// Process EnvFrom first then allow Env to replace existing values.
	for _, envFrom := range container.EnvFrom {
		switch {
		case envFrom.ConfigMapRef != nil:
			cm := envFrom.ConfigMapRef
			name := cm.Name
			configMap, ok := configMaps[name]
			if !ok {
				if kl.kubeClient == nil {
					return result, fmt.Errorf("couldn't get configMap %v/%v, no kubeClient defined", pod.Namespace, name)
				}
				optional := cm.Optional != nil && *cm.Optional
				configMap, err = kl.configMapManager.GetConfigMap(pod.Namespace, name)
				if err != nil {
					if errors.IsNotFound(err) && optional {
						// ignore error when marked optional
						continue
					}
					return result, err
				}
				configMaps[name] = configMap
			}

			invalidKeys := []string{}
			for k, v := range configMap.Data {
				if len(envFrom.Prefix) > 0 {
					k = envFrom.Prefix + k
				}
				if errMsgs := utilvalidation.IsEnvVarName(k); len(errMsgs) != 0 {
					invalidKeys = append(invalidKeys, k)
					continue
				}
				tmpEnv[k] = v
			}
			if len(invalidKeys) > 0 {
				sort.Strings(invalidKeys)
				kl.recorder.Eventf(pod, v1.EventTypeWarning, "InvalidEnvironmentVariableNames", "Keys [%s] from the EnvFrom configMap %s/%s were skipped since they are considered invalid environment variable names.", strings.Join(invalidKeys, ", "), pod.Namespace, name)
			}
		case envFrom.SecretRef != nil:
			s := envFrom.SecretRef
			name := s.Name
			secret, ok := secrets[name]
			if !ok {
				if kl.kubeClient == nil {
					return result, fmt.Errorf("couldn't get secret %v/%v, no kubeClient defined", pod.Namespace, name)
				}
				optional := s.Optional != nil && *s.Optional
				secret, err = kl.secretManager.GetSecret(pod.Namespace, name)
				if err != nil {
					if errors.IsNotFound(err) && optional {
						// ignore error when marked optional
						continue
					}
					return result, err
				}
				secrets[name] = secret
			}

			invalidKeys := []string{}
			for k, v := range secret.Data {
				if len(envFrom.Prefix) > 0 {
					k = envFrom.Prefix + k
				}
				if errMsgs := utilvalidation.IsEnvVarName(k); len(errMsgs) != 0 {
					invalidKeys = append(invalidKeys, k)
					continue
				}
				tmpEnv[k] = string(v)
			}
			if len(invalidKeys) > 0 {
				sort.Strings(invalidKeys)
				kl.recorder.Eventf(pod, v1.EventTypeWarning, "InvalidEnvironmentVariableNames", "Keys [%s] from the EnvFrom secret %s/%s were skipped since they are considered invalid environment variable names.", strings.Join(invalidKeys, ", "), pod.Namespace, name)
			}
		}
	}

	// Determine the final values of variables:
	//
	// 1.  Determine the final value of each variable:
	//     a.  If the variable's Value is set, expand the `$(var)` references to other
	//         variables in the .Value field; the sources of variables are the declared
	//         variables of the container and the service environment variables
	//     b.  If a source is defined for an environment variable, resolve the source
	// 2.  Create the container's environment in the order variables are declared
	// 3.  Add remaining service environment vars
	var (
		mappingFunc = expansion.MappingFuncFor(tmpEnv, serviceEnv)
	)
	for _, envVar := range container.Env {
		runtimeVal := envVar.Value
		if runtimeVal != "" {
			// Step 1a: expand variable references
			runtimeVal = expansion.Expand(runtimeVal, mappingFunc)
		} else if envVar.ValueFrom != nil {
			// Step 1b: resolve alternate env var sources
			switch {
			case envVar.ValueFrom.FieldRef != nil:
				runtimeVal, err = kl.podFieldSelectorRuntimeValue(envVar.ValueFrom.FieldRef, pod, podIP, podIPs)
				if err != nil {
					return result, err
				}
			case envVar.ValueFrom.ResourceFieldRef != nil:
				defaultedPod, defaultedContainer, err := kl.defaultPodLimitsForDownwardAPI(pod, container)
				if err != nil {
					return result, err
				}
				runtimeVal, err = containerResourceRuntimeValue(envVar.ValueFrom.ResourceFieldRef, defaultedPod, defaultedContainer)
				if err != nil {
					return result, err
				}
			case envVar.ValueFrom.ConfigMapKeyRef != nil:
				cm := envVar.ValueFrom.ConfigMapKeyRef
				name := cm.Name
				key := cm.Key
				optional := cm.Optional != nil && *cm.Optional
				configMap, ok := configMaps[name]
				if !ok {
					if kl.kubeClient == nil {
						return result, fmt.Errorf("couldn't get configMap %v/%v, no kubeClient defined", pod.Namespace, name)
					}
					configMap, err = kl.configMapManager.GetConfigMap(pod.Namespace, name)
					if err != nil {
						if errors.IsNotFound(err) && optional {
							// ignore error when marked optional
							continue
						}
						return result, err
					}
					configMaps[name] = configMap
				}
				runtimeVal, ok = configMap.Data[key]
				if !ok {
					if optional {
						continue
					}
					return result, fmt.Errorf("couldn't find key %v in ConfigMap %v/%v", key, pod.Namespace, name)
				}
			case envVar.ValueFrom.SecretKeyRef != nil:
				s := envVar.ValueFrom.SecretKeyRef
				name := s.Name
				key := s.Key
				optional := s.Optional != nil && *s.Optional
				secret, ok := secrets[name]
				if !ok {
					if kl.kubeClient == nil {
						return result, fmt.Errorf("couldn't get secret %v/%v, no kubeClient defined", pod.Namespace, name)
					}
					secret, err = kl.secretManager.GetSecret(pod.Namespace, name)
					if err != nil {
						if errors.IsNotFound(err) && optional {
							// ignore error when marked optional
							continue
						}
						return result, err
					}
					secrets[name] = secret
				}
				runtimeValBytes, ok := secret.Data[key]
				if !ok {
					if optional {
						continue
					}
					return result, fmt.Errorf("couldn't find key %v in Secret %v/%v", key, pod.Namespace, name)
				}
				runtimeVal = string(runtimeValBytes)
			}
		}

		tmpEnv[envVar.Name] = runtimeVal
	}

	// Append the env vars
	for k, v := range tmpEnv {
		result = append(result, kubecontainer.EnvVar{Name: k, Value: v})
	}

	// Append remaining service env vars.
	for k, v := range serviceEnv {
		// Accesses apiserver+Pods.
		// So, the master may set service env vars, or kubelet may.  In case both are doing
		// it, we skip the key from the kubelet-generated ones so we don't have duplicate
		// env vars.
		// TODO: remove this next line once all platforms use apiserver+Pods.
		if _, present := tmpEnv[k]; !present {
			result = append(result, kubecontainer.EnvVar{Name: k, Value: v})
		}
	}
	return result, nil
}

// podFieldSelectorRuntimeValue returns the runtime value of the given
// selector for a pod.
func (kl *Kubelet) podFieldSelectorRuntimeValue(fs *v1.ObjectFieldSelector, pod *v1.Pod, podIP string, podIPs []string) (string, error) {
	internalFieldPath, _, err := podshelper.ConvertDownwardAPIFieldLabel(fs.APIVersion, fs.FieldPath, "")
	if err != nil {
		return "", err
	}

	// make podIPs order match node IP family preference #97979
	podIPs = kl.sortPodIPs(podIPs)
	if len(podIPs) > 0 {
		podIP = podIPs[0]
	}

	switch internalFieldPath {
	case "spec.nodeName":
		return pod.Spec.NodeName, nil
	case "spec.serviceAccountName":
		return pod.Spec.ServiceAccountName, nil
	case "status.hostIP":
		hostIPs, err := kl.getHostIPsAnyWay()
		if err != nil {
			return "", err
		}
		return hostIPs[0].String(), nil
	case "status.podIP":
		return podIP, nil
	case "status.podIPs":
		return strings.Join(podIPs, ","), nil
	}
	return fieldpath.ExtractFieldPathAsString(pod, internalFieldPath)
}

// containerResourceRuntimeValue returns the value of the provided container resource
func containerResourceRuntimeValue(fs *v1.ResourceFieldSelector, pod *v1.Pod, container *v1.Container) (string, error) {
	containerName := fs.ContainerName
	if len(containerName) == 0 {
		return resource.ExtractContainerResourceValue(fs, container)
	}
	return resource.ExtractResourceValueByContainerName(fs, pod, containerName)
}

// killPod instructs the container runtime to kill the pod. This method requires that
// the pod status contains the result of the last syncPod, otherwise it may fail to
// terminate newly created containers and sandboxes.
func (kl *Kubelet) killPod(pod *v1.Pod, p kubecontainer.Pod, gracePeriodOverride *int64) error {
	// Call the container runtime KillPod method which stops all known running containers of the pod
	if err := kl.containerRuntime.KillPod(pod, p, gracePeriodOverride); err != nil {
		return err
	}
	if err := kl.containerManager.UpdateQOSCgroups(); err != nil {
		klog.V(2).InfoS("Failed to update QoS cgroups while killing pod", "err", err)
	}
	return nil
}

// makePodDataDirs creates the dirs for the pod datas.
func (kl *Kubelet) makePodDataDirs(pod *v1.Pod) error {
	uid := pod.UID
	if err := os.MkdirAll(kl.getPodDir(uid), 0750); err != nil && !os.IsExist(err) {
		return err
	}
	if err := os.MkdirAll(kl.getPodVolumesDir(uid), 0750); err != nil && !os.IsExist(err) {
		return err
	}
	if err := os.MkdirAll(kl.getPodPluginsDir(uid), 0750); err != nil && !os.IsExist(err) {
		return err
	}
	return nil
}

// getPullSecretsForPod inspects the Pod and retrieves the referenced pull
// secrets.
func (kl *Kubelet) getPullSecretsForPod(pod *v1.Pod) []v1.Secret {
	pullSecrets := []v1.Secret{}

	for _, secretRef := range pod.Spec.ImagePullSecrets {
		if len(secretRef.Name) == 0 {
			// API validation permitted entries with empty names (http://issue.k8s.io/99454#issuecomment-787838112).
			// Ignore to avoid unnecessary warnings.
			continue
		}
		secret, err := kl.secretManager.GetSecret(pod.Namespace, secretRef.Name)
		if err != nil {
			klog.InfoS("Unable to retrieve pull secret, the image pull may not succeed.", "pod", klog.KObj(pod), "secret", klog.KObj(secret), "err", err)
			continue
		}

		pullSecrets = append(pullSecrets, *secret)
	}

	return pullSecrets
}

func countRunningContainerStatus(status v1.PodStatus) int {
	var runningContainers int
	for _, c := range status.InitContainerStatuses {
		if c.State.Running != nil {
			runningContainers++
		}
	}
	for _, c := range status.ContainerStatuses {
		if c.State.Running != nil {
			runningContainers++
		}
	}
	for _, c := range status.EphemeralContainerStatuses {
		if c.State.Running != nil {
			runningContainers++
		}
	}
	return runningContainers
}

// PodResourcesAreReclaimed returns true if all required node-level resources that a pod was consuming have
// been reclaimed by the kubelet.  Reclaiming resources is a prerequisite to deleting a pod from the API server.
func (kl *Kubelet) PodResourcesAreReclaimed(pod *v1.Pod, status v1.PodStatus) bool {
	if kl.podWorkers.CouldHaveRunningContainers(pod.UID) {
		// We shouldn't delete pods that still have running containers
		klog.V(3).InfoS("Pod is terminated, but some containers are still running", "pod", klog.KObj(pod))
		return false
	}
	if count := countRunningContainerStatus(status); count > 0 {
		// We shouldn't delete pods until the reported pod status contains no more running containers (the previous
		// check ensures no more status can be generated, this check verifies we have seen enough of the status)
		klog.V(3).InfoS("Pod is terminated, but some container status has not yet been reported", "pod", klog.KObj(pod), "running", count)
		return false
	}
	if kl.podVolumesExist(pod.UID) && !kl.keepTerminatedPodVolumes {
		// We shouldn't delete pods whose volumes have not been cleaned up if we are not keeping terminated pod volumes
		klog.V(3).InfoS("Pod is terminated, but some volumes have not been cleaned up", "pod", klog.KObj(pod))
		return false
	}
	if kl.kubeletConfiguration.CgroupsPerQOS {
		pcm := kl.containerManager.NewPodContainerManager()
		if pcm.Exists(pod) {
			klog.V(3).InfoS("Pod is terminated, but pod cgroup sandbox has not been cleaned up", "pod", klog.KObj(pod))
			return false
		}
	}

	// Note: we leave pod containers to be reclaimed in the background since dockershim requires the
	// container for retrieving logs and we want to make sure logs are available until the pod is
	// physically deleted.

	klog.V(3).InfoS("Pod is terminated and all resources are reclaimed", "pod", klog.KObj(pod))
	return true
}

// podResourcesAreReclaimed simply calls PodResourcesAreReclaimed with the most up-to-date status.
func (kl *Kubelet) podResourcesAreReclaimed(pod *v1.Pod) bool {
	status, ok := kl.statusManager.GetPodStatus(pod.UID)
	if !ok {
		status = pod.Status
	}
	return kl.PodResourcesAreReclaimed(pod, status)
}

// filterOutTerminatedPods returns the pods that could still have running
// containers
func (kl *Kubelet) filterOutTerminatedPods(pods []*v1.Pod) []*v1.Pod {
	var filteredPods []*v1.Pod
	for _, p := range pods {
		if !kl.podWorkers.CouldHaveRunningContainers(p.UID) {
			continue
		}
		filteredPods = append(filteredPods, p)
	}
	return filteredPods
}

// removeOrphanedPodStatuses removes obsolete entries in podStatus where
// the pod is no longer considered bound to this node.
func (kl *Kubelet) removeOrphanedPodStatuses(pods []*v1.Pod, mirrorPods []*v1.Pod) {
	podUIDs := make(map[types.UID]bool)
	for _, pod := range pods {
		podUIDs[pod.UID] = true
	}
	for _, pod := range mirrorPods {
		podUIDs[pod.UID] = true
	}
	kl.statusManager.RemoveOrphanedStatuses(podUIDs)
}

// deleteOrphanedMirrorPods checks whether pod killer has done with orphaned mirror pod.
// If pod killing is done, podManager.DeleteMirrorPod() is called to delete mirror pod
// from the API server
func (kl *Kubelet) deleteOrphanedMirrorPods() {
	mirrorPods := kl.podManager.GetOrphanedMirrorPodNames()
	for _, podFullname := range mirrorPods {
		if !kl.podWorkers.IsPodForMirrorPodTerminatingByFullName(podFullname) {
			_, err := kl.podManager.DeleteMirrorPod(podFullname, nil)
			if err != nil {
				klog.ErrorS(err, "Encountered error when deleting mirror pod", "podName", podFullname)
			} else {
				klog.V(3).InfoS("Deleted pod", "podName", podFullname)
			}
		}
	}
}

// HandlePodCleanups performs a series of cleanup work, including terminating
// pod workers, killing unwanted pods, and removing orphaned volumes/pod
// directories. No config changes are sent to pod workers while this method
// is executing which means no new pods can appear.
// NOTE: This function is executed by the main sync loop, so it
// should not contain any blocking calls.
func (kl *Kubelet) HandlePodCleanups() error {
	// The kubelet lacks checkpointing, so we need to introspect the set of pods
	// in the cgroup tree prior to inspecting the set of pods in our pod manager.
	// this ensures our view of the cgroup tree does not mistakenly observe pods
	// that are added after the fact...
	var (
		cgroupPods map[types.UID]cm.CgroupName
		err        error
	)
	if kl.cgroupsPerQOS {
		pcm := kl.containerManager.NewPodContainerManager()
		cgroupPods, err = pcm.GetAllPodsFromCgroups()
		if err != nil {
			return fmt.Errorf("failed to get list of pods that still exist on cgroup mounts: %v", err)
		}
	}

	allPods, mirrorPods := kl.podManager.GetPodsAndMirrorPods()
	// Pod phase progresses monotonically. Once a pod has reached a final state,
	// it should never leave regardless of the restart policy. The statuses
	// of such pods should not be changed, and there is no need to sync them.
	// TODO: the logic here does not handle two cases:
	//   1. If the containers were removed immediately after they died, kubelet
	//      may fail to generate correct statuses, let alone filtering correctly.
	//   2. If kubelet restarted before writing the terminated status for a pod
	//      to the apiserver, it could still restart the terminated pod (even
	//      though the pod was not considered terminated by the apiserver).
	// These two conditions could be alleviated by checkpointing kubelet.

	// Stop the workers for terminated pods not in the config source
	klog.V(3).InfoS("Clean up pod workers for terminated pods")
	workingPods := kl.podWorkers.SyncKnownPods(allPods)

	allPodsByUID := make(map[types.UID]*v1.Pod)
	for _, pod := range allPods {
		allPodsByUID[pod.UID] = pod
	}

	// Identify the set of pods that have workers, which should be all pods
	// from config that are not terminated, as well as any terminating pods
	// that have already been removed from config. Pods that are terminating
	// will be added to possiblyRunningPods, to prevent overly aggressive
	// cleanup of pod cgroups.
	runningPods := make(map[types.UID]sets.Empty)
	possiblyRunningPods := make(map[types.UID]sets.Empty)
	for uid, sync := range workingPods {
		switch sync {
		case SyncPodWork:
			runningPods[uid] = struct{}{}
			possiblyRunningPods[uid] = struct{}{}
		case TerminatingPodWork:
			possiblyRunningPods[uid] = struct{}{}
		}
	}

	// Stop probing pods that are not running
	klog.V(3).InfoS("Clean up probes for terminating and terminated pods")
	kl.probeManager.CleanupPods(runningPods)

	// Terminate any pods that are observed in the runtime but not
	// present in the list of known running pods from config.
	runningRuntimePods, err := kl.runtimeCache.GetPods()
	if err != nil {
		klog.ErrorS(err, "Error listing containers")
		return err
	}
	for _, runningPod := range runningRuntimePods {
		switch workType, ok := workingPods[runningPod.ID]; {
		case ok && workType == SyncPodWork, ok && workType == TerminatingPodWork:
			// if the pod worker is already in charge of this pod, we don't need to do anything
			continue
		default:
			// If the pod isn't in the set that should be running and isn't already terminating, terminate
			// now. This termination is aggressive because all known pods should already be in a known state
			// (i.e. a removed static pod should already be terminating), so these are pods that were
			// orphaned due to kubelet restart or bugs. Since housekeeping blocks other config changes, we
			// know that another pod wasn't started in the background so we are safe to terminate the
			// unknown pods.
			if _, ok := allPodsByUID[runningPod.ID]; !ok {
				klog.V(3).InfoS("Clean up orphaned pod containers", "podUID", runningPod.ID)
				one := int64(1)
				kl.podWorkers.UpdatePod(UpdatePodOptions{
					UpdateType: kubetypes.SyncPodKill,
					RunningPod: runningPod,
					KillPodOptions: &KillPodOptions{
						PodTerminationGracePeriodSecondsOverride: &one,
					},
				})
			}
		}
	}

	// Remove orphaned pod statuses not in the total list of known config pods
	klog.V(3).InfoS("Clean up orphaned pod statuses")
	kl.removeOrphanedPodStatuses(allPods, mirrorPods)
	// Note that we just killed the unwanted pods. This may not have reflected
	// in the cache. We need to bypass the cache to get the latest set of
	// running pods to clean up the volumes.
	// TODO: Evaluate the performance impact of bypassing the runtime cache.
	runningRuntimePods, err = kl.containerRuntime.GetPods(false)
	if err != nil {
		klog.ErrorS(err, "Error listing containers")
		return err
	}

	// Remove orphaned volumes from pods that are known not to have any
	// containers. Note that we pass all pods (including terminated pods) to
	// the function, so that we don't remove volumes associated with terminated
	// but not yet deleted pods.
	// TODO: this method could more aggressively cleanup terminated pods
	// in the future (volumes, mount dirs, logs, and containers could all be
	// better separated)
	klog.V(3).InfoS("Clean up orphaned pod directories")
	err = kl.cleanupOrphanedPodDirs(allPods, runningRuntimePods)
	if err != nil {
		// We want all cleanup tasks to be run even if one of them failed. So
		// we just log an error here and continue other cleanup tasks.
		// This also applies to the other clean up tasks.
		klog.ErrorS(err, "Failed cleaning up orphaned pod directories")
	}

	// Remove any orphaned mirror pods (mirror pods are tracked by name via the
	// pod worker)
	klog.V(3).InfoS("Clean up orphaned mirror pods")
	kl.deleteOrphanedMirrorPods()

	// Remove any cgroups in the hierarchy for pods that are definitely no longer
	// running (not in the container runtime).
	if kl.cgroupsPerQOS {
		pcm := kl.containerManager.NewPodContainerManager()
		klog.V(3).InfoS("Clean up orphaned pod cgroups")
		kl.cleanupOrphanedPodCgroups(pcm, cgroupPods, possiblyRunningPods)
	}

	kl.backOff.GC()
	return nil
}

// validateContainerLogStatus returns the container ID for the desired container to retrieve logs for, based on the state
// of the container. The previous flag will only return the logs for the last terminated container, otherwise, the current
// running container is preferred over a previous termination. If info about the container is not available then a specific
// error is returned to the end user.
func (kl *Kubelet) validateContainerLogStatus(podName string, podStatus *v1.PodStatus, containerName string, previous bool) (containerID kubecontainer.ContainerID, err error) {
	var cID string

	cStatus, found := podutil.GetContainerStatus(podStatus.ContainerStatuses, containerName)
	if !found {
		cStatus, found = podutil.GetContainerStatus(podStatus.InitContainerStatuses, containerName)
	}
	if !found && utilfeature.DefaultFeatureGate.Enabled(features.EphemeralContainers) {
		cStatus, found = podutil.GetContainerStatus(podStatus.EphemeralContainerStatuses, containerName)
	}
	if !found {
		return kubecontainer.ContainerID{}, fmt.Errorf("container %q in pod %q is not available", containerName, podName)
	}
	lastState := cStatus.LastTerminationState
	waiting, running, terminated := cStatus.State.Waiting, cStatus.State.Running, cStatus.State.Terminated

	switch {
	case previous:
		if lastState.Terminated == nil || lastState.Terminated.ContainerID == "" {
			return kubecontainer.ContainerID{}, fmt.Errorf("previous terminated container %q in pod %q not found", containerName, podName)
		}
		cID = lastState.Terminated.ContainerID

	case running != nil:
		cID = cStatus.ContainerID

	case terminated != nil:
		// in cases where the next container didn't start, terminated.ContainerID will be empty, so get logs from the lastState.Terminated.
		if terminated.ContainerID == "" {
			if lastState.Terminated != nil && lastState.Terminated.ContainerID != "" {
				cID = lastState.Terminated.ContainerID
			} else {
				return kubecontainer.ContainerID{}, fmt.Errorf("container %q in pod %q is terminated", containerName, podName)
			}
		} else {
			cID = terminated.ContainerID
		}

	case lastState.Terminated != nil:
		if lastState.Terminated.ContainerID == "" {
			return kubecontainer.ContainerID{}, fmt.Errorf("container %q in pod %q is terminated", containerName, podName)
		}
		cID = lastState.Terminated.ContainerID

	case waiting != nil:
		// output some info for the most common pending failures
		switch reason := waiting.Reason; reason {
		case images.ErrImagePull.Error():
			return kubecontainer.ContainerID{}, fmt.Errorf("container %q in pod %q is waiting to start: image can't be pulled", containerName, podName)
		case images.ErrImagePullBackOff.Error():
			return kubecontainer.ContainerID{}, fmt.Errorf("container %q in pod %q is waiting to start: trying and failing to pull image", containerName, podName)
		default:
			return kubecontainer.ContainerID{}, fmt.Errorf("container %q in pod %q is waiting to start: %v", containerName, podName, reason)
		}
	default:
		// unrecognized state
		return kubecontainer.ContainerID{}, fmt.Errorf("container %q in pod %q is waiting to start - no logs yet", containerName, podName)
	}

	return kubecontainer.ParseContainerID(cID), nil
}

// GetKubeletContainerLogs returns logs from the container
// TODO: this method is returning logs of random container attempts, when it should be returning the most recent attempt
// or all of them.
func (kl *Kubelet) GetKubeletContainerLogs(ctx context.Context, podFullName, containerName string, logOptions *v1.PodLogOptions, stdout, stderr io.Writer) error {
	// Pod workers periodically write status to statusManager. If status is not
	// cached there, something is wrong (or kubelet just restarted and hasn't
	// caught up yet). Just assume the pod is not ready yet.
	name, namespace, err := kubecontainer.ParsePodFullName(podFullName)
	if err != nil {
		return fmt.Errorf("unable to parse pod full name %q: %v", podFullName, err)
	}

	pod, ok := kl.GetPodByName(namespace, name)
	if !ok {
		return fmt.Errorf("pod %q cannot be found - no logs available", name)
	}

	podUID := pod.UID
	if mirrorPod, ok := kl.podManager.GetMirrorPodByPod(pod); ok {
		podUID = mirrorPod.UID
	}
	podStatus, found := kl.statusManager.GetPodStatus(podUID)
	if !found {
		// If there is no cached status, use the status from the
		// apiserver. This is useful if kubelet has recently been
		// restarted.
		podStatus = pod.Status
	}

	// TODO: Consolidate the logic here with kuberuntime.GetContainerLogs, here we convert container name to containerID,
	// but inside kuberuntime we convert container id back to container name and restart count.
	// TODO: After separate container log lifecycle management, we should get log based on the existing log files
	// instead of container status.
	containerID, err := kl.validateContainerLogStatus(pod.Name, &podStatus, containerName, logOptions.Previous)
	if err != nil {
		return err
	}

	// Do a zero-byte write to stdout before handing off to the container runtime.
	// This ensures at least one Write call is made to the writer when copying starts,
	// even if we then block waiting for log output from the container.
	if _, err := stdout.Write([]byte{}); err != nil {
		return err
	}

	if kl.dockerLegacyService != nil {
		// dockerLegacyService should only be non-nil when we actually need it, so
		// inject it into the runtimeService.
		// TODO(random-liu): Remove this hack after deprecating unsupported log driver.
		return kl.dockerLegacyService.GetContainerLogs(ctx, pod, containerID, logOptions, stdout, stderr)
	}
	return kl.containerRuntime.GetContainerLogs(ctx, pod, containerID, logOptions, stdout, stderr)
}

// getPhase returns the phase of a pod given its container info.
func getPhase(spec *v1.PodSpec, info []v1.ContainerStatus) v1.PodPhase {
	pendingInitialization := 0
	failedInitialization := 0
	for _, container := range spec.InitContainers {
		containerStatus, ok := podutil.GetContainerStatus(info, container.Name)
		if !ok {
			pendingInitialization++
			continue
		}

		switch {
		case containerStatus.State.Running != nil:
			pendingInitialization++
		case containerStatus.State.Terminated != nil:
			if containerStatus.State.Terminated.ExitCode != 0 {
				failedInitialization++
			}
		case containerStatus.State.Waiting != nil:
			if containerStatus.LastTerminationState.Terminated != nil {
				if containerStatus.LastTerminationState.Terminated.ExitCode != 0 {
					failedInitialization++
				}
			} else {
				pendingInitialization++
			}
		default:
			pendingInitialization++
		}
	}

	unknown := 0
	running := 0
	waiting := 0
	stopped := 0
	succeeded := 0
	for _, container := range spec.Containers {
		containerStatus, ok := podutil.GetContainerStatus(info, container.Name)
		if !ok {
			unknown++
			continue
		}

		switch {
		case containerStatus.State.Running != nil:
			running++
		case containerStatus.State.Terminated != nil:
			stopped++
			if containerStatus.State.Terminated.ExitCode == 0 {
				succeeded++
			}
		case containerStatus.State.Waiting != nil:
			if containerStatus.LastTerminationState.Terminated != nil {
				stopped++
			} else {
				waiting++
			}
		default:
			unknown++
		}
	}

	if failedInitialization > 0 && spec.RestartPolicy == v1.RestartPolicyNever {
		return v1.PodFailed
	}

	switch {
	case pendingInitialization > 0:
		fallthrough
	case waiting > 0:
		klog.V(5).InfoS("Pod waiting > 0, pending")
		// One or more containers has not been started
		return v1.PodPending
	case running > 0 && unknown == 0:
		// All containers have been started, and at least
		// one container is running
		return v1.PodRunning
	case running == 0 && stopped > 0 && unknown == 0:
		// All containers are terminated
		if spec.RestartPolicy == v1.RestartPolicyAlways {
			// All containers are in the process of restarting
			return v1.PodRunning
		}
		if stopped == succeeded {
			// RestartPolicy is not Always, and all
			// containers are terminated in success
			return v1.PodSucceeded
		}
		if spec.RestartPolicy == v1.RestartPolicyNever {
			// RestartPolicy is Never, and all containers are
			// terminated with at least one in failure
			return v1.PodFailed
		}
		// RestartPolicy is OnFailure, and at least one in failure
		// and in the process of restarting
		return v1.PodRunning
	default:
		klog.V(5).InfoS("Pod default case, pending")
		return v1.PodPending
	}
}

// generateAPIPodStatus creates the final API pod status for a pod, given the
// internal pod status.
func (kl *Kubelet) generateAPIPodStatus(pod *v1.Pod, podStatus *kubecontainer.PodStatus) v1.PodStatus {
	klog.V(3).InfoS("Generating pod status", "pod", klog.KObj(pod))

	s := kl.convertStatusToAPIStatus(pod, podStatus)

	// check if an internal module has requested the pod is evicted.
	for _, podSyncHandler := range kl.PodSyncHandlers {
		if result := podSyncHandler.ShouldEvict(pod); result.Evict {
			s.Phase = v1.PodFailed
			s.Reason = result.Reason
			s.Message = result.Message
			return *s
		}
	}

	// Assume info is ready to process
	spec := &pod.Spec
	allStatus := append(append([]v1.ContainerStatus{}, s.ContainerStatuses...), s.InitContainerStatuses...)
	s.Phase = getPhase(spec, allStatus)
	klog.V(4).InfoS("Got phase for pod", "pod", klog.KObj(pod), "phase", s.Phase)
	// Check for illegal phase transition
	if pod.Status.Phase == v1.PodFailed || pod.Status.Phase == v1.PodSucceeded {
		// API server shows terminal phase; transitions are not allowed
		if s.Phase != pod.Status.Phase {
			klog.ErrorS(nil, "Pod attempted illegal phase transition", "pod", klog.KObj(pod), "originalStatusPhase", pod.Status.Phase, "apiStatusPhase", s.Phase, "apiStatus", s)
			// Force back to phase from the API server
			s.Phase = pod.Status.Phase
		}
	}
	kl.probeManager.UpdatePodStatus(pod.UID, s)
	s.Conditions = append(s.Conditions, status.GeneratePodInitializedCondition(spec, s.InitContainerStatuses, s.Phase))
	s.Conditions = append(s.Conditions, status.GeneratePodReadyCondition(spec, s.Conditions, s.ContainerStatuses, s.Phase))
	s.Conditions = append(s.Conditions, status.GenerateContainersReadyCondition(spec, s.ContainerStatuses, s.Phase))
	// Status manager will take care of the LastTransitionTimestamp, either preserve
	// the timestamp from apiserver, or set a new one. When kubelet sees the pod,
	// `PodScheduled` condition must be true.
	s.Conditions = append(s.Conditions, v1.PodCondition{
		Type:   v1.PodScheduled,
		Status: v1.ConditionTrue,
	})

	if kl.kubeClient != nil {
		hostIPs, err := kl.getHostIPsAnyWay()
		if err != nil {
			klog.V(4).InfoS("Cannot get host IPs", "err", err)
		} else {
			s.HostIP = hostIPs[0].String()
			if kubecontainer.IsHostNetworkPod(pod) && s.PodIP == "" {
				s.PodIP = hostIPs[0].String()
				s.PodIPs = []v1.PodIP{{IP: s.PodIP}}
				if utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) && len(hostIPs) == 2 {
					s.PodIPs = append(s.PodIPs, v1.PodIP{IP: hostIPs[1].String()})
				}
			}
		}
	}

	return *s
}

// sortPodIPs return the PodIPs sorted and truncated by the cluster IP family preference.
// The runtime pod status may have an arbitrary number of IPs, in an arbitrary order.
// PodIPs are obtained by: func (m *kubeGenericRuntimeManager) determinePodSandboxIPs()
// Pick out the first returned IP of the same IP family as the node IP
// first, followed by the first IP of the opposite IP family (if any)
// and use them for the Pod.Status.PodIPs and the Downward API environment variables
func (kl *Kubelet) sortPodIPs(podIPs []string) []string {
	ips := make([]string, 0, 2)
	var validPrimaryIP, validSecondaryIP func(ip string) bool
	if len(kl.nodeIPs) == 0 || utilnet.IsIPv4(kl.nodeIPs[0]) {
		validPrimaryIP = utilnet.IsIPv4String
		validSecondaryIP = utilnet.IsIPv6String
	} else {
		validPrimaryIP = utilnet.IsIPv6String
		validSecondaryIP = utilnet.IsIPv4String
	}
	for _, ip := range podIPs {
		if validPrimaryIP(ip) {
			ips = append(ips, ip)
			break
		}
	}
	for _, ip := range podIPs {
		if validSecondaryIP(ip) {
			ips = append(ips, ip)
			break
		}
	}
	return ips
}

// convertStatusToAPIStatus creates an api PodStatus for the given pod from
// the given internal pod status.  It is purely transformative and does not
// alter the kubelet state at all.
func (kl *Kubelet) convertStatusToAPIStatus(pod *v1.Pod, podStatus *kubecontainer.PodStatus) *v1.PodStatus {
	var apiPodStatus v1.PodStatus

	// copy pod status IPs to avoid race conditions with PodStatus #102806
	podIPs := make([]string, len(podStatus.IPs))
	for j, ip := range podStatus.IPs {
		podIPs[j] = ip
	}

	// make podIPs order match node IP family preference #97979
	podIPs = kl.sortPodIPs(podIPs)
	for _, ip := range podIPs {
		apiPodStatus.PodIPs = append(apiPodStatus.PodIPs, v1.PodIP{IP: ip})
	}
	if len(apiPodStatus.PodIPs) > 0 {
		apiPodStatus.PodIP = apiPodStatus.PodIPs[0].IP
	}

	// set status for Pods created on versions of kube older than 1.6
	apiPodStatus.QOSClass = v1qos.GetPodQOS(pod)

	oldPodStatus, found := kl.statusManager.GetPodStatus(pod.UID)
	if !found {
		oldPodStatus = pod.Status
	}

	apiPodStatus.ContainerStatuses = kl.convertToAPIContainerStatuses(
		pod, podStatus,
		oldPodStatus.ContainerStatuses,
		pod.Spec.Containers,
		len(pod.Spec.InitContainers) > 0,
		false,
	)
	apiPodStatus.InitContainerStatuses = kl.convertToAPIContainerStatuses(
		pod, podStatus,
		oldPodStatus.InitContainerStatuses,
		pod.Spec.InitContainers,
		len(pod.Spec.InitContainers) > 0,
		true,
	)
	if utilfeature.DefaultFeatureGate.Enabled(features.EphemeralContainers) {
		var ecSpecs []v1.Container
		for i := range pod.Spec.EphemeralContainers {
			ecSpecs = append(ecSpecs, v1.Container(pod.Spec.EphemeralContainers[i].EphemeralContainerCommon))
		}

		// #80875: By now we've iterated podStatus 3 times. We could refactor this to make a single
		// pass through podStatus.ContainerStatuses
		apiPodStatus.EphemeralContainerStatuses = kl.convertToAPIContainerStatuses(
			pod, podStatus,
			oldPodStatus.EphemeralContainerStatuses,
			ecSpecs,
			len(pod.Spec.InitContainers) > 0,
			false,
		)
	}

	// Preserves conditions not controlled by kubelet
	for _, c := range pod.Status.Conditions {
		if !kubetypes.PodConditionByKubelet(c.Type) {
			apiPodStatus.Conditions = append(apiPodStatus.Conditions, c)
		}
	}

	return &apiPodStatus
}

// convertToAPIContainerStatuses converts the given internal container
// statuses into API container statuses.
func (kl *Kubelet) convertToAPIContainerStatuses(pod *v1.Pod, podStatus *kubecontainer.PodStatus, previousStatus []v1.ContainerStatus, containers []v1.Container, hasInitContainers, isInitContainer bool) []v1.ContainerStatus {
	convertContainerStatus := func(cs *kubecontainer.Status, oldStatus *v1.ContainerStatus) *v1.ContainerStatus {
		cid := cs.ID.String()
		status := &v1.ContainerStatus{
			Name:         cs.Name,
			RestartCount: int32(cs.RestartCount),
			Image:        cs.Image,
			ImageID:      cs.ImageID,
			ContainerID:  cid,
		}
		switch {
		case cs.State == kubecontainer.ContainerStateRunning:
			status.State.Running = &v1.ContainerStateRunning{StartedAt: metav1.NewTime(cs.StartedAt)}
		case cs.State == kubecontainer.ContainerStateCreated:
			// Treat containers in the "created" state as if they are exited.
			// The pod workers are supposed start all containers it creates in
			// one sync (syncPod) iteration. There should not be any normal
			// "created" containers when the pod worker generates the status at
			// the beginning of a sync iteration.
			fallthrough
		case cs.State == kubecontainer.ContainerStateExited:
			status.State.Terminated = &v1.ContainerStateTerminated{
				ExitCode:    int32(cs.ExitCode),
				Reason:      cs.Reason,
				Message:     cs.Message,
				StartedAt:   metav1.NewTime(cs.StartedAt),
				FinishedAt:  metav1.NewTime(cs.FinishedAt),
				ContainerID: cid,
			}

		case cs.State == kubecontainer.ContainerStateUnknown &&
			oldStatus != nil && // we have an old status
			oldStatus.State.Running != nil: // our previous status was running
			// if this happens, then we know that this container was previously running and isn't anymore (assuming the CRI isn't failing to return running containers).
			// you can imagine this happening in cases where a container failed and the kubelet didn't ask about it in time to see the result.
			// in this case, the container should not to into waiting state immediately because that can make cases like runonce pods actually run
			// twice. "container never ran" is different than "container ran and failed".  This is handled differently in the kubelet
			// and it is handled differently in higher order logic like crashloop detection and handling
			status.State.Terminated = &v1.ContainerStateTerminated{
				Reason:   "ContainerStatusUnknown",
				Message:  "The container could not be located when the pod was terminated",
				ExitCode: 137, // this code indicates an error
			}
			// the restart count normally comes from the CRI (see near the top of this method), but since this is being added explicitly
			// for the case where the CRI did not return a status, we need to manually increment the restart count to be accurate.
			status.RestartCount = oldStatus.RestartCount + 1

		default:
			// this collapses any unknown state to container waiting.  If any container is waiting, then the pod status moves to pending even if it is running.
			// if I'm reading this correctly, then any failure to read status on any container results in the entire pod going pending even if the containers
			// are actually running.
			// see https://github.com/kubernetes/kubernetes/blob/5d1b3e26af73dde33ecb6a3e69fb5876ceab192f/pkg/kubelet/kuberuntime/kuberuntime_container.go#L497 to
			// https://github.com/kubernetes/kubernetes/blob/8976e3620f8963e72084971d9d4decbd026bf49f/pkg/kubelet/kuberuntime/helpers.go#L58-L71
			// and interpreted here https://github.com/kubernetes/kubernetes/blob/b27e78f590a0d43e4a23ca3b2bf1739ca4c6e109/pkg/kubelet/kubelet_pods.go#L1434-L1439
			status.State.Waiting = &v1.ContainerStateWaiting{}
		}
		return status
	}

	// Fetch old containers statuses from old pod status.
	oldStatuses := make(map[string]v1.ContainerStatus, len(containers))
	for _, status := range previousStatus {
		oldStatuses[status.Name] = status
	}

	// Set all container statuses to default waiting state
	statuses := make(map[string]*v1.ContainerStatus, len(containers))
	defaultWaitingState := v1.ContainerState{Waiting: &v1.ContainerStateWaiting{Reason: ContainerCreating}}
	if hasInitContainers {
		defaultWaitingState = v1.ContainerState{Waiting: &v1.ContainerStateWaiting{Reason: PodInitializing}}
	}

	for _, container := range containers {
		status := &v1.ContainerStatus{
			Name:  container.Name,
			Image: container.Image,
			State: defaultWaitingState,
		}
		oldStatus, found := oldStatuses[container.Name]
		if found {
			if oldStatus.State.Terminated != nil {
				status = &oldStatus
			} else {
				// Apply some values from the old statuses as the default values.
				status.RestartCount = oldStatus.RestartCount
				status.LastTerminationState = oldStatus.LastTerminationState
			}
		}
		statuses[container.Name] = status
	}

	for _, container := range containers {
		found := false
		for _, cStatus := range podStatus.ContainerStatuses {
			if container.Name == cStatus.Name {
				found = true
				break
			}
		}
		if found {
			continue
		}
		// if no container is found, then assuming it should be waiting seems plausible, but the status code requires
		// that a previous termination be present.  If we're offline long enough or something removed the container, then
		// the previous termination may not be present.  This next code block ensures that if the container was previously running
		// then when that container status disappears, we can infer that it terminated even if we don't know the status code.
		// By setting the lasttermination state we are able to leave the container status waiting and present more accurate
		// data via the API.

		oldStatus, ok := oldStatuses[container.Name]
		if !ok {
			continue
		}
		if oldStatus.State.Terminated != nil {
			// if the old container status was terminated, the lasttermination status is correct
			continue
		}
		if oldStatus.State.Running == nil {
			// if the old container status isn't running, then waiting is an appropriate status and we have nothing to do
			continue
		}

		// If we're here, we know the pod was previously running, but doesn't have a terminated status. We will check now to
		// see if it's in a pending state.
		status := statuses[container.Name]
		// If the status we're about to write indicates the default, the Waiting status will force this pod back into Pending.
		// That isn't true, we know the pod was previously running.
		isDefaultWaitingStatus := status.State.Waiting != nil && status.State.Waiting.Reason == ContainerCreating
		if hasInitContainers {
			isDefaultWaitingStatus = status.State.Waiting != nil && status.State.Waiting.Reason == PodInitializing
		}
		if !isDefaultWaitingStatus {
			// the status was written, don't override
			continue
		}
		if status.LastTerminationState.Terminated != nil {
			// if we already have a termination state, nothing to do
			continue
		}

		// setting this value ensures that we show as stopped here, not as waiting:
		// https://github.com/kubernetes/kubernetes/blob/90c9f7b3e198e82a756a68ffeac978a00d606e55/pkg/kubelet/kubelet_pods.go#L1440-L1445
		// This prevents the pod from becoming pending
		status.LastTerminationState.Terminated = &v1.ContainerStateTerminated{
			Reason:   "ContainerStatusUnknown",
			Message:  "The container could not be located when the pod was deleted.  The container used to be Running",
			ExitCode: 137,
		}

		// If the pod was not deleted, then it's been restarted. Increment restart count.
		if pod.DeletionTimestamp == nil {
			status.RestartCount += 1
		}

		statuses[container.Name] = status
	}

	// Copy the slice before sorting it
	containerStatusesCopy := make([]*kubecontainer.Status, len(podStatus.ContainerStatuses))
	copy(containerStatusesCopy, podStatus.ContainerStatuses)

	// Make the latest container status comes first.
	sort.Sort(sort.Reverse(kubecontainer.SortContainerStatusesByCreationTime(containerStatusesCopy)))
	// Set container statuses according to the statuses seen in pod status
	containerSeen := map[string]int{}
	for _, cStatus := range containerStatusesCopy {
		cName := cStatus.Name
		if _, ok := statuses[cName]; !ok {
			// This would also ignore the infra container.
			continue
		}
		if containerSeen[cName] >= 2 {
			continue
		}
		var oldStatusPtr *v1.ContainerStatus
		if oldStatus, ok := oldStatuses[cName]; ok {
			oldStatusPtr = &oldStatus
		}
		status := convertContainerStatus(cStatus, oldStatusPtr)
		if containerSeen[cName] == 0 {
			statuses[cName] = status
		} else {
			statuses[cName].LastTerminationState = status.State
		}
		containerSeen[cName] = containerSeen[cName] + 1
	}

	// Handle the containers failed to be started, which should be in Waiting state.
	for _, container := range containers {
		if isInitContainer {
			// If the init container is terminated with exit code 0, it won't be restarted.
			// TODO(random-liu): Handle this in a cleaner way.
			s := podStatus.FindContainerStatusByName(container.Name)
			if s != nil && s.State == kubecontainer.ContainerStateExited && s.ExitCode == 0 {
				continue
			}
		}
		// If a container should be restarted in next syncpod, it is *Waiting*.
		if !kubecontainer.ShouldContainerBeRestarted(&container, pod, podStatus) {
			continue
		}
		status := statuses[container.Name]
		reason, ok := kl.reasonCache.Get(pod.UID, container.Name)
		if !ok {
			// In fact, we could also apply Waiting state here, but it is less informative,
			// and the container will be restarted soon, so we prefer the original state here.
			// Note that with the current implementation of ShouldContainerBeRestarted the original state here
			// could be:
			//   * Waiting: There is no associated historical container and start failure reason record.
			//   * Terminated: The container is terminated.
			continue
		}
		if status.State.Terminated != nil {
			status.LastTerminationState = status.State
		}
		status.State = v1.ContainerState{
			Waiting: &v1.ContainerStateWaiting{
				Reason:  reason.Err.Error(),
				Message: reason.Message,
			},
		}
		statuses[container.Name] = status
	}

	// Sort the container statuses since clients of this interface expect the list
	// of containers in a pod has a deterministic order.
	if isInitContainer {
		return kubetypes.SortStatusesOfInitContainers(pod, statuses)
	}
	var containerStatuses []v1.ContainerStatus
	for _, status := range statuses {
		containerStatuses = append(containerStatuses, *status)
	}

	sort.Sort(kubetypes.SortedContainerStatuses(containerStatuses))
	return containerStatuses
}

// ServeLogs returns logs of current machine.
func (kl *Kubelet) ServeLogs(w http.ResponseWriter, req *http.Request) {
	// TODO: whitelist logs we are willing to serve
	kl.logServer.ServeHTTP(w, req)
}

// findContainer finds and returns the container with the given pod ID, full name, and container name.
// It returns nil if not found.
func (kl *Kubelet) findContainer(podFullName string, podUID types.UID, containerName string) (*kubecontainer.Container, error) {
	pods, err := kl.containerRuntime.GetPods(false)
	if err != nil {
		return nil, err
	}
	// Resolve and type convert back again.
	// We need the static pod UID but the kubecontainer API works with types.UID.
	podUID = types.UID(kl.podManager.TranslatePodUID(podUID))
	pod := kubecontainer.Pods(pods).FindPod(podFullName, podUID)
	return pod.FindContainerByName(containerName), nil
}

// RunInContainer runs a command in a container, returns the combined stdout, stderr as an array of bytes
func (kl *Kubelet) RunInContainer(podFullName string, podUID types.UID, containerName string, cmd []string) ([]byte, error) {
	container, err := kl.findContainer(podFullName, podUID, containerName)
	if err != nil {
		return nil, err
	}
	if container == nil {
		return nil, fmt.Errorf("container not found (%q)", containerName)
	}
	// TODO(tallclair): Pass a proper timeout value.
	return kl.runner.RunInContainer(container.ID, cmd, 0)
}

// GetExec gets the URL the exec will be served from, or nil if the Kubelet will serve it.
func (kl *Kubelet) GetExec(podFullName string, podUID types.UID, containerName string, cmd []string, streamOpts remotecommandserver.Options) (*url.URL, error) {
	container, err := kl.findContainer(podFullName, podUID, containerName)
	if err != nil {
		return nil, err
	}
	if container == nil {
		return nil, fmt.Errorf("container not found (%q)", containerName)
	}
	return kl.streamingRuntime.GetExec(container.ID, cmd, streamOpts.Stdin, streamOpts.Stdout, streamOpts.Stderr, streamOpts.TTY)
}

// GetAttach gets the URL the attach will be served from, or nil if the Kubelet will serve it.
func (kl *Kubelet) GetAttach(podFullName string, podUID types.UID, containerName string, streamOpts remotecommandserver.Options) (*url.URL, error) {
	container, err := kl.findContainer(podFullName, podUID, containerName)
	if err != nil {
		return nil, err
	}
	if container == nil {
		return nil, fmt.Errorf("container %s not found in pod %s", containerName, podFullName)
	}

	// The TTY setting for attach must match the TTY setting in the initial container configuration,
	// since whether the process is running in a TTY cannot be changed after it has started.  We
	// need the api.Pod to get the TTY status.
	pod, found := kl.GetPodByFullName(podFullName)
	if !found || (string(podUID) != "" && pod.UID != podUID) {
		return nil, fmt.Errorf("pod %s not found", podFullName)
	}
	containerSpec := kubecontainer.GetContainerSpec(pod, containerName)
	if containerSpec == nil {
		return nil, fmt.Errorf("container %s not found in pod %s", containerName, podFullName)
	}
	tty := containerSpec.TTY

	return kl.streamingRuntime.GetAttach(container.ID, streamOpts.Stdin, streamOpts.Stdout, streamOpts.Stderr, tty)
}

// GetPortForward gets the URL the port-forward will be served from, or nil if the Kubelet will serve it.
func (kl *Kubelet) GetPortForward(podName, podNamespace string, podUID types.UID, portForwardOpts portforward.V4Options) (*url.URL, error) {
	pods, err := kl.containerRuntime.GetPods(false)
	if err != nil {
		return nil, err
	}
	// Resolve and type convert back again.
	// We need the static pod UID but the kubecontainer API works with types.UID.
	podUID = types.UID(kl.podManager.TranslatePodUID(podUID))
	podFullName := kubecontainer.BuildPodFullName(podName, podNamespace)
	pod := kubecontainer.Pods(pods).FindPod(podFullName, podUID)
	if pod.IsEmpty() {
		return nil, fmt.Errorf("pod not found (%q)", podFullName)
	}

	return kl.streamingRuntime.GetPortForward(podName, podNamespace, podUID, portForwardOpts.Ports)
}

// cleanupOrphanedPodCgroups removes cgroups that should no longer exist.
// it reconciles the cached state of cgroupPods with the specified list of runningPods
func (kl *Kubelet) cleanupOrphanedPodCgroups(pcm cm.PodContainerManager, cgroupPods map[types.UID]cm.CgroupName, possiblyRunningPods map[types.UID]sets.Empty) {
	// Iterate over all the found pods to verify if they should be running
	for uid, val := range cgroupPods {
		// if the pod is in the running set, its not a candidate for cleanup
		if _, ok := possiblyRunningPods[uid]; ok {
			continue
		}

		// If volumes have not been unmounted/detached, do not delete the cgroup
		// so any memory backed volumes don't have their charges propagated to the
		// parent croup.  If the volumes still exist, reduce the cpu shares for any
		// process in the cgroup to the minimum value while we wait.  if the kubelet
		// is configured to keep terminated volumes, we will delete the cgroup and not block.
		if podVolumesExist := kl.podVolumesExist(uid); podVolumesExist && !kl.keepTerminatedPodVolumes {
			klog.V(3).InfoS("Orphaned pod found, but volumes not yet removed.  Reducing cpu to minimum", "podUID", uid)
			if err := pcm.ReduceCPULimits(val); err != nil {
				klog.InfoS("Failed to reduce cpu time for pod pending volume cleanup", "podUID", uid, "err", err)
			}
			continue
		}
		klog.V(3).InfoS("Orphaned pod found, removing pod cgroups", "podUID", uid)
		// Destroy all cgroups of pod that should not be running,
		// by first killing all the attached processes to these cgroups.
		// We ignore errors thrown by the method, as the housekeeping loop would
		// again try to delete these unwanted pod cgroups
		go pcm.Destroy(val)
	}
}

// enableHostUserNamespace determines if the host user namespace should be used by the container runtime.
// Returns true if the pod is using a host pid, pic, or network namespace, the pod is using a non-namespaced
// capability, the pod contains a privileged container, or the pod has a host path volume.
//
// NOTE: when if a container shares any namespace with another container it must also share the user namespace
// or it will not have the correct capabilities in the namespace.  This means that host user namespace
// is enabled per pod, not per container.
func (kl *Kubelet) enableHostUserNamespace(pod *v1.Pod) bool {
	if kubecontainer.HasPrivilegedContainer(pod) || hasHostNamespace(pod) ||
		hasHostVolume(pod) || hasNonNamespacedCapability(pod) || kl.hasHostMountPVC(pod) {
		return true
	}
	return false
}

// hasNonNamespacedCapability returns true if MKNOD, SYS_TIME, or SYS_MODULE is requested for any container.
func hasNonNamespacedCapability(pod *v1.Pod) bool {
	for _, c := range pod.Spec.Containers {
		if c.SecurityContext != nil && c.SecurityContext.Capabilities != nil {
			for _, cap := range c.SecurityContext.Capabilities.Add {
				if cap == "MKNOD" || cap == "SYS_TIME" || cap == "SYS_MODULE" {
					return true
				}
			}
		}
	}

	return false
}

// hasHostVolume returns true if the pod spec has a HostPath volume.
func hasHostVolume(pod *v1.Pod) bool {
	for _, v := range pod.Spec.Volumes {
		if v.HostPath != nil {
			return true
		}
	}
	return false
}

// hasHostNamespace returns true if hostIPC, hostNetwork, or hostPID are set to true.
func hasHostNamespace(pod *v1.Pod) bool {
	if pod.Spec.SecurityContext == nil {
		return false
	}
	return pod.Spec.HostIPC || pod.Spec.HostNetwork || pod.Spec.HostPID
}

// hasHostMountPVC returns true if a PVC is referencing a HostPath volume.
func (kl *Kubelet) hasHostMountPVC(pod *v1.Pod) bool {
	for _, volume := range pod.Spec.Volumes {
		if volume.PersistentVolumeClaim != nil {
			pvc, err := kl.kubeClient.CoreV1().PersistentVolumeClaims(pod.Namespace).Get(context.TODO(), volume.PersistentVolumeClaim.ClaimName, metav1.GetOptions{})
			if err != nil {
				klog.InfoS("Unable to retrieve pvc", "pvc", klog.KRef(pod.Namespace, volume.PersistentVolumeClaim.ClaimName), "err", err)
				continue
			}
			if pvc != nil {
				referencedVolume, err := kl.kubeClient.CoreV1().PersistentVolumes().Get(context.TODO(), pvc.Spec.VolumeName, metav1.GetOptions{})
				if err != nil {
					klog.InfoS("Unable to retrieve pv", "pvName", pvc.Spec.VolumeName, "err", err)
					continue
				}
				if referencedVolume != nil && referencedVolume.Spec.HostPath != nil {
					return true
				}
			}
		}
	}
	return false
}
