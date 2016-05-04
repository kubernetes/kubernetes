/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/cloudprovider"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/securitycontext"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/io"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/util/selinux"
	"k8s.io/kubernetes/pkg/util/sets"
	utilstrings "k8s.io/kubernetes/pkg/util/strings"
	"k8s.io/kubernetes/pkg/volume"
)

// This just exports required functions from kubelet proper, for use by volume
// plugins.
type volumeHost struct {
	kubelet *Kubelet
}

var _ volume.VolumeHost = &volumeHost{}

func (vh *volumeHost) GetPluginDir(pluginName string) string {
	return vh.kubelet.getPluginDir(pluginName)
}

func (vh *volumeHost) GetPodVolumeDir(podUID types.UID, pluginName string, volumeName string) string {
	return vh.kubelet.getPodVolumeDir(podUID, pluginName, volumeName)
}

func (vh *volumeHost) GetPodPluginDir(podUID types.UID, pluginName string) string {
	return vh.kubelet.getPodPluginDir(podUID, pluginName)
}

func (vh *volumeHost) GetKubeClient() clientset.Interface {
	return vh.kubelet.kubeClient
}

// NewWrapperMounter attempts to create a volume mounter
// from a volume Spec, pod and volume options.
// Returns a new volume Mounter or an error.
func (vh *volumeHost) NewWrapperMounter(volName string, spec volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	// The name of wrapper volume is set to "wrapped_{wrapped_volume_name}"
	wrapperVolumeName := "wrapped_" + volName
	if spec.Volume != nil {
		spec.Volume.Name = wrapperVolumeName
	}

	return vh.kubelet.newVolumeMounterFromPlugins(&spec, pod, opts)
}

// NewWrapperUnmounter attempts to create a volume unmounter
// from a volume name and pod uid.
// Returns a new volume Unmounter or an error.
func (vh *volumeHost) NewWrapperUnmounter(volName string, spec volume.Spec, podUID types.UID) (volume.Unmounter, error) {
	// The name of wrapper volume is set to "wrapped_{wrapped_volume_name}"
	wrapperVolumeName := "wrapped_" + volName
	if spec.Volume != nil {
		spec.Volume.Name = wrapperVolumeName
	}

	plugin, err := vh.kubelet.volumePluginMgr.FindPluginBySpec(&spec)
	if err != nil {
		return nil, err
	}

	return plugin.NewUnmounter(spec.Name(), podUID)
}

func (vh *volumeHost) GetCloudProvider() cloudprovider.Interface {
	return vh.kubelet.cloud
}

func (vh *volumeHost) GetMounter() mount.Interface {
	return vh.kubelet.mounter
}

func (vh *volumeHost) GetWriter() io.Writer {
	return vh.kubelet.writer
}

// Returns the hostname of the host kubelet is running on
func (vh *volumeHost) GetHostName() string {
	return vh.kubelet.hostname
}

// Compares the map of current volumes to the map of desired volumes.
// If an active volume does not have a respective desired volume, clean it up.
// This method is blocking:
// 1) it talks to API server to find volumes bound to persistent volume claims
// 2) it talks to cloud to detach volumes
func (kl *Kubelet) cleanupOrphanedVolumes(pods []*api.Pod, runningPods []*kubecontainer.Pod) error {
	desiredVolumes := kl.getDesiredVolumes(pods)
	currentVolumes := kl.getPodVolumesFromDisk()

	runningSet := sets.String{}
	for _, pod := range runningPods {
		runningSet.Insert(string(pod.ID))
	}

	for name, cleaner := range currentVolumes {
		if _, ok := desiredVolumes[name]; !ok {
			parts := strings.Split(name, "/")
			if runningSet.Has(parts[0]) {
				glog.Infof("volume %q, still has a container running %q, skipping teardown", name, parts[0])
				continue
			}
			//TODO (jonesdl) We should somehow differentiate between volumes that are supposed
			//to be deleted and volumes that are leftover after a crash.
			glog.Warningf("Orphaned volume %q found, tearing down volume", name)
			// TODO(yifan): Refactor this hacky string manipulation.
			kl.volumeManager.DeleteVolumes(types.UID(parts[0]))
			// Get path reference count
			refs, err := mount.GetMountRefs(kl.mounter, cleaner.Unmounter.GetPath())
			if err != nil {
				return fmt.Errorf("Could not get mount path references %v", err)
			}
			//TODO (jonesdl) This should not block other kubelet synchronization procedures
			err = cleaner.Unmounter.TearDown()
			if err != nil {
				glog.Errorf("Could not tear down volume %q: %v", name, err)
			}

			// volume is unmounted.  some volumes also require detachment from the node.
			if cleaner.Detacher != nil && len(refs) == 1 {
				detacher := *cleaner.Detacher
				err = detacher.Detach()
				if err != nil {
					glog.Errorf("Could not detach volume %q: %v", name, err)
				}
			}
		}
	}
	return nil
}

// Stores all volumes defined by the set of pods into a map.
// It stores real volumes there, i.e. persistent volume claims are resolved
// to volumes that are bound to them.
// Keys for each entry are in the format (POD_ID)/(VOLUME_NAME)
func (kl *Kubelet) getDesiredVolumes(pods []*api.Pod) map[string]api.Volume {
	desiredVolumes := make(map[string]api.Volume)
	for _, pod := range pods {
		for _, volume := range pod.Spec.Volumes {
			volumeName, err := kl.resolveVolumeName(pod, &volume)
			if err != nil {
				glog.V(3).Infof("%v", err)
				// Ignore the error and hope it's resolved next time
				continue
			}
			identifier := path.Join(string(pod.UID), volumeName)
			desiredVolumes[identifier] = volume
		}
	}
	return desiredVolumes
}

// cleanupOrphanedPodDirs removes the volumes of pods that should not be
// running and that have no containers running.
func (kl *Kubelet) cleanupOrphanedPodDirs(pods []*api.Pod, runningPods []*kubecontainer.Pod) error {
	active := sets.NewString()
	for _, pod := range pods {
		active.Insert(string(pod.UID))
	}
	for _, pod := range runningPods {
		active.Insert(string(pod.ID))
	}

	found, err := kl.listPodsFromDisk()
	if err != nil {
		return err
	}
	errlist := []error{}
	for _, uid := range found {
		if active.Has(string(uid)) {
			continue
		}
		if volumes, err := kl.getPodVolumes(uid); err != nil || len(volumes) != 0 {
			glog.V(3).Infof("Orphaned pod %q found, but volumes are not cleaned up; err: %v, volumes: %v ", uid, err, volumes)
			continue
		}

		glog.V(3).Infof("Orphaned pod %q found, removing", uid)
		if err := os.RemoveAll(kl.getPodDir(uid)); err != nil {
			errlist = append(errlist, err)
		}
	}
	return utilerrors.NewAggregate(errlist)
}

// resolveVolumeName returns the name of the persistent volume (PV) claimed by
// a persistent volume claim (PVC) or an error if the claim is not bound.
// Returns nil if the volume does not use a PVC.
func (kl *Kubelet) resolveVolumeName(pod *api.Pod, volume *api.Volume) (string, error) {
	claimSource := volume.VolumeSource.PersistentVolumeClaim
	if claimSource != nil {
		// resolve real volume behind the claim
		claim, err := kl.kubeClient.Core().PersistentVolumeClaims(pod.Namespace).Get(claimSource.ClaimName)
		if err != nil {
			return "", fmt.Errorf("Cannot find claim %s/%s for volume %s", pod.Namespace, claimSource.ClaimName, volume.Name)
		}
		if claim.Status.Phase != api.ClaimBound {
			return "", fmt.Errorf("Claim for volume %s/%s is not bound yet", pod.Namespace, claimSource.ClaimName)
		}
		// Use the real bound volume instead of PersistentVolume.Name
		return claim.Spec.VolumeName, nil
	}
	return volume.Name, nil
}

// mountExternalVolumes mounts the volumes declared in a pod, attaching them
// to the host if necessary, and returns a map containing information about
// the volumes for the pod or an error.  This method is run multiple times,
// and requires that implementations of Attach() and SetUp() be idempotent.
//
// Note, in the future, the attach-detach controller will handle attaching and
// detaching volumes; this call site will be maintained for backward-
// compatibility with current behavior of static pods and pods created via the
// Kubelet's http API.
func (kl *Kubelet) mountExternalVolumes(pod *api.Pod) (kubecontainer.VolumeMap, error) {
	podVolumes := make(kubecontainer.VolumeMap)
	for i := range pod.Spec.Volumes {
		volSpec := &pod.Spec.Volumes[i]
		var fsGroup *int64
		if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.FSGroup != nil {
			fsGroup = pod.Spec.SecurityContext.FSGroup
		}

		rootContext, err := kl.getRootDirContext()
		if err != nil {
			return nil, err
		}

		// Try to use a plugin for this volume.
		internal := volume.NewSpecFromVolume(volSpec)
		mounter, err := kl.newVolumeMounterFromPlugins(internal, pod, volume.VolumeOptions{RootContext: rootContext})
		if err != nil {
			glog.Errorf("Could not create volume mounter for pod %s: %v", pod.UID, err)
			return nil, err
		}

		// some volumes require attachment before mounter's setup.
		// The plugin can be nil, but non-nil errors are legitimate errors.
		// For non-nil plugins, Attachment to a node is required before Mounter's setup.
		attacher, err := kl.newVolumeAttacherFromPlugins(internal, pod, volume.VolumeOptions{RootContext: rootContext})
		if err != nil {
			glog.Errorf("Could not create volume attacher for pod %s: %v", pod.UID, err)
			return nil, err
		}
		if attacher != nil {
			err = attacher.Attach()
			if err != nil {
				return nil, err
			}
		}

		err = mounter.SetUp(fsGroup)
		if err != nil {
			return nil, err
		}
		podVolumes[volSpec.Name] = kubecontainer.VolumeInfo{Mounter: mounter}
	}
	return podVolumes, nil
}

// makeMounts determines the mount points for the given container.
func makeMounts(pod *api.Pod, podDir string, container *api.Container, hostName, hostDomain, podIP string, podVolumes kubecontainer.VolumeMap) ([]kubecontainer.Mount, error) {
	// Kubernetes only mounts on /etc/hosts if :
	// - container does not use hostNetwork and
	// - container is not a infrastructure(pause) container
	// - container is not already mounting on /etc/hosts
	// When the pause container is being created, its IP is still unknown. Hence, PodIP will not have been set.
	mountEtcHostsFile := (pod.Spec.SecurityContext == nil || !pod.Spec.SecurityContext.HostNetwork) && len(podIP) > 0
	glog.V(3).Infof("container: %v/%v/%v podIP: %q creating hosts mount: %v", pod.Namespace, pod.Name, container.Name, podIP, mountEtcHostsFile)
	mounts := []kubecontainer.Mount{}
	for _, mount := range container.VolumeMounts {
		mountEtcHostsFile = mountEtcHostsFile && (mount.MountPath != etcHostsPath)
		vol, ok := podVolumes[mount.Name]
		if !ok {
			glog.Warningf("Mount cannot be satisified for container %q, because the volume is missing: %q", container.Name, mount)
			continue
		}

		relabelVolume := false
		// If the volume supports SELinux and it has not been
		// relabeled already and it is not a read-only volume,
		// relabel it and mark it as labeled
		if vol.Mounter.GetAttributes().Managed && vol.Mounter.GetAttributes().SupportsSELinux && !vol.SELinuxLabeled {
			vol.SELinuxLabeled = true
			relabelVolume = true
		}
		mounts = append(mounts, kubecontainer.Mount{
			Name:           mount.Name,
			ContainerPath:  mount.MountPath,
			HostPath:       vol.Mounter.GetPath(),
			ReadOnly:       mount.ReadOnly,
			SELinuxRelabel: relabelVolume,
		})
	}
	if mountEtcHostsFile {
		hostsMount, err := makeHostsMount(podDir, podIP, hostName, hostDomain)
		if err != nil {
			return nil, err
		}
		mounts = append(mounts, *hostsMount)
	}
	return mounts, nil
}

// makeHostsMount makes the mountpoint for the hosts file that the containers
// in a pod are injected with.
func makeHostsMount(podDir, podIP, hostName, hostDomainName string) (*kubecontainer.Mount, error) {
	hostsFilePath := path.Join(podDir, "etc-hosts")
	if err := ensureHostsFile(hostsFilePath, podIP, hostName, hostDomainName); err != nil {
		return nil, err
	}
	return &kubecontainer.Mount{
		Name:          "k8s-managed-etc-hosts",
		ContainerPath: etcHostsPath,
		HostPath:      hostsFilePath,
		ReadOnly:      false,
	}, nil
}

// ensureHostsFile ensures that the given host file has an up-to-date ip, host
// name, and domain name.
func ensureHostsFile(fileName, hostIP, hostName, hostDomainName string) error {
	if _, err := os.Stat(fileName); os.IsExist(err) {
		glog.V(4).Infof("kubernetes-managed etc-hosts file exits. Will not be recreated: %q", fileName)
		return nil
	}
	var buffer bytes.Buffer
	buffer.WriteString("# Kubernetes-managed hosts file.\n")
	buffer.WriteString("127.0.0.1\tlocalhost\n")                      // ipv4 localhost
	buffer.WriteString("::1\tlocalhost ip6-localhost ip6-loopback\n") // ipv6 localhost
	buffer.WriteString("fe00::0\tip6-localnet\n")
	buffer.WriteString("fe00::0\tip6-mcastprefix\n")
	buffer.WriteString("fe00::1\tip6-allnodes\n")
	buffer.WriteString("fe00::2\tip6-allrouters\n")
	if len(hostDomainName) > 0 {
		buffer.WriteString(fmt.Sprintf("%s\t%s.%s\t%s\n", hostIP, hostName, hostDomainName, hostName))
	} else {
		buffer.WriteString(fmt.Sprintf("%s\t%s\n", hostIP, hostName))
	}
	return ioutil.WriteFile(fileName, buffer.Bytes(), 0644)
}

// relabelVolumes relabels SELinux volumes to match the pod's
// SELinuxOptions specification. This is only needed if the pod uses
// hostPID or hostIPC. Otherwise relabeling is delegated to docker.
func (kl *Kubelet) relabelVolumes(pod *api.Pod, volumes kubecontainer.VolumeMap) error {
	if pod.Spec.SecurityContext.SELinuxOptions == nil {
		return nil
	}

	rootDirContext, err := kl.getRootDirContext()
	if err != nil {
		return err
	}

	chconRunner := selinux.NewChconRunner()
	// Apply the pod's Level to the rootDirContext
	rootDirSELinuxOptions, err := securitycontext.ParseSELinuxOptions(rootDirContext)
	if err != nil {
		return err
	}

	rootDirSELinuxOptions.Level = pod.Spec.SecurityContext.SELinuxOptions.Level
	volumeContext := fmt.Sprintf("%s:%s:%s:%s", rootDirSELinuxOptions.User, rootDirSELinuxOptions.Role, rootDirSELinuxOptions.Type, rootDirSELinuxOptions.Level)

	for _, vol := range volumes {
		if vol.Mounter.GetAttributes().Managed && vol.Mounter.GetAttributes().SupportsSELinux {
			// Relabel the volume and its content to match the 'Level' of the pod
			err := filepath.Walk(vol.Mounter.GetPath(), func(path string, info os.FileInfo, err error) error {
				if err != nil {
					return err
				}
				return chconRunner.SetContext(path, volumeContext)
			})
			if err != nil {
				return err
			}
			vol.SELinuxLabeled = true
		}
	}
	return nil
}

type volumeTuple struct {
	Kind string
	Name string
}

// ListVolumesForPod returns a map of the volumes associated with the given pod
func (kl *Kubelet) ListVolumesForPod(podUID types.UID) (map[string]volume.Volume, bool) {
	result := map[string]volume.Volume{}
	vm, ok := kl.volumeManager.GetVolumes(podUID)
	if !ok {
		return result, false
	}
	for name, info := range vm {
		result[name] = info.Mounter
	}
	return result, true
}

// getPodVolumes examines the directory structure for a pod and returns
// information about the name and kind of each presently mounted volume, or an
// error.
func (kl *Kubelet) getPodVolumes(podUID types.UID) ([]*volumeTuple, error) {
	var volumes []*volumeTuple
	podVolDir := kl.getPodVolumesDir(podUID)
	volumeKindDirs, err := ioutil.ReadDir(podVolDir)
	if err != nil {
		glog.Errorf("Could not read directory %s: %v", podVolDir, err)
	}
	for _, volumeKindDir := range volumeKindDirs {
		volumeKind := volumeKindDir.Name()
		volumeKindPath := path.Join(podVolDir, volumeKind)
		// ioutil.ReadDir exits without returning any healthy dir when encountering the first lstat error
		// but skipping dirs means no cleanup for healthy volumes. switching to a no-exit api solves this problem
		volumeNameDirs, volumeNameDirsStat, err := util.ReadDirNoExit(volumeKindPath)
		if err != nil {
			return []*volumeTuple{}, fmt.Errorf("could not read directory %s: %v", volumeKindPath, err)
		}
		for i, volumeNameDir := range volumeNameDirs {
			if volumeNameDir != nil {
				volumes = append(volumes, &volumeTuple{Kind: volumeKind, Name: volumeNameDir.Name()})
			} else {
				glog.Errorf("Could not read directory %s: %v", podVolDir, volumeNameDirsStat[i])
			}
		}
	}
	return volumes, nil
}

// cleaner is a union struct to allow separating detaching from the cleaner.
// some volumes require detachment but not all.  Unmounter cannot be nil but Detacher is optional.
type cleaner struct {
	Unmounter volume.Unmounter
	Detacher  *volume.Detacher
}

// getPodVolumesFromDisk examines directory structure to determine volumes that
// are presently active and mounted. Returns a union struct containing a volume.Unmounter
// and potentially a volume.Detacher.
func (kl *Kubelet) getPodVolumesFromDisk() map[string]cleaner {
	currentVolumes := make(map[string]cleaner)
	podUIDs, err := kl.listPodsFromDisk()
	if err != nil {
		glog.Errorf("Could not get pods from disk: %v", err)
		return map[string]cleaner{}
	}
	// Find the volumes for each on-disk pod.
	for _, podUID := range podUIDs {
		volumes, err := kl.getPodVolumes(podUID)
		if err != nil {
			glog.Errorf("%v", err)
			continue
		}
		for _, volume := range volumes {
			identifier := fmt.Sprintf("%s/%s", podUID, volume.Name)
			glog.V(4).Infof("Making a volume.Unmounter for volume %s/%s of pod %s", volume.Kind, volume.Name, podUID)
			// TODO(thockin) This should instead return a reference to an extant
			// volume object, except that we don't actually hold on to pod specs
			// or volume objects.

			// Try to use a plugin for this volume.
			unmounter, err := kl.newVolumeUnmounterFromPlugins(volume.Kind, volume.Name, podUID)
			if err != nil {
				glog.Errorf("Could not create volume unmounter for %s: %v", volume.Name, err)
				continue
			}

			tuple := cleaner{Unmounter: unmounter}
			detacher, err := kl.newVolumeDetacherFromPlugins(volume.Kind, volume.Name, podUID)
			// plugin can be nil but a non-nil error is a legitimate error
			if err != nil {
				glog.Errorf("Could not create volume detacher for %s: %v", volume.Name, err)
				continue
			}
			if detacher != nil {
				tuple.Detacher = &detacher
			}
			currentVolumes[identifier] = tuple
		}
	}
	return currentVolumes
}

// newVolumeMounterFromPlugins attempts to find a plugin by volume spec, pod
// and volume options and then creates a Mounter.
// Returns a valid Unmounter or an error.
func (kl *Kubelet) newVolumeMounterFromPlugins(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Mounter, error) {
	plugin, err := kl.volumePluginMgr.FindPluginBySpec(spec)
	if err != nil {
		return nil, fmt.Errorf("can't use volume plugins for %s: %v", spec.Name(), err)
	}
	physicalMounter, err := plugin.NewMounter(spec, pod, opts)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate mounter for volume: %s using plugin: %s with a root cause: %v", spec.Name(), plugin.Name(), err)
	}
	glog.V(10).Infof("Used volume plugin %q to mount %s", plugin.Name(), spec.Name())
	return physicalMounter, nil
}

// newVolumeAttacherFromPlugins attempts to find a plugin from a volume spec
// and then create an Attacher.
// Returns:
//  - an attacher if one exists
//  - an error if no plugin was found for the volume
//    or the attacher failed to instantiate
//  - nil if there is no appropriate attacher for this volume
func (kl *Kubelet) newVolumeAttacherFromPlugins(spec *volume.Spec, pod *api.Pod, opts volume.VolumeOptions) (volume.Attacher, error) {
	plugin, err := kl.volumePluginMgr.FindAttachablePluginBySpec(spec)
	if err != nil {
		return nil, fmt.Errorf("can't use volume plugins for %s: %v", spec.Name(), err)
	}
	if plugin == nil {
		// Not found but not an error.
		return nil, nil
	}

	attacher, err := plugin.NewAttacher(spec)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate volume attacher for %s: %v", spec.Name(), err)
	}
	glog.V(3).Infof("Used volume plugin %q to attach %s/%s", plugin.Name(), spec.Name())
	return attacher, nil
}

// newVolumeUnmounterFromPlugins attempts to find a plugin by name and then
// create an Unmounter.
// Returns a valid Unmounter or an error.
func (kl *Kubelet) newVolumeUnmounterFromPlugins(kind string, name string, podUID types.UID) (volume.Unmounter, error) {
	plugName := utilstrings.UnescapeQualifiedNameForDisk(kind)
	plugin, err := kl.volumePluginMgr.FindPluginByName(plugName)
	if err != nil {
		// TODO: Maybe we should launch a cleanup of this dir?
		return nil, fmt.Errorf("can't use volume plugins for %s/%s: %v", podUID, kind, err)
	}

	unmounter, err := plugin.NewUnmounter(name, podUID)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate volume plugin for %s/%s: %v", podUID, kind, err)
	}
	glog.V(5).Infof("Used volume plugin %q to unmount %s/%s", plugin.Name(), podUID, kind)
	return unmounter, nil
}

// newVolumeDetacherFromPlugins attempts to find a plugin by a name and then
// create a Detacher.
// Returns:
//  - a detacher if one exists
//  - an error if no plugin was found for the volume
//    or the detacher failed to instantiate
//  - nil if there is no appropriate detacher for this volume
func (kl *Kubelet) newVolumeDetacherFromPlugins(kind string, name string, podUID types.UID) (volume.Detacher, error) {
	plugName := utilstrings.UnescapeQualifiedNameForDisk(kind)
	plugin, err := kl.volumePluginMgr.FindAttachablePluginByName(plugName)
	if err != nil {
		return nil, fmt.Errorf("can't use volume plugins for %s/%s: %v", podUID, kind, err)
	}
	if plugin == nil {
		// Not found but not an error.
		return nil, nil
	}

	detacher, err := plugin.NewDetacher(name, podUID)
	if err != nil {
		return nil, fmt.Errorf("failed to instantiate volume plugin for %s/%s: %v", podUID, kind, err)
	}
	glog.V(3).Infof("Used volume plugin %q to detach %s/%s", plugin.Name(), podUID, kind)
	return detacher, nil
}
