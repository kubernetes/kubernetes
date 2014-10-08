/*
Copyright 2014 Google Inc. All rights reserved.

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

package validation

import (
	"reflect"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	errs "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func validateVolumes(volumes []api.Volume) (util.StringSet, errs.ErrorList) {
	allErrs := errs.ErrorList{}

	allNames := util.StringSet{}
	for i := range volumes {
		vol := &volumes[i] // so we can set default values
		el := errs.ErrorList{}
		if vol.Source == nil {
			// TODO: Enforce that a source is set once we deprecate the implied form.
			vol.Source = &api.VolumeSource{
				EmptyDir: &api.EmptyDir{},
			}
		}
		el = validateSource(vol.Source).Prefix("source")
		if len(vol.Name) == 0 {
			el = append(el, errs.NewFieldRequired("name", vol.Name))
		} else if !util.IsDNSLabel(vol.Name) {
			el = append(el, errs.NewFieldInvalid("name", vol.Name))
		} else if allNames.Has(vol.Name) {
			el = append(el, errs.NewFieldDuplicate("name", vol.Name))
		}
		if len(el) == 0 {
			allNames.Insert(vol.Name)
		} else {
			allErrs = append(allErrs, el.PrefixIndex(i)...)
		}
	}
	return allNames, allErrs
}

func validateSource(source *api.VolumeSource) errs.ErrorList {
	numVolumes := 0
	allErrs := errs.ErrorList{}
	if source.HostDir != nil {
		numVolumes++
		allErrs = append(allErrs, validateHostDir(source.HostDir).Prefix("hostDirectory")...)
	}
	if source.EmptyDir != nil {
		numVolumes++
		//EmptyDirs have nothing to validate
	}
	if source.GCEPersistentDisk != nil {
		numVolumes++
		allErrs = append(allErrs, validateGCEPersistentDisk(source.GCEPersistentDisk)...)
	}
	if numVolumes != 1 {
		allErrs = append(allErrs, errs.NewFieldInvalid("", source))
	}
	return allErrs
}

func validateHostDir(hostDir *api.HostDir) errs.ErrorList {
	allErrs := errs.ErrorList{}
	if hostDir.Path == "" {
		allErrs = append(allErrs, errs.NewNotFound("path", hostDir.Path))
	}
	return allErrs
}

var supportedPortProtocols = util.NewStringSet(string(api.ProtocolTCP), string(api.ProtocolUDP))

func validateGCEPersistentDisk(PD *api.GCEPersistentDisk) errs.ErrorList {
	allErrs := errs.ErrorList{}
	if PD.PDName == "" {
		allErrs = append(allErrs, errs.NewFieldInvalid("PD.PDName", PD.PDName))
	}
	if PD.FSType == "" {
		allErrs = append(allErrs, errs.NewFieldInvalid("PD.FSType", PD.FSType))
	}
	if PD.Partition < 0 || PD.Partition > 255 {
		allErrs = append(allErrs, errs.NewFieldInvalid("PD.Partition", PD.Partition))
	}
	return allErrs
}

func validatePorts(ports []api.Port) errs.ErrorList {
	allErrs := errs.ErrorList{}

	allNames := util.StringSet{}
	for i := range ports {
		pErrs := errs.ErrorList{}
		port := &ports[i] // so we can set default values
		if len(port.Name) > 0 {
			if len(port.Name) > 63 || !util.IsDNSLabel(port.Name) {
				pErrs = append(pErrs, errs.NewFieldInvalid("name", port.Name))
			} else if allNames.Has(port.Name) {
				pErrs = append(pErrs, errs.NewFieldDuplicate("name", port.Name))
			} else {
				allNames.Insert(port.Name)
			}
		}
		if port.ContainerPort == 0 {
			pErrs = append(pErrs, errs.NewFieldRequired("containerPort", port.ContainerPort))
		} else if !util.IsValidPortNum(port.ContainerPort) {
			pErrs = append(pErrs, errs.NewFieldInvalid("containerPort", port.ContainerPort))
		}
		if port.HostPort != 0 && !util.IsValidPortNum(port.HostPort) {
			pErrs = append(pErrs, errs.NewFieldInvalid("hostPort", port.HostPort))
		}
		if len(port.Protocol) == 0 {
			port.Protocol = "TCP"
		} else if !supportedPortProtocols.Has(strings.ToUpper(string(port.Protocol))) {
			pErrs = append(pErrs, errs.NewFieldNotSupported("protocol", port.Protocol))
		}
		allErrs = append(allErrs, pErrs.PrefixIndex(i)...)
	}
	return allErrs
}

func validateEnv(vars []api.EnvVar) errs.ErrorList {
	allErrs := errs.ErrorList{}

	for i := range vars {
		vErrs := errs.ErrorList{}
		ev := &vars[i] // so we can set default values
		if len(ev.Name) == 0 {
			vErrs = append(vErrs, errs.NewFieldRequired("name", ev.Name))
		}
		if !util.IsCIdentifier(ev.Name) {
			vErrs = append(vErrs, errs.NewFieldInvalid("name", ev.Name))
		}
		allErrs = append(allErrs, vErrs.PrefixIndex(i)...)
	}
	return allErrs
}

func validateVolumeMounts(mounts []api.VolumeMount, volumes util.StringSet) errs.ErrorList {
	allErrs := errs.ErrorList{}

	for i := range mounts {
		mErrs := errs.ErrorList{}
		mnt := &mounts[i] // so we can set default values
		if len(mnt.Name) == 0 {
			mErrs = append(mErrs, errs.NewFieldRequired("name", mnt.Name))
		} else if !volumes.Has(mnt.Name) {
			mErrs = append(mErrs, errs.NewNotFound("name", mnt.Name))
		}
		if len(mnt.MountPath) == 0 {
			mErrs = append(mErrs, errs.NewFieldRequired("mountPath", mnt.MountPath))
		}
		allErrs = append(allErrs, mErrs.PrefixIndex(i)...)
	}
	return allErrs
}

// AccumulateUniquePorts runs an extraction function on each Port of each Container,
// accumulating the results and returning an error if any ports conflict.
func AccumulateUniquePorts(containers []api.Container, accumulator map[int]bool, extract func(*api.Port) int) errs.ErrorList {
	allErrs := errs.ErrorList{}

	for ci := range containers {
		cErrs := errs.ErrorList{}
		ctr := &containers[ci]
		for pi := range ctr.Ports {
			port := extract(&ctr.Ports[pi])
			if port == 0 {
				continue
			}
			if accumulator[port] {
				cErrs = append(cErrs, errs.NewFieldDuplicate("port", port))
			} else {
				accumulator[port] = true
			}
		}
		allErrs = append(allErrs, cErrs.PrefixIndex(ci)...)
	}
	return allErrs
}

// checkHostPortConflicts checks for colliding Port.HostPort values across
// a slice of containers.
func checkHostPortConflicts(containers []api.Container) errs.ErrorList {
	allPorts := map[int]bool{}
	return AccumulateUniquePorts(containers, allPorts, func(p *api.Port) int { return p.HostPort })
}

func validateExecAction(exec *api.ExecAction) errs.ErrorList {
	allErrors := errs.ErrorList{}
	if len(exec.Command) == 0 {
		allErrors = append(allErrors, errs.NewFieldRequired("command", exec.Command))
	}
	return allErrors
}

func validateHTTPGetAction(http *api.HTTPGetAction) errs.ErrorList {
	allErrors := errs.ErrorList{}
	if len(http.Path) == 0 {
		allErrors = append(allErrors, errs.NewFieldRequired("path", http.Path))
	}
	return allErrors
}

func validateHandler(handler *api.Handler) errs.ErrorList {
	allErrors := errs.ErrorList{}
	if handler.Exec != nil {
		allErrors = append(allErrors, validateExecAction(handler.Exec).Prefix("exec")...)
	} else if handler.HTTPGet != nil {
		allErrors = append(allErrors, validateHTTPGetAction(handler.HTTPGet).Prefix("httpGet")...)
	} else {
		allErrors = append(allErrors, errs.NewFieldInvalid("", handler))
	}
	return allErrors
}

func validateLifecycle(lifecycle *api.Lifecycle) errs.ErrorList {
	allErrs := errs.ErrorList{}
	if lifecycle.PostStart != nil {
		allErrs = append(allErrs, validateHandler(lifecycle.PostStart).Prefix("postStart")...)
	}
	if lifecycle.PreStop != nil {
		allErrs = append(allErrs, validateHandler(lifecycle.PreStop).Prefix("preStop")...)
	}
	return allErrs
}

func validateContainers(containers []api.Container, volumes util.StringSet) errs.ErrorList {
	allErrs := errs.ErrorList{}

	allNames := util.StringSet{}
	for i := range containers {
		cErrs := errs.ErrorList{}
		ctr := &containers[i] // so we can set default values
		capabilities := capabilities.Get()
		if len(ctr.Name) == 0 {
			cErrs = append(cErrs, errs.NewFieldRequired("name", ctr.Name))
		} else if !util.IsDNSLabel(ctr.Name) {
			cErrs = append(cErrs, errs.NewFieldInvalid("name", ctr.Name))
		} else if allNames.Has(ctr.Name) {
			cErrs = append(cErrs, errs.NewFieldDuplicate("name", ctr.Name))
		} else if ctr.Privileged && !capabilities.AllowPrivileged {
			cErrs = append(cErrs, errs.NewFieldForbidden("privileged", ctr.Privileged))
		} else {
			allNames.Insert(ctr.Name)
		}
		if len(ctr.Image) == 0 {
			cErrs = append(cErrs, errs.NewFieldRequired("image", ctr.Image))
		}
		if ctr.Lifecycle != nil {
			cErrs = append(cErrs, validateLifecycle(ctr.Lifecycle).Prefix("lifecycle")...)
		}
		cErrs = append(cErrs, validatePorts(ctr.Ports).Prefix("ports")...)
		cErrs = append(cErrs, validateEnv(ctr.Env).Prefix("env")...)
		cErrs = append(cErrs, validateVolumeMounts(ctr.VolumeMounts, volumes).Prefix("volumeMounts")...)
		allErrs = append(allErrs, cErrs.PrefixIndex(i)...)
	}
	// Check for colliding ports across all containers.
	// TODO(thockin): This really is dependent on the network config of the host (IP per pod?)
	// and the config of the new manifest.  But we have not specced that out yet, so we'll just
	// make some assumptions for now.  As of now, pods share a network namespace, which means that
	// every Port.HostPort across the whole pod must be unique.
	allErrs = append(allErrs, checkHostPortConflicts(containers)...)

	return allErrs
}

var supportedManifestVersions = util.NewStringSet("v1beta1", "v1beta2")

// ValidateManifest tests that the specified ContainerManifest has valid data.
// This includes checking formatting and uniqueness.  It also canonicalizes the
// structure by setting default values and implementing any backwards-compatibility
// tricks.
func ValidateManifest(manifest *api.ContainerManifest) errs.ErrorList {
	allErrs := errs.ErrorList{}

	if len(manifest.Version) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("version", manifest.Version))
	} else if !supportedManifestVersions.Has(strings.ToLower(manifest.Version)) {
		allErrs = append(allErrs, errs.NewFieldNotSupported("version", manifest.Version))
	}
	allVolumes, vErrs := validateVolumes(manifest.Volumes)
	allErrs = append(allErrs, vErrs.Prefix("volumes")...)
	allErrs = append(allErrs, validateContainers(manifest.Containers, allVolumes).Prefix("containers")...)
	allErrs = append(allErrs, validateRestartPolicy(&manifest.RestartPolicy).Prefix("restartPolicy")...)
	return allErrs
}

func validateRestartPolicy(restartPolicy *api.RestartPolicy) errs.ErrorList {
	numPolicies := 0
	allErrors := errs.ErrorList{}
	if restartPolicy.Always != nil {
		numPolicies++
	}
	if restartPolicy.OnFailure != nil {
		numPolicies++
	}
	if restartPolicy.Never != nil {
		numPolicies++
	}
	if numPolicies == 0 {
		restartPolicy.Always = &api.RestartPolicyAlways{}
	}
	if numPolicies > 1 {
		allErrors = append(allErrors, errs.NewFieldInvalid("", restartPolicy))
	}
	return allErrors
}

func ValidatePodState(podState *api.PodState) errs.ErrorList {
	allErrs := errs.ErrorList(ValidateManifest(&podState.Manifest)).Prefix("manifest")
	return allErrs
}

// ValidatePod tests if required fields in the pod are set.
func ValidatePod(pod *api.Pod) errs.ErrorList {
	allErrs := errs.ErrorList{}
	if len(pod.ID) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("id", pod.ID))
	}
	if !util.IsDNSSubdomain(pod.Namespace) {
		allErrs = append(allErrs, errs.NewFieldInvalid("namespace", pod.Namespace))
	}
	allErrs = append(allErrs, ValidatePodState(&pod.DesiredState).Prefix("desiredState")...)
	return allErrs
}

// ValidatePodUpdate tests to see if the update is legal
func ValidatePodUpdate(newPod, oldPod *api.Pod) errs.ErrorList {
	allErrs := errs.ErrorList{}

	if newPod.ID != oldPod.ID {
		allErrs = append(allErrs, errs.NewFieldInvalid("ID", newPod.ID))
	}

	if len(newPod.DesiredState.Manifest.Containers) != len(oldPod.DesiredState.Manifest.Containers) {
		allErrs = append(allErrs, errs.NewFieldInvalid("DesiredState.Manifest.Containers", newPod.DesiredState.Manifest.Containers))
		return allErrs
	}
	pod := *newPod
	pod.Labels = oldPod.Labels
	pod.TypeMeta.ResourceVersion = oldPod.TypeMeta.ResourceVersion
	// Tricky, we need to copy the container list so that we don't overwrite the update
	var newContainers []api.Container
	for ix, container := range pod.DesiredState.Manifest.Containers {
		container.Image = oldPod.DesiredState.Manifest.Containers[ix].Image
		newContainers = append(newContainers, container)
	}
	pod.DesiredState.Manifest.Containers = newContainers
	if !reflect.DeepEqual(pod.DesiredState.Manifest, oldPod.DesiredState.Manifest) {
		allErrs = append(allErrs, errs.NewFieldInvalid("DesiredState.Manifest.Containers", newPod.DesiredState.Manifest.Containers))
	}
	return allErrs
}

// ValidateService tests if required fields in the service are set.
func ValidateService(service *api.Service) errs.ErrorList {
	allErrs := errs.ErrorList{}
	if len(service.ID) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("id", service.ID))
	} else if !util.IsDNS952Label(service.ID) {
		allErrs = append(allErrs, errs.NewFieldInvalid("id", service.ID))
	}
	if !util.IsDNSSubdomain(service.Namespace) {
		allErrs = append(allErrs, errs.NewFieldInvalid("namespace", service.Namespace))
	}
	if !util.IsValidPortNum(service.Port) {
		allErrs = append(allErrs, errs.NewFieldInvalid("port", service.Port))
	}
	if len(service.Protocol) == 0 {
		service.Protocol = "TCP"
	} else if !supportedPortProtocols.Has(strings.ToUpper(string(service.Protocol))) {
		allErrs = append(allErrs, errs.NewFieldNotSupported("protocol", service.Protocol))
	}
	if labels.Set(service.Selector).AsSelector().Empty() {
		allErrs = append(allErrs, errs.NewFieldRequired("selector", service.Selector))
	}
	return allErrs
}

// ValidateReplicationController tests if required fields in the replication controller are set.
func ValidateReplicationController(controller *api.ReplicationController) errs.ErrorList {
	allErrs := errs.ErrorList{}
	if len(controller.ID) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("id", controller.ID))
	}
	if !util.IsDNSSubdomain(controller.Namespace) {
		allErrs = append(allErrs, errs.NewFieldInvalid("namespace", controller.Namespace))
	}
	allErrs = append(allErrs, ValidateReplicationControllerState(&controller.DesiredState).Prefix("desiredState")...)
	return allErrs
}

// ValidateReplicationControllerState tests if required fields in the replication controller state are set.
func ValidateReplicationControllerState(state *api.ReplicationControllerState) errs.ErrorList {
	allErrs := errs.ErrorList{}
	if labels.Set(state.ReplicaSelector).AsSelector().Empty() {
		allErrs = append(allErrs, errs.NewFieldRequired("replicaSelector", state.ReplicaSelector))
	}
	selector := labels.Set(state.ReplicaSelector).AsSelector()
	labels := labels.Set(state.PodTemplate.Labels)
	if !selector.Matches(labels) {
		allErrs = append(allErrs, errs.NewFieldInvalid("podTemplate.labels", state.PodTemplate))
	}
	if state.Replicas < 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("replicas", state.Replicas))
	}
	allErrs = append(allErrs, ValidateManifest(&state.PodTemplate.DesiredState.Manifest).Prefix("podTemplate.desiredState.manifest")...)
	allErrs = append(allErrs, ValidateReadOnlyPersistentDisks(state.PodTemplate.DesiredState.Manifest.Volumes).Prefix("podTemplate.desiredState.manifest")...)
	return allErrs
}
func ValidateReadOnlyPersistentDisks(volumes []api.Volume) errs.ErrorList {
	allErrs := errs.ErrorList{}
	for _, vol := range volumes {
		if vol.Source.GCEPersistentDisk != nil {
			if vol.Source.GCEPersistentDisk.ReadOnly == false {
				allErrs = append(allErrs, errs.NewFieldInvalid("GCEPersistentDisk.ReadOnly", false))
			}
		}
	}
	return allErrs
}

// ValidateBoundPod tests if required fields on a bound pod are set.
func ValidateBoundPod(pod *api.BoundPod) (errors []error) {
	if !util.IsDNSSubdomain(pod.ID) {
		errors = append(errors, errs.NewFieldInvalid("id", pod.ID))
	}
	if !util.IsDNSSubdomain(pod.Namespace) {
		errors = append(errors, errs.NewFieldInvalid("namespace", pod.Namespace))
	}
	containerManifest := &api.ContainerManifest{
		Version:       "v1beta2",
		ID:            pod.ID,
		UUID:          pod.UID,
		Containers:    pod.Spec.Containers,
		Volumes:       pod.Spec.Volumes,
		RestartPolicy: pod.Spec.RestartPolicy,
	}
	if errs := ValidateManifest(containerManifest); len(errs) != 0 {
		errors = append(errors, errs...)
	}
	return errors
}
