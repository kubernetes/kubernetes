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
	"fmt"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	errs "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/capabilities"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
)

// ValidateLabels validates that a set of labels are correctly defined.
func ValidateLabels(labels map[string]string, field string) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	for k := range labels {
		if !util.IsQualifiedName(k) {
			allErrs = append(allErrs, errs.NewFieldInvalid(field, k, ""))
		}
	}
	return allErrs
}

// ValidateAnnotations validates that a set of annotations are correctly defined.
func ValidateAnnotations(annotations map[string]string, field string) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	for k := range annotations {
		if !util.IsQualifiedName(strings.ToLower(k)) {
			allErrs = append(allErrs, errs.NewFieldInvalid(field, k, ""))
		}
	}
	return allErrs
}

// ValidateNameFunc validates that the provided name is valid for a given resource type.
// Not all resources have the same validation rules for names.
type ValidateNameFunc func(name string) (bool, string)

// nameIsDNSSubdomain is a ValidateNameFunc for names that must be a DNS subdomain.
func nameIsDNSSubdomain(name string) (bool, string) {
	if util.IsDNSSubdomain(name) {
		return true, ""
	}
	return false, "name must be lowercase letters and numbers, with inline dashes or periods"
}

// nameIsDNS952Label is a ValidateNameFunc for names that must be a DNS 952 label.
func nameIsDNS952Label(name string) (bool, string) {
	if util.IsDNS952Label(name) {
		return true, ""
	}
	return false, "name must be lowercase letters, numbers, and dashes"
}

// ValidateObjectMeta validates an object's metadata.
func ValidateObjectMeta(meta *api.ObjectMeta, requiresNamespace bool, nameFn ValidateNameFunc) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if len(meta.Name) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("name", meta.Name))
	} else {
		if ok, qualifier := nameFn(meta.Name); !ok {
			allErrs = append(allErrs, errs.NewFieldInvalid("name", meta.Name, qualifier))
		}
	}
	if requiresNamespace {
		if len(meta.Namespace) == 0 {
			allErrs = append(allErrs, errs.NewFieldRequired("namespace", meta.Namespace))
		} else if !util.IsDNSSubdomain(meta.Namespace) {
			allErrs = append(allErrs, errs.NewFieldInvalid("namespace", meta.Namespace, ""))
		}
	} else {
		if len(meta.Namespace) != 0 {
			allErrs = append(allErrs, errs.NewFieldInvalid("namespace", meta.Namespace, "namespace is not allowed on this type"))
		}
	}
	allErrs = append(allErrs, ValidateLabels(meta.Labels, "labels")...)
	allErrs = append(allErrs, ValidateAnnotations(meta.Annotations, "annotations")...)

	// Clear self link internally
	// TODO: move to its own area
	meta.SelfLink = ""

	return allErrs
}

// ValidateObjectMetaUpdate validates an object's metadata when updated
func ValidateObjectMetaUpdate(old, meta *api.ObjectMeta) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	// in the event it is left empty, set it, to allow clients more flexibility
	if len(meta.UID) == 0 {
		meta.UID = old.UID
	}
	if meta.CreationTimestamp.IsZero() {
		meta.CreationTimestamp = old.CreationTimestamp
	}

	if old.Name != meta.Name {
		allErrs = append(allErrs, errs.NewFieldInvalid("name", meta.Name, "field is immutable"))
	}
	if old.Namespace != meta.Namespace {
		allErrs = append(allErrs, errs.NewFieldInvalid("namespace", meta.Namespace, "field is immutable"))
	}
	if old.UID != meta.UID {
		allErrs = append(allErrs, errs.NewFieldInvalid("uid", meta.UID, "field is immutable"))
	}
	if old.CreationTimestamp != meta.CreationTimestamp {
		allErrs = append(allErrs, errs.NewFieldInvalid("creationTimestamp", meta.CreationTimestamp, "field is immutable"))
	}

	allErrs = append(allErrs, ValidateLabels(meta.Labels, "labels")...)
	allErrs = append(allErrs, ValidateAnnotations(meta.Annotations, "annotations")...)

	// Clear self link internally
	// TODO: move to its own area
	meta.SelfLink = ""

	return allErrs
}

func validateVolumes(volumes []api.Volume) (util.StringSet, errs.ValidationErrorList) {
	allErrs := errs.ValidationErrorList{}

	allNames := util.StringSet{}
	for i := range volumes {
		vol := &volumes[i] // so we can set default values
		el := validateSource(&vol.Source).Prefix("source")
		if len(vol.Name) == 0 {
			el = append(el, errs.NewFieldRequired("name", vol.Name))
		} else if !util.IsDNSLabel(vol.Name) {
			el = append(el, errs.NewFieldInvalid("name", vol.Name, ""))
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

func validateSource(source *api.VolumeSource) errs.ValidationErrorList {
	numVolumes := 0
	allErrs := errs.ValidationErrorList{}
	if source.HostPath != nil {
		numVolumes++
		allErrs = append(allErrs, validateHostPath(source.HostPath).Prefix("hostPath")...)
	}
	if source.EmptyDir != nil {
		numVolumes++
		// EmptyDirs have nothing to validate
	}
	if source.GitRepo != nil {
		numVolumes++
		allErrs = append(allErrs, validateGitRepo(source.GitRepo).Prefix("gitRepo")...)
	}
	if source.GCEPersistentDisk != nil {
		numVolumes++
		allErrs = append(allErrs, validateGCEPersistentDisk(source.GCEPersistentDisk).Prefix("persistentDisk")...)
	}
	if numVolumes == 0 {
		// TODO: Enforce that a source is set once we deprecate the implied form.
		source.EmptyDir = &api.EmptyDir{}
	} else if numVolumes != 1 {
		allErrs = append(allErrs, errs.NewFieldInvalid("", source, "exactly 1 volume type is required"))
	}
	return allErrs
}

func validateHostPath(hostDir *api.HostPath) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if hostDir.Path == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("path", hostDir.Path))
	}
	return allErrs
}

func validateGitRepo(gitRepo *api.GitRepo) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if gitRepo.Repository == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("repository", gitRepo.Repository))
	}
	return allErrs
}

func validateGCEPersistentDisk(PD *api.GCEPersistentDisk) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if PD.PDName == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("pdName", PD.PDName))
	}
	if PD.FSType == "" {
		allErrs = append(allErrs, errs.NewFieldRequired("fsType", PD.FSType))
	}
	if PD.Partition < 0 || PD.Partition > 255 {
		allErrs = append(allErrs, errs.NewFieldInvalid("partition", PD.Partition, ""))
	}
	return allErrs
}

var supportedPortProtocols = util.NewStringSet(string(api.ProtocolTCP), string(api.ProtocolUDP))

func validatePorts(ports []api.Port) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	allNames := util.StringSet{}
	for i := range ports {
		pErrs := errs.ValidationErrorList{}
		port := &ports[i] // so we can set default values
		if len(port.Name) > 0 {
			if len(port.Name) > 63 || !util.IsDNSLabel(port.Name) {
				pErrs = append(pErrs, errs.NewFieldInvalid("name", port.Name, ""))
			} else if allNames.Has(port.Name) {
				pErrs = append(pErrs, errs.NewFieldDuplicate("name", port.Name))
			} else {
				allNames.Insert(port.Name)
			}
		}
		if port.ContainerPort == 0 {
			pErrs = append(pErrs, errs.NewFieldRequired("containerPort", port.ContainerPort))
		} else if !util.IsValidPortNum(port.ContainerPort) {
			pErrs = append(pErrs, errs.NewFieldInvalid("containerPort", port.ContainerPort, ""))
		}
		if port.HostPort != 0 && !util.IsValidPortNum(port.HostPort) {
			pErrs = append(pErrs, errs.NewFieldInvalid("hostPort", port.HostPort, ""))
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

func validateEnv(vars []api.EnvVar) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	for i := range vars {
		vErrs := errs.ValidationErrorList{}
		ev := &vars[i] // so we can set default values
		if len(ev.Name) == 0 {
			vErrs = append(vErrs, errs.NewFieldRequired("name", ev.Name))
		}
		if !util.IsCIdentifier(ev.Name) {
			vErrs = append(vErrs, errs.NewFieldInvalid("name", ev.Name, ""))
		}
		allErrs = append(allErrs, vErrs.PrefixIndex(i)...)
	}
	return allErrs
}

func validateVolumeMounts(mounts []api.VolumeMount, volumes util.StringSet) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	for i := range mounts {
		mErrs := errs.ValidationErrorList{}
		mnt := &mounts[i] // so we can set default values
		if len(mnt.Name) == 0 {
			mErrs = append(mErrs, errs.NewFieldRequired("name", mnt.Name))
		} else if !volumes.Has(mnt.Name) {
			mErrs = append(mErrs, errs.NewFieldNotFound("name", mnt.Name))
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
func AccumulateUniquePorts(containers []api.Container, accumulator map[int]bool, extract func(*api.Port) int) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	for ci := range containers {
		cErrs := errs.ValidationErrorList{}
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
func checkHostPortConflicts(containers []api.Container) errs.ValidationErrorList {
	allPorts := map[int]bool{}
	return AccumulateUniquePorts(containers, allPorts, func(p *api.Port) int { return p.HostPort })
}

func validateExecAction(exec *api.ExecAction) errs.ValidationErrorList {
	allErrors := errs.ValidationErrorList{}
	if len(exec.Command) == 0 {
		allErrors = append(allErrors, errs.NewFieldRequired("command", exec.Command))
	}
	return allErrors
}

func validateHTTPGetAction(http *api.HTTPGetAction) errs.ValidationErrorList {
	allErrors := errs.ValidationErrorList{}
	if len(http.Path) == 0 {
		allErrors = append(allErrors, errs.NewFieldRequired("path", http.Path))
	}
	return allErrors
}

func validateHandler(handler *api.Handler) errs.ValidationErrorList {
	numHandlers := 0
	allErrors := errs.ValidationErrorList{}
	if handler.Exec != nil {
		numHandlers++
		allErrors = append(allErrors, validateExecAction(handler.Exec).Prefix("exec")...)
	}
	if handler.HTTPGet != nil {
		numHandlers++
		allErrors = append(allErrors, validateHTTPGetAction(handler.HTTPGet).Prefix("httpGet")...)
	}
	if numHandlers != 1 {
		allErrors = append(allErrors, errs.NewFieldInvalid("", handler, "exactly 1 handler type is required"))
	}
	return allErrors
}

func validateLifecycle(lifecycle *api.Lifecycle) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if lifecycle.PostStart != nil {
		allErrs = append(allErrs, validateHandler(lifecycle.PostStart).Prefix("postStart")...)
	}
	if lifecycle.PreStop != nil {
		allErrs = append(allErrs, validateHandler(lifecycle.PreStop).Prefix("preStop")...)
	}
	return allErrs
}

// TODO(dchen1107): Move this along with other defaulting values
func validatePullPolicyWithDefault(ctr *api.Container) errs.ValidationErrorList {
	allErrors := errs.ValidationErrorList{}

	switch ctr.ImagePullPolicy {
	case "":
		// TODO(dchen1107): Move ParseImageName code to pkg/util
		parts := strings.Split(ctr.Image, ":")
		// Check image tag
		if parts[len(parts)-1] == "latest" {
			ctr.ImagePullPolicy = api.PullAlways
		} else {
			ctr.ImagePullPolicy = api.PullIfNotPresent
		}
	case api.PullAlways, api.PullIfNotPresent, api.PullNever:
		break
	default:
		allErrors = append(allErrors, errs.NewFieldNotSupported("", ctr.ImagePullPolicy))
	}

	return allErrors
}

func validateContainers(containers []api.Container, volumes util.StringSet) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	allNames := util.StringSet{}
	for i := range containers {
		cErrs := errs.ValidationErrorList{}
		ctr := &containers[i] // so we can set default values
		capabilities := capabilities.Get()
		if len(ctr.Name) == 0 {
			cErrs = append(cErrs, errs.NewFieldRequired("name", ctr.Name))
		} else if !util.IsDNSLabel(ctr.Name) {
			cErrs = append(cErrs, errs.NewFieldInvalid("name", ctr.Name, ""))
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
		cErrs = append(cErrs, validatePullPolicyWithDefault(ctr).Prefix("pullPolicy")...)
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
// TODO: replaced by ValidatePodSpec
func ValidateManifest(manifest *api.ContainerManifest) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	if len(manifest.Version) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("version", manifest.Version))
	} else if !supportedManifestVersions.Has(strings.ToLower(manifest.Version)) {
		allErrs = append(allErrs, errs.NewFieldNotSupported("version", manifest.Version))
	}
	allVolumes, vErrs := validateVolumes(manifest.Volumes)
	allErrs = append(allErrs, vErrs.Prefix("volumes")...)
	allErrs = append(allErrs, validateContainers(manifest.Containers, allVolumes).Prefix("containers")...)
	allErrs = append(allErrs, validateRestartPolicy(&manifest.RestartPolicy).Prefix("restartPolicy")...)
	allErrs = append(allErrs, validateDNSPolicy(&manifest.DNSPolicy).Prefix("dnsPolicy")...)
	return allErrs
}

func validateRestartPolicy(restartPolicy *api.RestartPolicy) errs.ValidationErrorList {
	numPolicies := 0
	allErrors := errs.ValidationErrorList{}
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
		allErrors = append(allErrors, errs.NewFieldInvalid("", restartPolicy, "only 1 policy is allowed"))
	}
	return allErrors
}

func validateDNSPolicy(dnsPolicy *api.DNSPolicy) errs.ValidationErrorList {
	allErrors := errs.ValidationErrorList{}
	switch *dnsPolicy {
	case "":
		// TODO: move this out to standard defaulting logic, when that is ready.
		*dnsPolicy = api.DNSClusterFirst // Default value.
	case api.DNSClusterFirst, api.DNSDefault:
		break
	default:
		allErrors = append(allErrors, errs.NewFieldNotSupported("", dnsPolicy))
	}
	return allErrors
}

// ValidatePod tests if required fields in the pod are set.
func ValidatePod(pod *api.Pod) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&pod.ObjectMeta, true, nameIsDNSSubdomain).Prefix("metadata")...)
	allErrs = append(allErrs, ValidatePodSpec(&pod.Spec).Prefix("spec")...)

	return allErrs
}

// ValidatePodSpec tests that the specified PodSpec has valid data.
// This includes checking formatting and uniqueness.  It also canonicalizes the
// structure by setting default values and implementing any backwards-compatibility
// tricks.
func ValidatePodSpec(spec *api.PodSpec) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	allVolumes, vErrs := validateVolumes(spec.Volumes)
	allErrs = append(allErrs, vErrs.Prefix("volumes")...)
	allErrs = append(allErrs, validateContainers(spec.Containers, allVolumes).Prefix("containers")...)
	allErrs = append(allErrs, validateRestartPolicy(&spec.RestartPolicy).Prefix("restartPolicy")...)
	allErrs = append(allErrs, validateDNSPolicy(&spec.DNSPolicy).Prefix("dnsPolicy")...)
	allErrs = append(allErrs, ValidateLabels(spec.NodeSelector, "nodeSelector")...)
	return allErrs
}

// ValidatePodUpdate tests to see if the update is legal
func ValidatePodUpdate(newPod, oldPod *api.Pod) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldPod.ObjectMeta, &newPod.ObjectMeta).Prefix("metadata")...)

	if len(newPod.Spec.Containers) != len(oldPod.Spec.Containers) {
		allErrs = append(allErrs, errs.NewFieldInvalid("spec.containers", newPod.Spec.Containers, "may not add or remove containers"))
		return allErrs
	}
	pod := *newPod
	// Tricky, we need to copy the container list so that we don't overwrite the update
	var newContainers []api.Container
	for ix, container := range pod.Spec.Containers {
		container.Image = oldPod.Spec.Containers[ix].Image
		newContainers = append(newContainers, container)
	}
	pod.Spec.Containers = newContainers
	if !api.Semantic.DeepEqual(pod.Spec, oldPod.Spec) {
		// TODO: a better error would include all immutable fields explicitly.
		allErrs = append(allErrs, errs.NewFieldInvalid("spec.containers", newPod.Spec.Containers, "some fields are immutable"))
	}

	return allErrs
}

var supportedSessionAffinityType = util.NewStringSet(string(api.AffinityTypeClientIP), string(api.AffinityTypeNone))

// ValidateService tests if required fields in the service are set.
func ValidateService(service *api.Service) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&service.ObjectMeta, true, nameIsDNS952Label).Prefix("metadata")...)

	if !util.IsValidPortNum(service.Spec.Port) {
		allErrs = append(allErrs, errs.NewFieldInvalid("spec.port", service.Spec.Port, ""))
	}
	if len(service.Spec.Protocol) == 0 {
		service.Spec.Protocol = "TCP"
	} else if !supportedPortProtocols.Has(strings.ToUpper(string(service.Spec.Protocol))) {
		allErrs = append(allErrs, errs.NewFieldNotSupported("spec.protocol", service.Spec.Protocol))
	}

	if service.Spec.Selector != nil {
		allErrs = append(allErrs, ValidateLabels(service.Spec.Selector, "spec.selector")...)
	}

	if service.Spec.SessionAffinity == "" {
		service.Spec.SessionAffinity = api.AffinityTypeNone
	} else if !supportedSessionAffinityType.Has(string(service.Spec.SessionAffinity)) {
		allErrs = append(allErrs, errs.NewFieldNotSupported("spec.sessionAffinity", service.Spec.SessionAffinity))
	}

	return allErrs
}

// ValidateServiceUpdate tests if required fields in the service are set during an update
func ValidateServiceUpdate(oldService, service *api.Service) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldService.ObjectMeta, &service.ObjectMeta).Prefix("metadata")...)

	// TODO: PortalIP should be a Status field, since the system can set a value != to the user's value
	// PortalIP can only be set, not unset.
	if oldService.Spec.PortalIP != "" && service.Spec.PortalIP != oldService.Spec.PortalIP {
		allErrs = append(allErrs, errs.NewFieldInvalid("spec.portalIP", service.Spec.PortalIP, "field is immutable"))
	}

	return allErrs
}

// ValidateReplicationController tests if required fields in the replication controller are set.
func ValidateReplicationController(controller *api.ReplicationController) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&controller.ObjectMeta, true, nameIsDNSSubdomain).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateReplicationControllerSpec(&controller.Spec).Prefix("spec")...)

	return allErrs
}

// ValidateReplicationControllerUpdate tests if required fields in the replication controller are set.
func ValidateReplicationControllerUpdate(oldController, controller *api.ReplicationController) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldController.ObjectMeta, &controller.ObjectMeta).Prefix("metadata")...)
	allErrs = append(allErrs, ValidateReplicationControllerSpec(&controller.Spec).Prefix("spec")...)

	return allErrs
}

// ValidateReplicationControllerSpec tests if required fields in the replication controller spec are set.
func ValidateReplicationControllerSpec(spec *api.ReplicationControllerSpec) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	selector := labels.Set(spec.Selector).AsSelector()
	if selector.Empty() {
		allErrs = append(allErrs, errs.NewFieldRequired("selector", spec.Selector))
	}
	if spec.Replicas < 0 {
		allErrs = append(allErrs, errs.NewFieldInvalid("replicas", spec.Replicas, ""))
	}

	if spec.Template == nil {
		allErrs = append(allErrs, errs.NewFieldRequired("template", spec.Template))
	} else {
		labels := labels.Set(spec.Template.Labels)
		if !selector.Matches(labels) {
			allErrs = append(allErrs, errs.NewFieldInvalid("template.labels", spec.Template.Labels, "selector does not match template"))
		}
		allErrs = append(allErrs, ValidatePodTemplateSpec(spec.Template).Prefix("template")...)
		// RestartPolicy has already been first-order validated as per ValidatePodTemplateSpec().
		if spec.Template.Spec.RestartPolicy.Always == nil {
			// TODO: should probably be Unsupported
			// TODO: api.RestartPolicy should have a String() method for nicer printing
			allErrs = append(allErrs, errs.NewFieldInvalid("template.restartPolicy", spec.Template.Spec.RestartPolicy, "must be Always"))
		}
	}
	return allErrs
}

// ValidatePodTemplateSpec validates the spec of a pod template
func ValidatePodTemplateSpec(spec *api.PodTemplateSpec) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateLabels(spec.Labels, "labels")...)
	allErrs = append(allErrs, ValidateAnnotations(spec.Annotations, "annotations")...)
	allErrs = append(allErrs, ValidatePodSpec(&spec.Spec).Prefix("spec")...)
	allErrs = append(allErrs, ValidateReadOnlyPersistentDisks(spec.Spec.Volumes).Prefix("spec.volumes")...)
	return allErrs
}

func ValidateReadOnlyPersistentDisks(volumes []api.Volume) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	for _, vol := range volumes {
		if vol.Source.GCEPersistentDisk != nil {
			if vol.Source.GCEPersistentDisk.ReadOnly == false {
				allErrs = append(allErrs, errs.NewFieldInvalid("GCEPersistentDisk.ReadOnly", false, "must be true"))
			}
		}
	}
	return allErrs
}

// ValidateBoundPod tests if required fields on a bound pod are set.
// TODO: to be removed.
func ValidateBoundPod(pod *api.BoundPod) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if len(pod.Name) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("name", pod.Name))
	} else {
		if ok, qualifier := nameIsDNSSubdomain(pod.Name); !ok {
			allErrs = append(allErrs, errs.NewFieldInvalid("name", pod.Name, qualifier))
		}
	}
	if len(pod.Namespace) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("namespace", pod.Namespace))
	} else if !util.IsDNSSubdomain(pod.Namespace) {
		allErrs = append(allErrs, errs.NewFieldInvalid("namespace", pod.Namespace, ""))
	}
	allErrs = append(allErrs, ValidatePodSpec(&pod.Spec).Prefix("spec")...)
	return allErrs
}

// ValidateMinion tests if required fields in the node are set.
func ValidateMinion(node *api.Node) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMeta(&node.ObjectMeta, false, nameIsDNSSubdomain).Prefix("metadata")...)
	return allErrs
}

// ValidateMinionUpdate tests to make sure a minion update can be applied.  Modifies oldMinion.
func ValidateMinionUpdate(oldMinion *api.Node, minion *api.Node) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	allErrs = append(allErrs, ValidateObjectMetaUpdate(&oldMinion.ObjectMeta, &minion.ObjectMeta).Prefix("metadata")...)

	// TODO: Enable the code once we have better api object.status update model. Currently,
	// anyone can update node status.
	// if !api.Semantic.DeepEqual(minion.Status, api.NodeStatus{}) {
	// 	allErrs = append(allErrs, errs.NewFieldInvalid("status", minion.Status, "status must be empty"))
	// }

	// TODO: move reset function to its own location
	// Ignore metadata changes now that they have been tested
	oldMinion.ObjectMeta = minion.ObjectMeta
	// Allow users to update capacity
	oldMinion.Spec.Capacity = minion.Spec.Capacity
	// Clear status
	oldMinion.Status = minion.Status

	if !api.Semantic.DeepEqual(oldMinion, minion) {
		glog.V(4).Infof("Update failed validation %#v vs %#v", oldMinion, minion)
		allErrs = append(allErrs, fmt.Errorf("update contains more than labels or capacity changes"))
	}

	// TODO: validate Spec.Capacity
	return allErrs
}

// Typename is a generic representation for all compute resource typenames.
// Refer to docs/resources.md for more details.
func ValidateResourceName(str string) errs.ValidationErrorList {
	if !util.IsQualifiedName(str) {
		return errs.ValidationErrorList{fmt.Errorf("invalid compute resource typename format %q", str)}
	}

	parts := strings.Split(str, "/")
	switch len(parts) {
	case 1:
		if !api.IsStandardResourceName(parts[0]) {
			return errs.ValidationErrorList{fmt.Errorf("invalid compute resource typename. %q is neither a standard resource type nor is fully qualified", str)}
		}
		break
	case 2:
		if parts[0] == api.DefaultResourceNamespace {
			if !api.IsStandardResourceName(parts[1]) {
				return errs.ValidationErrorList{fmt.Errorf("invalid compute resource typename. %q contains a compute resource type not supported", str)}

			}
		}
		break
	}

	return errs.ValidationErrorList{}
}

// ValidateLimitRange tests if required fields in the LimitRange are set.
func ValidateLimitRange(limitRange *api.LimitRange) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if len(limitRange.Name) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("name", limitRange.Name))
	} else if !util.IsDNSSubdomain(limitRange.Name) {
		allErrs = append(allErrs, errs.NewFieldInvalid("name", limitRange.Name, ""))
	}
	if len(limitRange.Namespace) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("namespace", limitRange.Namespace))
	} else if !util.IsDNSSubdomain(limitRange.Namespace) {
		allErrs = append(allErrs, errs.NewFieldInvalid("namespace", limitRange.Namespace, ""))
	}
	// ensure resource names are properly qualified per docs/resources.md
	for i := range limitRange.Spec.Limits {
		limit := limitRange.Spec.Limits[i]
		for k := range limit.Max {
			allErrs = append(allErrs, ValidateResourceName(string(k))...)
		}
		for k := range limit.Min {
			allErrs = append(allErrs, ValidateResourceName(string(k))...)
		}
	}
	return allErrs
}

// ValidateResourceQuota tests if required fields in the ResourceQuota are set.
func ValidateResourceQuota(resourceQuota *api.ResourceQuota) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}
	if len(resourceQuota.Name) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("name", resourceQuota.Name))
	} else if !util.IsDNSSubdomain(resourceQuota.Name) {
		allErrs = append(allErrs, errs.NewFieldInvalid("name", resourceQuota.Name, ""))
	}
	if len(resourceQuota.Namespace) == 0 {
		allErrs = append(allErrs, errs.NewFieldRequired("namespace", resourceQuota.Namespace))
	} else if !util.IsDNSSubdomain(resourceQuota.Namespace) {
		allErrs = append(allErrs, errs.NewFieldInvalid("namespace", resourceQuota.Namespace, ""))
	}
	for k := range resourceQuota.Spec.Hard {
		allErrs = append(allErrs, ValidateResourceName(string(k))...)
	}
	for k := range resourceQuota.Status.Hard {
		allErrs = append(allErrs, ValidateResourceName(string(k))...)
	}
	for k := range resourceQuota.Status.Used {
		allErrs = append(allErrs, ValidateResourceName(string(k))...)
	}
	return allErrs
}
