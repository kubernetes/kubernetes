/*
Copyright 2020 The Kubernetes Authors.

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

package gvisor

import (
	"context"
	"fmt"
	"io"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/node"
)

const (
	// PluginName indicates name of admission plugin.
	PluginName = "Gvisor"

	// gvisorNodeKey is the key for gvisor node label and taint after beta.
	gvisorNodeKey = "sandbox.gke.io/runtime"
	// gvisorNodeValue is the value for gvisor node label and taint.
	gvisorNodeValue = "gvisor"

	// gvisorRuntimeClass is the name of the gvisor runtime class.
	gvisorRuntimeClass = "gvisor"

	// Annotation keys for gvisor mount.
	gvisorMountShareKey   = "dev.gvisor.spec.mount.%s.share"
	gvisorMountTypeKey    = "dev.gvisor.spec.mount.%s.type"
	gvisorMountOptionsKey = "dev.gvisor.spec.mount.%s.options"
)

var (
	// deprecatedAnnotations are old annotations not allowed anymore.
	deprecatedAnnotations = map[string]string{
		"runtime-handler.cri.kubernetes.io":    "gvisor", // Used during Alpha.
		"io.kubernetes.cri.untrusted-workload": "true",   // Used during Dogfood.
	}

	// capBlackList contains capabilities to be dropped if not explicitly added by users.
	capBlackList = []core.Capability{"NET_RAW"}
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewGvisor(), nil
	})
}

// NewGvisor creates a new Gvisor admission control handler
func NewGvisor() *Gvisor {
	return &Gvisor{
		Handler: admission.NewHandler(admission.Create, admission.Update),
	}
}

// Gvisor is an implementation of admission.Interface.
type Gvisor struct {
	*admission.Handler
}

var _ admission.MutationInterface = &Gvisor{}
var _ admission.ValidationInterface = &Gvisor{}

// checkDeprecatedAnnotation determines whether a pod contains any of the set of
// depcrecated, disallowed annotations
func checkDeprecatedAnnotation(pod *core.Pod) error {
	for k, v := range deprecatedAnnotations {
		if pod.Annotations[k] == v {
			return fmt.Errorf("annotation %q is deprecated. Use %q instead", k, "PodSpec.runtimeClassName: gvisor")
		}
	}
	return nil
}

// Admit determines whether a gvisor pod can be created, and modifies the pod
// in the request as needed
func (r *Gvisor) Admit(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) error {
	res := attributes.GetResource()
	op := attributes.GetOperation()
	if res.Group == "" && res.Resource == "pods" && op == admission.Create && len(attributes.GetSubresource()) == 0 {
		return admitPodCreate(attributes)
	}
	return nil
}

// Validate makes sure that a pod/runtimeclass adheres to Gvisor's definition
func (r *Gvisor) Validate(ctx context.Context, attributes admission.Attributes, o admission.ObjectInterfaces) error {
	res := attributes.GetResource()
	switch {
	case res.Group == "" && res.Resource == "pods" && len(attributes.GetSubresource()) == 0:
		switch attributes.GetOperation() {
		case admission.Create:
			return validatePodCreate(attributes)
		case admission.Update:
			return validatePodUpdate(attributes)
		default:
			return nil
		}
	case res.Group == "node.k8s.io" && res.Resource == "runtimeclasses" && len(attributes.GetSubresource()) == 0:
		return validateRuntimeClass(attributes)
	}
	return nil
}

// admitPodCreate performs some validation on the incoming object, and if
// it is a gvisor pod creation request, mutates the pod as necessary.
func admitPodCreate(attributes admission.Attributes) error {
	pod, err := getGvisorPod(attributes)
	if err != nil {
		return err
	}
	if pod == nil { // Pod is not a gvisor pod
		return nil
	}
	if err := validateGVisorPod(pod); err != nil {
		return admission.NewForbidden(attributes, err)
	}
	mutateGVisorPod(pod)
	return nil
}

// mutateGVisorPod is a helper function that modifies a pod object to meet
// gVisor specifications.
func mutateGVisorPod(pod *core.Pod) {
	updateCapabilities(pod.Spec.InitContainers)
	updateCapabilities(pod.Spec.Containers)
	updateVolumePodAnnotations(pod)
}

// updateCapabilities updates capabilities for given containers
func updateCapabilities(containers []core.Container) {
	for i := range containers {
		c := &containers[i]
		if c.SecurityContext == nil {
			c.SecurityContext = &core.SecurityContext{}
		}
		if c.SecurityContext.Capabilities == nil {
			c.SecurityContext.Capabilities = &core.Capabilities{}
		}
		if hasCapability("ALL", c.SecurityContext.Capabilities.Add) ||
			hasCapability("ALL", c.SecurityContext.Capabilities.Drop) {
			continue
		}
		for _, capToDrop := range capBlackList {
			if hasCapability(capToDrop, c.SecurityContext.Capabilities.Add) ||
				hasCapability(capToDrop, c.SecurityContext.Capabilities.Drop) {
				continue
			}
			c.SecurityContext.Capabilities.Drop = append(c.SecurityContext.Capabilities.Drop, capToDrop)
		}
	}
}

// hasCapability checks whether a given cap is in the cap list.
func hasCapability(cap core.Capability, caps []core.Capability) bool {
	for _, c := range caps {
		if strings.EqualFold(string(cap), string(c)) {
			return true
		}
	}
	return false
}

// updateVolumePodAnnotations puts volume information into gvisor specific pod
// annotations. This is mainly for optimizing volume performance. Currently only
// tmpfs based emptydir is handled.
func updateVolumePodAnnotations(pod *core.Pod) {
	var containers []core.Container
	containers = append(containers, pod.Spec.InitContainers...)
	containers = append(containers, pod.Spec.Containers...)

	for _, v := range pod.Spec.Volumes {
		info, ok := checkVolume(v, containers)
		if !ok {
			continue
		}
		if pod.Annotations == nil {
			pod.Annotations = make(map[string]string)
		}
		pod.Annotations[fmt.Sprintf(gvisorMountShareKey, v.Name)] = info.share
		pod.Annotations[fmt.Sprintf(gvisorMountTypeKey, v.Name)] = info.mountType
		pod.Annotations[fmt.Sprintf(gvisorMountOptionsKey, v.Name)] = strings.Join([]string{info.rwmode, info.propagation}, ",")
	}
}

// mountInfo holds mount info for gVisor specific mount.
type mountInfo struct {
	share       string
	mountType   string
	propagation string
	rwmode      string
}

// checkVolume checks whether a volume requires gVisor specific mount.
// For this optimization, gVisor currently only supports a
// single volume mounted into different containers with exactly
// the same mount option.
func checkVolume(v core.Volume, containers []core.Container) (mountInfo, bool) {
	if v.EmptyDir == nil {
		return mountInfo{}, false
	}

	mountType, err := getMountType(v.EmptyDir.Medium)
	if err != nil {
		return mountInfo{}, false
	}
	var (
		info     = mountInfo{mountType: mountType}
		mountNum int
	)
	for _, c := range containers {
		m := findVolumeMount(c, v.Name)
		if m == nil {
			continue
		}
		if m.SubPath != "" || m.SubPathExpr != "" {
			return mountInfo{}, false
		}

		crwmode := "rw"
		if m.ReadOnly {
			crwmode = "ro"
		}
		cpropagation := "rprivate"
		if m.MountPropagation != nil &&
			*m.MountPropagation == core.MountPropagationHostToContainer {
			cpropagation = "rslave"
		}
		mountNum++

		if (info.rwmode != "" && info.rwmode != crwmode) ||
			(info.propagation != "" && info.propagation != cpropagation) {
			return mountInfo{}, false
		}
		info.rwmode = crwmode
		info.propagation = cpropagation
	}

	switch {
	case mountNum > 1:
		info.share = "pod"
	case mountNum == 1:
		info.share = "container"
	default:
		// Skip this because no container mounts this volume.
		return mountInfo{}, false
	}
	return info, true
}

func findVolumeMount(c core.Container, name string) *core.VolumeMount {
	for _, m := range c.VolumeMounts {
		if m.Name == name {
			return &m
		}
	}
	return nil
}

func getMountType(medium core.StorageMedium) (string, error) {
	switch medium {
	case core.StorageMediumMemory, core.StorageMediumHugePages:
		// gVisor doesn't support huge pages, but tmpfs is the closest mapping.
		return "tmpfs", nil
	case core.StorageMediumDefault:
		return "bind", nil
	default:
		return "", fmt.Errorf("unsupported StorageMedium %v", medium)
	}
}

// getGvisorPod validates that an incoming request does indeed contain a gvisor
// pod. If both a nil pod and error are returned, then the pod was not a gvisor
// pod.
func getGvisorPod(attributes admission.Attributes) (*core.Pod, error) {
	// Verify that the object is indeed a Pod.
	pod, ok := attributes.GetObject().(*core.Pod)
	if !ok {
		return nil, apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}
	if err := checkDeprecatedAnnotation(pod); err != nil {
		return nil, admission.NewForbidden(attributes, fmt.Errorf("failed to validate pod object %s/%s: %v", pod.Namespace, pod.Name, err))
	}
	// Ignore if runtimeClassName is not present or selects a non-gvisor runtime.
	if rc := pod.Spec.RuntimeClassName; rc == nil || *rc != gvisorRuntimeClass {
		return nil, nil
	}
	return pod, nil
}

// validatePodCreate determines whether a pod create request is valid
func validatePodCreate(attributes admission.Attributes) error {
	pod, err := getGvisorPod(attributes)
	if err != nil {
		return err
	}
	if pod == nil { // Pod is not a gvisor pod
		return nil
	}
	if err := validateGVisorPod(pod); err != nil {
		return admission.NewForbidden(attributes, err)
	}
	return nil
}

// validatePodUpdate determines whether a pod update request is valid
func validatePodUpdate(attributes admission.Attributes) error {
	// Verify that the old object is indeed a Pod.
	_, ok := attributes.GetOldObject().(*core.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}
	// Verify that the object is indeed a Pod.
	pod, ok := attributes.GetObject().(*core.Pod)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}
	if err := checkDeprecatedAnnotation(pod); err != nil {
		return admission.NewForbidden(attributes, fmt.Errorf("failed to validate pod object %s/%s: %v", pod.Namespace, pod.Name, err))
	}
	// RuntimeClassName is immutable, so no need to perform additional validation
	return nil
}

// validateGVisorPod validates whether the pod is eligible to run in gVisor.
// 1) Pods with host path are not allowed.
// 2) Pods with host namespace are not allowed.
func validateGVisorPod(pod *core.Pod) error {
	if pod.Spec.NodeSelector != nil {
		if v, ok := pod.Spec.NodeSelector[gvisorNodeKey]; ok && v != gvisorNodeValue {
			return fmt.Errorf("conflict: pod.spec.nodeSelector[%q] = %q; it must be removed or set to %q", gvisorNodeKey, v, gvisorNodeValue)
		}
	}
	for _, v := range pod.Spec.Volumes {
		if v.HostPath != nil {
			return fmt.Errorf("HostPath is not allowed: %q", v.HostPath.Path)
		}
	}
	if pod.Spec.SecurityContext != nil {
		if pod.Spec.SecurityContext.HostNetwork {
			return fmt.Errorf("HostNetwork is not allowed")
		}
		if pod.Spec.SecurityContext.HostPID {
			return fmt.Errorf("HostPID is not allowed")
		}
		if pod.Spec.SecurityContext.HostIPC {
			return fmt.Errorf("HostIPC is not allowed")
		}
		if pod.Spec.SecurityContext != nil {
			if pod.Spec.SecurityContext.SELinuxOptions != nil {
				return fmt.Errorf("SELinuxOptions is not supported")
			}
			if pod.Spec.SecurityContext.FSGroup != nil {
				return fmt.Errorf("FSGroup is not supported")
			}
			if len(pod.Spec.SecurityContext.Sysctls) > 0 {
				return fmt.Errorf("Sysctls is not supported")
			}
		}
	}
	for k := range pod.Annotations {
		if strings.HasPrefix(k, "container.apparmor.security.beta.kubernetes.io") {
			return fmt.Errorf("Apparmor is not supported")
		}
		if strings.HasPrefix(k, "seccomp.security.alpha.kubernetes.io") {
			return fmt.Errorf("Seccomp is not supported")
		}
		if strings.HasPrefix(k, "container.seccomp.security.alpha.kubernetes.io") {
			return fmt.Errorf("Seccomp is not supported")
		}
	}
	var containers []core.Container
	containers = append(containers, pod.Spec.InitContainers...)
	containers = append(containers, pod.Spec.Containers...)
	for _, c := range containers {
		if c.SecurityContext != nil {
			if c.SecurityContext.Privileged != nil && *c.SecurityContext.Privileged {
				return fmt.Errorf("Privileged=true is not supported")
			}
			if c.SecurityContext.SELinuxOptions != nil {
				return fmt.Errorf("SELinuxOptions is not supported")
			}
			if c.SecurityContext.AllowPrivilegeEscalation != nil && *c.SecurityContext.AllowPrivilegeEscalation {
				return fmt.Errorf("AllowPrivilegeEscalation=true is not supported")
			}
			if c.SecurityContext.ProcMount != nil && *c.SecurityContext.ProcMount != core.DefaultProcMount {
				return fmt.Errorf("ProcMount=%s is not supported", *c.SecurityContext.ProcMount)
			}
		}
		if len(c.VolumeDevices) != 0 {
			return fmt.Errorf("VolumeDevices is not supported")
		}
		for _, m := range c.VolumeMounts {
			if m.MountPropagation != nil &&
				*m.MountPropagation != core.MountPropagationNone &&
				*m.MountPropagation != core.MountPropagationHostToContainer {
				return fmt.Errorf("MountPropagation=%s is not supported", *m.MountPropagation)
			}
		}
	}
	return nil
}

// validateRuntimeClass performs admission checks on RuntimeClass resources. We
// only care about the gVisor, so all other runtimeclasses are ignored.
func validateRuntimeClass(attributes admission.Attributes) error {
	// Ignore non-gvisor runtime classes.
	if attributes.GetName() != gvisorRuntimeClass {
		return nil
	}
	// Verify that the object is indeed a runtimeclass.
	rc, ok := attributes.GetObject().(*node.RuntimeClass)
	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind RuntimeClass but was unable to be converted")
	}
	// If this is the gvisor RuntimeClass, its runtimeHandler MUST also be gvisor.
	if rc.Handler != gvisorRuntimeClass {
		return admission.NewForbidden(attributes, admission.NewForbidden(attributes, fmt.Errorf("gvisor RuntimeClass cannot have a non-gvisor Handler: %s", rc.Handler)))
	}
	return nil
}

func stringPtr(p string) *string {
	return &p
}
