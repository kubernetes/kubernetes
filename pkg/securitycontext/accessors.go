/*
Copyright 2017 The Kubernetes Authors.

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

package securitycontext

import (
	"reflect"

	api "k8s.io/kubernetes/pkg/apis/core"
)

// PodSecurityContextAccessor allows reading the values of a PodSecurityContext object
type PodSecurityContextAccessor interface {
	HostNetwork() bool
	HostPID() bool
	HostIPC() bool
	SELinuxOptions() *api.SELinuxOptions
	RunAsUser() *int64
	RunAsGroup() *int64
	RunAsNonRoot() *bool
	SeccompProfile() *api.SeccompProfile
	SupplementalGroups() []int64
	FSGroup() *int64
}

// PodSecurityContextMutator allows reading and writing the values of a PodSecurityContext object
type PodSecurityContextMutator interface {
	PodSecurityContextAccessor

	SetHostNetwork(bool)
	SetHostPID(bool)
	SetHostIPC(bool)
	SetSELinuxOptions(*api.SELinuxOptions)
	SetRunAsUser(*int64)
	SetRunAsGroup(*int64)
	SetRunAsNonRoot(*bool)
	SetSeccompProfile(*api.SeccompProfile)
	SetSupplementalGroups([]int64)
	SetFSGroup(*int64)

	// PodSecurityContext returns the current PodSecurityContext object
	PodSecurityContext() *api.PodSecurityContext
}

// NewPodSecurityContextAccessor returns an accessor for the given pod security context.
// May be initialized with a nil PodSecurityContext.
func NewPodSecurityContextAccessor(podSC *api.PodSecurityContext) PodSecurityContextAccessor {
	return &podSecurityContextWrapper{podSC: podSC}
}

// NewPodSecurityContextMutator returns a mutator for the given pod security context.
// May be initialized with a nil PodSecurityContext.
func NewPodSecurityContextMutator(podSC *api.PodSecurityContext) PodSecurityContextMutator {
	return &podSecurityContextWrapper{podSC: podSC}
}

type podSecurityContextWrapper struct {
	podSC *api.PodSecurityContext
}

func (w *podSecurityContextWrapper) PodSecurityContext() *api.PodSecurityContext {
	return w.podSC
}

func (w *podSecurityContextWrapper) ensurePodSC() {
	if w.podSC == nil {
		w.podSC = &api.PodSecurityContext{}
	}
}

func (w *podSecurityContextWrapper) HostNetwork() bool {
	if w.podSC == nil {
		return false
	}
	return w.podSC.HostNetwork
}
func (w *podSecurityContextWrapper) SetHostNetwork(v bool) {
	if w.podSC == nil && v == false {
		return
	}
	w.ensurePodSC()
	w.podSC.HostNetwork = v
}
func (w *podSecurityContextWrapper) HostPID() bool {
	if w.podSC == nil {
		return false
	}
	return w.podSC.HostPID
}
func (w *podSecurityContextWrapper) SetHostPID(v bool) {
	if w.podSC == nil && v == false {
		return
	}
	w.ensurePodSC()
	w.podSC.HostPID = v
}
func (w *podSecurityContextWrapper) HostIPC() bool {
	if w.podSC == nil {
		return false
	}
	return w.podSC.HostIPC
}
func (w *podSecurityContextWrapper) SetHostIPC(v bool) {
	if w.podSC == nil && v == false {
		return
	}
	w.ensurePodSC()
	w.podSC.HostIPC = v
}
func (w *podSecurityContextWrapper) SELinuxOptions() *api.SELinuxOptions {
	if w.podSC == nil {
		return nil
	}
	return w.podSC.SELinuxOptions
}
func (w *podSecurityContextWrapper) SetSELinuxOptions(v *api.SELinuxOptions) {
	if w.podSC == nil && v == nil {
		return
	}
	w.ensurePodSC()
	w.podSC.SELinuxOptions = v
}
func (w *podSecurityContextWrapper) RunAsUser() *int64 {
	if w.podSC == nil {
		return nil
	}
	return w.podSC.RunAsUser
}
func (w *podSecurityContextWrapper) SetRunAsUser(v *int64) {
	if w.podSC == nil && v == nil {
		return
	}
	w.ensurePodSC()
	w.podSC.RunAsUser = v
}
func (w *podSecurityContextWrapper) RunAsGroup() *int64 {
	if w.podSC == nil {
		return nil
	}
	return w.podSC.RunAsGroup
}
func (w *podSecurityContextWrapper) SetRunAsGroup(v *int64) {
	if w.podSC == nil && v == nil {
		return
	}
	w.ensurePodSC()
	w.podSC.RunAsGroup = v
}

func (w *podSecurityContextWrapper) RunAsNonRoot() *bool {
	if w.podSC == nil {
		return nil
	}
	return w.podSC.RunAsNonRoot
}
func (w *podSecurityContextWrapper) SetRunAsNonRoot(v *bool) {
	if w.podSC == nil && v == nil {
		return
	}
	w.ensurePodSC()
	w.podSC.RunAsNonRoot = v
}
func (w *podSecurityContextWrapper) SeccompProfile() *api.SeccompProfile {
	if w.podSC == nil {
		return nil
	}
	return w.podSC.SeccompProfile
}
func (w *podSecurityContextWrapper) SetSeccompProfile(p *api.SeccompProfile) {
	if w.podSC == nil && p == nil {
		return
	}
	w.ensurePodSC()
	w.podSC.SeccompProfile = p
}
func (w *podSecurityContextWrapper) SupplementalGroups() []int64 {
	if w.podSC == nil {
		return nil
	}
	return w.podSC.SupplementalGroups
}
func (w *podSecurityContextWrapper) SetSupplementalGroups(v []int64) {
	if w.podSC == nil && len(v) == 0 {
		return
	}
	w.ensurePodSC()
	if len(v) == 0 && len(w.podSC.SupplementalGroups) == 0 {
		return
	}
	w.podSC.SupplementalGroups = v
}
func (w *podSecurityContextWrapper) FSGroup() *int64 {
	if w.podSC == nil {
		return nil
	}
	return w.podSC.FSGroup
}
func (w *podSecurityContextWrapper) SetFSGroup(v *int64) {
	if w.podSC == nil && v == nil {
		return
	}
	w.ensurePodSC()
	w.podSC.FSGroup = v
}

// ContainerSecurityContextAccessor allows reading the values of a SecurityContext object
type ContainerSecurityContextAccessor interface {
	Capabilities() *api.Capabilities
	Privileged() *bool
	ProcMount() api.ProcMountType
	SELinuxOptions() *api.SELinuxOptions
	RunAsUser() *int64
	RunAsGroup() *int64
	RunAsNonRoot() *bool
	ReadOnlyRootFilesystem() *bool
	SeccompProfile() *api.SeccompProfile
	AllowPrivilegeEscalation() *bool
}

// ContainerSecurityContextMutator allows reading and writing the values of a SecurityContext object
type ContainerSecurityContextMutator interface {
	ContainerSecurityContextAccessor

	ContainerSecurityContext() *api.SecurityContext

	SetCapabilities(*api.Capabilities)
	SetPrivileged(*bool)
	SetSELinuxOptions(*api.SELinuxOptions)
	SetRunAsUser(*int64)
	SetRunAsGroup(*int64)
	SetRunAsNonRoot(*bool)
	SetReadOnlyRootFilesystem(*bool)
	SetSeccompProfile(*api.SeccompProfile)
	SetAllowPrivilegeEscalation(*bool)
}

// NewContainerSecurityContextAccessor returns an accessor for the provided container security context
// May be initialized with a nil SecurityContext
func NewContainerSecurityContextAccessor(containerSC *api.SecurityContext) ContainerSecurityContextAccessor {
	return &containerSecurityContextWrapper{containerSC: containerSC}
}

// NewContainerSecurityContextMutator returns a mutator for the provided container security context
// May be initialized with a nil SecurityContext
func NewContainerSecurityContextMutator(containerSC *api.SecurityContext) ContainerSecurityContextMutator {
	return &containerSecurityContextWrapper{containerSC: containerSC}
}

type containerSecurityContextWrapper struct {
	containerSC *api.SecurityContext
}

func (w *containerSecurityContextWrapper) ContainerSecurityContext() *api.SecurityContext {
	return w.containerSC
}

func (w *containerSecurityContextWrapper) ensureContainerSC() {
	if w.containerSC == nil {
		w.containerSC = &api.SecurityContext{}
	}
}

func (w *containerSecurityContextWrapper) Capabilities() *api.Capabilities {
	if w.containerSC == nil {
		return nil
	}
	return w.containerSC.Capabilities
}
func (w *containerSecurityContextWrapper) SetCapabilities(v *api.Capabilities) {
	if w.containerSC == nil && v == nil {
		return
	}
	w.ensureContainerSC()
	w.containerSC.Capabilities = v
}
func (w *containerSecurityContextWrapper) Privileged() *bool {
	if w.containerSC == nil {
		return nil
	}
	return w.containerSC.Privileged
}
func (w *containerSecurityContextWrapper) SetPrivileged(v *bool) {
	if w.containerSC == nil && v == nil {
		return
	}
	w.ensureContainerSC()
	w.containerSC.Privileged = v
}
func (w *containerSecurityContextWrapper) ProcMount() api.ProcMountType {
	if w.containerSC == nil {
		return api.DefaultProcMount
	}
	if w.containerSC.ProcMount == nil {
		return api.DefaultProcMount
	}
	return *w.containerSC.ProcMount
}
func (w *containerSecurityContextWrapper) SELinuxOptions() *api.SELinuxOptions {
	if w.containerSC == nil {
		return nil
	}
	return w.containerSC.SELinuxOptions
}
func (w *containerSecurityContextWrapper) SetSELinuxOptions(v *api.SELinuxOptions) {
	if w.containerSC == nil && v == nil {
		return
	}
	w.ensureContainerSC()
	w.containerSC.SELinuxOptions = v
}
func (w *containerSecurityContextWrapper) RunAsUser() *int64 {
	if w.containerSC == nil {
		return nil
	}
	return w.containerSC.RunAsUser
}
func (w *containerSecurityContextWrapper) SetRunAsUser(v *int64) {
	if w.containerSC == nil && v == nil {
		return
	}
	w.ensureContainerSC()
	w.containerSC.RunAsUser = v
}
func (w *containerSecurityContextWrapper) RunAsGroup() *int64 {
	if w.containerSC == nil {
		return nil
	}
	return w.containerSC.RunAsGroup
}
func (w *containerSecurityContextWrapper) SetRunAsGroup(v *int64) {
	if w.containerSC == nil && v == nil {
		return
	}
	w.ensureContainerSC()
	w.containerSC.RunAsGroup = v
}

func (w *containerSecurityContextWrapper) RunAsNonRoot() *bool {
	if w.containerSC == nil {
		return nil
	}
	return w.containerSC.RunAsNonRoot
}
func (w *containerSecurityContextWrapper) SetRunAsNonRoot(v *bool) {
	if w.containerSC == nil && v == nil {
		return
	}
	w.ensureContainerSC()
	w.containerSC.RunAsNonRoot = v
}
func (w *containerSecurityContextWrapper) ReadOnlyRootFilesystem() *bool {
	if w.containerSC == nil {
		return nil
	}
	return w.containerSC.ReadOnlyRootFilesystem
}
func (w *containerSecurityContextWrapper) SetReadOnlyRootFilesystem(v *bool) {
	if w.containerSC == nil && v == nil {
		return
	}
	w.ensureContainerSC()
	w.containerSC.ReadOnlyRootFilesystem = v
}
func (w *containerSecurityContextWrapper) SeccompProfile() *api.SeccompProfile {
	if w.containerSC == nil {
		return nil
	}
	return w.containerSC.SeccompProfile
}
func (w *containerSecurityContextWrapper) SetSeccompProfile(p *api.SeccompProfile) {
	if w.containerSC == nil && p == nil {
		return
	}
	w.ensureContainerSC()
	w.containerSC.SeccompProfile = p
}

func (w *containerSecurityContextWrapper) AllowPrivilegeEscalation() *bool {
	if w.containerSC == nil {
		return nil
	}
	return w.containerSC.AllowPrivilegeEscalation
}
func (w *containerSecurityContextWrapper) SetAllowPrivilegeEscalation(v *bool) {
	if w.containerSC == nil && v == nil {
		return
	}
	w.ensureContainerSC()
	w.containerSC.AllowPrivilegeEscalation = v
}

// NewEffectiveContainerSecurityContextAccessor returns an accessor for reading effective values
// for the provided pod security context and container security context
func NewEffectiveContainerSecurityContextAccessor(podSC PodSecurityContextAccessor, containerSC ContainerSecurityContextMutator) ContainerSecurityContextAccessor {
	return &effectiveContainerSecurityContextWrapper{podSC: podSC, containerSC: containerSC}
}

// NewEffectiveContainerSecurityContextMutator returns a mutator for reading and writing effective values
// for the provided pod security context and container security context
func NewEffectiveContainerSecurityContextMutator(podSC PodSecurityContextAccessor, containerSC ContainerSecurityContextMutator) ContainerSecurityContextMutator {
	return &effectiveContainerSecurityContextWrapper{podSC: podSC, containerSC: containerSC}
}

type effectiveContainerSecurityContextWrapper struct {
	podSC       PodSecurityContextAccessor
	containerSC ContainerSecurityContextMutator
}

func (w *effectiveContainerSecurityContextWrapper) ContainerSecurityContext() *api.SecurityContext {
	return w.containerSC.ContainerSecurityContext()
}

func (w *effectiveContainerSecurityContextWrapper) Capabilities() *api.Capabilities {
	return w.containerSC.Capabilities()
}
func (w *effectiveContainerSecurityContextWrapper) SetCapabilities(v *api.Capabilities) {
	if !reflect.DeepEqual(w.Capabilities(), v) {
		w.containerSC.SetCapabilities(v)
	}
}
func (w *effectiveContainerSecurityContextWrapper) Privileged() *bool {
	return w.containerSC.Privileged()
}
func (w *effectiveContainerSecurityContextWrapper) SetPrivileged(v *bool) {
	if !reflect.DeepEqual(w.Privileged(), v) {
		w.containerSC.SetPrivileged(v)
	}
}
func (w *effectiveContainerSecurityContextWrapper) ProcMount() api.ProcMountType {
	return w.containerSC.ProcMount()
}
func (w *effectiveContainerSecurityContextWrapper) SELinuxOptions() *api.SELinuxOptions {
	if v := w.containerSC.SELinuxOptions(); v != nil {
		return v
	}
	return w.podSC.SELinuxOptions()
}
func (w *effectiveContainerSecurityContextWrapper) SetSELinuxOptions(v *api.SELinuxOptions) {
	if !reflect.DeepEqual(w.SELinuxOptions(), v) {
		w.containerSC.SetSELinuxOptions(v)
	}
}
func (w *effectiveContainerSecurityContextWrapper) RunAsUser() *int64 {
	if v := w.containerSC.RunAsUser(); v != nil {
		return v
	}
	return w.podSC.RunAsUser()
}
func (w *effectiveContainerSecurityContextWrapper) SetRunAsUser(v *int64) {
	if !reflect.DeepEqual(w.RunAsUser(), v) {
		w.containerSC.SetRunAsUser(v)
	}
}
func (w *effectiveContainerSecurityContextWrapper) RunAsGroup() *int64 {
	if v := w.containerSC.RunAsGroup(); v != nil {
		return v
	}
	return w.podSC.RunAsGroup()
}
func (w *effectiveContainerSecurityContextWrapper) SetRunAsGroup(v *int64) {
	if !reflect.DeepEqual(w.RunAsGroup(), v) {
		w.containerSC.SetRunAsGroup(v)
	}
}

func (w *effectiveContainerSecurityContextWrapper) RunAsNonRoot() *bool {
	if v := w.containerSC.RunAsNonRoot(); v != nil {
		return v
	}
	return w.podSC.RunAsNonRoot()
}
func (w *effectiveContainerSecurityContextWrapper) SetRunAsNonRoot(v *bool) {
	if !reflect.DeepEqual(w.RunAsNonRoot(), v) {
		w.containerSC.SetRunAsNonRoot(v)
	}
}
func (w *effectiveContainerSecurityContextWrapper) ReadOnlyRootFilesystem() *bool {
	return w.containerSC.ReadOnlyRootFilesystem()
}
func (w *effectiveContainerSecurityContextWrapper) SetReadOnlyRootFilesystem(v *bool) {
	if !reflect.DeepEqual(w.ReadOnlyRootFilesystem(), v) {
		w.containerSC.SetReadOnlyRootFilesystem(v)
	}
}
func (w *effectiveContainerSecurityContextWrapper) SeccompProfile() *api.SeccompProfile {
	return w.containerSC.SeccompProfile()
}
func (w *effectiveContainerSecurityContextWrapper) SetSeccompProfile(p *api.SeccompProfile) {
	if !reflect.DeepEqual(w.SeccompProfile(), p) {
		w.containerSC.SetSeccompProfile(p)
	}
}
func (w *effectiveContainerSecurityContextWrapper) AllowPrivilegeEscalation() *bool {
	return w.containerSC.AllowPrivilegeEscalation()
}
func (w *effectiveContainerSecurityContextWrapper) SetAllowPrivilegeEscalation(v *bool) {
	if !reflect.DeepEqual(w.AllowPrivilegeEscalation(), v) {
		w.containerSC.SetAllowPrivilegeEscalation(v)
	}
}
