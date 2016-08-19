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

package sysctl

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/validation/sysctl"
)

var defaultAllowedSysctls = map[string]sysctl.Namespace{
	"kernel.shmall":                sysctl.IpcNamespace,
	"kernel.shmmax":                sysctl.IpcNamespace,
	"kernel.shmmni":                sysctl.IpcNamespace,
	"kernel.shm_rmid_forced":       sysctl.IpcNamespace,
	"net.ipv4.ip_local_port_range": sysctl.NetNamespace,
	"net.ipv4.tcp_max_syn_backlog": sysctl.NetNamespace,
	"net.ipv4.tcp_syncookies":      sysctl.NetNamespace,
}

var defaultAllowedPrefixes = map[string]sysctl.Namespace{}

type Whitelist struct {
	custom   []string
	sysctls  map[string]sysctl.Namespace
	prefixes map[string]sysctl.Namespace
}

// NewWhitelist takes a custom whitelist, consisting of sysctls and sysctls patterns
// (ending in *). It can be checked whether a sysctl is either part of the default
// whitelist or part of the custom whitelist.
func NewWhitelist(custom []string) (*Whitelist, error) {
	w := &Whitelist{
		custom:   custom,
		sysctls:  map[string]sysctl.Namespace{},
		prefixes: map[string]sysctl.Namespace{},
	}
	for s, ns := range defaultAllowedSysctls {
		w.sysctls[s] = ns
	}
	for p, ns := range defaultAllowedPrefixes {
		w.prefixes[p] = ns
	}
	for _, s := range custom {
		if strings.HasSuffix(s, "*") {
			prefix := s[:len(s)-1]
			ns := sysctl.NamespacedBy(prefix)
			if ns == sysctl.UnknownNamespace {
				return nil, fmt.Errorf("the sysctls %q are not known to be namespaced", s)
			}
			w.prefixes[prefix] = ns
		} else {
			ns := sysctl.NamespacedBy(s)
			if ns == sysctl.UnknownNamespace {
				return nil, fmt.Errorf("the sysctl %q are not known to be namespaced", s)
			}
			w.sysctls[s] = ns
		}
	}
	return w, nil
}

// CustomWhitelist return the custom whitelist slace with sysctls and
// sysctl pattern (ending in *).
func (w *Whitelist) CustomWhitelist() []string {
	return w.custom
}

// Valid checks that a sysctl is whitelisted because it is known
// to be namespaced by the Linux kernel. Note that being whitelisted is required, but not
// sufficient: the container runtime might have a stricter check and refuse to launch a pod.
//
// The parameters hostNet and hostIPC are used to forbid sysctls for pod sharing the
// respective namespaces with the host. This check is only possible for sysctls on
// the static default whitelist, not those on the custom whitelist provided by the admin.
func (w *Whitelist) Valid(val string, hostNet, hostIPC bool) bool {
	if ns, found := w.sysctls[val]; found {
		if ns == sysctl.IpcNamespace && hostIPC {
			return false
		}
		if ns == sysctl.NetNamespace && hostNet {
			return false
		}
		return true
	}
	for p, ns := range w.prefixes {
		if strings.HasPrefix(val, p) {
			if ns == sysctl.IpcNamespace && hostIPC {
				return false
			}
			if ns == sysctl.NetNamespace && hostNet {
				return false
			}
			return true
		}
	}
	return false
}

// Validate checks that all sysctls given in a api.SysctlsPodAnnotationKey annotation
// are valid according to the whitelist.
func (w *Whitelist) Validate(pod *api.Pod) error {
	a := pod.Annotations[api.SysctlsPodAnnotationKey]
	if a == "" {
		return nil
	}

	sysctls, err := api.SysctlsFromPodAnnotation(a)
	if err != nil {
		return fmt.Errorf("pod with UID %q specified an invalid %s annotation: %v", pod.UID, api.SysctlsPodAnnotationKey, err)
	}

	for _, s := range sysctls {
		if !w.Valid(s.Name, pod.Spec.SecurityContext.HostNetwork, pod.Spec.SecurityContext.HostIPC) {
			return fmt.Errorf("pod with UID %q specified a not whitelisted sysctl: %s", pod.UID, s)
		}
	}

	return nil
}
