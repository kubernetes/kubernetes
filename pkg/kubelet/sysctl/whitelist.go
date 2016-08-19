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
)

// Whitelist provides a list of allowed sysctls and sysctl patterns (ending in *)
// and a function to check whether a given sysctl matches this list.
type Whitelist struct {
	sysctls  map[string]Namespace
	prefixes map[string]Namespace
}

// NewWhitelist creates a new Whitelist from a list of sysctls and sysctl pattern (ending in *).
func NewWhitelist(sysctls []string) (*Whitelist, error) {
	w := &Whitelist{
		sysctls:  map[string]Namespace{},
		prefixes: map[string]Namespace{},
	}

	for _, s := range sysctls {
		if strings.HasSuffix(s, "*") {
			prefix := s[:len(s)-1]
			ns := NamespacedBy(prefix)
			if ns == UnknownNamespace {
				return nil, fmt.Errorf("the sysctls %q are not known to be namespaced", s)
			}
			w.prefixes[prefix] = ns
		} else {
			ns := NamespacedBy(s)
			if ns == UnknownNamespace {
				return nil, fmt.Errorf("the sysctl %q are not known to be namespaced", s)
			}
			w.sysctls[s] = ns
		}
	}
	return w, nil
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
		if ns == IpcNamespace && hostIPC {
			return false
		}
		if ns == NetNamespace && hostNet {
			return false
		}
		return true
	}
	for p, ns := range w.prefixes {
		if strings.HasPrefix(val, p) {
			if ns == IpcNamespace && hostIPC {
				return false
			}
			if ns == NetNamespace && hostNet {
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
