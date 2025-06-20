//go:build !windows
// +build !windows

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

package kuberuntime

import (
	"context"
	"fmt"
	"math"
	"strings"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/securitycontext"
)

// verifyRunAsNonRoot verifies RunAsNonRoot.
func verifyRunAsNonRoot(ctx context.Context, pod *v1.Pod, container *v1.Container, uid *int64, username string) error {
	effectiveSc := securitycontext.DetermineEffectiveSecurityContext(pod, container)
	// If the option is not set, or if running as root is allowed, return nil.
	if effectiveSc == nil || effectiveSc.RunAsNonRoot == nil || !*effectiveSc.RunAsNonRoot {
		return nil
	}

	if effectiveSc.RunAsUser != nil {
		if *effectiveSc.RunAsUser == 0 {
			return fmt.Errorf("container's runAsUser breaks non-root policy (pod: %q, container: %s)", format.Pod(pod), container.Name)
		}
		return nil
	}

	switch {
	case uid == nil && len(username) > 0:
		return fmt.Errorf("container has runAsNonRoot and image has non-numeric user (%s), cannot verify user is non-root (pod: %q, container: %s)", username, format.Pod(pod), container.Name)
	case uid != nil:
		if *uid == 0 {
			return fmt.Errorf("container has runAsNonRoot and image will run as root (pod: %q, container: %s)", format.Pod(pod), container.Name)
		}
		if errs := validation.IsValidUserID(*uid); len(errs) > 0 {
			return fmt.Errorf(
				"container has runAsNonRoot and image has an invalid user id (%d). (Must be 1-(%d)): %s (pod: %q, container: %s)",
				*uid,
				math.MaxInt32,
				strings.Join(errs, "; "),
				format.Pod(pod),
				container.Name,
			)
		}
		return nil
	default:
		return nil
	}
}
