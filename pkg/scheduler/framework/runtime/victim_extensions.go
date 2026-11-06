/*
Copyright The Kubernetes Authors.

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

package runtime

import (
	"context"

	v1 "k8s.io/api/core/v1"
)

// RunVictimExtensionsAssumeVictim calls AssumeVictim of all plugins.
func (f *frameworkImpl) RunVictimExtensionsAssumeVictim(ctx context.Context, victim *v1.Pod) {
	for _, ext := range f.victimExtensions {
		ext.AssumeVictim(extensionContext(ctx, "VictimExtensions", ext), victim)
	}
}

// RunVictimExtensionsPodEligibleToPreempt returns the result of the first
// plugin which reports that the pod is not eligible to preempt.
func (f *frameworkImpl) RunVictimExtensionsPodEligibleToPreempt(ctx context.Context, pod *v1.Pod, nodeName string) (bool, string) {
	for _, ext := range f.victimExtensions {
		if eligible, reason := ext.PodEligibleToPreempt(extensionContext(ctx, "VictimExtensions", ext), pod, nodeName); !eligible {
			return false, reason
		}
	}
	return true, ""
}
