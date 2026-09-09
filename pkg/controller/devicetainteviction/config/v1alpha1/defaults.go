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

package v1alpha1

import (
	kubectrlmgrconfigv1alpha1 "k8s.io/kube-controller-manager/config/v1alpha1"
)

// RecommendedDefaultDeviceTaintEvictionControllerConfiguration defaults a pointer to a
// DeviceTaintEvictionControllerConfiguration struct. This will set the recommended default
// values, but they may be subject to change between API versions. This function
// is intentionally not registered in the scheme as a "normal" `SetDefaults_Foo`
// function to allow consumers of this type to set whatever defaults for their
// embedded configs. Forcing consumers to use these defaults would be problematic
// as defaulting in the scheme is done as part of the conversion, and there would
// be no easy way to opt-out. Instead, if you want to use this defaulting method
// run it in your wrapper struct of this type in its `SetDefaults_` method.
func RecommendedDefaultDeviceTaintEvictionControllerConfiguration(obj *kubectrlmgrconfigv1alpha1.DeviceTaintEvictionControllerConfiguration) {
	if obj.ConcurrentSyncs == 0 {
		// This is a compromise between getting work done and not overwhelming the apiserver
		// and pod informers. Integration testing with 100 workers modified pods so quickly
		// that a watch in the integration test couldn't keep up:
		//   cacher.go:855] cacher (pods): 100 objects queued in incoming channel.
		//   cache_watcher.go:203] Forcing pods watcher close due to unresponsiveness: key: "/pods/", labels: "", fields: "". len(c.input) = 10, len(c.result) = 10, graceful = false
		obj.ConcurrentSyncs = 8
	}
}
