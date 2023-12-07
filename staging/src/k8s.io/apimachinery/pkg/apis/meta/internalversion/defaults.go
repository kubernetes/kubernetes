/*
Copyright 2023 The Kubernetes Authors.

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

package internalversion

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// SetListOptionsDefaults sets defaults on the provided ListOptions if applicable.
//
// TODO(#115478): once the watch-list fg is always on we register this function in the scheme (via AddTypeDefaultingFunc).
// TODO(#115478): when the function is registered in the scheme remove all callers of this method.
func SetListOptionsDefaults(obj *ListOptions, isWatchListFeatureEnabled bool) {
	if !isWatchListFeatureEnabled {
		return
	}
	if obj.SendInitialEvents != nil || len(obj.ResourceVersionMatch) != 0 {
		return
	}
	legacy := obj.ResourceVersion == "" || obj.ResourceVersion == "0"
	if obj.Watch && legacy {
		turnOnInitialEvents := true
		obj.SendInitialEvents = &turnOnInitialEvents
		obj.ResourceVersionMatch = metav1.ResourceVersionMatchNotOlderThan
	}
}
