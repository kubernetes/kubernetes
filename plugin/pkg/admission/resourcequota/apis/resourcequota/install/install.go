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

// Package install installs the experimental API group, making it available as
// an option to all of the API encoding/decoding machinery.
package install

import (
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	resourcequotaapi "k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota"
	resourcequotav1 "k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota/v1"
	resourcequotav1alpha1 "k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota/v1alpha1"
	resourcequotav1beta1 "k8s.io/kubernetes/plugin/pkg/admission/resourcequota/apis/resourcequota/v1beta1"
)

// Install registers the API group and adds types to a scheme
func Install(scheme *runtime.Scheme) {
	utilruntime.Must(resourcequotaapi.AddToScheme(scheme))

	// v1beta1 and v1alpha1 are in the k8s.io-suffixed group
	utilruntime.Must(resourcequotav1beta1.AddToScheme(scheme))
	utilruntime.Must(resourcequotav1alpha1.AddToScheme(scheme))
	utilruntime.Must(scheme.SetVersionPriority(resourcequotav1beta1.SchemeGroupVersion, resourcequotav1alpha1.SchemeGroupVersion))

	// v1 is in the config.k8s.io-suffixed group
	utilruntime.Must(resourcequotav1.AddToScheme(scheme))
	utilruntime.Must(scheme.SetVersionPriority(resourcequotav1.SchemeGroupVersion))
}
