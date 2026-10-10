/*
Copyright 2022 The Kubernetes Authors.

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

// Package install installs the resource API, making it available as an
// option to all of the API encoding/decoding machinery.
package install

import (
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/resource"
	v1 "k8s.io/kubernetes/pkg/apis/resource/v1"
	"k8s.io/kubernetes/pkg/apis/resource/v1alpha3"
	"k8s.io/kubernetes/pkg/apis/resource/v1beta1"
	"k8s.io/kubernetes/pkg/apis/resource/v1beta2"
	"k8s.io/kubernetes/pkg/features"
)

func init() {
	Install(legacyscheme.Scheme)
}

// Install registers the API group and adds types to a scheme
func Install(scheme *runtime.Scheme) {
	utilruntime.Must(resource.AddToScheme(scheme))
	utilruntime.Must(v1alpha3.AddToScheme(scheme))
	utilruntime.Must(v1beta2.AddToScheme(scheme))
	utilruntime.Must(v1.AddToScheme(scheme))
	// v1beta1 is intentionally excluded here: if it gets added later,
	// it'll have a lower priority than any version listed here.
	// We never want it to be used.
	utilruntime.Must(scheme.SetVersionPriority(v1.SchemeGroupVersion, v1beta2.SchemeGroupVersion, v1alpha3.SchemeGroupVersion))

	// Installing resource.k8s.io/v1beta1 is deferred to a scheme init func because
	// it depends on the DRAResourceV1beta1API feature gate, which is not
	// initialized yet when package init functions run.
	scheme.AddInitFunc(func(logger klog.Logger, scheme *runtime.Scheme) error {
		if !utilfeature.DefaultFeatureGate.Enabled(features.DRAResourceV1beta1API) {
			logger.V(4).Info("Not installing resource.k8s.io/v1beta1, DRAResourceV1beta1API feature gate is disabled")
			return nil
		}
		return v1beta1.AddToScheme(scheme)
	})
}
