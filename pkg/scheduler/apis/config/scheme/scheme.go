/*
Copyright 2018 The Kubernetes Authors.

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

package scheme

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	kubeschedulerconfigv1 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1"
	kubeschedulerconfigv1alpha1 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1alpha1"
	kubeschedulerconfigv1alpha2 "k8s.io/kubernetes/pkg/scheduler/apis/config/v1alpha2"
)

var (
	// Scheme is the runtime.Scheme to which all kubescheduler api types are registered.
	Scheme = runtime.NewScheme()

	// Codecs provides access to encoding and decoding for the scheme.
	Codecs = serializer.NewCodecFactory(Scheme, serializer.EnableStrict)
)

func init() {
	AddToScheme(Scheme)
}

// AddToScheme builds the kubescheduler scheme using all known versions of the kubescheduler api.
func AddToScheme(scheme *runtime.Scheme) {
	utilruntime.Must(kubeschedulerconfig.AddToScheme(scheme))
	utilruntime.Must(kubeschedulerconfigv1.AddToScheme(scheme))
	utilruntime.Must(kubeschedulerconfigv1alpha1.AddToScheme(scheme))
	utilruntime.Must(kubeschedulerconfigv1alpha2.AddToScheme(scheme))
	utilruntime.Must(scheme.SetVersionPriority(kubeschedulerconfigv1alpha2.SchemeGroupVersion, kubeschedulerconfigv1alpha1.SchemeGroupVersion))
}
