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

package polymorphichelpers

import (
	"errors"
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	appsv1beta2 "k8s.io/api/apps/v1beta2"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

func defaultObjectResumer(obj runtime.Object) ([]byte, error) {
	switch obj := obj.(type) {
	case *extensions.Deployment:
		if !obj.Spec.Paused {
			return nil, errors.New("is not paused")
		}
		obj.Spec.Paused = false
		return runtime.Encode(internalVersionJSONEncoder(), obj)

	case *extensionsv1beta1.Deployment:
		if !obj.Spec.Paused {
			return nil, errors.New("is not paused")
		}
		obj.Spec.Paused = false
		return runtime.Encode(scheme.Codecs.LegacyCodec(extensionsv1beta1.SchemeGroupVersion), obj)

	case *appsv1.Deployment:
		if !obj.Spec.Paused {
			return nil, errors.New("is not paused")
		}
		obj.Spec.Paused = false
		return runtime.Encode(scheme.Codecs.LegacyCodec(appsv1.SchemeGroupVersion), obj)

	case *appsv1beta2.Deployment:
		if !obj.Spec.Paused {
			return nil, errors.New("is not paused")
		}
		obj.Spec.Paused = false
		return runtime.Encode(scheme.Codecs.LegacyCodec(appsv1beta2.SchemeGroupVersion), obj)

	case *appsv1beta1.Deployment:
		if !obj.Spec.Paused {
			return nil, errors.New("is not paused")
		}
		obj.Spec.Paused = false
		return runtime.Encode(scheme.Codecs.LegacyCodec(appsv1beta1.SchemeGroupVersion), obj)

	default:
		return nil, fmt.Errorf("resuming is not supported")
	}
}
