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

package pod

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/utils/ktesting/kobject"
)

// Get creates a function which retrieves the pod anew each time the function
// is called. Fatal errors are detected by framework.GetObject and cause
// polling to stop.
func Get(c clientset.Interface, pod kobject.Object) framework.GetFunc[*v1.Pod] {
	return framework.GetObject(c.CoreV1().Pods(pod.GetNamespace()).Get, pod.GetName(), metav1.GetOptions{})
}
