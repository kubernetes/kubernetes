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

package kubeadm

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SetDefaultTimeouts sets an internal Timeouts struct to its default values.
func SetDefaultTimeouts(t **Timeouts) {
	*t = &Timeouts{
		ControlPlaneComponentHealthCheck: &metav1.Duration{Duration: 4 * time.Minute},
		KubeletHealthCheck:               &metav1.Duration{Duration: 4 * time.Minute},
		KubernetesAPICall:                &metav1.Duration{Duration: 1 * time.Minute},
		EtcdAPICall:                      &metav1.Duration{Duration: 2 * time.Minute},
		TLSBootstrap:                     &metav1.Duration{Duration: 5 * time.Minute},
		Discovery:                        &metav1.Duration{Duration: 5 * time.Minute},
	}
}
