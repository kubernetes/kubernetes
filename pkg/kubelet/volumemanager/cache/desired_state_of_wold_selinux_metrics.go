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

package cache

import (
	"sync"

	compbasemetrics "k8s.io/component-base/metrics"
	"k8s.io/component-base/metrics/legacyregistry"
)

var (
	// TODO: add plugin name + access mode labels to all these metrics
	seLinuxContainerContextErrors = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "volume_manager_selinux_container_errors_total",
			Help:           "Number of errors when kubelet cannot compute SELinux context for a container. Kubelet can't start such a Pod then and it will retry, therefore value of this metric may not represent the actual nr. of containers.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"access_mode"},
	)
	seLinuxContainerContextWarnings = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "volume_manager_selinux_container_warnings_total",
			StabilityLevel: compbasemetrics.ALPHA,
			Help:           "Number of errors when kubelet cannot compute SELinux context for a container that are ignored. They will become real errors when SELinuxMountReadWriteOncePod feature is expanded to all volume access modes.",
		},
		[]string{"access_mode"},
	)
	seLinuxPodContextMismatchErrors = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "volume_manager_selinux_pod_context_mismatch_errors_total",
			Help:           "Number of errors when a Pod defines different SELinux contexts for its containers that use the same volume. Kubelet can't start such a Pod then and it will retry, therefore value of this metric may not represent the actual nr. of Pods.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"access_mode"},
	)
	seLinuxPodContextMismatchWarnings = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "volume_manager_selinux_pod_context_mismatch_warnings_total",
			Help:           "Number of errors when a Pod defines different SELinux contexts for its containers that use the same volume. They are not errors yet, but they will become real errors when SELinuxMountReadWriteOncePod feature is expanded to all volume access modes.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"access_mode"},
	)
	seLinuxVolumeContextMismatchErrors = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "volume_manager_selinux_volume_context_mismatch_errors_total",
			Help:           "Number of errors when a Pod uses a volume that is already mounted with a different SELinux context than the Pod needs. Kubelet can't start such a Pod then and it will retry, therefore value of this metric may not represent the actual nr. of Pods.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"volume_plugin", "access_mode"},
	)
	seLinuxVolumeContextMismatchWarnings = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "volume_manager_selinux_volume_context_mismatch_warnings_total",
			Help:           "Number of errors when a Pod uses a volume that is already mounted with a different SELinux context than the Pod needs. They are not errors yet, but they will become real errors when SELinuxMountReadWriteOncePod feature is expanded to all volume access modes.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"volume_plugin", "access_mode"},
	)
	seLinuxVolumesAdmitted = compbasemetrics.NewGaugeVec(
		&compbasemetrics.GaugeOpts{
			Name:           "volume_manager_selinux_volumes_admitted_total",
			Help:           "Number of volumes whose SELinux context was fine and will be mounted with mount -o context option.",
			StabilityLevel: compbasemetrics.ALPHA,
		},
		[]string{"volume_plugin", "access_mode"},
	)

	registerMetrics sync.Once
)

func registerSELinuxMetrics() {
	registerMetrics.Do(func() {
		legacyregistry.MustRegister(seLinuxContainerContextErrors)
		legacyregistry.MustRegister(seLinuxContainerContextWarnings)
		legacyregistry.MustRegister(seLinuxPodContextMismatchErrors)
		legacyregistry.MustRegister(seLinuxPodContextMismatchWarnings)
		legacyregistry.MustRegister(seLinuxVolumeContextMismatchErrors)
		legacyregistry.MustRegister(seLinuxVolumeContextMismatchWarnings)
		legacyregistry.MustRegister(seLinuxVolumesAdmitted)
	})
}
