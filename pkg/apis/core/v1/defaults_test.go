/*
Copyright 2015 The Kubernetes Authors.

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

package v1_test

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/features"
	utilpointer "k8s.io/utils/pointer"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

// TestWorkloadDefaults detects changes to defaults within PodTemplateSpec.
// Defaulting changes within this type can cause spurious rollouts of workloads on API server update.
func TestWorkloadDefaults(t *testing.T) {
	t.Run("enabled_features", func(t *testing.T) { testWorkloadDefaults(t, true) })
	t.Run("disabled_features", func(t *testing.T) { testWorkloadDefaults(t, false) })
}
func testWorkloadDefaults(t *testing.T, featuresEnabled bool) {
	allFeatures := utilfeature.DefaultFeatureGate.DeepCopy().GetAll()
	for feature, featureSpec := range allFeatures {
		if !featureSpec.LockToDefault {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, feature, featuresEnabled)()
		}
	}
	// New defaults under PodTemplateSpec are only acceptable if they would not be applied when reading data from a previous release.
	// Forbidden: adding a new field `MyField *bool` and defaulting it to a non-nil value
	// Forbidden: defaulting an existing field `MyField *bool` when it was previously not defaulted
	// Forbidden: changing an existing default value
	// Allowed: adding a new field `MyContainer *MyType` and defaulting a child of that type (e.g. `MyContainer.MyChildField`) if and only if MyContainer is non-nil
	expectedDefaults := map[string]string{
		".Spec.Containers[0].Env[0].ValueFrom.FieldRef.APIVersion":       `"v1"`,
		".Spec.Containers[0].ImagePullPolicy":                            `"IfNotPresent"`,
		".Spec.Containers[0].Lifecycle.PostStart.HTTPGet.Path":           `"/"`,
		".Spec.Containers[0].Lifecycle.PostStart.HTTPGet.Scheme":         `"HTTP"`,
		".Spec.Containers[0].Lifecycle.PreStop.HTTPGet.Path":             `"/"`,
		".Spec.Containers[0].Lifecycle.PreStop.HTTPGet.Scheme":           `"HTTP"`,
		".Spec.Containers[0].LivenessProbe.FailureThreshold":             `3`,
		".Spec.Containers[0].LivenessProbe.ProbeHandler.HTTPGet.Path":    `"/"`,
		".Spec.Containers[0].LivenessProbe.ProbeHandler.HTTPGet.Scheme":  `"HTTP"`,
		".Spec.Containers[0].LivenessProbe.PeriodSeconds":                `10`,
		".Spec.Containers[0].LivenessProbe.SuccessThreshold":             `1`,
		".Spec.Containers[0].LivenessProbe.TimeoutSeconds":               `1`,
		".Spec.Containers[0].Ports[0].Protocol":                          `"TCP"`,
		".Spec.Containers[0].ReadinessProbe.FailureThreshold":            `3`,
		".Spec.Containers[0].ReadinessProbe.ProbeHandler.HTTPGet.Path":   `"/"`,
		".Spec.Containers[0].ReadinessProbe.ProbeHandler.HTTPGet.Scheme": `"HTTP"`,
		".Spec.Containers[0].ReadinessProbe.PeriodSeconds":               `10`,
		".Spec.Containers[0].ReadinessProbe.SuccessThreshold":            `1`,
		".Spec.Containers[0].ReadinessProbe.TimeoutSeconds":              `1`,
		".Spec.Containers[0].StartupProbe.FailureThreshold":              "3",
		".Spec.Containers[0].StartupProbe.ProbeHandler.HTTPGet.Path":     `"/"`,
		".Spec.Containers[0].StartupProbe.ProbeHandler.HTTPGet.Scheme":   `"HTTP"`,
		".Spec.Containers[0].StartupProbe.PeriodSeconds":                 "10",
		".Spec.Containers[0].StartupProbe.SuccessThreshold":              "1",
		".Spec.Containers[0].StartupProbe.TimeoutSeconds":                "1",
		".Spec.Containers[0].TerminationMessagePath":                     `"/dev/termination-log"`,
		".Spec.Containers[0].TerminationMessagePolicy":                   `"File"`,
		".Spec.Containers[0].LivenessProbe.ProbeHandler.GRPC.Service":    `""`,
		".Spec.Containers[0].ReadinessProbe.ProbeHandler.GRPC.Service":   `""`,
		".Spec.Containers[0].StartupProbe.ProbeHandler.GRPC.Service":     `""`,
		".Spec.DNSPolicy": `"ClusterFirst"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.ProbeHandler.HTTPGet.Path":    `"/"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.ProbeHandler.HTTPGet.Scheme":  `"HTTP"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.ProbeHandler.HTTPGet.Path":   `"/"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.ProbeHandler.HTTPGet.Scheme": `"HTTP"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.ProbeHandler.HTTPGet.Path":     `"/"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.ProbeHandler.HTTPGet.Scheme":   `"HTTP"`,
		".Spec.InitContainers[0].LivenessProbe.ProbeHandler.HTTPGet.Path":                                  `"/"`,
		".Spec.InitContainers[0].LivenessProbe.ProbeHandler.HTTPGet.Scheme":                                `"HTTP"`,
		".Spec.InitContainers[0].ReadinessProbe.ProbeHandler.HTTPGet.Path":                                 `"/"`,
		".Spec.InitContainers[0].ReadinessProbe.ProbeHandler.HTTPGet.Scheme":                               `"HTTP"`,
		".Spec.InitContainers[0].StartupProbe.ProbeHandler.HTTPGet.Path":                                   `"/"`,
		".Spec.InitContainers[0].StartupProbe.ProbeHandler.HTTPGet.Scheme":                                 `"HTTP"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.Env[0].ValueFrom.FieldRef.APIVersion":       `"v1"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ImagePullPolicy":                            `"IfNotPresent"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.Lifecycle.PostStart.HTTPGet.Path":           `"/"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.Lifecycle.PostStart.HTTPGet.Scheme":         `"HTTP"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.Lifecycle.PreStop.HTTPGet.Path":             `"/"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.Lifecycle.PreStop.HTTPGet.Scheme":           `"HTTP"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.FailureThreshold":             "3",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.ProbeHandler.GRPC.Service":    `""`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.PeriodSeconds":                "10",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.SuccessThreshold":             "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.TimeoutSeconds":               "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.Ports[0].Protocol":                          `"TCP"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.FailureThreshold":            "3",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.ProbeHandler.GRPC.Service":   `""`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.PeriodSeconds":               "10",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.SuccessThreshold":            "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.TimeoutSeconds":              "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.FailureThreshold":              "3",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.ProbeHandler.GRPC.Service":     `""`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.PeriodSeconds":                 "10",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.SuccessThreshold":              "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.TimeoutSeconds":                "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.TerminationMessagePath":                     `"/dev/termination-log"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.TerminationMessagePolicy":                   `"File"`,
		".Spec.InitContainers[0].Env[0].ValueFrom.FieldRef.APIVersion":                                     `"v1"`,
		".Spec.InitContainers[0].ImagePullPolicy":                                                          `"IfNotPresent"`,
		".Spec.InitContainers[0].Lifecycle.PostStart.HTTPGet.Path":                                         `"/"`,
		".Spec.InitContainers[0].Lifecycle.PostStart.HTTPGet.Scheme":                                       `"HTTP"`,
		".Spec.InitContainers[0].Lifecycle.PreStop.HTTPGet.Path":                                           `"/"`,
		".Spec.InitContainers[0].Lifecycle.PreStop.HTTPGet.Scheme":                                         `"HTTP"`,
		".Spec.InitContainers[0].LivenessProbe.FailureThreshold":                                           `3`,
		".Spec.InitContainers[0].LivenessProbe.ProbeHandler.GRPC.Service":                                  `""`,
		".Spec.InitContainers[0].LivenessProbe.PeriodSeconds":                                              `10`,
		".Spec.InitContainers[0].LivenessProbe.SuccessThreshold":                                           `1`,
		".Spec.InitContainers[0].LivenessProbe.TimeoutSeconds":                                             `1`,
		".Spec.InitContainers[0].Ports[0].Protocol":                                                        `"TCP"`,
		".Spec.InitContainers[0].ReadinessProbe.FailureThreshold":                                          `3`,
		".Spec.InitContainers[0].ReadinessProbe.ProbeHandler.GRPC.Service":                                 `""`,
		".Spec.InitContainers[0].ReadinessProbe.PeriodSeconds":                                             `10`,
		".Spec.InitContainers[0].ReadinessProbe.SuccessThreshold":                                          `1`,
		".Spec.InitContainers[0].ReadinessProbe.TimeoutSeconds":                                            `1`,
		".Spec.InitContainers[0].StartupProbe.FailureThreshold":                                            "3",
		".Spec.InitContainers[0].StartupProbe.ProbeHandler.GRPC.Service":                                   `""`,
		".Spec.InitContainers[0].StartupProbe.PeriodSeconds":                                               "10",
		".Spec.InitContainers[0].StartupProbe.SuccessThreshold":                                            "1",
		".Spec.InitContainers[0].StartupProbe.TimeoutSeconds":                                              "1",
		".Spec.InitContainers[0].TerminationMessagePath":                                                   `"/dev/termination-log"`,
		".Spec.InitContainers[0].TerminationMessagePolicy":                                                 `"File"`,
		".Spec.RestartPolicy":                                                                         `"Always"`,
		".Spec.SchedulerName":                                                                         `"default-scheduler"`,
		".Spec.SecurityContext":                                                                       `{}`,
		".Spec.TerminationGracePeriodSeconds":                                                         `30`,
		".Spec.Volumes[0].VolumeSource.AzureDisk.CachingMode":                                         `"ReadWrite"`,
		".Spec.Volumes[0].VolumeSource.AzureDisk.FSType":                                              `"ext4"`,
		".Spec.Volumes[0].VolumeSource.AzureDisk.Kind":                                                `"Shared"`,
		".Spec.Volumes[0].VolumeSource.AzureDisk.ReadOnly":                                            `false`,
		".Spec.Volumes[0].VolumeSource.ConfigMap.DefaultMode":                                         `420`,
		".Spec.Volumes[0].VolumeSource.DownwardAPI.DefaultMode":                                       `420`,
		".Spec.Volumes[0].VolumeSource.DownwardAPI.Items[0].FieldRef.APIVersion":                      `"v1"`,
		".Spec.Volumes[0].VolumeSource.EmptyDir":                                                      `{}`,
		".Spec.Volumes[0].VolumeSource.Ephemeral.VolumeClaimTemplate.Spec.VolumeMode":                 `"Filesystem"`,
		".Spec.Volumes[0].VolumeSource.HostPath.Type":                                                 `""`,
		".Spec.Volumes[0].VolumeSource.ISCSI.ISCSIInterface":                                          `"default"`,
		".Spec.Volumes[0].VolumeSource.Projected.DefaultMode":                                         `420`,
		".Spec.Volumes[0].VolumeSource.Projected.Sources[0].DownwardAPI.Items[0].FieldRef.APIVersion": `"v1"`,
		".Spec.Volumes[0].VolumeSource.Projected.Sources[0].ServiceAccountToken.ExpirationSeconds":    `3600`,
		".Spec.Volumes[0].VolumeSource.RBD.Keyring":                                                   `"/etc/ceph/keyring"`,
		".Spec.Volumes[0].VolumeSource.RBD.RBDPool":                                                   `"rbd"`,
		".Spec.Volumes[0].VolumeSource.RBD.RadosUser":                                                 `"admin"`,
		".Spec.Volumes[0].VolumeSource.ScaleIO.FSType":                                                `"xfs"`,
		".Spec.Volumes[0].VolumeSource.ScaleIO.StorageMode":                                           `"ThinProvisioned"`,
		".Spec.Volumes[0].VolumeSource.Secret.DefaultMode":                                            `420`,
	}
	t.Run("empty PodTemplateSpec", func(t *testing.T) {
		rc := &v1.ReplicationController{Spec: v1.ReplicationControllerSpec{Template: &v1.PodTemplateSpec{}}}
		template := rc.Spec.Template
		defaults := detectDefaults(t, rc, reflect.ValueOf(template))
		if !reflect.DeepEqual(expectedDefaults, defaults) {
			t.Errorf("Defaults for PodTemplateSpec changed. This can cause spurious rollouts of workloads on API server upgrade.")
			t.Logf(cmp.Diff(expectedDefaults, defaults))
		}
	})
	t.Run("hostnet PodTemplateSpec with ports", func(t *testing.T) {
		rc := &v1.ReplicationController{
			Spec: v1.ReplicationControllerSpec{
				Template: &v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						HostNetwork: true,
						Containers: []v1.Container{{
							Ports: []v1.ContainerPort{{
								ContainerPort: 12345,
								Protocol:      v1.ProtocolTCP,
							}},
						}},
					},
				},
			},
		}
		template := rc.Spec.Template
		defaults := detectDefaults(t, rc, reflect.ValueOf(template))
		expected := func() map[string]string {
			// Set values that are known inputs
			m := map[string]string{
				".Spec.HostNetwork":                          "true",
				".Spec.Containers[0].Ports[0].ContainerPort": "12345",
			}
			if utilfeature.DefaultFeatureGate.Enabled(features.DefaultHostNetworkHostPortsInPodTemplates) {
				m[".Spec.Containers"] = `[{"name":"","ports":[{"hostPort":12345,"containerPort":12345,"protocol":"TCP"}],"resources":{},"terminationMessagePath":"/dev/termination-log","terminationMessagePolicy":"File","imagePullPolicy":"IfNotPresent"}]`
				m[".Spec.Containers[0].Ports"] = `[{"hostPort":12345,"containerPort":12345,"protocol":"TCP"}]`
				m[".Spec.Containers[0].Ports[0].HostPort"] = "12345"
			} else {
				m[".Spec.Containers"] = `[{"name":"","ports":[{"containerPort":12345,"protocol":"TCP"}],"resources":{},"terminationMessagePath":"/dev/termination-log","terminationMessagePolicy":"File","imagePullPolicy":"IfNotPresent"}]`
				m[".Spec.Containers[0].Ports"] = `[{"containerPort":12345,"protocol":"TCP"}]`
			}
			for k, v := range expectedDefaults {
				if _, found := m[k]; !found {
					m[k] = v
				}
			}
			return m
		}()
		if !reflect.DeepEqual(expected, defaults) {
			t.Errorf("Defaults for PodTemplateSpec changed. This can cause spurious rollouts of workloads on API server upgrade.")
			t.Logf(cmp.Diff(expected, defaults))
		}
	})
}

// TestPodDefaults detects changes to defaults within PodSpec.
// Defaulting changes within this type (*especially* within containers) can cause kubelets to restart containers on API server update.
func TestPodDefaults(t *testing.T) {
	t.Run("enabled_features", func(t *testing.T) { testPodDefaults(t, true) })
	t.Run("disabled_features", func(t *testing.T) { testPodDefaults(t, false) })
}
func testPodDefaults(t *testing.T, featuresEnabled bool) {
	features := utilfeature.DefaultFeatureGate.DeepCopy().GetAll()
	for feature, featureSpec := range features {
		if !featureSpec.LockToDefault {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, feature, featuresEnabled)()
		}
	}
	pod := &v1.Pod{}
	// New defaults under PodSpec are only acceptable if they would not be applied when reading data from a previous release.
	// Forbidden: adding a new field `MyField *bool` and defaulting it to a non-nil value
	// Forbidden: defaulting an existing field `MyField *bool` when it was previously not defaulted
	// Forbidden: changing an existing default value
	// Allowed: adding a new field `MyContainer *MyType` and defaulting a child of that type (e.g. `MyContainer.MyChildField`) if and only if MyContainer is non-nil
	expectedDefaults := map[string]string{
		".Spec.Containers[0].Env[0].ValueFrom.FieldRef.APIVersion":       `"v1"`,
		".Spec.Containers[0].ImagePullPolicy":                            `"IfNotPresent"`,
		".Spec.Containers[0].Lifecycle.PostStart.HTTPGet.Path":           `"/"`,
		".Spec.Containers[0].Lifecycle.PostStart.HTTPGet.Scheme":         `"HTTP"`,
		".Spec.Containers[0].Lifecycle.PreStop.HTTPGet.Path":             `"/"`,
		".Spec.Containers[0].Lifecycle.PreStop.HTTPGet.Scheme":           `"HTTP"`,
		".Spec.Containers[0].LivenessProbe.FailureThreshold":             `3`,
		".Spec.Containers[0].LivenessProbe.ProbeHandler.HTTPGet.Path":    `"/"`,
		".Spec.Containers[0].LivenessProbe.ProbeHandler.HTTPGet.Scheme":  `"HTTP"`,
		".Spec.Containers[0].LivenessProbe.PeriodSeconds":                `10`,
		".Spec.Containers[0].LivenessProbe.SuccessThreshold":             `1`,
		".Spec.Containers[0].LivenessProbe.TimeoutSeconds":               `1`,
		".Spec.Containers[0].Ports[0].Protocol":                          `"TCP"`,
		".Spec.Containers[0].ReadinessProbe.FailureThreshold":            `3`,
		".Spec.Containers[0].ReadinessProbe.ProbeHandler.HTTPGet.Path":   `"/"`,
		".Spec.Containers[0].ReadinessProbe.ProbeHandler.HTTPGet.Scheme": `"HTTP"`,
		".Spec.Containers[0].ReadinessProbe.PeriodSeconds":               `10`,
		".Spec.Containers[0].ReadinessProbe.SuccessThreshold":            `1`,
		".Spec.Containers[0].ReadinessProbe.TimeoutSeconds":              `1`,
		".Spec.Containers[0].Resources.Requests":                         `{"":"0"}`, // this gets defaulted from the limits field
		".Spec.Containers[0].StartupProbe.FailureThreshold":              "3",
		".Spec.Containers[0].StartupProbe.ProbeHandler.HTTPGet.Path":     `"/"`,
		".Spec.Containers[0].StartupProbe.ProbeHandler.HTTPGet.Scheme":   `"HTTP"`,
		".Spec.Containers[0].StartupProbe.PeriodSeconds":                 "10",
		".Spec.Containers[0].StartupProbe.SuccessThreshold":              "1",
		".Spec.Containers[0].StartupProbe.TimeoutSeconds":                "1",
		".Spec.Containers[0].TerminationMessagePath":                     `"/dev/termination-log"`,
		".Spec.Containers[0].TerminationMessagePolicy":                   `"File"`,
		".Spec.Containers[0].LivenessProbe.ProbeHandler.GRPC.Service":    `""`,
		".Spec.Containers[0].ReadinessProbe.ProbeHandler.GRPC.Service":   `""`,
		".Spec.Containers[0].StartupProbe.ProbeHandler.GRPC.Service":     `""`,
		".Spec.DNSPolicy":          `"ClusterFirst"`,
		".Spec.EnableServiceLinks": `true`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.Env[0].ValueFrom.FieldRef.APIVersion":       `"v1"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ImagePullPolicy":                            `"IfNotPresent"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.Lifecycle.PostStart.HTTPGet.Path":           `"/"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.Lifecycle.PostStart.HTTPGet.Scheme":         `"HTTP"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.Lifecycle.PreStop.HTTPGet.Path":             `"/"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.Lifecycle.PreStop.HTTPGet.Scheme":           `"HTTP"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.FailureThreshold":             "3",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.PeriodSeconds":                "10",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.SuccessThreshold":             "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.TimeoutSeconds":               "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.Ports[0].Protocol":                          `"TCP"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.FailureThreshold":            "3",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.PeriodSeconds":               "10",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.SuccessThreshold":            "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.TimeoutSeconds":              "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.FailureThreshold":              "3",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.PeriodSeconds":                 "10",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.SuccessThreshold":              "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.TimeoutSeconds":                "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.TerminationMessagePath":                     `"/dev/termination-log"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.TerminationMessagePolicy":                   `"File"`,
		".Spec.InitContainers[0].Env[0].ValueFrom.FieldRef.APIVersion":                                     `"v1"`,
		".Spec.InitContainers[0].ImagePullPolicy":                                                          `"IfNotPresent"`,
		".Spec.InitContainers[0].Lifecycle.PostStart.HTTPGet.Path":                                         `"/"`,
		".Spec.InitContainers[0].Lifecycle.PostStart.HTTPGet.Scheme":                                       `"HTTP"`,
		".Spec.InitContainers[0].Lifecycle.PreStop.HTTPGet.Path":                                           `"/"`,
		".Spec.InitContainers[0].Lifecycle.PreStop.HTTPGet.Scheme":                                         `"HTTP"`,
		".Spec.InitContainers[0].LivenessProbe.FailureThreshold":                                           `3`,
		".Spec.InitContainers[0].LivenessProbe.PeriodSeconds":                                              `10`,
		".Spec.InitContainers[0].LivenessProbe.SuccessThreshold":                                           `1`,
		".Spec.InitContainers[0].LivenessProbe.TimeoutSeconds":                                             `1`,
		".Spec.InitContainers[0].Ports[0].Protocol":                                                        `"TCP"`,
		".Spec.InitContainers[0].ReadinessProbe.FailureThreshold":                                          `3`,
		".Spec.InitContainers[0].ReadinessProbe.PeriodSeconds":                                             `10`,
		".Spec.InitContainers[0].ReadinessProbe.SuccessThreshold":                                          `1`,
		".Spec.InitContainers[0].ReadinessProbe.TimeoutSeconds":                                            `1`,
		".Spec.InitContainers[0].Resources.Requests":                                                       `{"":"0"}`, // this gets defaulted from the limits field
		".Spec.InitContainers[0].TerminationMessagePath":                                                   `"/dev/termination-log"`,
		".Spec.InitContainers[0].TerminationMessagePolicy":                                                 `"File"`,
		".Spec.InitContainers[0].StartupProbe.FailureThreshold":                                            "3",
		".Spec.InitContainers[0].StartupProbe.PeriodSeconds":                                               "10",
		".Spec.InitContainers[0].StartupProbe.SuccessThreshold":                                            "1",
		".Spec.InitContainers[0].StartupProbe.TimeoutSeconds":                                              "1",
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.ProbeHandler.GRPC.Service":    `""`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.ProbeHandler.HTTPGet.Path":    `"/"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.LivenessProbe.ProbeHandler.HTTPGet.Scheme":  `"HTTP"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.ProbeHandler.GRPC.Service":   `""`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.ProbeHandler.HTTPGet.Path":   `"/"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.ReadinessProbe.ProbeHandler.HTTPGet.Scheme": `"HTTP"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.ProbeHandler.GRPC.Service":     `""`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.ProbeHandler.HTTPGet.Path":     `"/"`,
		".Spec.EphemeralContainers[0].EphemeralContainerCommon.StartupProbe.ProbeHandler.HTTPGet.Scheme":   `"HTTP"`,
		".Spec.InitContainers[0].LivenessProbe.ProbeHandler.GRPC.Service":                                  `""`,
		".Spec.InitContainers[0].LivenessProbe.ProbeHandler.HTTPGet.Path":                                  `"/"`,
		".Spec.InitContainers[0].LivenessProbe.ProbeHandler.HTTPGet.Scheme":                                `"HTTP"`,
		".Spec.InitContainers[0].ReadinessProbe.ProbeHandler.GRPC.Service":                                 `""`,
		".Spec.InitContainers[0].ReadinessProbe.ProbeHandler.HTTPGet.Path":                                 `"/"`,
		".Spec.InitContainers[0].ReadinessProbe.ProbeHandler.HTTPGet.Scheme":                               `"HTTP"`,
		".Spec.InitContainers[0].StartupProbe.ProbeHandler.GRPC.Service":                                   `""`,
		".Spec.InitContainers[0].StartupProbe.ProbeHandler.HTTPGet.Path":                                   `"/"`,
		".Spec.InitContainers[0].StartupProbe.ProbeHandler.HTTPGet.Scheme":                                 `"HTTP"`,
		".Spec.RestartPolicy":                                                                         `"Always"`,
		".Spec.SchedulerName":                                                                         `"default-scheduler"`,
		".Spec.SecurityContext":                                                                       `{}`,
		".Spec.TerminationGracePeriodSeconds":                                                         `30`,
		".Spec.Volumes[0].VolumeSource.AzureDisk.CachingMode":                                         `"ReadWrite"`,
		".Spec.Volumes[0].VolumeSource.AzureDisk.FSType":                                              `"ext4"`,
		".Spec.Volumes[0].VolumeSource.AzureDisk.Kind":                                                `"Shared"`,
		".Spec.Volumes[0].VolumeSource.AzureDisk.ReadOnly":                                            `false`,
		".Spec.Volumes[0].VolumeSource.ConfigMap.DefaultMode":                                         `420`,
		".Spec.Volumes[0].VolumeSource.DownwardAPI.DefaultMode":                                       `420`,
		".Spec.Volumes[0].VolumeSource.DownwardAPI.Items[0].FieldRef.APIVersion":                      `"v1"`,
		".Spec.Volumes[0].VolumeSource.EmptyDir":                                                      `{}`,
		".Spec.Volumes[0].VolumeSource.Ephemeral.VolumeClaimTemplate.Spec.VolumeMode":                 `"Filesystem"`,
		".Spec.Volumes[0].VolumeSource.HostPath.Type":                                                 `""`,
		".Spec.Volumes[0].VolumeSource.ISCSI.ISCSIInterface":                                          `"default"`,
		".Spec.Volumes[0].VolumeSource.Projected.DefaultMode":                                         `420`,
		".Spec.Volumes[0].VolumeSource.Projected.Sources[0].DownwardAPI.Items[0].FieldRef.APIVersion": `"v1"`,
		".Spec.Volumes[0].VolumeSource.Projected.Sources[0].ServiceAccountToken.ExpirationSeconds":    `3600`,
		".Spec.Volumes[0].VolumeSource.RBD.Keyring":                                                   `"/etc/ceph/keyring"`,
		".Spec.Volumes[0].VolumeSource.RBD.RBDPool":                                                   `"rbd"`,
		".Spec.Volumes[0].VolumeSource.RBD.RadosUser":                                                 `"admin"`,
		".Spec.Volumes[0].VolumeSource.ScaleIO.FSType":                                                `"xfs"`,
		".Spec.Volumes[0].VolumeSource.ScaleIO.StorageMode":                                           `"ThinProvisioned"`,
		".Spec.Volumes[0].VolumeSource.Secret.DefaultMode":                                            `420`,
	}
	defaults := detectDefaults(t, pod, reflect.ValueOf(pod))
	if !reflect.DeepEqual(expectedDefaults, defaults) {
		t.Errorf("Defaults for PodSpec changed. This can cause spurious restarts of containers on API server upgrade.")
		t.Logf(cmp.Diff(expectedDefaults, defaults))
	}
}

func TestPodHostNetworkDefaults(t *testing.T) {
	cases := []struct {
		name                 string
		gate                 bool
		hostNet              bool
		expectPodDefault     bool
		expectPodSpecDefault bool
	}{{
		name:                 "gate disabled, hostNetwork=false",
		gate:                 false,
		hostNet:              false,
		expectPodDefault:     false,
		expectPodSpecDefault: false,
	}, {
		name:                 "gate disabled, hostNetwork=true",
		gate:                 false,
		hostNet:              true,
		expectPodDefault:     true,
		expectPodSpecDefault: false,
	}, {
		name:                 "gate enabled, hostNetwork=false",
		gate:                 true,
		hostNet:              false,
		expectPodDefault:     false,
		expectPodSpecDefault: false,
	}, {
		name:                 "gate enabled, hostNetwork=true",
		gate:                 true,
		hostNet:              true,
		expectPodDefault:     true,
		expectPodSpecDefault: true,
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DefaultHostNetworkHostPortsInPodTemplates, tc.gate)()

			const portNum = 12345
			spec := v1.PodSpec{
				HostNetwork: tc.hostNet,
				Containers: []v1.Container{{
					Ports: []v1.ContainerPort{{
						ContainerPort: portNum,
						Protocol:      v1.ProtocolTCP,
						// Note: HostPort is not set
					}},
				}},
			}

			// Test Pod defaulting.
			p := v1.Pod{Spec: *spec.DeepCopy()}
			corev1.SetDefaults_Pod(&p)
			if got := p.Spec.Containers[0].Ports[0].HostPort; tc.expectPodDefault && got == 0 {
				t.Errorf("expected Pod HostPort to be defaulted, got %v", got)
			}
			if got := p.Spec.Containers[0].Ports[0].HostPort; !tc.expectPodDefault && got != 0 {
				t.Errorf("expected Pod HostPort to be 0, got %v", got)
			}

			// Test PodSpec defaulting.
			s := spec.DeepCopy()
			corev1.SetDefaults_PodSpec(s)
			if got := s.Containers[0].Ports[0].HostPort; tc.expectPodSpecDefault && got == 0 {
				t.Errorf("expected PodSpec HostPort to be defaulted, got %v", got)
			}
			if got := s.Containers[0].Ports[0].HostPort; !tc.expectPodSpecDefault && got != 0 {
				t.Errorf("expected PodSpec HostPort to be 0, got %v", got)
			}
		})
	}
}

type testPath struct {
	path  string
	value reflect.Value
}

func detectDefaults(t *testing.T, obj runtime.Object, v reflect.Value) map[string]string {
	defaults := map[string]string{}
	toVisit := []testPath{{path: "", value: v}}

	for len(toVisit) > 0 {
		visit := toVisit[0]
		toVisit = toVisit[1:]

		legacyscheme.Scheme.Default(obj)
		defaultedV := visit.value
		zeroV := reflect.Zero(visit.value.Type())

		switch {
		case visit.value.Kind() == reflect.Struct:
			for fi := 0; fi < visit.value.NumField(); fi++ {
				structField := visit.value.Type().Field(fi)
				valueField := visit.value.Field(fi)
				if valueField.CanSet() {
					toVisit = append(toVisit, testPath{path: visit.path + "." + structField.Name, value: valueField})
				}
			}

		case visit.value.Kind() == reflect.Slice:
			if !visit.value.IsNil() {
				// if we already have a value, we either got defaulted or there
				// was a fixed input - flag it and see if we can descend
				// anyway.
				marshaled, _ := json.Marshal(defaultedV.Interface())
				defaults[visit.path] = string(marshaled)
				toVisit = append(toVisit, testPath{path: visit.path + "[0]", value: visit.value.Index(0)})
			} else if visit.value.Type().Elem().Kind() == reflect.Struct {
				if strings.HasPrefix(visit.path, ".ObjectMeta.ManagedFields[") {
					break
				}
				// if we don't already have a value, and contain structs, add an empty item so we can recurse
				item := reflect.New(visit.value.Type().Elem()).Elem()
				visit.value.Set(reflect.Append(visit.value, item))
				toVisit = append(toVisit, testPath{path: visit.path + "[0]", value: visit.value.Index(0)})
			} else if !isPrimitive(visit.value.Type().Elem().Kind()) {
				t.Logf("unhandled non-primitive slice type %s: %s", visit.path, visit.value.Type().Elem())
			}

		case visit.value.Kind() == reflect.Map:
			if !visit.value.IsNil() {
				// if we already have a value, we got defaulted
				marshaled, _ := json.Marshal(defaultedV.Interface())
				defaults[visit.path] = string(marshaled)
			} else if visit.value.Type().Key().Kind() == reflect.String && visit.value.Type().Elem().Kind() == reflect.Struct {
				if strings.HasPrefix(visit.path, ".ObjectMeta.ManagedFields[") {
					break
				}
				// if we don't already have a value, and contain structs, add an empty item so we can recurse
				item := reflect.New(visit.value.Type().Elem()).Elem()
				visit.value.Set(reflect.MakeMap(visit.value.Type()))
				visit.value.SetMapIndex(reflect.New(visit.value.Type().Key()).Elem(), item)
				toVisit = append(toVisit, testPath{path: visit.path + "[*]", value: item})
			} else if !isPrimitive(visit.value.Type().Elem().Kind()) {
				t.Logf("unhandled non-primitive map type %s: %s", visit.path, visit.value.Type().Elem())
			}

		case visit.value.Kind() == reflect.Pointer:
			if visit.value.IsNil() {
				if visit.value.Type().Elem().Kind() == reflect.Struct {
					visit.value.Set(reflect.New(visit.value.Type().Elem()))
					toVisit = append(toVisit, testPath{path: visit.path, value: visit.value.Elem()})
				} else if !isPrimitive(visit.value.Type().Elem().Kind()) {
					t.Errorf("unhandled non-primitive nil ptr: %s: %s", visit.path, visit.value.Type())
				}
			} else {
				if visit.path != "" {
					marshaled, _ := json.Marshal(defaultedV.Interface())
					defaults[visit.path] = string(marshaled)
				}
				toVisit = append(toVisit, testPath{path: visit.path, value: visit.value.Elem()})
			}

		case isPrimitive(visit.value.Kind()):
			if !reflect.DeepEqual(defaultedV.Interface(), zeroV.Interface()) {
				marshaled, _ := json.Marshal(defaultedV.Interface())
				defaults[visit.path] = string(marshaled)
			}

		default:
			t.Errorf("unhandled kind: %s: %s", visit.path, visit.value.Type())
		}

	}
	return defaults
}

func isPrimitive(k reflect.Kind) bool {
	switch k {
	case reflect.String, reflect.Bool, reflect.Int32, reflect.Int64, reflect.Int:
		return true
	default:
		return false
	}
}

func roundTrip(t *testing.T, obj runtime.Object) runtime.Object {
	codec := legacyscheme.Codecs.LegacyCodec(corev1.SchemeGroupVersion)
	data, err := runtime.Encode(codec, obj)
	if err != nil {
		t.Errorf("%v\n %#v", err, obj)
		return nil
	}
	obj2, err := runtime.Decode(codec, data)
	if err != nil {
		t.Errorf("%v\nData: %s\nSource: %#v", err, string(data), obj)
		return nil
	}
	obj3 := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = legacyscheme.Scheme.Convert(obj2, obj3, nil)
	if err != nil {
		t.Errorf("%v\nSource: %#v", err, obj2)
		return nil
	}
	return obj3
}

func TestSetDefaultReplicationController(t *testing.T) {
	tests := []struct {
		rc             *v1.ReplicationController
		expectLabels   bool
		expectSelector bool
	}{
		{
			rc: &v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   true,
			expectSelector: true,
		},
		{
			rc: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   false,
			expectSelector: true,
		},
		{
			rc: &v1.ReplicationController{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"bar": "foo",
					},
				},
				Spec: v1.ReplicationControllerSpec{
					Selector: map[string]string{
						"some": "other",
					},
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   false,
			expectSelector: false,
		},
		{
			rc: &v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Selector: map[string]string{
						"some": "other",
					},
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectLabels:   true,
			expectSelector: false,
		},
	}

	for _, test := range tests {
		rc := test.rc
		obj2 := roundTrip(t, runtime.Object(rc))
		rc2, ok := obj2.(*v1.ReplicationController)
		if !ok {
			t.Errorf("unexpected object: %v", rc2)
			t.FailNow()
		}
		if test.expectSelector != reflect.DeepEqual(rc2.Spec.Selector, rc2.Spec.Template.Labels) {
			if test.expectSelector {
				t.Errorf("expected: %v, got: %v", rc2.Spec.Template.Labels, rc2.Spec.Selector)
			} else {
				t.Errorf("unexpected equality: %v", rc.Spec.Selector)
			}
		}
		if test.expectLabels != reflect.DeepEqual(rc2.Labels, rc2.Spec.Template.Labels) {
			if test.expectLabels {
				t.Errorf("expected: %v, got: %v", rc2.Spec.Template.Labels, rc2.Labels)
			} else {
				t.Errorf("unexpected equality: %v", rc.Labels)
			}
		}
	}
}

func TestSetDefaultReplicationControllerReplicas(t *testing.T) {
	tests := []struct {
		rc             v1.ReplicationController
		expectReplicas int32
	}{
		{
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectReplicas: 1,
		},
		{
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Replicas: utilpointer.Int32(0),
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectReplicas: 0,
		},
		{
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Replicas: utilpointer.Int32(3),
					Template: &v1.PodTemplateSpec{
						ObjectMeta: metav1.ObjectMeta{
							Labels: map[string]string{
								"foo": "bar",
							},
						},
					},
				},
			},
			expectReplicas: 3,
		},
	}

	for _, test := range tests {
		rc := &test.rc
		obj2 := roundTrip(t, runtime.Object(rc))
		rc2, ok := obj2.(*v1.ReplicationController)
		if !ok {
			t.Errorf("unexpected object: %v", rc2)
			t.FailNow()
		}
		if rc2.Spec.Replicas == nil {
			t.Errorf("unexpected nil Replicas")
		} else if test.expectReplicas != *rc2.Spec.Replicas {
			t.Errorf("expected: %d replicas, got: %d", test.expectReplicas, *rc2.Spec.Replicas)
		}
	}
}

type InitContainerValidator func(got, expected *v1.Container) error

func TestSetDefaultReplicationControllerInitContainers(t *testing.T) {
	assertEnvFieldRef := func(got, expected *v1.Container) error {
		if len(got.Env) != len(expected.Env) {
			return fmt.Errorf("different number of env: got <%v>, expected <%v>", len(got.Env), len(expected.Env))
		}

		for j := range got.Env {
			ge := &got.Env[j]
			ee := &expected.Env[j]

			if ge.Name != ee.Name {
				return fmt.Errorf("different name of env: got <%v>, expected <%v>", ge.Name, ee.Name)
			}

			if ge.ValueFrom.FieldRef.APIVersion != ee.ValueFrom.FieldRef.APIVersion {
				return fmt.Errorf("different api version of FieldRef <%v>: got <%v>, expected <%v>",
					ge.Name, ge.ValueFrom.FieldRef.APIVersion, ee.ValueFrom.FieldRef.APIVersion)
			}
		}
		return nil
	}

	assertImagePullPolicy := func(got, expected *v1.Container) error {
		if got.ImagePullPolicy != expected.ImagePullPolicy {
			return fmt.Errorf("different image pull policy: got <%v>, expected <%v>", got.ImagePullPolicy, expected.ImagePullPolicy)
		}
		return nil
	}

	assertContainerPort := func(got, expected *v1.Container) error {
		if len(got.Ports) != len(expected.Ports) {
			return fmt.Errorf("different number of ports: got <%v>, expected <%v>", len(got.Ports), len(expected.Ports))
		}

		for i := range got.Ports {
			gp := &got.Ports[i]
			ep := &expected.Ports[i]

			if gp.Name != ep.Name {
				return fmt.Errorf("different name of port: got <%v>, expected <%v>", gp.Name, ep.Name)
			}

			if gp.Protocol != ep.Protocol {
				return fmt.Errorf("different port protocol <%v>: got <%v>, expected <%v>", gp.Name, gp.Protocol, ep.Protocol)
			}
		}

		return nil
	}

	assertResource := func(got, expected *v1.Container) error {
		if len(got.Resources.Limits) != len(expected.Resources.Limits) {
			return fmt.Errorf("different number of resources.Limits: got <%v>, expected <%v>", len(got.Resources.Limits), (expected.Resources.Limits))
		}

		for k, v := range got.Resources.Limits {
			if ev, found := expected.Resources.Limits[v1.ResourceName(k)]; !found {
				return fmt.Errorf("failed to find resource <%v> in expected resources.Limits.", k)
			} else {
				if ev.Value() != v.Value() {
					return fmt.Errorf("different resource.Limits: got <%v>, expected <%v>.", v.Value(), ev.Value())
				}
			}
		}

		if len(got.Resources.Requests) != len(expected.Resources.Requests) {
			return fmt.Errorf("different number of resources.Requests: got <%v>, expected <%v>", len(got.Resources.Requests), (expected.Resources.Requests))
		}

		for k, v := range got.Resources.Requests {
			if ev, found := expected.Resources.Requests[v1.ResourceName(k)]; !found {
				return fmt.Errorf("failed to find resource <%v> in expected resources.Requests.", k)
			} else {
				if ev.Value() != v.Value() {
					return fmt.Errorf("different resource.Requests: got <%v>, expected <%v>.", v.Value(), ev.Value())
				}
			}
		}

		return nil
	}

	assertProb := func(got, expected *v1.Container) error {
		// Assert LivenessProbe
		if got.LivenessProbe.ProbeHandler.HTTPGet.Path != expected.LivenessProbe.ProbeHandler.HTTPGet.Path ||
			got.LivenessProbe.ProbeHandler.HTTPGet.Scheme != expected.LivenessProbe.ProbeHandler.HTTPGet.Scheme ||
			got.LivenessProbe.FailureThreshold != expected.LivenessProbe.FailureThreshold ||
			got.LivenessProbe.SuccessThreshold != expected.LivenessProbe.SuccessThreshold ||
			got.LivenessProbe.PeriodSeconds != expected.LivenessProbe.PeriodSeconds ||
			got.LivenessProbe.TimeoutSeconds != expected.LivenessProbe.TimeoutSeconds {
			return fmt.Errorf("different LivenessProbe: got <%v>, expected <%v>", got.LivenessProbe, expected.LivenessProbe)
		}

		// Assert ReadinessProbe
		if got.ReadinessProbe.ProbeHandler.HTTPGet.Path != expected.ReadinessProbe.ProbeHandler.HTTPGet.Path ||
			got.ReadinessProbe.ProbeHandler.HTTPGet.Scheme != expected.ReadinessProbe.ProbeHandler.HTTPGet.Scheme ||
			got.ReadinessProbe.FailureThreshold != expected.ReadinessProbe.FailureThreshold ||
			got.ReadinessProbe.SuccessThreshold != expected.ReadinessProbe.SuccessThreshold ||
			got.ReadinessProbe.PeriodSeconds != expected.ReadinessProbe.PeriodSeconds ||
			got.ReadinessProbe.TimeoutSeconds != expected.ReadinessProbe.TimeoutSeconds {
			return fmt.Errorf("different ReadinessProbe: got <%v>, expected <%v>", got.ReadinessProbe, expected.ReadinessProbe)
		}

		return nil
	}

	assertLifeCycle := func(got, expected *v1.Container) error {
		if got.Lifecycle.PostStart.HTTPGet.Path != expected.Lifecycle.PostStart.HTTPGet.Path ||
			got.Lifecycle.PostStart.HTTPGet.Scheme != expected.Lifecycle.PostStart.HTTPGet.Scheme {
			return fmt.Errorf("different LifeCycle: got <%v>, expected <%v>", got.Lifecycle, expected.Lifecycle)
		}

		return nil
	}

	cpu, _ := resource.ParseQuantity("100m")
	mem, _ := resource.ParseQuantity("100Mi")

	tests := []struct {
		name       string
		rc         v1.ReplicationController
		expected   []v1.Container
		validators []InitContainerValidator
	}{
		{
			name: "imagePullIPolicy",
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							InitContainers: []v1.Container{
								{
									Name:  "install",
									Image: "busybox",
								},
							},
						},
					},
				},
			},
			expected: []v1.Container{
				{
					ImagePullPolicy: v1.PullAlways,
				},
			},
			validators: []InitContainerValidator{assertImagePullPolicy},
		},
		{
			name: "FieldRef",
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							InitContainers: []v1.Container{
								{
									Name:  "fun",
									Image: "alpine",
									Env: []v1.EnvVar{
										{
											Name: "MY_POD_IP",
											ValueFrom: &v1.EnvVarSource{
												FieldRef: &v1.ObjectFieldSelector{
													APIVersion: "",
													FieldPath:  "status.podIP",
												},
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: []v1.Container{
				{
					Env: []v1.EnvVar{
						{
							Name: "MY_POD_IP",
							ValueFrom: &v1.EnvVarSource{
								FieldRef: &v1.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "status.podIP",
								},
							},
						},
					},
				},
			},
			validators: []InitContainerValidator{assertEnvFieldRef},
		},
		{
			name: "ContainerPort",
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							InitContainers: []v1.Container{
								{
									Name:  "fun",
									Image: "alpine",
									Ports: []v1.ContainerPort{
										{
											Name: "default",
										},
									},
								},
							},
						},
					},
				},
			},
			expected: []v1.Container{
				{
					Ports: []v1.ContainerPort{
						{
							Name:     "default",
							Protocol: v1.ProtocolTCP,
						},
					},
				},
			},
			validators: []InitContainerValidator{assertContainerPort},
		},
		{
			name: "Resources",
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							InitContainers: []v1.Container{
								{
									Name:  "fun",
									Image: "alpine",
									Resources: v1.ResourceRequirements{
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("100m"),
											v1.ResourceMemory: resource.MustParse("100Mi"),
										},
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("100m"),
											v1.ResourceMemory: resource.MustParse("100Mi"),
										},
									},
								},
							},
						},
					},
				},
			},
			expected: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU:    cpu,
							v1.ResourceMemory: mem,
						},
						Requests: v1.ResourceList{
							v1.ResourceCPU:    cpu,
							v1.ResourceMemory: mem,
						},
					},
				},
			},
			validators: []InitContainerValidator{assertResource},
		},
		{
			name: "Probe",
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							InitContainers: []v1.Container{
								{
									Name:  "fun",
									Image: "alpine",
									LivenessProbe: &v1.Probe{
										ProbeHandler: v1.ProbeHandler{
											HTTPGet: &v1.HTTPGetAction{
												Host: "localhost",
											},
										},
									},
									ReadinessProbe: &v1.Probe{
										ProbeHandler: v1.ProbeHandler{
											HTTPGet: &v1.HTTPGetAction{
												Host: "localhost",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: []v1.Container{
				{
					LivenessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{
							HTTPGet: &v1.HTTPGetAction{
								Path:   "/",
								Scheme: v1.URISchemeHTTP,
							},
						},
						TimeoutSeconds:   1,
						PeriodSeconds:    10,
						SuccessThreshold: 1,
						FailureThreshold: 3,
					},
					ReadinessProbe: &v1.Probe{
						ProbeHandler: v1.ProbeHandler{
							HTTPGet: &v1.HTTPGetAction{
								Path:   "/",
								Scheme: v1.URISchemeHTTP,
							},
						},
						TimeoutSeconds:   1,
						PeriodSeconds:    10,
						SuccessThreshold: 1,
						FailureThreshold: 3,
					},
				},
			},
			validators: []InitContainerValidator{assertProb},
		},
		{
			name: "LifeCycle",
			rc: v1.ReplicationController{
				Spec: v1.ReplicationControllerSpec{
					Template: &v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							InitContainers: []v1.Container{
								{
									Name:  "fun",
									Image: "alpine",
									Ports: []v1.ContainerPort{
										{
											Name: "default",
										},
									},
									Lifecycle: &v1.Lifecycle{
										PostStart: &v1.LifecycleHandler{
											HTTPGet: &v1.HTTPGetAction{
												Host: "localhost",
											},
										},
										PreStop: &v1.LifecycleHandler{
											HTTPGet: &v1.HTTPGetAction{
												Host: "localhost",
											},
										},
									},
								},
							},
						},
					},
				},
			},
			expected: []v1.Container{
				{
					Lifecycle: &v1.Lifecycle{
						PostStart: &v1.LifecycleHandler{
							HTTPGet: &v1.HTTPGetAction{
								Path:   "/",
								Scheme: v1.URISchemeHTTP,
							},
						},
						PreStop: &v1.LifecycleHandler{
							HTTPGet: &v1.HTTPGetAction{
								Path:   "/",
								Scheme: v1.URISchemeHTTP,
							},
						},
					},
				},
			},
			validators: []InitContainerValidator{assertLifeCycle},
		},
	}

	assertInitContainers := func(got, expected []v1.Container, validators []InitContainerValidator) error {
		if len(got) != len(expected) {
			return fmt.Errorf("different number of init container: got <%d>, expected <%d>",
				len(got), len(expected))
		}

		for i := range got {
			g := &got[i]
			e := &expected[i]

			for _, validator := range validators {
				if err := validator(g, e); err != nil {
					return err
				}
			}
		}

		return nil
	}

	for _, test := range tests {
		rc := &test.rc
		obj2 := roundTrip(t, runtime.Object(rc))
		rc2, ok := obj2.(*v1.ReplicationController)
		if !ok {
			t.Errorf("unexpected object: %v", rc2)
			t.FailNow()
		}

		if err := assertInitContainers(rc2.Spec.Template.Spec.InitContainers, test.expected, test.validators); err != nil {
			t.Errorf("test %v failed: %v", test.name, err)
		}
	}
}

func TestSetDefaultService(t *testing.T) {
	svc := &v1.Service{}
	obj2 := roundTrip(t, runtime.Object(svc))
	svc2 := obj2.(*v1.Service)
	if svc2.Spec.SessionAffinity != v1.ServiceAffinityNone {
		t.Errorf("Expected default session affinity type:%s, got: %s", v1.ServiceAffinityNone, svc2.Spec.SessionAffinity)
	}
	if svc2.Spec.SessionAffinityConfig != nil {
		t.Errorf("Expected empty session affinity config when session affinity type: %s, got: %v", v1.ServiceAffinityNone, svc2.Spec.SessionAffinityConfig)
	}
	if svc2.Spec.Type != v1.ServiceTypeClusterIP {
		t.Errorf("Expected default type:%s, got: %s", v1.ServiceTypeClusterIP, svc2.Spec.Type)
	}
}

func TestSetDefaultServiceSessionAffinityConfig(t *testing.T) {
	testCases := map[string]v1.Service{
		"SessionAffinityConfig is empty": {
			Spec: v1.ServiceSpec{
				SessionAffinity:       v1.ServiceAffinityClientIP,
				SessionAffinityConfig: nil,
			},
		},
		"ClientIP is empty": {
			Spec: v1.ServiceSpec{
				SessionAffinity: v1.ServiceAffinityClientIP,
				SessionAffinityConfig: &v1.SessionAffinityConfig{
					ClientIP: nil,
				},
			},
		},
		"TimeoutSeconds is empty": {
			Spec: v1.ServiceSpec{
				SessionAffinity: v1.ServiceAffinityClientIP,
				SessionAffinityConfig: &v1.SessionAffinityConfig{
					ClientIP: &v1.ClientIPConfig{
						TimeoutSeconds: nil,
					},
				},
			},
		},
	}
	for name, test := range testCases {
		obj2 := roundTrip(t, runtime.Object(&test))
		svc2 := obj2.(*v1.Service)
		if svc2.Spec.SessionAffinityConfig == nil || svc2.Spec.SessionAffinityConfig.ClientIP == nil || svc2.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds == nil {
			t.Fatalf("Case: %s, unexpected empty SessionAffinityConfig/ClientIP/TimeoutSeconds when session affinity type: %s, got: %v", name, v1.ServiceAffinityClientIP, svc2.Spec.SessionAffinityConfig)
		}
		if *svc2.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds != v1.DefaultClientIPServiceAffinitySeconds {
			t.Errorf("Case: %s, default TimeoutSeconds should be %d when session affinity type: %s, got: %d", name, v1.DefaultClientIPServiceAffinitySeconds, v1.ServiceAffinityClientIP, *svc2.Spec.SessionAffinityConfig.ClientIP.TimeoutSeconds)
		}
	}
}

func TestSetDefaultServiceLoadbalancerIPMode(t *testing.T) {
	modeVIP := v1.LoadBalancerIPModeVIP
	modeProxy := v1.LoadBalancerIPModeProxy
	testCases := []struct {
		name           string
		ipModeEnabled  bool
		svc            *v1.Service
		expectedIPMode []*v1.LoadBalancerIPMode
	}{
		{
			name:          "Set IP but not set IPMode with LoadbalancerIPMode disabled",
			ipModeEnabled: false,
			svc: &v1.Service{
				Spec: v1.ServiceSpec{Type: v1.ServiceTypeLoadBalancer},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{{
							IP: "1.2.3.4",
						}},
					},
				}},
			expectedIPMode: []*v1.LoadBalancerIPMode{nil},
		}, {
			name:          "Set IP but bot set IPMode with LoadbalancerIPMode enabled",
			ipModeEnabled: true,
			svc: &v1.Service{
				Spec: v1.ServiceSpec{Type: v1.ServiceTypeLoadBalancer},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{{
							IP: "1.2.3.4",
						}},
					},
				}},
			expectedIPMode: []*v1.LoadBalancerIPMode{&modeVIP},
		}, {
			name:          "Both IP and IPMode are set with LoadbalancerIPMode enabled",
			ipModeEnabled: true,
			svc: &v1.Service{
				Spec: v1.ServiceSpec{Type: v1.ServiceTypeLoadBalancer},
				Status: v1.ServiceStatus{
					LoadBalancer: v1.LoadBalancerStatus{
						Ingress: []v1.LoadBalancerIngress{{
							IP:     "1.2.3.4",
							IPMode: &modeProxy,
						}},
					},
				}},
			expectedIPMode: []*v1.LoadBalancerIPMode{&modeProxy},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.LoadBalancerIPMode, tc.ipModeEnabled)()
			obj := roundTrip(t, runtime.Object(tc.svc))
			svc := obj.(*v1.Service)
			for i, s := range svc.Status.LoadBalancer.Ingress {
				got := s.IPMode
				expected := tc.expectedIPMode[i]
				if !reflect.DeepEqual(got, expected) {
					t.Errorf("Expected IPMode %v, got %v", tc.expectedIPMode[i], s.IPMode)
				}
			}
		})
	}
}

func TestSetDefaultSecretVolumeSource(t *testing.T) {
	s := v1.PodSpec{}
	s.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				Secret: &v1.SecretVolumeSource{},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*v1.Pod)
	defaultMode := pod2.Spec.Volumes[0].VolumeSource.Secret.DefaultMode
	expectedMode := v1.SecretVolumeSourceDefaultMode

	if defaultMode == nil || *defaultMode != expectedMode {
		t.Errorf("Expected secret DefaultMode %v, got %v", expectedMode, defaultMode)
	}
}

func TestSetDefaultConfigMapVolumeSource(t *testing.T) {
	s := v1.PodSpec{}
	s.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				ConfigMap: &v1.ConfigMapVolumeSource{},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*v1.Pod)
	defaultMode := pod2.Spec.Volumes[0].VolumeSource.ConfigMap.DefaultMode
	expectedMode := v1.ConfigMapVolumeSourceDefaultMode

	if defaultMode == nil || *defaultMode != expectedMode {
		t.Errorf("Expected v1.ConfigMap DefaultMode %v, got %v", expectedMode, defaultMode)
	}
}

func TestSetDefaultDownwardAPIVolumeSource(t *testing.T) {
	s := v1.PodSpec{}
	s.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				DownwardAPI: &v1.DownwardAPIVolumeSource{},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*v1.Pod)
	defaultMode := pod2.Spec.Volumes[0].VolumeSource.DownwardAPI.DefaultMode
	expectedMode := v1.DownwardAPIVolumeSourceDefaultMode

	if defaultMode == nil || *defaultMode != expectedMode {
		t.Errorf("Expected DownwardAPI DefaultMode %v, got %v", expectedMode, defaultMode)
	}
}

func TestSetDefaultProjectedVolumeSource(t *testing.T) {
	s := v1.PodSpec{}
	s.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				Projected: &v1.ProjectedVolumeSource{},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*v1.Pod)
	defaultMode := pod2.Spec.Volumes[0].VolumeSource.Projected.DefaultMode
	expectedMode := v1.ProjectedVolumeSourceDefaultMode

	if defaultMode == nil || *defaultMode != expectedMode {
		t.Errorf("Expected v1.ProjectedVolumeSource DefaultMode %v, got %v", expectedMode, defaultMode)
	}
}

func TestSetDefaultSecret(t *testing.T) {
	s := &v1.Secret{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*v1.Secret)

	if s2.Type != v1.SecretTypeOpaque {
		t.Errorf("Expected secret type %v, got %v", v1.SecretTypeOpaque, s2.Type)
	}
}

func TestSetDefaultPersistentVolume(t *testing.T) {
	fsMode := v1.PersistentVolumeFilesystem
	blockMode := v1.PersistentVolumeBlock

	tests := []struct {
		name               string
		volumeMode         *v1.PersistentVolumeMode
		expectedVolumeMode v1.PersistentVolumeMode
	}{
		{
			name:               "volume mode nil",
			volumeMode:         nil,
			expectedVolumeMode: v1.PersistentVolumeFilesystem,
		},
		{
			name:               "volume mode filesystem",
			volumeMode:         &fsMode,
			expectedVolumeMode: v1.PersistentVolumeFilesystem,
		},
		{
			name:               "volume mode block",
			volumeMode:         &blockMode,
			expectedVolumeMode: v1.PersistentVolumeBlock,
		},
	}

	for _, test := range tests {
		pv := &v1.PersistentVolume{
			Spec: v1.PersistentVolumeSpec{
				VolumeMode: test.volumeMode,
			},
		}
		obj1 := roundTrip(t, runtime.Object(pv))
		pv1 := obj1.(*v1.PersistentVolume)
		if pv1.Status.Phase != v1.VolumePending {
			t.Errorf("Expected claim phase %v, got %v", v1.ClaimPending, pv1.Status.Phase)
		}
		if pv1.Spec.PersistentVolumeReclaimPolicy != v1.PersistentVolumeReclaimRetain {
			t.Errorf("Expected pv reclaim policy %v, got %v", v1.PersistentVolumeReclaimRetain, pv1.Spec.PersistentVolumeReclaimPolicy)
		}
		if *pv1.Spec.VolumeMode != test.expectedVolumeMode {
			t.Errorf("Test %s failed, Expected VolumeMode: %v, but got %v", test.name, test.volumeMode, *pv1.Spec.VolumeMode)
		}
	}
}

func TestSetDefaultPersistentVolumeClaim(t *testing.T) {
	fsMode := v1.PersistentVolumeFilesystem
	blockMode := v1.PersistentVolumeBlock

	tests := []struct {
		name               string
		volumeMode         *v1.PersistentVolumeMode
		expectedVolumeMode v1.PersistentVolumeMode
	}{
		{
			name:               "volume mode nil",
			volumeMode:         nil,
			expectedVolumeMode: v1.PersistentVolumeFilesystem,
		},
		{
			name:               "volume mode filesystem",
			volumeMode:         &fsMode,
			expectedVolumeMode: v1.PersistentVolumeFilesystem,
		},
		{
			name:               "volume mode block",
			volumeMode:         &blockMode,
			expectedVolumeMode: v1.PersistentVolumeBlock,
		},
	}

	for _, test := range tests {
		pvc := &v1.PersistentVolumeClaim{
			Spec: v1.PersistentVolumeClaimSpec{
				VolumeMode: test.volumeMode,
			},
		}
		obj1 := roundTrip(t, runtime.Object(pvc))
		pvc1 := obj1.(*v1.PersistentVolumeClaim)
		if pvc1.Status.Phase != v1.ClaimPending {
			t.Errorf("Expected claim phase %v, got %v", v1.ClaimPending, pvc1.Status.Phase)
		}
		if *pvc1.Spec.VolumeMode != test.expectedVolumeMode {
			t.Errorf("Test %s failed, Expected VolumeMode: %v, but got %v", test.name, test.volumeMode, *pvc1.Spec.VolumeMode)
		}
	}
}

func TestSetDefaultEphemeral(t *testing.T) {
	fsMode := v1.PersistentVolumeFilesystem
	blockMode := v1.PersistentVolumeBlock

	tests := []struct {
		name               string
		volumeMode         *v1.PersistentVolumeMode
		expectedVolumeMode v1.PersistentVolumeMode
	}{
		{
			name:               "volume mode nil",
			volumeMode:         nil,
			expectedVolumeMode: v1.PersistentVolumeFilesystem,
		},
		{
			name:               "volume mode filesystem",
			volumeMode:         &fsMode,
			expectedVolumeMode: v1.PersistentVolumeFilesystem,
		},
		{
			name:               "volume mode block",
			volumeMode:         &blockMode,
			expectedVolumeMode: v1.PersistentVolumeBlock,
		},
	}

	for _, test := range tests {
		pod := &v1.Pod{
			Spec: v1.PodSpec{
				Volumes: []v1.Volume{
					{
						VolumeSource: v1.VolumeSource{
							Ephemeral: &v1.EphemeralVolumeSource{
								VolumeClaimTemplate: &v1.PersistentVolumeClaimTemplate{
									Spec: v1.PersistentVolumeClaimSpec{
										VolumeMode: test.volumeMode,
									},
								},
							},
						},
					},
				},
			},
		}
		obj1 := roundTrip(t, runtime.Object(pod))
		pod1 := obj1.(*v1.Pod)
		if *pod1.Spec.Volumes[0].VolumeSource.Ephemeral.VolumeClaimTemplate.Spec.VolumeMode != test.expectedVolumeMode {
			t.Errorf("Test %s failed, Expected VolumeMode: %v, but got %v", test.name, test.volumeMode, *pod1.Spec.Volumes[0].VolumeSource.Ephemeral.VolumeClaimTemplate.Spec.VolumeMode)
		}
	}
}

func TestSetDefaultEndpointsProtocol(t *testing.T) {
	in := &v1.Endpoints{Subsets: []v1.EndpointSubset{
		{Ports: []v1.EndpointPort{{}, {Protocol: "UDP"}, {}}},
	}}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*v1.Endpoints)

	for i := range out.Subsets {
		for j := range out.Subsets[i].Ports {
			if in.Subsets[i].Ports[j].Protocol == "" {
				if out.Subsets[i].Ports[j].Protocol != v1.ProtocolTCP {
					t.Errorf("Expected protocol %s, got %s", v1.ProtocolTCP, out.Subsets[i].Ports[j].Protocol)
				}
			} else {
				if out.Subsets[i].Ports[j].Protocol != in.Subsets[i].Ports[j].Protocol {
					t.Errorf("Expected protocol %s, got %s", in.Subsets[i].Ports[j].Protocol, out.Subsets[i].Ports[j].Protocol)
				}
			}
		}
	}
}

func TestSetDefaultServiceTargetPort(t *testing.T) {
	in := &v1.Service{Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 1234}}}}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*v1.Service)
	if out.Spec.Ports[0].TargetPort != intstr.FromInt32(1234) {
		t.Errorf("Expected TargetPort to be defaulted, got %v", out.Spec.Ports[0].TargetPort)
	}

	in = &v1.Service{Spec: v1.ServiceSpec{Ports: []v1.ServicePort{{Port: 1234, TargetPort: intstr.FromInt32(5678)}}}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*v1.Service)
	if out.Spec.Ports[0].TargetPort != intstr.FromInt32(5678) {
		t.Errorf("Expected TargetPort to be unchanged, got %v", out.Spec.Ports[0].TargetPort)
	}
}

func TestSetDefaultServicePort(t *testing.T) {
	// Unchanged if set.
	in := &v1.Service{Spec: v1.ServiceSpec{
		Ports: []v1.ServicePort{
			{Protocol: "UDP", Port: 9376, TargetPort: intstr.FromString("p")},
			{Protocol: "UDP", Port: 8675, TargetPort: intstr.FromInt32(309)},
		},
	}}
	out := roundTrip(t, runtime.Object(in)).(*v1.Service)
	if out.Spec.Ports[0].Protocol != v1.ProtocolUDP {
		t.Errorf("Expected protocol %s, got %s", v1.ProtocolUDP, out.Spec.Ports[0].Protocol)
	}
	if out.Spec.Ports[0].TargetPort != intstr.FromString("p") {
		t.Errorf("Expected port %v, got %v", in.Spec.Ports[0].Port, out.Spec.Ports[0].TargetPort)
	}
	if out.Spec.Ports[1].Protocol != v1.ProtocolUDP {
		t.Errorf("Expected protocol %s, got %s", v1.ProtocolUDP, out.Spec.Ports[1].Protocol)
	}
	if out.Spec.Ports[1].TargetPort != intstr.FromInt32(309) {
		t.Errorf("Expected port %v, got %v", in.Spec.Ports[1].Port, out.Spec.Ports[1].TargetPort)
	}

	// Defaulted.
	in = &v1.Service{Spec: v1.ServiceSpec{
		Ports: []v1.ServicePort{
			{Protocol: "", Port: 9376, TargetPort: intstr.FromString("")},
			{Protocol: "", Port: 8675, TargetPort: intstr.FromInt32(0)},
		},
	}}
	out = roundTrip(t, runtime.Object(in)).(*v1.Service)
	if out.Spec.Ports[0].Protocol != v1.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", v1.ProtocolTCP, out.Spec.Ports[0].Protocol)
	}
	if out.Spec.Ports[0].TargetPort != intstr.FromInt32(in.Spec.Ports[0].Port) {
		t.Errorf("Expected port %v, got %v", in.Spec.Ports[0].Port, out.Spec.Ports[0].TargetPort)
	}
	if out.Spec.Ports[1].Protocol != v1.ProtocolTCP {
		t.Errorf("Expected protocol %s, got %s", v1.ProtocolTCP, out.Spec.Ports[1].Protocol)
	}
	if out.Spec.Ports[1].TargetPort != intstr.FromInt32(in.Spec.Ports[1].Port) {
		t.Errorf("Expected port %v, got %v", in.Spec.Ports[1].Port, out.Spec.Ports[1].TargetPort)
	}
}

func TestSetDefaultServiceExternalTraffic(t *testing.T) {
	in := &v1.Service{}
	obj := roundTrip(t, runtime.Object(in))
	out := obj.(*v1.Service)
	if out.Spec.ExternalTrafficPolicy != "" {
		t.Errorf("Expected ExternalTrafficPolicy to be empty, got %v", out.Spec.ExternalTrafficPolicy)
	}

	in = &v1.Service{Spec: v1.ServiceSpec{Type: v1.ServiceTypeNodePort}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*v1.Service)
	if out.Spec.ExternalTrafficPolicy != v1.ServiceExternalTrafficPolicyCluster {
		t.Errorf("Expected ExternalTrafficPolicy to be %v, got %v", v1.ServiceExternalTrafficPolicyCluster, out.Spec.ExternalTrafficPolicy)
	}

	in = &v1.Service{Spec: v1.ServiceSpec{Type: v1.ServiceTypeLoadBalancer}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*v1.Service)
	if out.Spec.ExternalTrafficPolicy != v1.ServiceExternalTrafficPolicyCluster {
		t.Errorf("Expected ExternalTrafficPolicy to be %v, got %v", v1.ServiceExternalTrafficPolicyCluster, out.Spec.ExternalTrafficPolicy)
	}

	in = &v1.Service{Spec: v1.ServiceSpec{Type: v1.ServiceTypeClusterIP, ExternalIPs: []string{"1.2.3.4"}}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*v1.Service)
	if out.Spec.ExternalTrafficPolicy != v1.ServiceExternalTrafficPolicyCluster {
		t.Errorf("Expected ExternalTrafficPolicy to be %v, got %v", v1.ServiceExternalTrafficPolicyCluster, out.Spec.ExternalTrafficPolicy)
	}

	in = &v1.Service{Spec: v1.ServiceSpec{Type: v1.ServiceTypeClusterIP}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*v1.Service)
	if out.Spec.ExternalTrafficPolicy != "" {
		t.Errorf("Expected ExternalTrafficPolicy to be empty, got %v", out.Spec.ExternalTrafficPolicy)
	}

	in = &v1.Service{Spec: v1.ServiceSpec{Type: v1.ServiceTypeExternalName}}
	obj = roundTrip(t, runtime.Object(in))
	out = obj.(*v1.Service)
	if out.Spec.ExternalTrafficPolicy != "" {
		t.Errorf("Expected ExternalTrafficPolicy to be empty, got %v", out.Spec.ExternalTrafficPolicy)
	}
}

func TestSetDefaultNamespace(t *testing.T) {
	s := &v1.Namespace{}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*v1.Namespace)

	if s2.Status.Phase != v1.NamespaceActive {
		t.Errorf("Expected phase %v, got %v", v1.NamespaceActive, s2.Status.Phase)
	}
}

func TestSetDefaultNamespaceLabels(t *testing.T) {
	theNs := "default-ns-labels-are-great"
	s := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: theNs,
		},
	}
	obj2 := roundTrip(t, runtime.Object(s))
	s2 := obj2.(*v1.Namespace)

	if s2.ObjectMeta.Labels[v1.LabelMetadataName] != theNs {
		t.Errorf("Expected default namespace label value of %v, but got %v", theNs, s2.ObjectMeta.Labels[v1.LabelMetadataName])
	}
}

func TestSetDefaultPodSpecHostNetwork(t *testing.T) {
	portNum := int32(8080)
	s := v1.PodSpec{}
	s.HostNetwork = true
	s.Containers = []v1.Container{
		{
			Ports: []v1.ContainerPort{
				{
					ContainerPort: portNum,
				},
			},
		},
	}
	s.InitContainers = []v1.Container{
		{
			Ports: []v1.ContainerPort{
				{
					ContainerPort: portNum,
				},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	obj2 := roundTrip(t, runtime.Object(pod))
	pod2 := obj2.(*v1.Pod)
	s2 := pod2.Spec

	hostPortNum := s2.Containers[0].Ports[0].HostPort
	if hostPortNum != portNum {
		t.Errorf("Expected container port to be defaulted, was made %d instead of %d", hostPortNum, portNum)
	}

	hostPortNum = s2.InitContainers[0].Ports[0].HostPort
	if hostPortNum != portNum {
		t.Errorf("Expected container port to be defaulted, was made %d instead of %d", hostPortNum, portNum)
	}
}

func TestSetDefaultNodeStatusAllocatable(t *testing.T) {
	capacity := v1.ResourceList{
		v1.ResourceCPU:    resource.MustParse("1000m"),
		v1.ResourceMemory: resource.MustParse("10G"),
	}
	allocatable := v1.ResourceList{
		v1.ResourceCPU:    resource.MustParse("500m"),
		v1.ResourceMemory: resource.MustParse("5G"),
	}
	tests := []struct {
		capacity            v1.ResourceList
		allocatable         v1.ResourceList
		expectedAllocatable v1.ResourceList
	}{{ // Everything set, no defaulting.
		capacity:            capacity,
		allocatable:         allocatable,
		expectedAllocatable: allocatable,
	}, { // Allocatable set, no defaulting.
		capacity:            nil,
		allocatable:         allocatable,
		expectedAllocatable: allocatable,
	}, { // Capacity set, allocatable defaults to capacity.
		capacity:            capacity,
		allocatable:         nil,
		expectedAllocatable: capacity,
	}, { // Nothing set, allocatable "defaults" to capacity.
		capacity:            nil,
		allocatable:         nil,
		expectedAllocatable: nil,
	}}

	copyResourceList := func(rl v1.ResourceList) v1.ResourceList {
		if rl == nil {
			return nil
		}
		copy := make(v1.ResourceList, len(rl))
		for k, v := range rl {
			copy[k] = v.DeepCopy()
		}
		return copy
	}

	resourceListsEqual := func(a v1.ResourceList, b v1.ResourceList) bool {
		if len(a) != len(b) {
			return false
		}
		for k, v := range a {
			vb, found := b[k]
			if !found {
				return false
			}
			if v.Cmp(vb) != 0 {
				return false
			}
		}
		return true
	}

	for i, testcase := range tests {
		node := v1.Node{
			Status: v1.NodeStatus{
				Capacity:    copyResourceList(testcase.capacity),
				Allocatable: copyResourceList(testcase.allocatable),
			},
		}
		node2 := roundTrip(t, runtime.Object(&node)).(*v1.Node)
		actual := node2.Status.Allocatable
		expected := testcase.expectedAllocatable
		if !resourceListsEqual(expected, actual) {
			t.Errorf("[%d] Expected v1.NodeStatus.Allocatable: %+v; Got: %+v", i, expected, actual)
		}
	}
}

func TestSetDefaultObjectFieldSelectorAPIVersion(t *testing.T) {
	s := v1.PodSpec{
		Containers: []v1.Container{
			{
				Env: []v1.EnvVar{
					{
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{},
						},
					},
				},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	obj2 := roundTrip(t, runtime.Object(pod))
	pod2 := obj2.(*v1.Pod)
	s2 := pod2.Spec

	apiVersion := s2.Containers[0].Env[0].ValueFrom.FieldRef.APIVersion
	if apiVersion != "v1" {
		t.Errorf("Expected default APIVersion v1, got: %v", apiVersion)
	}
}

func TestSetMinimumScalePod(t *testing.T) {
	// verify we default if limits are specified (and that request=0 is preserved)
	s := v1.PodSpec{}
	s.Containers = []v1.Container{
		{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("1n"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("2n"),
				},
			},
		},
	}
	s.InitContainers = []v1.Container{
		{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("1n"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("2n"),
				},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	corev1.SetObjectDefaults_Pod(pod)

	if expect := resource.MustParse("1m"); expect.Cmp(pod.Spec.Containers[0].Resources.Requests[v1.ResourceMemory]) != 0 {
		t.Errorf("did not round resources: %#v", pod.Spec.Containers[0].Resources)
	}
	if expect := resource.MustParse("1m"); expect.Cmp(pod.Spec.InitContainers[0].Resources.Requests[v1.ResourceMemory]) != 0 {
		t.Errorf("did not round resources: %#v", pod.Spec.InitContainers[0].Resources)
	}
}

func TestSetDefaultRequestsPod(t *testing.T) {
	// verify we default if limits are specified (and that request=0 is preserved)
	s := v1.PodSpec{}
	s.Containers = []v1.Container{
		{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("0"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("1Gi"),
				},
			},
		},
	}
	s.InitContainers = []v1.Container{
		{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("0"),
				},
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("1Gi"),
				},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*v1.Pod)
	defaultRequest := pod2.Spec.Containers[0].Resources.Requests
	if requestValue := defaultRequest[v1.ResourceCPU]; requestValue.String() != "100m" {
		t.Errorf("Expected request cpu: %s, got: %s", "100m", requestValue.String())
	}
	if requestValue := defaultRequest[v1.ResourceMemory]; requestValue.String() != "0" {
		t.Errorf("Expected request memory: %s, got: %s", "0", requestValue.String())
	}
	defaultRequest = pod2.Spec.InitContainers[0].Resources.Requests
	if requestValue := defaultRequest[v1.ResourceCPU]; requestValue.String() != "100m" {
		t.Errorf("Expected request cpu: %s, got: %s", "100m", requestValue.String())
	}
	if requestValue := defaultRequest[v1.ResourceMemory]; requestValue.String() != "0" {
		t.Errorf("Expected request memory: %s, got: %s", "0", requestValue.String())
	}

	// verify we do nothing if no limits are specified
	s = v1.PodSpec{}
	s.Containers = []v1.Container{{}}
	s.InitContainers = []v1.Container{{}}
	pod = &v1.Pod{
		Spec: s,
	}
	output = roundTrip(t, runtime.Object(pod))
	pod2 = output.(*v1.Pod)
	defaultRequest = pod2.Spec.Containers[0].Resources.Requests
	if requestValue := defaultRequest[v1.ResourceCPU]; requestValue.String() != "0" {
		t.Errorf("Expected 0 request value, got: %s", requestValue.String())
	}
	defaultRequest = pod2.Spec.InitContainers[0].Resources.Requests
	if requestValue := defaultRequest[v1.ResourceCPU]; requestValue.String() != "0" {
		t.Errorf("Expected 0 request value, got: %s", requestValue.String())
	}
}

func TestDefaultRequestIsNotSetForReplicationController(t *testing.T) {
	s := v1.PodSpec{}
	s.Containers = []v1.Container{
		{
			Resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("100m"),
				},
			},
		},
	}
	rc := &v1.ReplicationController{
		Spec: v1.ReplicationControllerSpec{
			Replicas: utilpointer.Int32(3),
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{
						"foo": "bar",
					},
				},
				Spec: s,
			},
		},
	}
	output := roundTrip(t, runtime.Object(rc))
	rc2 := output.(*v1.ReplicationController)
	defaultRequest := rc2.Spec.Template.Spec.Containers[0].Resources.Requests
	requestValue := defaultRequest[v1.ResourceCPU]
	if requestValue.String() != "0" {
		t.Errorf("Expected 0 request value, got: %s", requestValue.String())
	}
}

func TestSetDefaultLimitRangeItem(t *testing.T) {
	limitRange := &v1.LimitRange{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-defaults",
		},
		Spec: v1.LimitRangeSpec{
			Limits: []v1.LimitRangeItem{{
				Type: v1.LimitTypeContainer,
				Max: v1.ResourceList{
					v1.ResourceCPU: resource.MustParse("100m"),
				},
				Min: v1.ResourceList{
					v1.ResourceMemory: resource.MustParse("100Mi"),
				},
				Default:        v1.ResourceList{},
				DefaultRequest: v1.ResourceList{},
			}},
		},
	}

	output := roundTrip(t, runtime.Object(limitRange))
	limitRange2 := output.(*v1.LimitRange)
	defaultLimit := limitRange2.Spec.Limits[0].Default
	defaultRequest := limitRange2.Spec.Limits[0].DefaultRequest

	// verify that default cpu was set to the max
	defaultValue := defaultLimit[v1.ResourceCPU]
	if defaultValue.String() != "100m" {
		t.Errorf("Expected default cpu: %s, got: %s", "100m", defaultValue.String())
	}
	// verify that default request was set to the limit
	requestValue := defaultRequest[v1.ResourceCPU]
	if requestValue.String() != "100m" {
		t.Errorf("Expected request cpu: %s, got: %s", "100m", requestValue.String())
	}
	// verify that if a min is provided, it will be the default if no limit is specified
	requestMinValue := defaultRequest[v1.ResourceMemory]
	if requestMinValue.String() != "100Mi" {
		t.Errorf("Expected request memory: %s, got: %s", "100Mi", requestMinValue.String())
	}
}

func TestSetDefaultProbe(t *testing.T) {
	originalProbe := v1.Probe{}
	expectedProbe := v1.Probe{
		InitialDelaySeconds: 0,
		TimeoutSeconds:      1,
		PeriodSeconds:       10,
		SuccessThreshold:    1,
		FailureThreshold:    3,
	}

	pod := &v1.Pod{
		Spec: v1.PodSpec{
			Containers: []v1.Container{{LivenessProbe: &originalProbe}},
		},
	}

	output := roundTrip(t, runtime.Object(pod)).(*v1.Pod)
	actualProbe := *output.Spec.Containers[0].LivenessProbe
	if actualProbe != expectedProbe {
		t.Errorf("Expected probe: %+v\ngot: %+v\n", expectedProbe, actualProbe)
	}
}

func TestSetDefaultSchedulerName(t *testing.T) {
	pod := &v1.Pod{}

	output := roundTrip(t, runtime.Object(pod)).(*v1.Pod)
	if output.Spec.SchedulerName != v1.DefaultSchedulerName {
		t.Errorf("Expected scheduler name: %+v\ngot: %+v\n", v1.DefaultSchedulerName, output.Spec.SchedulerName)
	}
}

func TestSetDefaultHostPathVolumeSource(t *testing.T) {
	s := v1.PodSpec{}
	s.Volumes = []v1.Volume{
		{
			VolumeSource: v1.VolumeSource{
				HostPath: &v1.HostPathVolumeSource{Path: "foo"},
			},
		},
	}
	pod := &v1.Pod{
		Spec: s,
	}
	output := roundTrip(t, runtime.Object(pod))
	pod2 := output.(*v1.Pod)
	defaultType := pod2.Spec.Volumes[0].VolumeSource.HostPath.Type
	expectedType := v1.HostPathUnset

	if defaultType == nil || *defaultType != expectedType {
		t.Errorf("Expected v1.HostPathVolumeSource default type %v, got %v", expectedType, defaultType)
	}
}

func TestSetDefaultEnableServiceLinks(t *testing.T) {
	pod := &v1.Pod{}
	output := roundTrip(t, runtime.Object(pod)).(*v1.Pod)
	if output.Spec.EnableServiceLinks == nil || *output.Spec.EnableServiceLinks != v1.DefaultEnableServiceLinks {
		t.Errorf("Expected enableServiceLinks value: %+v\ngot: %+v\n", v1.DefaultEnableServiceLinks, *output.Spec.EnableServiceLinks)
	}
}

func TestSetDefaultServiceInternalTrafficPolicy(t *testing.T) {
	cluster := v1.ServiceInternalTrafficPolicyCluster
	local := v1.ServiceInternalTrafficPolicyLocal
	testCases := []struct {
		name                          string
		expectedInternalTrafficPolicy *v1.ServiceInternalTrafficPolicy
		svc                           v1.Service
	}{
		{
			name:                          "must set default internalTrafficPolicy",
			expectedInternalTrafficPolicy: &cluster,
			svc:                           v1.Service{},
		},
		{
			name:                          "must not set default internalTrafficPolicy when it's cluster",
			expectedInternalTrafficPolicy: &cluster,
			svc: v1.Service{
				Spec: v1.ServiceSpec{
					InternalTrafficPolicy: &cluster,
				},
			},
		},
		{
			name:                          "must not set default internalTrafficPolicy when type is ExternalName",
			expectedInternalTrafficPolicy: nil,
			svc: v1.Service{
				Spec: v1.ServiceSpec{
					Type: v1.ServiceTypeExternalName,
				},
			},
		},
		{
			name:                          "must not set default internalTrafficPolicy when it's local",
			expectedInternalTrafficPolicy: &local,
			svc: v1.Service{
				Spec: v1.ServiceSpec{
					InternalTrafficPolicy: &local,
				},
			},
		},
	}
	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			obj := roundTrip(t, runtime.Object(&test.svc))
			svc := obj.(*v1.Service)

			if !reflect.DeepEqual(svc.Spec.InternalTrafficPolicy, test.expectedInternalTrafficPolicy) {
				t.Errorf("expected .spec.internalTrafficPolicy: %v got %v", test.expectedInternalTrafficPolicy, svc.Spec.InternalTrafficPolicy)
			}
		})
	}
}

func TestSetDefaultResizePolicy(t *testing.T) {
	// verify we default to NotRequired restart policy for resize when resources are specified
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)()

	for desc, tc := range map[string]struct {
		testContainer        v1.Container
		expectedResizePolicy []v1.ContainerResizePolicy
	}{
		"CPU and memory limits are specified": {
			testContainer: v1.Container{
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("200Mi"),
					},
				},
			},
			expectedResizePolicy: []v1.ContainerResizePolicy{
				{
					ResourceName:  v1.ResourceCPU,
					RestartPolicy: v1.NotRequired,
				},
				{
					ResourceName:  v1.ResourceMemory,
					RestartPolicy: v1.NotRequired,
				},
			},
		},
		"CPU requests are specified": {
			testContainer: v1.Container{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("100m"),
					},
				},
			},
			expectedResizePolicy: []v1.ContainerResizePolicy{
				{
					ResourceName:  v1.ResourceCPU,
					RestartPolicy: v1.NotRequired,
				},
			},
		},
		"Memory limits are specified": {
			testContainer: v1.Container{
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse("200Mi"),
					},
				},
			},
			expectedResizePolicy: []v1.ContainerResizePolicy{
				{
					ResourceName:  v1.ResourceMemory,
					RestartPolicy: v1.NotRequired,
				},
			},
		},
		"No resources are specified": {
			testContainer:        v1.Container{Name: "besteffort"},
			expectedResizePolicy: nil,
		},
		"CPU and memory limits are specified with restartContainer resize policy for memory": {
			testContainer: v1.Container{
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("200Mi"),
					},
				},
				ResizePolicy: []v1.ContainerResizePolicy{
					{
						ResourceName:  v1.ResourceMemory,
						RestartPolicy: v1.RestartContainer,
					},
				},
			},
			expectedResizePolicy: []v1.ContainerResizePolicy{
				{
					ResourceName:  v1.ResourceMemory,
					RestartPolicy: v1.RestartContainer,
				},
				{
					ResourceName:  v1.ResourceCPU,
					RestartPolicy: v1.NotRequired,
				},
			},
		},
		"CPU requests and memory limits are specified with restartContainer resize policy for CPU": {
			testContainer: v1.Container{
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse("200Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("100m"),
					},
				},
				ResizePolicy: []v1.ContainerResizePolicy{
					{
						ResourceName:  v1.ResourceCPU,
						RestartPolicy: v1.RestartContainer,
					},
				},
			},
			expectedResizePolicy: []v1.ContainerResizePolicy{
				{
					ResourceName:  v1.ResourceCPU,
					RestartPolicy: v1.RestartContainer,
				},
				{
					ResourceName:  v1.ResourceMemory,
					RestartPolicy: v1.NotRequired,
				},
			},
		},
		"CPU and memory requests are specified with restartContainer resize policy for both": {
			testContainer: v1.Container{
				Resources: v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("200Mi"),
					},
				},
				ResizePolicy: []v1.ContainerResizePolicy{
					{
						ResourceName:  v1.ResourceCPU,
						RestartPolicy: v1.RestartContainer,
					},
					{
						ResourceName:  v1.ResourceMemory,
						RestartPolicy: v1.RestartContainer,
					},
				},
			},
			expectedResizePolicy: []v1.ContainerResizePolicy{
				{
					ResourceName:  v1.ResourceCPU,
					RestartPolicy: v1.RestartContainer,
				},
				{
					ResourceName:  v1.ResourceMemory,
					RestartPolicy: v1.RestartContainer,
				},
			},
		},
		"Ephemeral storage limits are specified": {
			testContainer: v1.Container{
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceEphemeralStorage: resource.MustParse("500Mi"),
					},
				},
			},
			expectedResizePolicy: nil,
		},
		"Ephemeral storage requests and CPU limits are specified": {
			testContainer: v1.Container{
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU: resource.MustParse("100m"),
					},
					Requests: v1.ResourceList{
						v1.ResourceEphemeralStorage: resource.MustParse("500Mi"),
					},
				},
			},
			expectedResizePolicy: []v1.ContainerResizePolicy{
				{
					ResourceName:  v1.ResourceCPU,
					RestartPolicy: v1.NotRequired,
				},
			},
		},
		"Ephemeral storage requests and limits, memory requests with restartContainer policy are specified": {
			testContainer: v1.Container{
				Resources: v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceEphemeralStorage: resource.MustParse("500Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceEphemeralStorage: resource.MustParse("500Mi"),
						v1.ResourceMemory:           resource.MustParse("200Mi"),
					},
				},
				ResizePolicy: []v1.ContainerResizePolicy{
					{
						ResourceName:  v1.ResourceMemory,
						RestartPolicy: v1.RestartContainer,
					},
				},
			},
			expectedResizePolicy: []v1.ContainerResizePolicy{
				{
					ResourceName:  v1.ResourceMemory,
					RestartPolicy: v1.RestartContainer,
				},
			},
		},
	} {
		t.Run(desc, func(t *testing.T) {
			testPod := v1.Pod{}
			testPod.Spec.Containers = append(testPod.Spec.Containers, tc.testContainer)
			output := roundTrip(t, runtime.Object(&testPod))
			pod2 := output.(*v1.Pod)
			if !cmp.Equal(pod2.Spec.Containers[0].ResizePolicy, tc.expectedResizePolicy) {
				t.Errorf("expected resize policy %+v, but got %+v", tc.expectedResizePolicy, pod2.Spec.Containers[0].ResizePolicy)
			}
		})
	}
}
