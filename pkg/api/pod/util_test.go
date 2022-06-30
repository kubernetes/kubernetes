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

package pod

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/pointer"
)

func TestVisitContainers(t *testing.T) {
	setAllFeatureEnabledContainersDuringTest := ContainerType(0)
	testCases := []struct {
		desc                       string
		spec                       *api.PodSpec
		wantContainers             []string
		mask                       ContainerType
		ephemeralContainersEnabled bool
	}{
		{
			desc:           "empty podspec",
			spec:           &api.PodSpec{},
			wantContainers: []string{},
			mask:           AllContainers,
		},
		{
			desc: "regular containers",
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2"},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e1"}},
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e2"}},
				},
			},
			wantContainers: []string{"c1", "c2"},
			mask:           Containers,
		},
		{
			desc: "init containers",
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2"},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e1"}},
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e2"}},
				},
			},
			wantContainers: []string{"i1", "i2"},
			mask:           InitContainers,
		},
		{
			desc: "ephemeral containers",
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2"},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e1"}},
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e2"}},
				},
			},
			wantContainers: []string{"e1", "e2"},
			mask:           EphemeralContainers,
		},
		{
			desc: "all container types",
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2"},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e1"}},
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e2"}},
				},
			},
			wantContainers: []string{"i1", "i2", "c1", "c2", "e1", "e2"},
			mask:           AllContainers,
		},
		{
			desc: "all feature enabled container types with ephemeral containers disabled",
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2"},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e1"}},
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e2"}},
				},
			},
			wantContainers: []string{"i1", "i2", "c1", "c2"},
			mask:           setAllFeatureEnabledContainersDuringTest,
		},
		{
			desc: "all feature enabled container types with ephemeral containers enabled",
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2", SecurityContext: &api.SecurityContext{}},
				},
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2", SecurityContext: &api.SecurityContext{}},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e1"}},
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e2"}},
				},
			},
			wantContainers:             []string{"i1", "i2", "c1", "c2", "e1", "e2"},
			mask:                       setAllFeatureEnabledContainersDuringTest,
			ephemeralContainersEnabled: true,
		},
		{
			desc: "dropping fields",
			spec: &api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2", SecurityContext: &api.SecurityContext{}},
				},
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2", SecurityContext: &api.SecurityContext{}},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e1"}},
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e2", SecurityContext: &api.SecurityContext{}}},
				},
			},
			wantContainers: []string{"i1", "i2", "c1", "c2", "e1", "e2"},
			mask:           AllContainers,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EphemeralContainers, tc.ephemeralContainersEnabled)()

			if tc.mask == setAllFeatureEnabledContainersDuringTest {
				tc.mask = AllFeatureEnabledContainers()
			}

			gotContainers := []string{}
			VisitContainers(tc.spec, tc.mask, func(c *api.Container, containerType ContainerType) bool {
				gotContainers = append(gotContainers, c.Name)
				if c.SecurityContext != nil {
					c.SecurityContext = nil
				}
				return true
			})
			if !cmp.Equal(gotContainers, tc.wantContainers) {
				t.Errorf("VisitContainers() = %+v, want %+v", gotContainers, tc.wantContainers)
			}
			for _, c := range tc.spec.Containers {
				if c.SecurityContext != nil {
					t.Errorf("VisitContainers() did not drop SecurityContext for container %q", c.Name)
				}
			}
			for _, c := range tc.spec.InitContainers {
				if c.SecurityContext != nil {
					t.Errorf("VisitContainers() did not drop SecurityContext for init container %q", c.Name)
				}
			}
			for _, c := range tc.spec.EphemeralContainers {
				if c.SecurityContext != nil {
					t.Errorf("VisitContainers() did not drop SecurityContext for ephemeral container %q", c.Name)
				}
			}
		})
	}
}

func TestPodSecrets(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EphemeralContainers, true)()

	// Stub containing all possible secret references in a pod.
	// The names of the referenced secrets match struct paths detected by reflection.
	pod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{{
				EnvFrom: []api.EnvFromSource{{
					SecretRef: &api.SecretEnvSource{
						LocalObjectReference: api.LocalObjectReference{
							Name: "Spec.Containers[*].EnvFrom[*].SecretRef"}}}},
				Env: []api.EnvVar{{
					ValueFrom: &api.EnvVarSource{
						SecretKeyRef: &api.SecretKeySelector{
							LocalObjectReference: api.LocalObjectReference{
								Name: "Spec.Containers[*].Env[*].ValueFrom.SecretKeyRef"}}}}}}},
			ImagePullSecrets: []api.LocalObjectReference{{
				Name: "Spec.ImagePullSecrets"}},
			InitContainers: []api.Container{{
				EnvFrom: []api.EnvFromSource{{
					SecretRef: &api.SecretEnvSource{
						LocalObjectReference: api.LocalObjectReference{
							Name: "Spec.InitContainers[*].EnvFrom[*].SecretRef"}}}},
				Env: []api.EnvVar{{
					ValueFrom: &api.EnvVarSource{
						SecretKeyRef: &api.SecretKeySelector{
							LocalObjectReference: api.LocalObjectReference{
								Name: "Spec.InitContainers[*].Env[*].ValueFrom.SecretKeyRef"}}}}}}},
			Volumes: []api.Volume{{
				VolumeSource: api.VolumeSource{
					AzureFile: &api.AzureFileVolumeSource{
						SecretName: "Spec.Volumes[*].VolumeSource.AzureFile.SecretName"}}}, {
				VolumeSource: api.VolumeSource{
					CephFS: &api.CephFSVolumeSource{
						SecretRef: &api.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.CephFS.SecretRef"}}}}, {
				VolumeSource: api.VolumeSource{
					Cinder: &api.CinderVolumeSource{
						SecretRef: &api.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.Cinder.SecretRef"}}}}, {
				VolumeSource: api.VolumeSource{
					FlexVolume: &api.FlexVolumeSource{
						SecretRef: &api.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.FlexVolume.SecretRef"}}}}, {
				VolumeSource: api.VolumeSource{
					Projected: &api.ProjectedVolumeSource{
						Sources: []api.VolumeProjection{{
							Secret: &api.SecretProjection{
								LocalObjectReference: api.LocalObjectReference{
									Name: "Spec.Volumes[*].VolumeSource.Projected.Sources[*].Secret"}}}}}}}, {
				VolumeSource: api.VolumeSource{
					RBD: &api.RBDVolumeSource{
						SecretRef: &api.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.RBD.SecretRef"}}}}, {
				VolumeSource: api.VolumeSource{
					Secret: &api.SecretVolumeSource{
						SecretName: "Spec.Volumes[*].VolumeSource.Secret.SecretName"}}}, {
				VolumeSource: api.VolumeSource{
					Secret: &api.SecretVolumeSource{
						SecretName: "Spec.Volumes[*].VolumeSource.Secret"}}}, {
				VolumeSource: api.VolumeSource{
					ScaleIO: &api.ScaleIOVolumeSource{
						SecretRef: &api.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.ScaleIO.SecretRef"}}}}, {
				VolumeSource: api.VolumeSource{
					ISCSI: &api.ISCSIVolumeSource{
						SecretRef: &api.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.ISCSI.SecretRef"}}}}, {
				VolumeSource: api.VolumeSource{
					StorageOS: &api.StorageOSVolumeSource{
						SecretRef: &api.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.StorageOS.SecretRef"}}}}, {
				VolumeSource: api.VolumeSource{
					CSI: &api.CSIVolumeSource{
						NodePublishSecretRef: &api.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.CSI.NodePublishSecretRef"}}}}},
			EphemeralContainers: []api.EphemeralContainer{{
				EphemeralContainerCommon: api.EphemeralContainerCommon{
					EnvFrom: []api.EnvFromSource{{
						SecretRef: &api.SecretEnvSource{
							LocalObjectReference: api.LocalObjectReference{
								Name: "Spec.EphemeralContainers[*].EphemeralContainerCommon.EnvFrom[*].SecretRef"}}}},
					Env: []api.EnvVar{{
						ValueFrom: &api.EnvVarSource{
							SecretKeyRef: &api.SecretKeySelector{
								LocalObjectReference: api.LocalObjectReference{
									Name: "Spec.EphemeralContainers[*].EphemeralContainerCommon.Env[*].ValueFrom.SecretKeyRef"}}}}}}}},
		},
	}
	extractedNames := sets.NewString()
	VisitPodSecretNames(pod, func(name string) bool {
		extractedNames.Insert(name)
		return true
	}, AllContainers)

	// excludedSecretPaths holds struct paths to fields with "secret" in the name that are not actually references to secret API objects
	excludedSecretPaths := sets.NewString(
		"Spec.Volumes[*].VolumeSource.CephFS.SecretFile",
	)
	// expectedSecretPaths holds struct paths to fields with "secret" in the name that are references to secret API objects.
	// every path here should be represented as an example in the Pod stub above, with the secret name set to the path.
	expectedSecretPaths := sets.NewString(
		"Spec.Containers[*].EnvFrom[*].SecretRef",
		"Spec.Containers[*].Env[*].ValueFrom.SecretKeyRef",
		"Spec.EphemeralContainers[*].EphemeralContainerCommon.EnvFrom[*].SecretRef",
		"Spec.EphemeralContainers[*].EphemeralContainerCommon.Env[*].ValueFrom.SecretKeyRef",
		"Spec.ImagePullSecrets",
		"Spec.InitContainers[*].EnvFrom[*].SecretRef",
		"Spec.InitContainers[*].Env[*].ValueFrom.SecretKeyRef",
		"Spec.Volumes[*].VolumeSource.AzureFile.SecretName",
		"Spec.Volumes[*].VolumeSource.CephFS.SecretRef",
		"Spec.Volumes[*].VolumeSource.Cinder.SecretRef",
		"Spec.Volumes[*].VolumeSource.FlexVolume.SecretRef",
		"Spec.Volumes[*].VolumeSource.Projected.Sources[*].Secret",
		"Spec.Volumes[*].VolumeSource.RBD.SecretRef",
		"Spec.Volumes[*].VolumeSource.Secret",
		"Spec.Volumes[*].VolumeSource.Secret.SecretName",
		"Spec.Volumes[*].VolumeSource.ScaleIO.SecretRef",
		"Spec.Volumes[*].VolumeSource.ISCSI.SecretRef",
		"Spec.Volumes[*].VolumeSource.StorageOS.SecretRef",
		"Spec.Volumes[*].VolumeSource.CSI.NodePublishSecretRef",
	)
	secretPaths := collectResourcePaths(t, "secret", nil, "", reflect.TypeOf(&api.Pod{}))
	secretPaths = secretPaths.Difference(excludedSecretPaths)
	if missingPaths := expectedSecretPaths.Difference(secretPaths); len(missingPaths) > 0 {
		t.Logf("Missing expected secret paths:\n%s", strings.Join(missingPaths.List(), "\n"))
		t.Error("Missing expected secret paths. Verify VisitPodSecretNames() is correctly finding the missing paths, then correct expectedSecretPaths")
	}
	if extraPaths := secretPaths.Difference(expectedSecretPaths); len(extraPaths) > 0 {
		t.Logf("Extra secret paths:\n%s", strings.Join(extraPaths.List(), "\n"))
		t.Error("Extra fields with 'secret' in the name found. Verify VisitPodSecretNames() is including these fields if appropriate, then correct expectedSecretPaths")
	}

	if missingNames := expectedSecretPaths.Difference(extractedNames); len(missingNames) > 0 {
		t.Logf("Missing expected secret names:\n%s", strings.Join(missingNames.List(), "\n"))
		t.Error("Missing expected secret names. Verify the pod stub above includes these references, then verify VisitPodSecretNames() is correctly finding the missing names")
	}
	if extraNames := extractedNames.Difference(expectedSecretPaths); len(extraNames) > 0 {
		t.Logf("Extra secret names:\n%s", strings.Join(extraNames.List(), "\n"))
		t.Error("Extra secret names extracted. Verify VisitPodSecretNames() is correctly extracting secret names")
	}

	// emptyPod is a stub containing empty object names
	emptyPod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{{
				EnvFrom: []api.EnvFromSource{{
					SecretRef: &api.SecretEnvSource{
						LocalObjectReference: api.LocalObjectReference{
							Name: ""}}}}}},
		},
	}
	VisitPodSecretNames(emptyPod, func(name string) bool {
		t.Fatalf("expected no empty names collected, got %q", name)
		return false
	}, AllContainers)
}

// collectResourcePaths traverses the object, computing all the struct paths that lead to fields with resourcename in the name.
func collectResourcePaths(t *testing.T, resourcename string, path *field.Path, name string, tp reflect.Type) sets.String {
	resourcename = strings.ToLower(resourcename)
	resourcePaths := sets.NewString()

	if tp.Kind() == reflect.Pointer {
		resourcePaths.Insert(collectResourcePaths(t, resourcename, path, name, tp.Elem()).List()...)
		return resourcePaths
	}

	if strings.Contains(strings.ToLower(name), resourcename) {
		resourcePaths.Insert(path.String())
	}

	switch tp.Kind() {
	case reflect.Pointer:
		resourcePaths.Insert(collectResourcePaths(t, resourcename, path, name, tp.Elem()).List()...)
	case reflect.Struct:
		// ObjectMeta is generic and therefore should never have a field with a specific resource's name;
		// it contains cycles so it's easiest to just skip it.
		if name == "ObjectMeta" {
			break
		}
		for i := 0; i < tp.NumField(); i++ {
			field := tp.Field(i)
			resourcePaths.Insert(collectResourcePaths(t, resourcename, path.Child(field.Name), field.Name, field.Type).List()...)
		}
	case reflect.Interface:
		t.Errorf("cannot find %s fields in interface{} field %s", resourcename, path.String())
	case reflect.Map:
		resourcePaths.Insert(collectResourcePaths(t, resourcename, path.Key("*"), "", tp.Elem()).List()...)
	case reflect.Slice:
		resourcePaths.Insert(collectResourcePaths(t, resourcename, path.Key("*"), "", tp.Elem()).List()...)
	default:
		// all primitive types
	}

	return resourcePaths
}

func TestPodConfigmaps(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EphemeralContainers, true)()

	// Stub containing all possible ConfigMap references in a pod.
	// The names of the referenced ConfigMaps match struct paths detected by reflection.
	pod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{{
				EnvFrom: []api.EnvFromSource{{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{
							Name: "Spec.Containers[*].EnvFrom[*].ConfigMapRef"}}}},
				Env: []api.EnvVar{{
					ValueFrom: &api.EnvVarSource{
						ConfigMapKeyRef: &api.ConfigMapKeySelector{
							LocalObjectReference: api.LocalObjectReference{
								Name: "Spec.Containers[*].Env[*].ValueFrom.ConfigMapKeyRef"}}}}}}},
			EphemeralContainers: []api.EphemeralContainer{{
				EphemeralContainerCommon: api.EphemeralContainerCommon{
					EnvFrom: []api.EnvFromSource{{
						ConfigMapRef: &api.ConfigMapEnvSource{
							LocalObjectReference: api.LocalObjectReference{
								Name: "Spec.EphemeralContainers[*].EphemeralContainerCommon.EnvFrom[*].ConfigMapRef"}}}},
					Env: []api.EnvVar{{
						ValueFrom: &api.EnvVarSource{
							ConfigMapKeyRef: &api.ConfigMapKeySelector{
								LocalObjectReference: api.LocalObjectReference{
									Name: "Spec.EphemeralContainers[*].EphemeralContainerCommon.Env[*].ValueFrom.ConfigMapKeyRef"}}}}}}}},
			InitContainers: []api.Container{{
				EnvFrom: []api.EnvFromSource{{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{
							Name: "Spec.InitContainers[*].EnvFrom[*].ConfigMapRef"}}}},
				Env: []api.EnvVar{{
					ValueFrom: &api.EnvVarSource{
						ConfigMapKeyRef: &api.ConfigMapKeySelector{
							LocalObjectReference: api.LocalObjectReference{
								Name: "Spec.InitContainers[*].Env[*].ValueFrom.ConfigMapKeyRef"}}}}}}},
			Volumes: []api.Volume{{
				VolumeSource: api.VolumeSource{
					Projected: &api.ProjectedVolumeSource{
						Sources: []api.VolumeProjection{{
							ConfigMap: &api.ConfigMapProjection{
								LocalObjectReference: api.LocalObjectReference{
									Name: "Spec.Volumes[*].VolumeSource.Projected.Sources[*].ConfigMap"}}}}}}}, {
				VolumeSource: api.VolumeSource{
					ConfigMap: &api.ConfigMapVolumeSource{
						LocalObjectReference: api.LocalObjectReference{
							Name: "Spec.Volumes[*].VolumeSource.ConfigMap"}}}}},
		},
	}
	extractedNames := sets.NewString()
	VisitPodConfigmapNames(pod, func(name string) bool {
		extractedNames.Insert(name)
		return true
	}, AllContainers)

	// expectedPaths holds struct paths to fields with "ConfigMap" in the name that are references to ConfigMap API objects.
	// every path here should be represented as an example in the Pod stub above, with the ConfigMap name set to the path.
	expectedPaths := sets.NewString(
		"Spec.Containers[*].EnvFrom[*].ConfigMapRef",
		"Spec.Containers[*].Env[*].ValueFrom.ConfigMapKeyRef",
		"Spec.EphemeralContainers[*].EphemeralContainerCommon.EnvFrom[*].ConfigMapRef",
		"Spec.EphemeralContainers[*].EphemeralContainerCommon.Env[*].ValueFrom.ConfigMapKeyRef",
		"Spec.InitContainers[*].EnvFrom[*].ConfigMapRef",
		"Spec.InitContainers[*].Env[*].ValueFrom.ConfigMapKeyRef",
		"Spec.Volumes[*].VolumeSource.Projected.Sources[*].ConfigMap",
		"Spec.Volumes[*].VolumeSource.ConfigMap",
	)
	collectPaths := collectResourcePaths(t, "ConfigMap", nil, "", reflect.TypeOf(&api.Pod{}))
	if missingPaths := expectedPaths.Difference(collectPaths); len(missingPaths) > 0 {
		t.Logf("Missing expected paths:\n%s", strings.Join(missingPaths.List(), "\n"))
		t.Error("Missing expected paths. Verify VisitPodConfigmapNames() is correctly finding the missing paths, then correct expectedPaths")
	}
	if extraPaths := collectPaths.Difference(expectedPaths); len(extraPaths) > 0 {
		t.Logf("Extra paths:\n%s", strings.Join(extraPaths.List(), "\n"))
		t.Error("Extra fields with resource in the name found. Verify VisitPodConfigmapNames() is including these fields if appropriate, then correct expectedPaths")
	}

	if missingNames := expectedPaths.Difference(extractedNames); len(missingNames) > 0 {
		t.Logf("Missing expected names:\n%s", strings.Join(missingNames.List(), "\n"))
		t.Error("Missing expected names. Verify the pod stub above includes these references, then verify VisitPodConfigmapNames() is correctly finding the missing names")
	}
	if extraNames := extractedNames.Difference(expectedPaths); len(extraNames) > 0 {
		t.Logf("Extra names:\n%s", strings.Join(extraNames.List(), "\n"))
		t.Error("Extra names extracted. Verify VisitPodConfigmapNames() is correctly extracting resource names")
	}

	// emptyPod is a stub containing empty object names
	emptyPod := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{{
				EnvFrom: []api.EnvFromSource{{
					ConfigMapRef: &api.ConfigMapEnvSource{
						LocalObjectReference: api.LocalObjectReference{
							Name: ""}}}}}},
		},
	}
	VisitPodConfigmapNames(emptyPod, func(name string) bool {
		t.Fatalf("expected no empty names collected, got %q", name)
		return false
	}, AllContainers)
}

func TestDropFSGroupFields(t *testing.T) {
	nofsGroupPod := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "container1",
						Image: "testimage",
					},
				},
			},
		}
	}

	var podFSGroup int64 = 100
	changePolicy := api.FSGroupChangeAlways

	fsGroupPod := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "container1",
						Image: "testimage",
					},
				},
				SecurityContext: &api.PodSecurityContext{
					FSGroup:             &podFSGroup,
					FSGroupChangePolicy: &changePolicy,
				},
			},
		}
	}
	podInfos := []struct {
		description                  string
		newPodHasFSGroupChangePolicy bool
		pod                          func() *api.Pod
		expectPolicyInPod            bool
	}{
		{
			description:                  "oldPod.FSGroupChangePolicy=nil, feature=true, newPod.FSGroupChangePolicy=true",
			pod:                          nofsGroupPod,
			newPodHasFSGroupChangePolicy: true,
			expectPolicyInPod:            true,
		},
		{
			description:                  "oldPod=nil, feature=true, newPod.FSGroupChangePolicy=true",
			pod:                          func() *api.Pod { return nil },
			newPodHasFSGroupChangePolicy: true,
			expectPolicyInPod:            true,
		},
		{
			description:                  "oldPod.FSGroupChangePolicy=true, feature=true, newPod.FSGroupChangePolicy=false",
			pod:                          fsGroupPod,
			newPodHasFSGroupChangePolicy: false,
			expectPolicyInPod:            false,
		},
	}
	for _, podInfo := range podInfos {
		t.Run(podInfo.description, func(t *testing.T) {
			oldPod := podInfo.pod()
			newPod := oldPod.DeepCopy()
			if oldPod == nil && podInfo.newPodHasFSGroupChangePolicy {
				newPod = fsGroupPod()
			}

			if oldPod != nil {
				if podInfo.newPodHasFSGroupChangePolicy {
					newPod.Spec.SecurityContext = &api.PodSecurityContext{
						FSGroup:             &podFSGroup,
						FSGroupChangePolicy: &changePolicy,
					}
				} else {
					newPod.Spec.SecurityContext = &api.PodSecurityContext{}
				}
			}
			DropDisabledPodFields(newPod, oldPod)

			if podInfo.expectPolicyInPod {
				secContext := newPod.Spec.SecurityContext
				if secContext == nil || secContext.FSGroupChangePolicy == nil {
					t.Errorf("for %s, expected fsGroupChangepolicy found none", podInfo.description)
				}
			} else {
				secConext := newPod.Spec.SecurityContext
				if secConext != nil && secConext.FSGroupChangePolicy != nil {
					t.Errorf("for %s, unexpected fsGroupChangepolicy set", podInfo.description)
				}
			}
		})
	}

}

func TestDropProcMount(t *testing.T) {
	procMount := api.UnmaskedProcMount
	defaultProcMount := api.DefaultProcMount
	podWithProcMount := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:  api.RestartPolicyNever,
				Containers:     []api.Container{{Name: "container1", Image: "testimage", SecurityContext: &api.SecurityContext{ProcMount: &procMount}}},
				InitContainers: []api.Container{{Name: "container1", Image: "testimage", SecurityContext: &api.SecurityContext{ProcMount: &procMount}}},
			},
		}
	}
	podWithDefaultProcMount := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:  api.RestartPolicyNever,
				Containers:     []api.Container{{Name: "container1", Image: "testimage", SecurityContext: &api.SecurityContext{ProcMount: &defaultProcMount}}},
				InitContainers: []api.Container{{Name: "container1", Image: "testimage", SecurityContext: &api.SecurityContext{ProcMount: &defaultProcMount}}},
			},
		}
	}
	podWithoutProcMount := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:  api.RestartPolicyNever,
				Containers:     []api.Container{{Name: "container1", Image: "testimage", SecurityContext: &api.SecurityContext{ProcMount: nil}}},
				InitContainers: []api.Container{{Name: "container1", Image: "testimage", SecurityContext: &api.SecurityContext{ProcMount: nil}}},
			},
		}
	}

	podInfo := []struct {
		description  string
		hasProcMount bool
		pod          func() *api.Pod
	}{
		{
			description:  "has ProcMount",
			hasProcMount: true,
			pod:          podWithProcMount,
		},
		{
			description:  "has default ProcMount",
			hasProcMount: false,
			pod:          podWithDefaultProcMount,
		},
		{
			description:  "does not have ProcMount",
			hasProcMount: false,
			pod:          podWithoutProcMount,
		},
		{
			description:  "is nil",
			hasProcMount: false,
			pod:          func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasProcMount, oldPod := oldPodInfo.hasProcMount, oldPodInfo.pod()
				newPodHasProcMount, newPod := newPodInfo.hasProcMount, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ProcMountType, enabled)()

					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", cmp.Diff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasProcMount:
						// new pod should not be changed if the feature is enabled, or if the old pod had ProcMount
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					case newPodHasProcMount:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have ProcMount
						if procMountInUse(&newPod.Spec) {
							t.Errorf("new pod had ProcMount: %#v", &newPod.Spec)
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}

func TestDropEmptyDirSizeLimit(t *testing.T) {
	sizeLimit := resource.MustParse("1Gi")
	podWithEmptyDirSizeLimit := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyNever,
				Volumes: []api.Volume{
					{
						Name: "a",
						VolumeSource: api.VolumeSource{
							EmptyDir: &api.EmptyDirVolumeSource{
								Medium:    "memory",
								SizeLimit: &sizeLimit,
							},
						},
					},
				},
			},
		}
	}
	podWithoutEmptyDirSizeLimit := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyNever,
				Volumes: []api.Volume{
					{
						Name: "a",
						VolumeSource: api.VolumeSource{
							EmptyDir: &api.EmptyDirVolumeSource{
								Medium: "memory",
							},
						},
					},
				},
			},
		}
	}

	podInfo := []struct {
		description          string
		hasEmptyDirSizeLimit bool
		pod                  func() *api.Pod
	}{
		{
			description:          "has EmptyDir Size Limit",
			hasEmptyDirSizeLimit: true,
			pod:                  podWithEmptyDirSizeLimit,
		},
		{
			description:          "does not have EmptyDir Size Limit",
			hasEmptyDirSizeLimit: false,
			pod:                  podWithoutEmptyDirSizeLimit,
		},
		{
			description:          "is nil",
			hasEmptyDirSizeLimit: false,
			pod:                  func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasEmptyDirSizeLimit, oldPod := oldPodInfo.hasEmptyDirSizeLimit, oldPodInfo.pod()
				newPodHasEmptyDirSizeLimit, newPod := newPodInfo.hasEmptyDirSizeLimit, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.LocalStorageCapacityIsolation, enabled)()

					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", cmp.Diff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasEmptyDirSizeLimit:
						// new pod should not be changed if the feature is enabled, or if the old pod had EmptyDir SizeLimit
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					case newPodHasEmptyDirSizeLimit:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have EmptyDir SizeLimit
						if !reflect.DeepEqual(newPod, podWithoutEmptyDirSizeLimit()) {
							t.Errorf("new pod had EmptyDir SizeLimit: %v", cmp.Diff(newPod, podWithoutEmptyDirSizeLimit()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}

func TestDropAppArmor(t *testing.T) {
	podWithAppArmor := func() *api.Pod {
		return &api.Pod{
			ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{"a": "1", v1.AppArmorBetaContainerAnnotationKeyPrefix + "foo": "default"}},
			Spec:       api.PodSpec{},
		}
	}
	podWithoutAppArmor := func() *api.Pod {
		return &api.Pod{
			ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{"a": "1"}},
			Spec:       api.PodSpec{},
		}
	}

	podInfo := []struct {
		description string
		hasAppArmor bool
		pod         func() *api.Pod
	}{
		{
			description: "has AppArmor",
			hasAppArmor: true,
			pod:         podWithAppArmor,
		},
		{
			description: "does not have AppArmor",
			hasAppArmor: false,
			pod:         podWithoutAppArmor,
		},
		{
			description: "is nil",
			hasAppArmor: false,
			pod:         func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasAppArmor, oldPod := oldPodInfo.hasAppArmor, oldPodInfo.pod()
				newPodHasAppArmor, newPod := newPodInfo.hasAppArmor, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AppArmor, enabled)()

					DropDisabledPodFields(newPod, oldPod)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", cmp.Diff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasAppArmor:
						// new pod should not be changed if the feature is enabled, or if the old pod had AppArmor
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					case newPodHasAppArmor:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have AppArmor
						if !reflect.DeepEqual(newPod, podWithoutAppArmor()) {
							t.Errorf("new pod had EmptyDir SizeLimit: %v", cmp.Diff(newPod, podWithoutAppArmor()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}

func TestDropProbeGracePeriod(t *testing.T) {
	podWithProbeGracePeriod := func() *api.Pod {
		livenessGracePeriod := int64(10)
		livenessProbe := api.Probe{TerminationGracePeriodSeconds: &livenessGracePeriod}
		startupGracePeriod := int64(10)
		startupProbe := api.Probe{TerminationGracePeriodSeconds: &startupGracePeriod}
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyNever,
				Containers:    []api.Container{{Name: "container1", Image: "testimage", LivenessProbe: &livenessProbe, StartupProbe: &startupProbe}},
			},
		}
	}
	podWithoutProbeGracePeriod := func() *api.Pod {
		p := podWithProbeGracePeriod()
		p.Spec.Containers[0].LivenessProbe.TerminationGracePeriodSeconds = nil
		p.Spec.Containers[0].StartupProbe.TerminationGracePeriodSeconds = nil
		return p
	}

	podInfo := []struct {
		description    string
		hasGracePeriod bool
		pod            func() *api.Pod
	}{
		{
			description:    "has probe-level terminationGracePeriod",
			hasGracePeriod: true,
			pod:            podWithProbeGracePeriod,
		},
		{
			description:    "does not have probe-level terminationGracePeriod",
			hasGracePeriod: false,
			pod:            podWithoutProbeGracePeriod,
		},
		{
			description:    "only has liveness probe-level terminationGracePeriod",
			hasGracePeriod: true,
			pod: func() *api.Pod {
				p := podWithProbeGracePeriod()
				p.Spec.Containers[0].StartupProbe.TerminationGracePeriodSeconds = nil
				return p
			},
		},
		{
			description:    "is nil",
			hasGracePeriod: false,
			pod:            func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasGracePeriod, oldPod := oldPodInfo.hasGracePeriod, oldPodInfo.pod()
				newPodHasGracePeriod, newPod := newPodInfo.hasGracePeriod, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ProbeTerminationGracePeriod, enabled)()

					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", cmp.Diff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasGracePeriod:
						// new pod should not be changed if the feature is enabled, or if the old pod had terminationGracePeriod
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					case newPodHasGracePeriod:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have terminationGracePeriod
						if !reflect.DeepEqual(newPod, podWithoutProbeGracePeriod()) {
							t.Errorf("new pod had probe-level terminationGracePeriod: %v", cmp.Diff(newPod, podWithoutProbeGracePeriod()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}

func TestDropEphemeralContainers(t *testing.T) {
	podWithEphemeralContainers := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:       api.RestartPolicyNever,
				EphemeralContainers: []api.EphemeralContainer{{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "container1", Image: "testimage"}}},
			},
		}
	}
	podWithoutEphemeralContainers := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyNever,
			},
		}
	}

	podInfo := []struct {
		description            string
		hasEphemeralContainers bool
		pod                    func() *api.Pod
	}{
		{
			description:            "has ephemeral containers",
			hasEphemeralContainers: true,
			pod:                    podWithEphemeralContainers,
		},
		{
			description:            "does not have ephemeral containers",
			hasEphemeralContainers: false,
			pod:                    podWithoutEphemeralContainers,
		},
		{
			description:            "is nil",
			hasEphemeralContainers: false,
			pod:                    func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasEphemeralContainers, oldPod := oldPodInfo.hasEphemeralContainers, oldPodInfo.pod()
				newPodHasEphemeralContainers, newPod := newPodInfo.hasEphemeralContainers, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EphemeralContainers, enabled)()

					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", cmp.Diff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasEphemeralContainers:
						// new pod should not be changed if the feature is enabled, or if the old pod had subpaths
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					case newPodHasEphemeralContainers:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have subpaths
						if !reflect.DeepEqual(newPod, podWithoutEphemeralContainers()) {
							t.Errorf("new pod had subpaths: %v", cmp.Diff(newPod, podWithoutEphemeralContainers()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}

func TestValidatePodDeletionCostOption(t *testing.T) {
	testCases := []struct {
		name                            string
		oldPodMeta                      *metav1.ObjectMeta
		featureEnabled                  bool
		wantAllowInvalidPodDeletionCost bool
	}{
		{
			name:                            "CreateFeatureEnabled",
			featureEnabled:                  true,
			wantAllowInvalidPodDeletionCost: false,
		},
		{
			name:                            "CreateFeatureDisabled",
			featureEnabled:                  false,
			wantAllowInvalidPodDeletionCost: true,
		},
		{
			name:                            "UpdateFeatureDisabled",
			oldPodMeta:                      &metav1.ObjectMeta{Annotations: map[string]string{api.PodDeletionCost: "100"}},
			featureEnabled:                  false,
			wantAllowInvalidPodDeletionCost: true,
		},
		{
			name:                            "UpdateFeatureEnabledValidOldValue",
			oldPodMeta:                      &metav1.ObjectMeta{Annotations: map[string]string{api.PodDeletionCost: "100"}},
			featureEnabled:                  true,
			wantAllowInvalidPodDeletionCost: false,
		},
		{
			name:                            "UpdateFeatureEnabledValidOldValue",
			oldPodMeta:                      &metav1.ObjectMeta{Annotations: map[string]string{api.PodDeletionCost: "invalid-value"}},
			featureEnabled:                  true,
			wantAllowInvalidPodDeletionCost: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDeletionCost, tc.featureEnabled)()
			// The new pod doesn't impact the outcome.
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(nil, nil, nil, tc.oldPodMeta)
			if tc.wantAllowInvalidPodDeletionCost != gotOptions.AllowInvalidPodDeletionCost {
				t.Errorf("unexpected diff, want: %v, got: %v", tc.wantAllowInvalidPodDeletionCost, gotOptions.AllowInvalidPodDeletionCost)
			}
		})
	}
}

func TestHaveSameExpandedDNSConfig(t *testing.T) {
	testCases := []struct {
		desc       string
		podSpec    *api.PodSpec
		oldPodSpec *api.PodSpec
		want       bool
	}{
		{
			desc:       "nil DNSConfig",
			podSpec:    &api.PodSpec{},
			oldPodSpec: &api.PodSpec{},
			want:       false,
		},
		{
			desc: "empty DNSConfig",
			podSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{},
			},
			oldPodSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{},
			},
			want: false,
		},
		{
			desc: "same legacy DNSConfig",
			podSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{
					Searches: []string{
						"foo.com",
						"bar.io",
						"3.com",
						"4.com",
						"5.com",
						"6.com",
					},
				},
			},
			oldPodSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{
					Searches: []string{
						"foo.com",
						"bar.io",
						"3.com",
						"4.com",
						"5.com",
						"6.com",
					},
				},
			},
			want: false,
		},
		{
			desc: "update legacy DNSConfig",
			podSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{
					Searches: []string{
						"foo.com",
						"bar.io",
						"baz.com",
						"4.com",
						"5.com",
						"6.com",
					},
				},
			},
			oldPodSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{
					Searches: []string{
						"foo.com",
						"bar.io",
						"3.com",
						"4.com",
						"5.com",
						"6.com",
					},
				},
			},
			want: false,
		},
		{
			desc: "same expanded DNSConfig",
			podSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{
					Searches: []string{
						"foo.com",
						"bar.io",
						"3.com",
						"4.com",
						"5.com",
						"6.com",
						"7.expanded.com",
						"8.expanded.com",
						"9.expanded.com",
						"10.expanded.com",
						"11.expanded.com",
						"12.expanded.com",
						"13.expanded.com",
						"14.expanded.com",
						"15.expanded.com",
						"16.expanded.com",
						"17.expanded.com",
						"18.expanded.com",
						"19.expanded.com",
						"20.expanded.com",
						"21.expanded.com",
						"22.expanded.com",
						"23.expanded.com",
						"24.expanded.com",
						"25.expanded.com",
						"26.expanded.com",
						"27.expanded.com",
						"28.expanded.com",
						"29.expanded.com",
						"30.expanded.com",
						"31.expanded.com",
						"32.expanded.com",
					},
				},
			},
			oldPodSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{
					Searches: []string{
						"foo.com",
						"bar.io",
						"3.com",
						"4.com",
						"5.com",
						"6.com",
						"7.expanded.com",
						"8.expanded.com",
						"9.expanded.com",
						"10.expanded.com",
						"11.expanded.com",
						"12.expanded.com",
						"13.expanded.com",
						"14.expanded.com",
						"15.expanded.com",
						"16.expanded.com",
						"17.expanded.com",
						"18.expanded.com",
						"19.expanded.com",
						"20.expanded.com",
						"21.expanded.com",
						"22.expanded.com",
						"23.expanded.com",
						"24.expanded.com",
						"25.expanded.com",
						"26.expanded.com",
						"27.expanded.com",
						"28.expanded.com",
						"29.expanded.com",
						"30.expanded.com",
						"31.expanded.com",
						"32.expanded.com",
					},
				},
			},
			want: true,
		},
		{
			desc: "update expanded DNSConfig",
			podSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{
					Searches: []string{
						"foo.com",
						"bar.io",
						"3.com",
						"4.com",
						"5.com",
						"6.com",
						"baz.expanded.com",
						"8.expanded.com",
						"9.expanded.com",
						"10.expanded.com",
						"11.expanded.com",
						"12.expanded.com",
						"13.expanded.com",
						"14.expanded.com",
						"15.expanded.com",
						"16.expanded.com",
						"17.expanded.com",
						"18.expanded.com",
						"19.expanded.com",
						"20.expanded.com",
						"21.expanded.com",
						"22.expanded.com",
						"23.expanded.com",
						"24.expanded.com",
						"25.expanded.com",
						"26.expanded.com",
						"27.expanded.com",
						"28.expanded.com",
						"29.expanded.com",
						"30.expanded.com",
						"31.expanded.com",
						"32.expanded.com",
					},
				},
			},
			oldPodSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{
					Searches: []string{
						"foo.com",
						"bar.io",
						"3.com",
						"4.com",
						"5.com",
						"6.com",
						"7.expanded.com",
						"8.expanded.com",
						"9.expanded.com",
						"10.expanded.com",
						"11.expanded.com",
						"12.expanded.com",
						"13.expanded.com",
						"14.expanded.com",
						"15.expanded.com",
						"16.expanded.com",
						"17.expanded.com",
						"18.expanded.com",
						"19.expanded.com",
						"20.expanded.com",
						"21.expanded.com",
						"22.expanded.com",
						"23.expanded.com",
						"24.expanded.com",
						"25.expanded.com",
						"26.expanded.com",
						"27.expanded.com",
						"28.expanded.com",
						"29.expanded.com",
						"30.expanded.com",
						"31.expanded.com",
						"32.expanded.com",
					},
				},
			},
			want: false,
		},
		{
			desc: "update to legacy DNSConfig",
			podSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{
					Searches: []string{
						"foo.com",
						"bar.io",
						"baz.com",
						"4.com",
						"5.com",
						"6.com",
					},
				},
			},
			oldPodSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{
					Searches: []string{
						"foo.com",
						"bar.io",
						"3.com",
						"4.com",
						"5.com",
						"6.com",
						"7.expanded.com",
						"8.expanded.com",
						"9.expanded.com",
						"10.expanded.com",
						"11.expanded.com",
						"12.expanded.com",
						"13.expanded.com",
						"14.expanded.com",
						"15.expanded.com",
						"16.expanded.com",
						"17.expanded.com",
						"18.expanded.com",
						"19.expanded.com",
						"20.expanded.com",
						"21.expanded.com",
						"22.expanded.com",
						"23.expanded.com",
						"24.expanded.com",
						"25.expanded.com",
						"26.expanded.com",
						"27.expanded.com",
						"28.expanded.com",
						"29.expanded.com",
						"30.expanded.com",
						"31.expanded.com",
						"32.expanded.com",
					},
				},
			},
			want: false,
		},
		{
			desc: "update to expanded DNSConfig",
			podSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{
					Searches: []string{
						"foo.com",
						"bar.io",
						"3.com",
						"4.com",
						"5.com",
						"6.com",
						"baz.expanded.com",
						"8.expanded.com",
						"9.expanded.com",
						"10.expanded.com",
						"11.expanded.com",
						"12.expanded.com",
						"13.expanded.com",
						"14.expanded.com",
						"15.expanded.com",
						"16.expanded.com",
						"17.expanded.com",
						"18.expanded.com",
						"19.expanded.com",
						"20.expanded.com",
						"21.expanded.com",
						"22.expanded.com",
						"23.expanded.com",
						"24.expanded.com",
						"25.expanded.com",
						"26.expanded.com",
						"27.expanded.com",
						"28.expanded.com",
						"29.expanded.com",
						"30.expanded.com",
						"31.expanded.com",
						"32.expanded.com",
					},
				},
			},
			oldPodSpec: &api.PodSpec{
				DNSConfig: &api.PodDNSConfig{
					Searches: []string{
						"foo.com",
						"bar.io",
						"3.com",
						"4.com",
						"5.com",
						"6.com",
					},
				},
			},
			want: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			got := haveSameExpandedDNSConfig(tc.podSpec, tc.oldPodSpec)
			if tc.want != got {
				t.Errorf("unexpected diff, want: %v, got: %v", tc.want, got)
			}
		})
	}
}

func TestDropDisabledTopologySpreadConstraintsFields(t *testing.T) {
	testCases := []struct {
		name        string
		enabled     bool
		podSpec     *api.PodSpec
		oldPodSpec  *api.PodSpec
		wantPodSpec *api.PodSpec
	}{
		{
			name:        "TopologySpreadConstraints is nil",
			podSpec:     &api.PodSpec{},
			oldPodSpec:  &api.PodSpec{},
			wantPodSpec: &api.PodSpec{},
		},
		{
			name:        "TopologySpreadConstraints is empty",
			podSpec:     &api.PodSpec{TopologySpreadConstraints: []api.TopologySpreadConstraint{}},
			oldPodSpec:  &api.PodSpec{TopologySpreadConstraints: []api.TopologySpreadConstraint{}},
			wantPodSpec: &api.PodSpec{TopologySpreadConstraints: []api.TopologySpreadConstraint{}},
		},
		{
			name: "TopologySpreadConstraints is not empty, but all constraints don't have minDomains",
			podSpec: &api.PodSpec{TopologySpreadConstraints: []api.TopologySpreadConstraint{
				{
					MinDomains: nil,
				},
				{
					MinDomains: nil,
				},
			}},
			oldPodSpec: &api.PodSpec{TopologySpreadConstraints: []api.TopologySpreadConstraint{
				{
					MinDomains: nil,
				},
				{
					MinDomains: nil,
				},
			}},
			wantPodSpec: &api.PodSpec{TopologySpreadConstraints: []api.TopologySpreadConstraint{
				{
					MinDomains: nil,
				},
				{
					MinDomains: nil,
				},
			}},
		},
		{
			name: "one constraint in podSpec has non-empty minDomains, feature gate is disabled " +
				"and all constraint in oldPodSpec doesn't have minDomains",
			enabled: false,
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						MinDomains: pointer.Int32(2),
					},
					{
						MinDomains: nil,
					},
				},
			},
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						MinDomains: nil,
					},
					{
						MinDomains: nil,
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						// cleared.
						MinDomains: nil,
					},
					{
						MinDomains: nil,
					},
				},
			},
		},
		{
			name: "one constraint in podSpec has non-empty minDomains, feature gate is disabled " +
				"and one constraint in oldPodSpec has minDomains",
			enabled: false,
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						MinDomains: pointer.Int32(2),
					},
					{
						MinDomains: nil,
					},
				},
			},
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						MinDomains: pointer.Int32(2),
					},
					{
						MinDomains: nil,
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						// not cleared.
						MinDomains: pointer.Int32(2),
					},
					{
						MinDomains: nil,
					},
				},
			},
		},
		{
			name: "one constraint in podSpec has non-empty minDomains, feature gate is enabled" +
				"and all constraint in oldPodSpec doesn't have minDomains",
			enabled: true,
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						MinDomains: pointer.Int32(2),
					},
					{
						MinDomains: nil,
					},
				},
			},
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						MinDomains: nil,
					},
					{
						MinDomains: nil,
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						// not cleared.
						MinDomains: pointer.Int32(2),
					},
					{
						MinDomains: nil,
					},
				},
			},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MinDomainsInPodTopologySpread, tc.enabled)()
			dropDisabledFields(tc.podSpec, nil, tc.oldPodSpec, nil)
			if diff := cmp.Diff(tc.wantPodSpec, tc.podSpec); diff != "" {
				t.Errorf("unexpected pod spec (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestDropOSField(t *testing.T) {
	podWithOSField := func() *api.Pod {
		osField := api.PodOS{Name: "linux"}
		return &api.Pod{
			Spec: api.PodSpec{
				OS: &osField,
			},
		}
	}
	podWithoutOSField := func() *api.Pod { return &api.Pod{} }
	podInfo := []struct {
		description   string
		hasPodOSField bool
		pod           func() *api.Pod
	}{
		{
			description:   "has PodOS field",
			hasPodOSField: true,
			pod:           podWithOSField,
		},
		{
			description:   "does not have PodOS field",
			hasPodOSField: false,
			pod:           podWithoutOSField,
		},
		{
			description:   "is nil",
			hasPodOSField: false,
			pod:           func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasOsField, oldPod := oldPodInfo.hasPodOSField, oldPodInfo.pod()
				newPodHasOSField, newPod := newPodInfo.hasPodOSField, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IdentifyPodOS, enabled)()

					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", cmp.Diff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasOsField:
						// new pod should not be changed if the feature is enabled, or if the old pod had subpaths
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					case newPodHasOSField:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have OSfield
						if !reflect.DeepEqual(newPod, podWithoutOSField()) {
							t.Errorf("new pod has OS field: %v", cmp.Diff(newPod, podWithoutOSField()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}

func TestDropNodeInclusionPolicyFields(t *testing.T) {
	ignore := api.NodeInclusionPolicyIgnore
	honor := api.NodeInclusionPolicyHonor

	tests := []struct {
		name        string
		enabled     bool
		podSpec     *api.PodSpec
		oldPodSpec  *api.PodSpec
		wantPodSpec *api.PodSpec
	}{
		{
			name:    "feature disabled, both pods don't use the fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
		},
		{
			name:    "feature disabled, only old pod use NodeAffinityPolicy field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{NodeAffinityPolicy: &honor},
				},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
		},
		{
			name:    "feature disabled, only old pod use NodeTaintsPolicy field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{NodeTaintsPolicy: &ignore},
				},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
		},
		{
			name:    "feature disabled, only current pod use NodeAffinityPolicy field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{NodeAffinityPolicy: &honor},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{{
					NodeAffinityPolicy: nil,
				}},
			},
		},
		{
			name:    "feature disabled, only current pod use NodeTaintsPolicy field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{NodeTaintsPolicy: &ignore},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{NodeTaintsPolicy: nil},
				},
			},
		},
		{
			name:    "feature disabled, both pods use NodeAffinityPolicy fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{NodeAffinityPolicy: &honor},
				},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{NodeAffinityPolicy: &ignore},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{NodeAffinityPolicy: &ignore},
				},
			},
		},
		{
			name:    "feature disabled, both pods use NodeTaintsPolicy fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{NodeTaintsPolicy: &ignore},
				},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{NodeTaintsPolicy: &honor},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{NodeTaintsPolicy: &honor},
				},
			},
		},
		{
			name:    "feature enabled, both pods use the fields",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						NodeAffinityPolicy: &ignore,
						NodeTaintsPolicy:   &honor,
					},
				},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						NodeAffinityPolicy: &honor,
						NodeTaintsPolicy:   &ignore,
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						NodeAffinityPolicy: &honor,
						NodeTaintsPolicy:   &ignore,
					},
				},
			},
		},
		{
			name:    "feature enabled, only old pod use NodeAffinityPolicy field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						NodeAffinityPolicy: &honor,
					},
				},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
		},
		{
			name:    "feature enabled, only old pod use NodeTaintsPolicy field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						NodeTaintsPolicy: &ignore,
					},
				},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
		},
		{
			name:    "feature enabled, only current pod use NodeAffinityPolicy field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						NodeAffinityPolicy: &ignore,
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						NodeAffinityPolicy: &ignore,
					},
				},
			},
		},
		{
			name:    "feature enabled, only current pod use NodeTaintsPolicy field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						NodeTaintsPolicy: &honor,
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						NodeTaintsPolicy: &honor,
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeInclusionPolicyInPodTopologySpread, test.enabled)()

			dropDisabledFields(test.podSpec, nil, test.oldPodSpec, nil)
			if diff := cmp.Diff(test.wantPodSpec, test.podSpec); diff != "" {
				t.Errorf("unexpected pod spec (-want, +got):\n%s", diff)
			}
		})
	}
}
