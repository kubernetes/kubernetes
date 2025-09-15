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
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"k8s.io/component-base/featuregate"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/utils/ptr"
)

func TestVisitContainers(t *testing.T) {
	setAllFeatureEnabledContainersDuringTest := ContainerType(0)
	testCases := []struct {
		desc           string
		spec           *api.PodSpec
		wantContainers []string
		mask           ContainerType
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
			wantContainers: []string{"i1", "i2", "c1", "c2", "e1", "e2"},
			mask:           setAllFeatureEnabledContainersDuringTest,
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

func TestContainerIter(t *testing.T) {
	testCases := []struct {
		desc           string
		spec           *api.PodSpec
		wantContainers []string
		mask           ContainerType
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
			desc: "init + main containers",
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
			mask:           InitContainers | Containers,
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
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			gotContainers := []string{}
			for c, containerType := range ContainerIter(tc.spec, tc.mask) {
				gotContainers = append(gotContainers, c.Name)

				switch containerType {
				case InitContainers:
					if c.Name[0] != 'i' {
						t.Errorf("ContainerIter() yielded container type InitContainers for container %q", c.Name)
					}
				case Containers:
					if c.Name[0] != 'c' {
						t.Errorf("ContainerIter() yielded container type Containers for container %q", c.Name)
					}
				case EphemeralContainers:
					if c.Name[0] != 'e' {
						t.Errorf("ContainerIter() yielded container type EphemeralContainers for container %q", c.Name)
					}
				default:
					t.Errorf("ContainerIter() yielded unknown container type %d", containerType)
				}
			}

			if !cmp.Equal(gotContainers, tc.wantContainers) {
				t.Errorf("ContainerIter() = %+v, want %+v", gotContainers, tc.wantContainers)
			}
		})
	}
}

func TestPodSecrets(t *testing.T) {
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
	extractedNames := sets.New[string]()
	VisitPodSecretNames(pod, func(name string) bool {
		extractedNames.Insert(name)
		return true
	}, AllContainers)

	// excludedSecretPaths holds struct paths to fields with "secret" in the name that are not actually references to secret API objects
	excludedSecretPaths := sets.New[string](
		"Spec.Volumes[*].VolumeSource.CephFS.SecretFile",
	)
	// expectedSecretPaths holds struct paths to fields with "secret" in the name that are references to secret API objects.
	// every path here should be represented as an example in the Pod stub above, with the secret name set to the path.
	expectedSecretPaths := sets.New[string](
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
		t.Logf("Missing expected secret paths:\n%s", strings.Join(sets.List[string](missingPaths), "\n"))
		t.Error("Missing expected secret paths. Verify VisitPodSecretNames() is correctly finding the missing paths, then correct expectedSecretPaths")
	}
	if extraPaths := secretPaths.Difference(expectedSecretPaths); len(extraPaths) > 0 {
		t.Logf("Extra secret paths:\n%s", strings.Join(sets.List[string](extraPaths), "\n"))
		t.Error("Extra fields with 'secret' in the name found. Verify VisitPodSecretNames() is including these fields if appropriate, then correct expectedSecretPaths")
	}

	if missingNames := expectedSecretPaths.Difference(extractedNames); len(missingNames) > 0 {
		t.Logf("Missing expected secret names:\n%s", strings.Join(sets.List[string](missingNames), "\n"))
		t.Error("Missing expected secret names. Verify the pod stub above includes these references, then verify VisitPodSecretNames() is correctly finding the missing names")
	}
	if extraNames := extractedNames.Difference(expectedSecretPaths); len(extraNames) > 0 {
		t.Logf("Extra secret names:\n%s", strings.Join(sets.List[string](extraNames), "\n"))
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
func collectResourcePaths(t *testing.T, resourcename string, path *field.Path, name string, tp reflect.Type) sets.Set[string] {
	resourcename = strings.ToLower(resourcename)
	resourcePaths := sets.New[string]()

	if tp.Kind() == reflect.Pointer {
		resourcePaths.Insert(sets.List[string](collectResourcePaths(t, resourcename, path, name, tp.Elem()))...)
		return resourcePaths
	}

	if strings.Contains(strings.ToLower(name), resourcename) {
		resourcePaths.Insert(path.String())
	}

	switch tp.Kind() {
	case reflect.Pointer:
		resourcePaths.Insert(sets.List[string](collectResourcePaths(t, resourcename, path, name, tp.Elem()))...)
	case reflect.Struct:
		// ObjectMeta is generic and therefore should never have a field with a specific resource's name;
		// it contains cycles so it's easiest to just skip it.
		if name == "ObjectMeta" {
			break
		}
		for i := 0; i < tp.NumField(); i++ {
			field := tp.Field(i)
			resourcePaths.Insert(sets.List[string](collectResourcePaths(t, resourcename, path.Child(field.Name), field.Name, field.Type))...)
		}
	case reflect.Interface:
		t.Errorf("cannot find %s fields in interface{} field %s", resourcename, path.String())
	case reflect.Map:
		resourcePaths.Insert(sets.List[string](collectResourcePaths(t, resourcename, path.Key("*"), "", tp.Elem()))...)
	case reflect.Slice:
		resourcePaths.Insert(sets.List[string](collectResourcePaths(t, resourcename, path.Key("*"), "", tp.Elem()))...)
	default:
		// all primitive types
	}

	return resourcePaths
}

func TestPodConfigmaps(t *testing.T) {
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
	extractedNames := sets.New[string]()
	VisitPodConfigmapNames(pod, func(name string) bool {
		extractedNames.Insert(name)
		return true
	}, AllContainers)

	// expectedPaths holds struct paths to fields with "ConfigMap" in the name that are references to ConfigMap API objects.
	// every path here should be represented as an example in the Pod stub above, with the ConfigMap name set to the path.
	expectedPaths := sets.New[string](
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
		t.Logf("Missing expected paths:\n%s", strings.Join(sets.List[string](missingPaths), "\n"))
		t.Error("Missing expected paths. Verify VisitPodConfigmapNames() is correctly finding the missing paths, then correct expectedPaths")
	}
	if extraPaths := collectPaths.Difference(expectedPaths); len(extraPaths) > 0 {
		t.Logf("Extra paths:\n%s", strings.Join(sets.List[string](extraPaths), "\n"))
		t.Error("Extra fields with resource in the name found. Verify VisitPodConfigmapNames() is including these fields if appropriate, then correct expectedPaths")
	}

	if missingNames := expectedPaths.Difference(extractedNames); len(missingNames) > 0 {
		t.Logf("Missing expected names:\n%s", strings.Join(sets.List[string](missingNames), "\n"))
		t.Error("Missing expected names. Verify the pod stub above includes these references, then verify VisitPodConfigmapNames() is correctly finding the missing names")
	}
	if extraNames := extractedNames.Difference(expectedPaths); len(extraNames) > 0 {
		t.Logf("Extra names:\n%s", strings.Join(sets.List[string](extraNames), "\n"))
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
				secContext := newPod.Spec.SecurityContext
				if secContext != nil && secContext.FSGroupChangePolicy != nil {
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
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ProcMountType, enabled)

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

func TestDropDynamicResourceAllocation(t *testing.T) {
	resourceClaimName := "external-claim"

	podWithClaims := &api.Pod{
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Resources: api.ResourceRequirements{
						Claims: []api.ResourceClaim{{Name: "my-claim"}},
					},
				},
			},
			InitContainers: []api.Container{
				{
					Resources: api.ResourceRequirements{
						Claims: []api.ResourceClaim{{Name: "my-claim"}},
					},
				},
			},
			EphemeralContainers: []api.EphemeralContainer{
				{
					EphemeralContainerCommon: api.EphemeralContainerCommon{
						Resources: api.ResourceRequirements{
							Claims: []api.ResourceClaim{{Name: "my-claim"}},
						},
					},
				},
			},
			ResourceClaims: []api.PodResourceClaim{
				{
					Name:              "my-claim",
					ResourceClaimName: &resourceClaimName,
				},
			},
		},
		Status: api.PodStatus{
			ResourceClaimStatuses: []api.PodResourceClaimStatus{
				{Name: "my-claim", ResourceClaimName: ptr.To("pod-my-claim")},
			},
		},
	}
	podWithoutClaims := &api.Pod{
		Spec: api.PodSpec{
			Containers:          []api.Container{{}},
			InitContainers:      []api.Container{{}},
			EphemeralContainers: []api.EphemeralContainer{{}},
		},
	}

	var noPod *api.Pod

	testcases := []struct {
		description string
		enabled     bool
		oldPod      *api.Pod
		newPod      *api.Pod
		wantPod     *api.Pod
	}{
		{
			description: "old with claims / new with claims / disabled",
			oldPod:      podWithClaims,
			newPod:      podWithClaims,
			wantPod:     podWithClaims,
		},
		{
			description: "old without claims / new with claims / disabled",
			oldPod:      podWithoutClaims,
			newPod:      podWithClaims,
			wantPod:     podWithoutClaims,
		},
		{
			description: "no old pod/ new with claims / disabled",
			oldPod:      noPod,
			newPod:      podWithClaims,
			wantPod:     podWithoutClaims,
		},

		{
			description: "old with claims / new without claims / disabled",
			oldPod:      podWithClaims,
			newPod:      podWithoutClaims,
			wantPod:     podWithoutClaims,
		},
		{
			description: "old without claims / new without claims / disabled",
			oldPod:      podWithoutClaims,
			newPod:      podWithoutClaims,
			wantPod:     podWithoutClaims,
		},
		{
			description: "no old pod/ new without claims / disabled",
			oldPod:      noPod,
			newPod:      podWithoutClaims,
			wantPod:     podWithoutClaims,
		},

		{
			description: "old with claims / new with claims / enabled",
			enabled:     true,
			oldPod:      podWithClaims,
			newPod:      podWithClaims,
			wantPod:     podWithClaims,
		},
		{
			description: "old without claims / new with claims / enabled",
			enabled:     true,
			oldPod:      podWithoutClaims,
			newPod:      podWithClaims,
			wantPod:     podWithClaims,
		},
		{
			description: "no old pod/ new with claims / enabled",
			enabled:     true,
			oldPod:      noPod,
			newPod:      podWithClaims,
			wantPod:     podWithClaims,
		},

		{
			description: "old with claims / new without claims / enabled",
			enabled:     true,
			oldPod:      podWithClaims,
			newPod:      podWithoutClaims,
			wantPod:     podWithoutClaims,
		},
		{
			description: "old without claims / new without claims / enabled",
			enabled:     true,
			oldPod:      podWithoutClaims,
			newPod:      podWithoutClaims,
			wantPod:     podWithoutClaims,
		},
		{
			description: "no old pod/ new without claims / enabled",
			enabled:     true,
			oldPod:      noPod,
			newPod:      podWithoutClaims,
			wantPod:     podWithoutClaims,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DynamicResourceAllocation, tc.enabled)

			oldPod := tc.oldPod.DeepCopy()
			newPod := tc.newPod.DeepCopy()
			wantPod := tc.wantPod
			DropDisabledPodFields(newPod, oldPod)

			// old pod should never be changed
			if diff := cmp.Diff(oldPod, tc.oldPod); diff != "" {
				t.Errorf("old pod changed: %s", diff)
			}

			if diff := cmp.Diff(wantPod, newPod); diff != "" {
				t.Errorf("new pod changed (- want, + got): %s", diff)
			}
		})
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
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodDeletionCost, tc.featureEnabled)
			// The new pod doesn't impact the outcome.
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(nil, nil, nil, tc.oldPodMeta)
			if tc.wantAllowInvalidPodDeletionCost != gotOptions.AllowInvalidPodDeletionCost {
				t.Errorf("unexpected diff, want: %v, got: %v", tc.wantAllowInvalidPodDeletionCost, gotOptions.AllowInvalidPodDeletionCost)
			}
		})
	}
}

func TestDropDisabledPodStatusFields_HostIPs(t *testing.T) {
	podWithHostIPs := func() *api.PodStatus {
		return &api.PodStatus{
			HostIPs: makeHostIPs("10.0.0.1", "fd00:10::1"),
		}
	}

	podWithoutHostIPs := func() *api.PodStatus {
		return &api.PodStatus{
			HostIPs: nil,
		}
	}

	tests := []struct {
		name          string
		podStatus     *api.PodStatus
		oldPodStatus  *api.PodStatus
		wantPodStatus *api.PodStatus
	}{
		{
			name:         "old=without, new=without",
			oldPodStatus: podWithoutHostIPs(),
			podStatus:    podWithoutHostIPs(),

			wantPodStatus: podWithoutHostIPs(),
		},
		{
			name:         "old=without, new=with",
			oldPodStatus: podWithoutHostIPs(),
			podStatus:    podWithHostIPs(),

			wantPodStatus: podWithHostIPs(),
		},
		{
			name:         "old=with, new=without",
			oldPodStatus: podWithHostIPs(),
			podStatus:    podWithoutHostIPs(),

			wantPodStatus: podWithoutHostIPs(),
		},
		{
			name:         "old=with, new=with",
			oldPodStatus: podWithHostIPs(),
			podStatus:    podWithHostIPs(),

			wantPodStatus: podWithHostIPs(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dropDisabledPodStatusFields(tt.podStatus, tt.oldPodStatus, &api.PodSpec{}, &api.PodSpec{})

			if !reflect.DeepEqual(tt.podStatus, tt.wantPodStatus) {
				t.Errorf("dropDisabledStatusFields() = %v, want %v", tt.podStatus, tt.wantPodStatus)
			}
		})
	}
}

func makeHostIPs(ips ...string) []api.HostIP {
	ret := []api.HostIP{}
	for _, ip := range ips {
		ret = append(ret, api.HostIP{IP: ip})
	}
	return ret
}

func TestDropDisabledPodStatusFields_ObservedGeneration(t *testing.T) {
	now := metav1.NewTime(time.Now())

	podWithObservedGen := func() *api.PodStatus {
		return &api.PodStatus{
			ObservedGeneration: 1,
			Conditions: []api.PodCondition{{
				LastProbeTime:      now,
				LastTransitionTime: now,
			}},
		}
	}
	podWithObservedGenInConditions := func() *api.PodStatus {
		return &api.PodStatus{
			Conditions: []api.PodCondition{{
				LastProbeTime:      now,
				LastTransitionTime: now,
				ObservedGeneration: 1,
			}},
		}
	}

	podWithoutObservedGen := func() *api.PodStatus {
		return &api.PodStatus{
			Conditions: []api.PodCondition{{
				LastProbeTime:      now,
				LastTransitionTime: now,
			}},
		}
	}

	tests := []struct {
		name          string
		podStatus     *api.PodStatus
		oldPodStatus  *api.PodStatus
		featureGateOn bool
		wantPodStatus *api.PodStatus
	}{
		{
			name:          "old=without, new=without / feature gate off",
			oldPodStatus:  podWithoutObservedGen(),
			podStatus:     podWithoutObservedGen(),
			featureGateOn: false,
			wantPodStatus: podWithoutObservedGen(),
		},
		{
			name:          "old=without, new=without / feature gate on",
			oldPodStatus:  podWithoutObservedGen(),
			podStatus:     podWithoutObservedGen(),
			featureGateOn: true,
			wantPodStatus: podWithoutObservedGen(),
		},
		{
			name:          "old=without, new=with / feature gate off",
			oldPodStatus:  podWithoutObservedGen(),
			podStatus:     podWithObservedGen(),
			featureGateOn: false,
			wantPodStatus: podWithoutObservedGen(),
		},
		{
			name:          "old=with, new=without / feature gate on",
			oldPodStatus:  podWithObservedGen(),
			podStatus:     podWithoutObservedGen(),
			featureGateOn: true,
			wantPodStatus: podWithoutObservedGen(),
		},
		{
			name:          "old=with, new=with / feature gate off",
			oldPodStatus:  podWithObservedGen(),
			podStatus:     podWithObservedGen(),
			featureGateOn: false,
			wantPodStatus: podWithObservedGen(),
		},
		{
			name:          "old=with, new=with / feature gate on",
			oldPodStatus:  podWithObservedGen(),
			podStatus:     podWithObservedGen(),
			featureGateOn: true,
			wantPodStatus: podWithObservedGen(),
		},
		{
			name:          "old=without, new=withInConditions / feature gate off",
			oldPodStatus:  podWithoutObservedGen(),
			podStatus:     podWithObservedGenInConditions(),
			featureGateOn: false,
			wantPodStatus: podWithoutObservedGen(),
		},
		{
			name:          "old=without, new=withInConditions / feature gate on",
			oldPodStatus:  podWithoutObservedGen(),
			podStatus:     podWithObservedGenInConditions(),
			featureGateOn: true,
			wantPodStatus: podWithObservedGenInConditions(),
		},
		{
			name:          "old=withInConditions, new=without / feature gate off",
			oldPodStatus:  podWithObservedGenInConditions(),
			podStatus:     podWithoutObservedGen(),
			featureGateOn: false,
			wantPodStatus: podWithoutObservedGen(),
		},
		{
			name:          "old=withInConditions, new=without / feature gate on",
			oldPodStatus:  podWithObservedGenInConditions(),
			podStatus:     podWithoutObservedGen(),
			featureGateOn: true,
			wantPodStatus: podWithoutObservedGen(),
		},
		{
			name:          "old=withInConditions, new=withInCondtions / feature gate off",
			oldPodStatus:  podWithObservedGenInConditions(),
			podStatus:     podWithObservedGenInConditions(),
			featureGateOn: false,
			wantPodStatus: podWithObservedGenInConditions(),
		},
		{
			name:          "old=withInConditions, new=withInCondtions / feature gate on",
			oldPodStatus:  podWithObservedGenInConditions(),
			podStatus:     podWithObservedGenInConditions(),
			featureGateOn: true,
			wantPodStatus: podWithObservedGenInConditions(),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodObservedGenerationTracking, tt.featureGateOn)
			dropDisabledPodStatusFields(tt.podStatus, tt.oldPodStatus, &api.PodSpec{}, &api.PodSpec{})
			if !reflect.DeepEqual(tt.podStatus, tt.wantPodStatus) {
				t.Errorf("dropDisabledStatusFields() = %v, want %v", tt.podStatus, tt.wantPodStatus)
			}
		})
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
			if !test.enabled {
				// TODO: this will be removed in 1.36
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.32"))
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.NodeInclusionPolicyInPodTopologySpread, test.enabled)
			}

			dropDisabledFields(test.podSpec, nil, test.oldPodSpec, nil)
			if diff := cmp.Diff(test.wantPodSpec, test.podSpec); diff != "" {
				t.Errorf("unexpected pod spec (-want, +got):\n%s", diff)
			}
		})
	}
}

func Test_dropDisabledMatchLabelKeysFieldInPodAffinity(t *testing.T) {
	tests := []struct {
		name        string
		enabled     bool
		podSpec     *api.PodSpec
		oldPodSpec  *api.PodSpec
		wantPodSpec *api.PodSpec
	}{
		{
			name:    "[PodAffinity/required] feature disabled, both pods don't use MatchLabelKeys/MismatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/required] feature disabled, only old pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/required] feature disabled, only current pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{{}},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/required] feature disabled, both pods use MatchLabelKeys/MismatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/required] feature enabled, only old pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/required] feature enabled, only current pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/required] feature enabled, both pods use MatchLabelKeys/MismatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/preferred] feature disabled, both pods don't use MatchLabelKeys/MismatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/preferred] feature disabled, only old pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/preferred] feature disabled, only current pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{{}},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/preferred] feature disabled, both pods use MatchLabelKeys/MismatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/preferred] feature enabled, only old pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/preferred] feature enabled, only current pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
		},
		{
			name:    "[PodAffinity/preferred] feature enabled, both pods use MatchLabelKeys/MismatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAffinity: &api.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/required] feature disabled, both pods don't use MatchLabelKeys/MismatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/required] feature disabled, only old pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/required] feature disabled, only current pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{{}},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/required] feature disabled, both pods use MatchLabelKeys/MismatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/required] feature enabled, only old pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/required] feature enabled, only current pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/required] feature enabled, both pods use MatchLabelKeys/MismatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []api.PodAffinityTerm{
							{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
						},
					},
				},
			},
		},

		{
			name:    "[PodAntiAffinity/preferred] feature disabled, both pods don't use MatchLabelKeys/MismatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/preferred] feature disabled, only old pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/preferred] feature disabled, only current pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{{}},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/preferred] feature disabled, both pods use MatchLabelKeys/MismatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/preferred] feature enabled, only old pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/preferred] feature enabled, only current pod uses MatchLabelKeys/MismatchLabelKeys field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
		},
		{
			name:    "[PodAntiAffinity/preferred] feature enabled, both pods use MatchLabelKeys/MismatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			podSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
			wantPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					PodAntiAffinity: &api.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []api.WeightedPodAffinityTerm{
							{
								PodAffinityTerm: api.PodAffinityTerm{MatchLabelKeys: []string{"foo"}, MismatchLabelKeys: []string{"foo"}},
							},
						},
					},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if !test.enabled {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.32"))
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MatchLabelKeysInPodAffinity, false)
			}

			dropDisabledFields(test.podSpec, nil, test.oldPodSpec, nil)
			if diff := cmp.Diff(test.wantPodSpec, test.podSpec); diff != "" {
				t.Errorf("unexpected pod spec (-want, +got):\n%s", diff)
			}
		})
	}
}

func Test_dropDisabledMatchLabelKeysFieldInTopologySpread(t *testing.T) {
	tests := []struct {
		name        string
		enabled     bool
		podSpec     *api.PodSpec
		oldPodSpec  *api.PodSpec
		wantPodSpec *api.PodSpec
	}{
		{
			name:    "feature disabled, both pods don't use MatchLabelKeys fields",
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
			name:    "feature disabled, only old pod uses MatchLabelKeys field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{MatchLabelKeys: []string{"foo"}},
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
			name:    "feature disabled, only current pod uses MatchLabelKeys field",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{MatchLabelKeys: []string{"foo"}},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{{}},
			},
		},
		{
			name:    "feature disabled, both pods use MatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{MatchLabelKeys: []string{"foo"}},
				},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{MatchLabelKeys: []string{"foo"}},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{MatchLabelKeys: []string{"foo"}},
				},
			},
		},
		{
			name:    "feature enabled, only old pod uses MatchLabelKeys field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{MatchLabelKeys: []string{"foo"}},
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
			name:    "feature enabled, only current pod uses MatchLabelKeys field",
			enabled: true,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{MatchLabelKeys: []string{"foo"}},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{MatchLabelKeys: []string{"foo"}},
				},
			},
		},
		{
			name:    "feature enabled, both pods use MatchLabelKeys fields",
			enabled: false,
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{MatchLabelKeys: []string{"foo"}},
				},
			},
			podSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{MatchLabelKeys: []string{"foo"}},
				},
			},
			wantPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{MatchLabelKeys: []string{"foo"}},
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MatchLabelKeysInPodTopologySpread, test.enabled)

			dropDisabledFields(test.podSpec, nil, test.oldPodSpec, nil)
			if diff := cmp.Diff(test.wantPodSpec, test.podSpec); diff != "" {
				t.Errorf("unexpected pod spec (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestDropHostUsers(t *testing.T) {
	falseVar := false
	trueVar := true

	podWithoutHostUsers := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{}},
		}
	}
	podWithHostUsersFalse := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{
					HostUsers: &falseVar,
				},
			},
		}
	}
	podWithHostUsersTrue := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{
					HostUsers: &trueVar,
				},
			},
		}
	}

	podInfo := []struct {
		description  string
		hasHostUsers bool
		pod          func() *api.Pod
	}{
		{
			description:  "with hostUsers=true",
			hasHostUsers: true,
			pod:          podWithHostUsersTrue,
		},
		{
			description:  "with hostUsers=false",
			hasHostUsers: true,
			pod:          podWithHostUsersFalse,
		},
		{
			description: "with hostUsers=nil",
			pod:         func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasHostUsers, oldPod := oldPodInfo.hasHostUsers, oldPodInfo.pod()
				newPodHasHostUsers, newPod := newPodInfo.hasHostUsers, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.UserNamespacesSupport, enabled)

					DropDisabledPodFields(newPod, oldPod)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", cmp.Diff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasHostUsers:
						// new pod should not be changed if the feature is enabled, or if the old pod had hostUsers
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					case newPodHasHostUsers:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have hostUsers
						if exp := podWithoutHostUsers(); !reflect.DeepEqual(newPod, exp) {
							t.Errorf("new pod had hostUsers: %v", cmp.Diff(newPod, exp))
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

func TestValidateTopologySpreadConstraintLabelSelectorOption(t *testing.T) {
	testCases := []struct {
		name       string
		oldPodSpec *api.PodSpec
		wantOption bool
	}{
		{
			name:       "Create",
			wantOption: false,
		},
		{
			name: "UpdateInvalidLabelSelector",
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						LabelSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"NoUppercaseOrSpecialCharsLike=Equals": "foo"},
						},
					},
				},
			},
			wantOption: true,
		},
		{
			name: "UpdateValidLabelSelector",
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						LabelSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"foo": "foo"},
						},
					},
				},
			},
			wantOption: false,
		},
		{
			name: "UpdateEmptyLabelSelector",
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{
					{
						LabelSelector: nil,
					},
				},
			},
			wantOption: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Pod meta doesn't impact the outcome.
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&api.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.AllowInvalidTopologySpreadConstraintLabelSelector {
				t.Errorf("Got AllowInvalidLabelValueInSelector=%t, want %t", gotOptions.AllowInvalidTopologySpreadConstraintLabelSelector, tc.wantOption)
			}
		})
	}
}

func TestOldPodViolatesMatchLabelKeysValidationOption(t *testing.T) {
	testCases := []struct {
		name                               string
		oldPodSpec                         *api.PodSpec
		matchLabelKeysEnabled              bool
		matchLabelKeysSelectorMergeEnabled bool
		wantOption                         bool
	}{
		{
			name:                               "Create",
			oldPodSpec:                         &api.PodSpec{},
			matchLabelKeysEnabled:              true,
			matchLabelKeysSelectorMergeEnabled: true,
			wantOption:                         false,
		},
		{
			name: "UpdateInvalidMatchLabelKeys",
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{{
					MatchLabelKeys: []string{"foo"},
					LabelSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "foo",
								Operator: metav1.LabelSelectorOpNotIn,
								Values:   []string{"value1"},
							},
							{
								Key:      "foo",
								Operator: metav1.LabelSelectorOpNotIn,
								Values:   []string{"value2", "value3"},
							},
						},
					},
				}},
			},
			matchLabelKeysEnabled:              true,
			matchLabelKeysSelectorMergeEnabled: true,
			wantOption:                         true,
		},
		{
			name: "UpdateValidMatchLabelKeys",
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{{
					MatchLabelKeys: []string{"foo"},
					LabelSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "foo",
								Operator: metav1.LabelSelectorOpNotIn,
								Values:   []string{"value1"},
							},
						},
					},
				}},
			},
			matchLabelKeysEnabled:              true,
			matchLabelKeysSelectorMergeEnabled: true,
			wantOption:                         false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MatchLabelKeysInPodTopologySpread, tc.matchLabelKeysEnabled)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MatchLabelKeysInPodTopologySpreadSelectorMerge, tc.matchLabelKeysSelectorMergeEnabled)
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&api.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.OldPodViolatesMatchLabelKeysValidation {
				t.Errorf("Got OldPodViolatesMatchLabelKeysValidation=%t, want %t", gotOptions.OldPodViolatesMatchLabelKeysValidation, tc.wantOption)
			}
		})
	}
}

func TestOldPodViolatesLegacyMatchLabelKeysValidationOption(t *testing.T) {
	testCases := []struct {
		name                               string
		oldPodSpec                         *api.PodSpec
		matchLabelKeysEnabled              bool
		matchLabelKeysSelectorMergeEnabled bool
		wantOption                         bool
	}{
		{
			name:                               "Create",
			oldPodSpec:                         &api.PodSpec{},
			matchLabelKeysEnabled:              true,
			matchLabelKeysSelectorMergeEnabled: false,
			wantOption:                         false,
		},
		{
			name: "UpdateInvalidMatchLabelKeys",
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{{
					MatchLabelKeys: []string{"foo"},
					LabelSelector: &metav1.LabelSelector{
						MatchExpressions: []metav1.LabelSelectorRequirement{
							{
								Key:      "foo",
								Operator: metav1.LabelSelectorOpNotIn,
								Values:   []string{"value1"},
							},
						},
					},
				}},
			},
			matchLabelKeysEnabled:              true,
			matchLabelKeysSelectorMergeEnabled: false,
			wantOption:                         true,
		},
		{
			name: "UpdateValidMatchLabelKeys",
			oldPodSpec: &api.PodSpec{
				TopologySpreadConstraints: []api.TopologySpreadConstraint{{
					MatchLabelKeys: []string{"foo"},
					LabelSelector:  &metav1.LabelSelector{},
				}},
			},
			matchLabelKeysEnabled:              true,
			matchLabelKeysSelectorMergeEnabled: false,
			wantOption:                         false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MatchLabelKeysInPodTopologySpread, tc.matchLabelKeysEnabled)
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MatchLabelKeysInPodTopologySpreadSelectorMerge, tc.matchLabelKeysSelectorMergeEnabled)
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&api.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.OldPodViolatesLegacyMatchLabelKeysValidation {
				t.Errorf("Got OldPodViolatesLegacyMatchLabelKeysValidation=%t, want %t", gotOptions.OldPodViolatesLegacyMatchLabelKeysValidation, tc.wantOption)
			}
		})
	}
}
func TestValidateAllowNonLocalProjectedTokenPathOption(t *testing.T) {
	testCases := []struct {
		name       string
		oldPodSpec *api.PodSpec
		wantOption bool
	}{
		{
			name:       "Create",
			wantOption: false,
		},
		{
			name: "UpdateInvalidProjectedTokenPath",
			oldPodSpec: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										ServiceAccountToken: &api.ServiceAccountTokenProjection{
											Path: "../foo",
										},
									},
								},
							},
						},
					},
				},
			},
			wantOption: true,
		},
		{
			name: "UpdateValidProjectedTokenPath",
			oldPodSpec: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										ServiceAccountToken: &api.ServiceAccountTokenProjection{
											Path: "foo",
										},
									},
								},
							},
						},
					},
				},
			},
			wantOption: false,
		},
		{
			name: "UpdateEmptyProjectedTokenPath",
			oldPodSpec: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: nil,
							HostPath:  &api.HostPathVolumeSource{Path: "foo"},
						},
					},
				},
			},
			wantOption: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Pod meta doesn't impact the outcome.
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&api.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.AllowNonLocalProjectedTokenPath {
				t.Errorf("Got AllowNonLocalProjectedTokenPath=%t, want %t", gotOptions.AllowNonLocalProjectedTokenPath, tc.wantOption)
			}
		})
	}
}

func TestDropInPlacePodVerticalScaling(t *testing.T) {
	podWithInPlaceVerticalScaling := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "c1",
						Image: "image",
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("200m")},
						},
						ResizePolicy: []api.ContainerResizePolicy{
							{ResourceName: api.ResourceCPU, RestartPolicy: api.NotRequired},
							{ResourceName: api.ResourceMemory, RestartPolicy: api.RestartContainer},
						},
					},
				},
			},
			Status: api.PodStatus{
				ContainerStatuses: []api.ContainerStatus{
					{
						Name:               "c1",
						Image:              "image",
						AllocatedResources: api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
						Resources: &api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("200m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("300m")},
						},
					},
				},
			},
		}
	}
	podWithoutInPlaceVerticalScaling := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "c1",
						Image: "image",
						Resources: api.ResourceRequirements{
							Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
							Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("200m")},
						},
					},
				},
			},
			Status: api.PodStatus{
				ContainerStatuses: []api.ContainerStatus{
					{
						Name:  "c1",
						Image: "image",
					},
				},
			},
		}
	}

	podInfo := []struct {
		description               string
		hasInPlaceVerticalScaling bool
		pod                       func() *api.Pod
	}{
		{
			description:               "has in-place vertical scaling enabled with resources",
			hasInPlaceVerticalScaling: true,
			pod:                       podWithInPlaceVerticalScaling,
		},
		{
			description:               "has in-place vertical scaling disabled",
			hasInPlaceVerticalScaling: false,
			pod:                       podWithoutInPlaceVerticalScaling,
		},
		{
			description:               "is nil",
			hasInPlaceVerticalScaling: false,
			pod:                       func() *api.Pod { return nil },
		},
	}

	for _, ippvsEnabled := range []bool{true, false} {
		t.Run(fmt.Sprintf("InPlacePodVerticalScaling=%t", ippvsEnabled), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, ippvsEnabled)

			for _, oldPodInfo := range podInfo {
				for _, newPodInfo := range podInfo {
					oldPodHasInPlaceVerticalScaling, oldPod := oldPodInfo.hasInPlaceVerticalScaling, oldPodInfo.pod()
					newPodHasInPlaceVerticalScaling, newPod := newPodInfo.hasInPlaceVerticalScaling, newPodInfo.pod()
					if newPod == nil {
						continue
					}

					t.Run(fmt.Sprintf("old pod %v, new pod %v", oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
						var oldPodSpec *api.PodSpec
						var oldPodStatus *api.PodStatus
						if oldPod != nil {
							oldPodSpec = &oldPod.Spec
							oldPodStatus = &oldPod.Status
						}
						dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)
						dropDisabledPodStatusFields(&newPod.Status, oldPodStatus, &newPod.Spec, oldPodSpec)

						// old pod should never be changed
						if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
							t.Errorf("old pod changed: %v", cmp.Diff(oldPod, oldPodInfo.pod()))
						}

						switch {
						case ippvsEnabled || oldPodHasInPlaceVerticalScaling:
							// new pod shouldn't change if feature enabled or if old pod has ResizePolicy set
							expected := newPodInfo.pod()
							if !reflect.DeepEqual(newPod, expected) {
								t.Errorf("new pod changed: %v", cmp.Diff(newPod, expected))
							}
						case newPodHasInPlaceVerticalScaling:
							// new pod should be changed
							if reflect.DeepEqual(newPod, newPodInfo.pod()) {
								t.Errorf("new pod was not changed")
							}
							// new pod should not have ResizePolicy
							if !reflect.DeepEqual(newPod, podWithoutInPlaceVerticalScaling()) {
								t.Errorf("new pod has ResizePolicy: %v", cmp.Diff(newPod, podWithoutInPlaceVerticalScaling()))
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

		})
	}
}

func TestDropPodLevelResources(t *testing.T) {
	containers := []api.Container{
		{
			Name:  "c1",
			Image: "image",
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{api.ResourceCPU: resource.MustParse("100m")},
				Limits:   api.ResourceList{api.ResourceCPU: resource.MustParse("200m")},
			},
		},
	}
	podWithPodLevelResources := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				Resources: &api.ResourceRequirements{
					Requests: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("100m"),
						api.ResourceMemory: resource.MustParse("50Gi"),
					},
					Limits: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("100m"),
						api.ResourceMemory: resource.MustParse("50Gi"),
					},
				},
				Containers: containers,
			},
		}
	}

	podWithoutPodLevelResources := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				Containers: containers,
			},
		}
	}

	podInfo := []struct {
		description          string
		hasPodLevelResources bool
		pod                  func() *api.Pod
	}{
		{
			description:          "has pod-level resources",
			hasPodLevelResources: true,
			pod:                  podWithPodLevelResources,
		},
		{
			description:          "does not have pod-level resources",
			hasPodLevelResources: false,
			pod:                  podWithoutPodLevelResources,
		},
		{
			description:          "is nil",
			hasPodLevelResources: false,
			pod:                  func() *api.Pod { return nil },
		},
		{
			description:          "is empty struct",
			hasPodLevelResources: false,
			// refactor to generalize and use podWithPodLevelResources()
			pod: func() *api.Pod {
				return &api.Pod{
					Spec: api.PodSpec{
						Resources:  &api.ResourceRequirements{},
						Containers: containers,
					},
				}
			},
		},
		{
			description:          "is empty Requests list",
			hasPodLevelResources: false,
			pod: func() *api.Pod {
				return &api.Pod{
					Spec: api.PodSpec{Resources: &api.ResourceRequirements{
						Requests: api.ResourceList{},
					}}}
			},
		},
		{
			description:          "is empty Limits list",
			hasPodLevelResources: false,
			pod: func() *api.Pod {
				return &api.Pod{
					Spec: api.PodSpec{Resources: &api.ResourceRequirements{
						Limits: api.ResourceList{},
					}}}
			},
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasPodLevelResources, oldPod := oldPodInfo.hasPodLevelResources, oldPodInfo.pod()
				newPodHasPodLevelResources, newPod := newPodInfo.hasPodLevelResources, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelResources, enabled)

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
					case enabled || oldPodHasPodLevelResources:
						// new pod shouldn't change if feature enabled or if old pod has
						// any pod level resources
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					case newPodHasPodLevelResources:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have any pod-level resources
						if !reflect.DeepEqual(newPod, podWithoutPodLevelResources()) {
							t.Errorf("new pod has pod-level resources: %v", cmp.Diff(newPod, podWithoutPodLevelResources()))
						}
					default:
						if newPod.Spec.Resources != nil {
							t.Errorf("expected nil, got: %v", newPod.Spec.Resources)
						}
					}
				})
			}
		}
	}
}

func TestDropSidecarContainers(t *testing.T) {
	containerRestartPolicyAlways := api.ContainerRestartPolicyAlways

	podWithSidecarContainers := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				InitContainers: []api.Container{
					{
						Name:          "c1",
						Image:         "image",
						RestartPolicy: &containerRestartPolicyAlways,
					},
				},
			},
		}
	}

	podWithoutSidecarContainers := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				InitContainers: []api.Container{
					{
						Name:  "c1",
						Image: "image",
					},
				},
			},
		}
	}

	podInfo := []struct {
		description         string
		hasSidecarContainer bool
		pod                 func() *api.Pod
	}{
		{
			description:         "has a sidecar container",
			hasSidecarContainer: true,
			pod:                 podWithSidecarContainers,
		},
		{
			description:         "does not have a sidecar container",
			hasSidecarContainer: false,
			pod:                 podWithoutSidecarContainers,
		},
		{
			description:         "is nil",
			hasSidecarContainer: false,
			pod:                 func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasSidecarContainer, oldPod := oldPodInfo.hasSidecarContainer, oldPodInfo.pod()
				newPodHasSidecarContainer, newPod := newPodInfo.hasSidecarContainer, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					if !enabled {
						// TODO: Remove this in v1.36
						featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.32"))
						featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SidecarContainers, false)
					}

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
					case enabled || oldPodHasSidecarContainer:
						// new pod shouldn't change if feature enabled or if old pod has
						// any sidecar container
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
						}
					case newPodHasSidecarContainer:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have any sidecar container
						if !reflect.DeepEqual(newPod, podWithoutSidecarContainers()) {
							t.Errorf("new pod has a sidecar container: %v", cmp.Diff(newPod, podWithoutSidecarContainers()))
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

func TestDropClusterTrustBundleProjectedVolumes(t *testing.T) {
	testCases := []struct {
		description                         string
		clusterTrustBundleProjectionEnabled bool
		oldPod                              *api.PodSpec
		newPod                              *api.PodSpec
		wantPod                             *api.PodSpec
	}{
		{
			description: "feature gate disabled, cannot add CTB volume to pod",
			oldPod: &api.PodSpec{
				Volumes: []api.Volume{},
			},
			newPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										ClusterTrustBundle: &api.ClusterTrustBundleProjection{
											Name: ptr.To("foo"),
										},
									},
								},
							}},
					},
				},
			},
			wantPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{},
								},
							}},
					},
				},
			},
		},
		{
			description: "feature gate disabled, can keep CTB volume on pod",
			oldPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										ClusterTrustBundle: &api.ClusterTrustBundleProjection{
											Name: ptr.To("foo"),
										},
									},
								},
							}},
					},
				},
			},
			newPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										ClusterTrustBundle: &api.ClusterTrustBundleProjection{
											Name: ptr.To("foo"),
										},
									},
								},
							}},
					},
				},
			},
			wantPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										ClusterTrustBundle: &api.ClusterTrustBundleProjection{
											Name: ptr.To("foo"),
										},
									},
								},
							}},
					},
				},
			},
		},
		{
			description:                         "feature gate enabled, can add CTB volume to pod",
			clusterTrustBundleProjectionEnabled: true,
			oldPod: &api.PodSpec{
				Volumes: []api.Volume{},
			},
			newPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										ClusterTrustBundle: &api.ClusterTrustBundleProjection{
											Name: ptr.To("foo"),
										},
									},
								},
							}},
					},
				},
			},
			wantPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										ClusterTrustBundle: &api.ClusterTrustBundleProjection{
											Name: ptr.To("foo"),
										},
									},
								},
							}},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ClusterTrustBundleProjection, tc.clusterTrustBundleProjectionEnabled)

			dropDisabledClusterTrustBundleProjection(tc.newPod, tc.oldPod)
			if diff := cmp.Diff(tc.newPod, tc.wantPod); diff != "" {
				t.Fatalf("Unexpected modification to new pod; diff (-got +want)\n%s", diff)
			}
		})
	}
}

func TestDropPodCertificateProjectedVolumes(t *testing.T) {
	testCases := []struct {
		description                     string
		podCertificateProjectionEnabled bool
		oldPod                          *api.PodSpec
		newPod                          *api.PodSpec
		wantPod                         *api.PodSpec
	}{
		{
			description: "feature gate disabled, cannot add volume to pod",
			oldPod: &api.PodSpec{
				Volumes: []api.Volume{},
			},
			newPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										PodCertificate: &api.PodCertificateProjection{
											SignerName: "foo.example.com/bar",
										},
									},
								},
							}},
					},
				},
			},
			wantPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{},
								},
							}},
					},
				},
			},
		},
		{
			description: "feature gate disabled, can keep volume on pod",
			oldPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										PodCertificate: &api.PodCertificateProjection{
											SignerName: "foo.example.com/bar",
										},
									},
								},
							}},
					},
				},
			},
			newPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										PodCertificate: &api.PodCertificateProjection{
											SignerName: "foo.example.com/bar",
										},
									},
								},
							}},
					},
				},
			},
			wantPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										PodCertificate: &api.PodCertificateProjection{
											SignerName: "foo.example.com/bar",
										},
									},
								},
							}},
					},
				},
			},
		},
		{
			description:                     "feature gate enabled, can add volume to pod",
			podCertificateProjectionEnabled: true,
			oldPod: &api.PodSpec{
				Volumes: []api.Volume{},
			},
			newPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										PodCertificate: &api.PodCertificateProjection{
											SignerName: "foo.example.com/bar",
										},
									},
								},
							}},
					},
				},
			},
			wantPod: &api.PodSpec{
				Volumes: []api.Volume{
					{
						Name: "foo",
						VolumeSource: api.VolumeSource{
							Projected: &api.ProjectedVolumeSource{
								Sources: []api.VolumeProjection{
									{
										PodCertificate: &api.PodCertificateProjection{
											SignerName: "foo.example.com/bar",
										},
									},
								},
							}},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodCertificateRequest, tc.podCertificateProjectionEnabled)

			dropDisabledPodCertificateProjection(tc.newPod, tc.oldPod)
			if diff := cmp.Diff(tc.newPod, tc.wantPod); diff != "" {
				t.Fatalf("Unexpected modification to new pod; diff (-got +want)\n%s", diff)
			}
		})
	}
}

func TestDropPodLifecycleSleepAction(t *testing.T) {
	makeSleepHandler := func() *api.LifecycleHandler {
		return &api.LifecycleHandler{
			Sleep: &api.SleepAction{Seconds: 1},
		}
	}

	makeExecHandler := func() *api.LifecycleHandler {
		return &api.LifecycleHandler{
			Exec: &api.ExecAction{Command: []string{"foo"}},
		}
	}

	makeHTTPGetHandler := func() *api.LifecycleHandler {
		return &api.LifecycleHandler{
			HTTPGet: &api.HTTPGetAction{Host: "foo"},
		}
	}

	makeContainer := func(preStop, postStart *api.LifecycleHandler) api.Container {
		container := api.Container{Name: "foo"}
		if preStop != nil || postStart != nil {
			container.Lifecycle = &api.Lifecycle{
				PostStart: postStart,
				PreStop:   preStop,
			}
		}
		return container
	}

	makeEphemeralContainer := func(preStop, postStart *api.LifecycleHandler) api.EphemeralContainer {
		container := api.EphemeralContainer{
			EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "foo"},
		}
		if preStop != nil || postStart != nil {
			container.Lifecycle = &api.Lifecycle{
				PostStart: postStart,
				PreStop:   preStop,
			}
		}
		return container
	}

	makePod := func(containers []api.Container, initContainers []api.Container, ephemeralContainers []api.EphemeralContainer) *api.PodSpec {
		return &api.PodSpec{
			Containers:          containers,
			InitContainers:      initContainers,
			EphemeralContainers: ephemeralContainers,
		}
	}

	testCases := []struct {
		gateEnabled            bool
		oldLifecycleHandler    *api.LifecycleHandler
		newLifecycleHandler    *api.LifecycleHandler
		expectLifecycleHandler *api.LifecycleHandler
	}{
		// nil -> nil
		{
			gateEnabled:            false,
			oldLifecycleHandler:    nil,
			newLifecycleHandler:    nil,
			expectLifecycleHandler: nil,
		},
		{
			gateEnabled:            true,
			oldLifecycleHandler:    nil,
			newLifecycleHandler:    nil,
			expectLifecycleHandler: nil,
		},
		// nil -> exec
		{
			gateEnabled:            false,
			oldLifecycleHandler:    nil,
			newLifecycleHandler:    makeExecHandler(),
			expectLifecycleHandler: makeExecHandler(),
		},
		{
			gateEnabled:            true,
			oldLifecycleHandler:    nil,
			newLifecycleHandler:    makeExecHandler(),
			expectLifecycleHandler: makeExecHandler(),
		},
		// nil -> sleep
		{
			gateEnabled:            false,
			oldLifecycleHandler:    nil,
			newLifecycleHandler:    makeSleepHandler(),
			expectLifecycleHandler: nil,
		},
		{
			gateEnabled:            true,
			oldLifecycleHandler:    nil,
			newLifecycleHandler:    makeSleepHandler(),
			expectLifecycleHandler: makeSleepHandler(),
		},
		// exec -> exec
		{
			gateEnabled:            false,
			oldLifecycleHandler:    makeExecHandler(),
			newLifecycleHandler:    makeExecHandler(),
			expectLifecycleHandler: makeExecHandler(),
		},
		{
			gateEnabled:            true,
			oldLifecycleHandler:    makeExecHandler(),
			newLifecycleHandler:    makeExecHandler(),
			expectLifecycleHandler: makeExecHandler(),
		},
		// exec -> http
		{
			gateEnabled:            false,
			oldLifecycleHandler:    makeExecHandler(),
			newLifecycleHandler:    makeHTTPGetHandler(),
			expectLifecycleHandler: makeHTTPGetHandler(),
		},
		{
			gateEnabled:            true,
			oldLifecycleHandler:    makeExecHandler(),
			newLifecycleHandler:    makeHTTPGetHandler(),
			expectLifecycleHandler: makeHTTPGetHandler(),
		},
		// exec -> sleep
		{
			gateEnabled:            false,
			oldLifecycleHandler:    makeExecHandler(),
			newLifecycleHandler:    makeSleepHandler(),
			expectLifecycleHandler: nil,
		},
		{
			gateEnabled:            true,
			oldLifecycleHandler:    makeExecHandler(),
			newLifecycleHandler:    makeSleepHandler(),
			expectLifecycleHandler: makeSleepHandler(),
		},
		// sleep -> exec
		{
			gateEnabled:            false,
			oldLifecycleHandler:    makeSleepHandler(),
			newLifecycleHandler:    makeExecHandler(),
			expectLifecycleHandler: makeExecHandler(),
		},
		{
			gateEnabled:            true,
			oldLifecycleHandler:    makeSleepHandler(),
			newLifecycleHandler:    makeExecHandler(),
			expectLifecycleHandler: makeExecHandler(),
		},
		// sleep -> sleep
		{
			gateEnabled:            false,
			oldLifecycleHandler:    makeSleepHandler(),
			newLifecycleHandler:    makeSleepHandler(),
			expectLifecycleHandler: makeSleepHandler(),
		},
		{
			gateEnabled:            true,
			oldLifecycleHandler:    makeSleepHandler(),
			newLifecycleHandler:    makeSleepHandler(),
			expectLifecycleHandler: makeSleepHandler(),
		},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprintf("test_%d", i), func(t *testing.T) {
			featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.33"))
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLifecycleSleepAction, tc.gateEnabled)

			// preStop
			// container
			{
				oldPod := makePod([]api.Container{makeContainer(tc.oldLifecycleHandler.DeepCopy(), nil)}, nil, nil)
				newPod := makePod([]api.Container{makeContainer(tc.newLifecycleHandler.DeepCopy(), nil)}, nil, nil)
				expectPod := makePod([]api.Container{makeContainer(tc.expectLifecycleHandler.DeepCopy(), nil)}, nil, nil)
				dropDisabledFields(newPod, nil, oldPod, nil)
				if diff := cmp.Diff(expectPod, newPod); diff != "" {
					t.Fatalf("Unexpected modification to new pod; diff (-got +want)\n%s", diff)
				}
			}
			// InitContainer
			{
				oldPod := makePod(nil, []api.Container{makeContainer(tc.oldLifecycleHandler.DeepCopy(), nil)}, nil)
				newPod := makePod(nil, []api.Container{makeContainer(tc.newLifecycleHandler.DeepCopy(), nil)}, nil)
				expectPod := makePod(nil, []api.Container{makeContainer(tc.expectLifecycleHandler.DeepCopy(), nil)}, nil)
				dropDisabledFields(newPod, nil, oldPod, nil)
				if diff := cmp.Diff(expectPod, newPod); diff != "" {
					t.Fatalf("Unexpected modification to new pod; diff (-got +want)\n%s", diff)
				}
			}
			// EphemeralContainer
			{
				oldPod := makePod(nil, nil, []api.EphemeralContainer{makeEphemeralContainer(tc.oldLifecycleHandler.DeepCopy(), nil)})
				newPod := makePod(nil, nil, []api.EphemeralContainer{makeEphemeralContainer(tc.newLifecycleHandler.DeepCopy(), nil)})
				expectPod := makePod(nil, nil, []api.EphemeralContainer{makeEphemeralContainer(tc.expectLifecycleHandler.DeepCopy(), nil)})
				dropDisabledFields(newPod, nil, oldPod, nil)
				if diff := cmp.Diff(expectPod, newPod); diff != "" {
					t.Fatalf("Unexpected modification to new pod; diff (-got +want)\n%s", diff)
				}
			}
			// postStart
			// container
			{
				oldPod := makePod([]api.Container{makeContainer(nil, tc.oldLifecycleHandler.DeepCopy())}, nil, nil)
				newPod := makePod([]api.Container{makeContainer(nil, tc.newLifecycleHandler.DeepCopy())}, nil, nil)
				expectPod := makePod([]api.Container{makeContainer(nil, tc.expectLifecycleHandler.DeepCopy())}, nil, nil)
				dropDisabledFields(newPod, nil, oldPod, nil)
				if diff := cmp.Diff(expectPod, newPod); diff != "" {
					t.Fatalf("Unexpected modification to new pod; diff (-got +want)\n%s", diff)
				}
			}
			// InitContainer
			{
				oldPod := makePod(nil, []api.Container{makeContainer(nil, tc.oldLifecycleHandler.DeepCopy())}, nil)
				newPod := makePod(nil, []api.Container{makeContainer(nil, tc.newLifecycleHandler.DeepCopy())}, nil)
				expectPod := makePod(nil, []api.Container{makeContainer(nil, tc.expectLifecycleHandler.DeepCopy())}, nil)
				dropDisabledFields(newPod, nil, oldPod, nil)
				if diff := cmp.Diff(expectPod, newPod); diff != "" {
					t.Fatalf("Unexpected modification to new pod; diff (-got +want)\n%s", diff)
				}
			}
			// EphemeralContainer
			{
				oldPod := makePod(nil, nil, []api.EphemeralContainer{makeEphemeralContainer(nil, tc.oldLifecycleHandler.DeepCopy())})
				newPod := makePod(nil, nil, []api.EphemeralContainer{makeEphemeralContainer(nil, tc.newLifecycleHandler.DeepCopy())})
				expectPod := makePod(nil, nil, []api.EphemeralContainer{makeEphemeralContainer(nil, tc.expectLifecycleHandler.DeepCopy())})
				dropDisabledFields(newPod, nil, oldPod, nil)
				if diff := cmp.Diff(expectPod, newPod); diff != "" {
					t.Fatalf("Unexpected modification to new pod; diff (-got +want)\n%s", diff)
				}
			}
		})
	}
}

func TestDropContainerStopSignals(t *testing.T) {
	makeContainer := func(lifecycle *api.Lifecycle) api.Container {
		container := api.Container{Name: "foo"}
		if lifecycle != nil {
			container.Lifecycle = lifecycle
		}
		return container
	}

	makeEphemeralContainer := func(lifecycle *api.Lifecycle) api.EphemeralContainer {
		container := api.EphemeralContainer{
			EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "foo"},
		}
		if lifecycle != nil {
			container.Lifecycle = lifecycle
		}
		return container
	}

	makePod := func(os api.OSName, containers []api.Container, initContainers []api.Container, ephemeralContainers []api.EphemeralContainer) *api.PodSpec {
		return &api.PodSpec{
			OS:                  &api.PodOS{Name: os},
			Containers:          containers,
			InitContainers:      initContainers,
			EphemeralContainers: ephemeralContainers,
		}
	}

	testCases := []struct {
		featuregateEnabled bool
		oldLifecycle       *api.Lifecycle
		newLifecycle       *api.Lifecycle
		expectedLifecycle  *api.Lifecycle
	}{
		// feature gate is turned on and stopsignal is not in use - Lifecycle stays nil
		{
			featuregateEnabled: true,
			oldLifecycle:       nil,
			newLifecycle:       nil,
			expectedLifecycle:  nil,
		},
		// feature gate is turned off and StopSignal is in use - StopSignal is not dropped
		{
			featuregateEnabled: false,
			oldLifecycle:       &api.Lifecycle{StopSignal: ptr.To(api.SIGTERM)},
			newLifecycle:       &api.Lifecycle{StopSignal: ptr.To(api.SIGTERM)},
			expectedLifecycle:  &api.Lifecycle{StopSignal: ptr.To(api.SIGTERM)},
		},
		// feature gate is turned off and StopSignal is not in use - Entire lifecycle is dropped
		{
			featuregateEnabled: false,
			oldLifecycle:       &api.Lifecycle{StopSignal: nil},
			newLifecycle:       &api.Lifecycle{StopSignal: ptr.To(api.SIGTERM)},
			expectedLifecycle:  nil,
		},
		// feature gate is turned on and StopSignal is in use - StopSignal is not dropped
		{
			featuregateEnabled: true,
			oldLifecycle:       &api.Lifecycle{StopSignal: nil},
			newLifecycle:       &api.Lifecycle{StopSignal: ptr.To(api.SIGTERM)},
			expectedLifecycle:  &api.Lifecycle{StopSignal: ptr.To(api.SIGTERM)},
		},
		// feature gate is turned off and PreStop is in use - StopSignal alone is dropped
		{
			featuregateEnabled: false,
			oldLifecycle: &api.Lifecycle{StopSignal: nil, PreStop: &api.LifecycleHandler{
				Exec: &api.ExecAction{Command: []string{"foo"}},
			}},
			newLifecycle: &api.Lifecycle{StopSignal: ptr.To(api.SIGTERM), PreStop: &api.LifecycleHandler{
				Exec: &api.ExecAction{Command: []string{"foo"}},
			}},
			expectedLifecycle: &api.Lifecycle{StopSignal: nil, PreStop: &api.LifecycleHandler{
				Exec: &api.ExecAction{Command: []string{"foo"}},
			}},
		},
		// feature gate is turned on and PreStop is in use - StopSignal is not dropped
		{
			featuregateEnabled: true,
			oldLifecycle: &api.Lifecycle{StopSignal: nil, PreStop: &api.LifecycleHandler{
				Exec: &api.ExecAction{Command: []string{"foo"}},
			}},
			newLifecycle: &api.Lifecycle{StopSignal: ptr.To(api.SIGTERM), PreStop: &api.LifecycleHandler{
				Exec: &api.ExecAction{Command: []string{"foo"}},
			}},
			expectedLifecycle: &api.Lifecycle{StopSignal: ptr.To(api.SIGTERM), PreStop: &api.LifecycleHandler{
				Exec: &api.ExecAction{Command: []string{"foo"}},
			}},
		},
		// feature gate is turned off and PreStop and StopSignal are in use - nothing is dropped
		{
			featuregateEnabled: true,
			oldLifecycle: &api.Lifecycle{StopSignal: ptr.To(api.SIGTERM), PreStop: &api.LifecycleHandler{
				Exec: &api.ExecAction{Command: []string{"foo"}},
			}},
			newLifecycle: &api.Lifecycle{StopSignal: ptr.To(api.SIGTERM), PreStop: &api.LifecycleHandler{
				Exec: &api.ExecAction{Command: []string{"foo"}},
			}},
			expectedLifecycle: &api.Lifecycle{StopSignal: ptr.To(api.SIGTERM), PreStop: &api.LifecycleHandler{
				Exec: &api.ExecAction{Command: []string{"foo"}},
			}},
		},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprintf("test_%d", i), func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ContainerStopSignals, tc.featuregateEnabled)
			// Containers
			{
				oldPod := makePod(api.Linux, []api.Container{makeContainer(tc.oldLifecycle.DeepCopy())}, nil, nil)
				newPod := makePod(api.Linux, []api.Container{makeContainer(tc.newLifecycle.DeepCopy())}, nil, nil)
				expectedPod := makePod(api.Linux, []api.Container{makeContainer(tc.expectedLifecycle.DeepCopy())}, nil, nil)
				dropDisabledFields(newPod, nil, oldPod, nil)

				if diff := cmp.Diff(expectedPod, newPod); diff != "" {
					t.Fatalf("Unexpected modification to new pod; diff (-got +want)\n%s", diff)
				}
			}
			// InitContainers
			{
				oldPod := makePod(api.Linux, nil, []api.Container{makeContainer(tc.oldLifecycle.DeepCopy())}, nil)
				newPod := makePod(api.Linux, nil, []api.Container{makeContainer(tc.newLifecycle.DeepCopy())}, nil)
				expectPod := makePod(api.Linux, nil, []api.Container{makeContainer(tc.expectedLifecycle.DeepCopy())}, nil)
				dropDisabledFields(newPod, nil, oldPod, nil)
				if diff := cmp.Diff(expectPod, newPod); diff != "" {
					t.Fatalf("Unexpected modification to new pod; diff (-got +want)\n%s", diff)
				}
			}
			// EphemeralContainers
			{
				oldPod := makePod(api.Linux, nil, nil, []api.EphemeralContainer{makeEphemeralContainer(tc.oldLifecycle.DeepCopy())})
				newPod := makePod(api.Linux, nil, nil, []api.EphemeralContainer{makeEphemeralContainer(tc.newLifecycle.DeepCopy())})
				expectPod := makePod(api.Linux, nil, nil, []api.EphemeralContainer{makeEphemeralContainer(tc.expectedLifecycle.DeepCopy())})
				dropDisabledFields(newPod, nil, oldPod, nil)
				if diff := cmp.Diff(expectPod, newPod); diff != "" {
					t.Fatalf("Unexpected modification to new pod; diff (-got +want)\n%s", diff)
				}
			}

		})
	}
}

func TestDropSupplementalGroupsPolicy(t *testing.T) {
	supplementalGroupsPolicyMerge := api.SupplementalGroupsPolicyMerge
	podWithSupplementalGroupsPolicy := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{
					SupplementalGroupsPolicy: &supplementalGroupsPolicyMerge,
				},
				Containers: []api.Container{
					{
						Name:  "c1",
						Image: "image",
					},
				},
			},
			Status: api.PodStatus{
				ContainerStatuses: []api.ContainerStatus{
					{
						Name:  "c1",
						Image: "image",
						User: &api.ContainerUser{
							Linux: &api.LinuxContainerUser{
								UID:                0,
								GID:                0,
								SupplementalGroups: []int64{0, 1000},
							},
						},
					},
				},
			},
		}
	}
	podWithoutSupplementalGroupsPolicy := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{},
				Containers: []api.Container{
					{
						Name:  "c1",
						Image: "image",
					},
				},
			},
			Status: api.PodStatus{
				ContainerStatuses: []api.ContainerStatus{
					{
						Name:  "c1",
						Image: "image",
					},
				},
			},
		}
	}

	podInfo := []struct {
		description                 string
		hasSupplementalGroupsPolicy bool
		pod                         func() *api.Pod
	}{
		{
			description:                 "with SupplementalGroupsPolicy and User",
			hasSupplementalGroupsPolicy: true,
			pod:                         podWithSupplementalGroupsPolicy,
		},
		{
			description:                 "without SupplementalGroupsPolicy and User",
			hasSupplementalGroupsPolicy: false,
			pod:                         podWithoutSupplementalGroupsPolicy,
		},
		{
			description:                 "is nil",
			hasSupplementalGroupsPolicy: false,
			pod:                         func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasSupplementalGroupsPolicy, oldPod := oldPodInfo.hasSupplementalGroupsPolicy, oldPodInfo.pod()
				newPodHasSupplementalGroupsPolicy, newPod := newPodInfo.hasSupplementalGroupsPolicy, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(
					fmt.Sprintf(
						"feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description,
					),
					func(t *testing.T) {
						featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SupplementalGroupsPolicy, enabled)

						var oldPodSpec *api.PodSpec
						var oldPodStatus *api.PodStatus
						if oldPod != nil {
							oldPodSpec = &oldPod.Spec
							oldPodStatus = &oldPod.Status
						}
						dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)
						dropDisabledPodStatusFields(&newPod.Status, oldPodStatus, &newPod.Spec, oldPodSpec)

						// old pod should never be changed
						if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
							t.Errorf("old pod changed: %v", cmp.Diff(oldPod, oldPodInfo.pod()))
						}
						switch {
						case enabled || oldPodHasSupplementalGroupsPolicy:
							// new pod shouldn't change if feature enabled or if old pod has SupplementalGroupsPolicy and User set
							if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
								t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
							}
						case newPodHasSupplementalGroupsPolicy:
							// new pod should be changed
							if reflect.DeepEqual(newPod, newPodInfo.pod()) {
								t.Errorf("new pod was not changed")
							}
							// new pod should not have SupplementalGroupsPolicy and User fields
							if !reflect.DeepEqual(newPod, podWithoutSupplementalGroupsPolicy()) {
								t.Errorf("new pod has SupplementalGroups and User: %v", cmp.Diff(newPod, podWithoutSupplementalGroupsPolicy()))
							}
						default:
							// new pod should not need to be changed
							if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
								t.Errorf("new pod changed: %v", cmp.Diff(newPod, newPodInfo.pod()))
							}
						}
					},
				)
			}
		}
	}
}

func TestDropImageVolumes(t *testing.T) {
	const (
		volumeNameImage = "volume"
		volumeNameOther = "volume-other"
	)
	anotherVolume := api.Volume{Name: volumeNameOther, VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{}}}
	podWithVolume := &api.Pod{
		Spec: api.PodSpec{
			Volumes: []api.Volume{
				{Name: volumeNameImage, VolumeSource: api.VolumeSource{Image: &api.ImageVolumeSource{}}},
				anotherVolume,
			},
			Containers: []api.Container{{
				VolumeMounts: []api.VolumeMount{{Name: volumeNameImage}, {Name: volumeNameOther}},
			}},
			InitContainers: []api.Container{{
				VolumeMounts: []api.VolumeMount{{Name: volumeNameImage}},
			}},
			EphemeralContainers: []api.EphemeralContainer{
				{EphemeralContainerCommon: api.EphemeralContainerCommon{
					VolumeMounts: []api.VolumeMount{{Name: volumeNameImage}},
				}},
			},
		},
	}

	podWithoutVolume := &api.Pod{
		Spec: api.PodSpec{
			Volumes:             []api.Volume{anotherVolume},
			Containers:          []api.Container{{VolumeMounts: []api.VolumeMount{{Name: volumeNameOther}}}},
			InitContainers:      []api.Container{{}},
			EphemeralContainers: []api.EphemeralContainer{{}},
		},
	}

	noPod := &api.Pod{}

	testcases := []struct {
		description string
		enabled     bool
		oldPod      *api.Pod
		newPod      *api.Pod
		wantPod     *api.Pod
	}{
		{
			description: "old with volume / new with volume / disabled",
			oldPod:      podWithVolume,
			newPod:      podWithVolume,
			wantPod:     podWithVolume,
		},
		{
			description: "old without volume / new with volume / disabled",
			oldPod:      podWithoutVolume,
			newPod:      podWithVolume,
			wantPod:     podWithoutVolume,
		},
		{
			description: "no old pod/ new with volume / disabled",
			oldPod:      noPod,
			newPod:      podWithVolume,
			wantPod:     podWithoutVolume,
		},
		{
			description: "nil old pod/ new with volume / disabled",
			oldPod:      nil,
			newPod:      podWithVolume,
			wantPod:     podWithoutVolume,
		},
		{
			description: "old with volume / new without volume / disabled",
			oldPod:      podWithVolume,
			newPod:      podWithoutVolume,
			wantPod:     podWithoutVolume,
		},
		{
			description: "old without volume / new without volume / disabled",
			oldPod:      podWithoutVolume,
			newPod:      podWithoutVolume,
			wantPod:     podWithoutVolume,
		},
		{
			description: "no old pod/ new without volume / disabled",
			oldPod:      noPod,
			newPod:      podWithoutVolume,
			wantPod:     podWithoutVolume,
		},

		{
			description: "old with volume / new with volume / enabled",
			enabled:     true,
			oldPod:      podWithVolume,
			newPod:      podWithVolume,
			wantPod:     podWithVolume,
		},
		{
			description: "old without volume / new with volume / enabled",
			enabled:     true,
			oldPod:      podWithoutVolume,
			newPod:      podWithVolume,
			wantPod:     podWithVolume,
		},
		{
			description: "no old pod/ new with volume / enabled",
			enabled:     true,
			oldPod:      noPod,
			newPod:      podWithVolume,
			wantPod:     podWithVolume,
		},

		{
			description: "old with volume / new without volume / enabled",
			enabled:     true,
			oldPod:      podWithVolume,
			newPod:      podWithoutVolume,
			wantPod:     podWithoutVolume,
		},
		{
			description: "old without volume / new without volume / enabled",
			enabled:     true,
			oldPod:      podWithoutVolume,
			newPod:      podWithoutVolume,
			wantPod:     podWithoutVolume,
		},
		{
			description: "no old pod/ new without volume / enabled",
			enabled:     true,
			oldPod:      noPod,
			newPod:      podWithoutVolume,
			wantPod:     podWithoutVolume,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ImageVolume, tc.enabled)

			oldPod := tc.oldPod.DeepCopy()
			newPod := tc.newPod.DeepCopy()
			wantPod := tc.wantPod
			DropDisabledPodFields(newPod, oldPod)

			// old pod should never be changed
			if diff := cmp.Diff(oldPod, tc.oldPod); diff != "" {
				t.Errorf("old pod changed: %s", diff)
			}

			if diff := cmp.Diff(wantPod, newPod); diff != "" {
				t.Errorf("new pod changed (- want, + got): %s", diff)
			}
		})
	}
}

func TestDropSELinuxChangePolicy(t *testing.T) {
	podRecursive := &api.Pod{
		Spec: api.PodSpec{
			SecurityContext: &api.PodSecurityContext{
				SELinuxChangePolicy: ptr.To(api.SELinuxChangePolicyRecursive),
			},
		},
	}
	podMountOption := &api.Pod{
		Spec: api.PodSpec{
			SecurityContext: &api.PodSecurityContext{
				SELinuxChangePolicy: ptr.To(api.SELinuxChangePolicyMountOption),
			},
		},
	}
	podNull := &api.Pod{
		Spec: api.PodSpec{
			SecurityContext: &api.PodSecurityContext{
				SELinuxChangePolicy: nil,
			},
		},
	}

	tests := []struct {
		name    string
		oldPod  *api.Pod
		newPod  *api.Pod
		gates   []featuregate.Feature
		wantPod *api.Pod
	}{
		{
			name:    "no old pod, new pod with Recursive, all features disabled",
			oldPod:  nil,
			newPod:  podRecursive,
			gates:   nil,
			wantPod: podNull,
		},
		{
			name:    "no old pod, new pod with MountOption, all features disabled",
			oldPod:  nil,
			newPod:  podMountOption,
			gates:   nil,
			wantPod: podNull,
		},
		{
			name:    "old pod with Recursive, new pod with Recursive, all features disabled",
			oldPod:  podRecursive,
			newPod:  podRecursive,
			gates:   nil,
			wantPod: podRecursive,
		},
		{
			name:    "old pod with MountOption, new pod with Recursive, all features disabled",
			oldPod:  podMountOption,
			newPod:  podRecursive,
			gates:   nil,
			wantPod: podRecursive,
		},
		{
			name:    "no old pod, new pod with Recursive, SELinuxChangePolicy feature enabled",
			oldPod:  nil,
			newPod:  podRecursive,
			gates:   []featuregate.Feature{features.SELinuxChangePolicy},
			wantPod: podRecursive,
		},
		{
			name:    "no old pod, new pod with MountOption, SELinuxChangePolicy feature enabled",
			oldPod:  nil,
			newPod:  podMountOption,
			gates:   []featuregate.Feature{features.SELinuxChangePolicy},
			wantPod: podMountOption,
		},
		{
			name:    "old pod with Recursive, new pod with Recursive, SELinuxChangePolicy feature enabled",
			oldPod:  podRecursive,
			newPod:  podRecursive,
			gates:   []featuregate.Feature{features.SELinuxChangePolicy},
			wantPod: podRecursive,
		},
		{
			name:    "old pod with MountOption, new pod with Recursive, SELinuxChangePolicy feature enabled",
			oldPod:  podMountOption,
			newPod:  podRecursive,
			gates:   []featuregate.Feature{features.SELinuxChangePolicy},
			wantPod: podRecursive,
		},
		// In theory, SELinuxMount does not have any effect on dropping SELinuxChangePolicy field, but for completeness:
		{
			name:    "no old pod, new pod with Recursive, SELinuxChangePolicy + SELinuxMount features enabled",
			oldPod:  nil,
			newPod:  podRecursive,
			gates:   []featuregate.Feature{features.SELinuxChangePolicy, features.SELinuxMount},
			wantPod: podRecursive,
		},
		{
			name:    "no old pod, new pod with MountOption, SELinuxChangePolicy + SELinuxMount features enabled",
			oldPod:  nil,
			newPod:  podMountOption,
			gates:   []featuregate.Feature{features.SELinuxChangePolicy, features.SELinuxMount},
			wantPod: podMountOption,
		},
		{
			name:    "old pod with Recursive, new pod with Recursive, SELinuxChangePolicy + SELinuxMount features enabled",
			oldPod:  podRecursive,
			newPod:  podRecursive,
			gates:   []featuregate.Feature{features.SELinuxChangePolicy, features.SELinuxMount},
			wantPod: podRecursive,
		},
		{
			name:    "old pod with MountOption, new pod with Recursive, SELinuxChangePolicy + SELinuxMount features enabled",
			oldPod:  podMountOption,
			newPod:  podRecursive,
			gates:   []featuregate.Feature{features.SELinuxChangePolicy, features.SELinuxMount},
			wantPod: podRecursive,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			// Set feature gates for the test. *Disable* those that are not in tc.gates.
			allGates := []featuregate.Feature{features.SELinuxChangePolicy, features.SELinuxMount}
			enabledGates := sets.New(tc.gates...)
			for _, gate := range allGates {
				enable := enabledGates.Has(gate)
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, gate, enable)
			}

			oldPod := tc.oldPod.DeepCopy()
			newPod := tc.newPod.DeepCopy()
			DropDisabledPodFields(newPod, oldPod)

			// old pod should never be changed
			if diff := cmp.Diff(oldPod, tc.oldPod); diff != "" {
				t.Errorf("old pod changed: %s", diff)
			}

			if diff := cmp.Diff(tc.wantPod, newPod); diff != "" {
				t.Errorf("new pod changed (- want, + got): %s", diff)
			}
		})
	}
}

func TestValidateAllowSidecarResizePolicy(t *testing.T) {
	restartPolicyAlways := api.ContainerRestartPolicyAlways
	testCases := []struct {
		name       string
		oldPodSpec *api.PodSpec
		wantOption bool
	}{
		{
			name:       "old pod spec is nil",
			wantOption: false,
		},
		{
			name: "one sidecar container + one regular init container, no resize policy set on any of them",
			oldPodSpec: &api.PodSpec{
				InitContainers: []api.Container{
					{
						Name:          "c1-restartable-init",
						Image:         "image",
						RestartPolicy: &restartPolicyAlways,
					},
					{
						Name:  "c1-init",
						Image: "image",
					},
				},
			},
			wantOption: false,
		},
		{
			name: "one sidecar container + one regular init container, resize policy set on regular init container",
			oldPodSpec: &api.PodSpec{
				InitContainers: []api.Container{
					{
						Name:          "c1-restartable-init",
						Image:         "image",
						RestartPolicy: &restartPolicyAlways,
					},
					{
						Name:  "c1-init",
						Image: "image",
						ResizePolicy: []api.ContainerResizePolicy{
							{ResourceName: api.ResourceCPU, RestartPolicy: api.NotRequired},
						},
					},
				},
			},
			wantOption: false,
		},
		{
			name: "one sidecar container + one regular init container, resize policy set on sidecar container",
			oldPodSpec: &api.PodSpec{
				InitContainers: []api.Container{
					{
						Name:          "c1-restartable-init",
						Image:         "image",
						RestartPolicy: &restartPolicyAlways,
						ResizePolicy: []api.ContainerResizePolicy{
							{ResourceName: api.ResourceCPU, RestartPolicy: api.NotRequired},
						},
					},
					{
						Name:  "c1-init",
						Image: "image",
					},
				},
			},
			wantOption: true,
		},
		{
			name: "one sidecar container + one regular init container, resize policy set on both of them",
			oldPodSpec: &api.PodSpec{
				InitContainers: []api.Container{
					{
						Name:          "c1-restartable-init",
						Image:         "image",
						RestartPolicy: &restartPolicyAlways,
						ResizePolicy: []api.ContainerResizePolicy{
							{ResourceName: api.ResourceCPU, RestartPolicy: api.NotRequired},
						},
					},
					{
						Name:  "c1-init",
						Image: "image",
						ResizePolicy: []api.ContainerResizePolicy{
							{ResourceName: api.ResourceCPU, RestartPolicy: api.NotRequired},
						},
					},
				},
			},
			wantOption: true,
		},
		{
			name: "two sidecar containers, resize policy set on one of them",
			oldPodSpec: &api.PodSpec{
				InitContainers: []api.Container{
					{
						Name:          "c1-restartable-init",
						Image:         "image",
						RestartPolicy: &restartPolicyAlways,
						ResizePolicy: []api.ContainerResizePolicy{
							{ResourceName: api.ResourceCPU, RestartPolicy: api.NotRequired},
						},
					},
					{
						Name:          "c2-restartable-init",
						Image:         "image",
						RestartPolicy: &restartPolicyAlways,
					},
				},
			},
			wantOption: true,
		},
		{
			name: "two regular init containers, resize policy set on both of them",
			oldPodSpec: &api.PodSpec{
				InitContainers: []api.Container{
					{
						Name:  "c1-init",
						Image: "image",
						ResizePolicy: []api.ContainerResizePolicy{
							{ResourceName: api.ResourceCPU, RestartPolicy: api.NotRequired},
						},
					},
					{
						Name:  "c2-init",
						Image: "image",
						ResizePolicy: []api.ContainerResizePolicy{
							{ResourceName: api.ResourceCPU, RestartPolicy: api.NotRequired},
						},
					},
				},
			},
			wantOption: false,
		},
		{
			name: "two non-init containers, resize policy set on both of them",
			oldPodSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name:  "c1",
						Image: "image",
						ResizePolicy: []api.ContainerResizePolicy{
							{ResourceName: api.ResourceCPU, RestartPolicy: api.NotRequired},
						},
					},
					{
						Name:  "c2",
						Image: "image",
						ResizePolicy: []api.ContainerResizePolicy{
							{ResourceName: api.ResourceCPU, RestartPolicy: api.NotRequired},
						},
					},
				},
			},
			wantOption: false,
		},
	}

	for _, tc := range testCases {
		for _, ippvsEnabled := range []bool{true, false} {
			t.Run(fmt.Sprintf("%s/%t", tc.name, ippvsEnabled), func(t *testing.T) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, ippvsEnabled)

				gotOptions := GetValidationOptionsFromPodSpecAndMeta(&api.PodSpec{}, tc.oldPodSpec, nil, nil)
				expected := tc.wantOption || ippvsEnabled
				assert.Equal(t, expected, gotOptions.AllowSidecarResizePolicy, "AllowSidecarResizePolicy")
			})
		}
	}
}

func TestValidateInvalidLabelValueInNodeSelectorOption(t *testing.T) {
	testCases := []struct {
		name       string
		oldPodSpec *api.PodSpec
		wantOption bool
	}{
		{
			name:       "Create",
			wantOption: false,
		},
		{
			name: "UpdateInvalidLabelSelector",
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					NodeAffinity: &api.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{{
								MatchExpressions: []api.NodeSelectorRequirement{{
									Key:      "foo",
									Operator: api.NodeSelectorOpIn,
									Values:   []string{"-1"},
								}},
							}},
						},
					},
				},
			},
			wantOption: true,
		},
		{
			name: "UpdateValidLabelSelector",
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					NodeAffinity: &api.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{{
								MatchExpressions: []api.NodeSelectorRequirement{{
									Key:      "foo",
									Operator: api.NodeSelectorOpIn,
									Values:   []string{"bar"},
								}},
							}},
						},
					},
				},
			},
			wantOption: false,
		},
		{
			name: "UpdateEmptyLabelSelector",
			oldPodSpec: &api.PodSpec{
				Affinity: &api.Affinity{
					NodeAffinity: &api.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &api.NodeSelector{
							NodeSelectorTerms: []api.NodeSelectorTerm{},
						},
					},
				},
			},
			wantOption: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Pod meta doesn't impact the outcome.
			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&api.PodSpec{}, tc.oldPodSpec, nil, nil)
			if tc.wantOption != gotOptions.AllowInvalidLabelValueInRequiredNodeAffinity {
				t.Errorf("Got AllowInvalidLabelValueInRequiredNodeAffinity=%t, want %t", gotOptions.AllowInvalidLabelValueInRequiredNodeAffinity, tc.wantOption)
			}
		})
	}
}

func TestValidateAllowPodLifecycleSleepActionZeroValue(t *testing.T) {
	testCases := []struct {
		name                                        string
		podSpec                                     *api.PodSpec
		featureEnabled                              bool
		expectAllowPodLifecycleSleepActionZeroValue bool
	}{
		{
			name:           "no lifecycle hooks",
			podSpec:        &api.PodSpec{},
			featureEnabled: true,
			expectAllowPodLifecycleSleepActionZeroValue: true,
		},
		{
			name: "Prestop with non-zero second duration",
			podSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Lifecycle: &api.Lifecycle{
							PreStop: &api.LifecycleHandler{
								Sleep: &api.SleepAction{
									Seconds: 1,
								},
							},
						},
					},
				},
			},
			featureEnabled: true,
			expectAllowPodLifecycleSleepActionZeroValue: true,
		},
		{
			name: "PostStart with non-zero second duration",
			podSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Lifecycle: &api.Lifecycle{
							PostStart: &api.LifecycleHandler{
								Sleep: &api.SleepAction{
									Seconds: 1,
								},
							},
						},
					},
				},
			},
			featureEnabled: true,
			expectAllowPodLifecycleSleepActionZeroValue: true,
		},
		{
			name: "PreStop with zero seconds",
			podSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Lifecycle: &api.Lifecycle{
							PreStop: &api.LifecycleHandler{
								Sleep: &api.SleepAction{
									Seconds: 0,
								},
							},
						},
					},
				},
			},
			featureEnabled: true,
			expectAllowPodLifecycleSleepActionZeroValue: true,
		},
		{
			name: "PostStart with zero seconds",
			podSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Lifecycle: &api.Lifecycle{
							PostStart: &api.LifecycleHandler{
								Sleep: &api.SleepAction{
									Seconds: 0,
								},
							},
						},
					},
				},
			},
			featureEnabled: true,
			expectAllowPodLifecycleSleepActionZeroValue: true,
		},
		{
			name:           "no lifecycle hooks with feature gate disabled",
			podSpec:        &api.PodSpec{},
			featureEnabled: false,
			expectAllowPodLifecycleSleepActionZeroValue: false,
		},
		{
			name: "Prestop with non-zero second duration with feature gate disabled",
			podSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Lifecycle: &api.Lifecycle{
							PreStop: &api.LifecycleHandler{
								Sleep: &api.SleepAction{
									Seconds: 1,
								},
							},
						},
					},
				},
			},
			featureEnabled: false,
			expectAllowPodLifecycleSleepActionZeroValue: false,
		},
		{
			name: "PostStart with non-zero second duration with feature gate disabled",
			podSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Lifecycle: &api.Lifecycle{
							PostStart: &api.LifecycleHandler{
								Sleep: &api.SleepAction{
									Seconds: 1,
								},
							},
						},
					},
				},
			},
			featureEnabled: false,
			expectAllowPodLifecycleSleepActionZeroValue: false,
		},
		{
			name: "PreStop with zero seconds with feature gate disabled",
			podSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Lifecycle: &api.Lifecycle{
							PreStop: &api.LifecycleHandler{
								Sleep: &api.SleepAction{
									Seconds: 0,
								},
							},
						},
					},
				},
			},
			featureEnabled: false,
			expectAllowPodLifecycleSleepActionZeroValue: true,
		},
		{
			name: "PostStart with zero seconds with feature gate disabled",
			podSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Lifecycle: &api.Lifecycle{
							PostStart: &api.LifecycleHandler{
								Sleep: &api.SleepAction{
									Seconds: 0,
								},
							},
						},
					},
				},
			},
			featureEnabled: false,
			expectAllowPodLifecycleSleepActionZeroValue: true,
		},
	}
	featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.33"))

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLifecycleSleepActionAllowZero, tc.featureEnabled)

			gotOptions := GetValidationOptionsFromPodSpecAndMeta(&api.PodSpec{}, tc.podSpec, nil, nil)
			assert.Equal(t, tc.expectAllowPodLifecycleSleepActionZeroValue, gotOptions.AllowPodLifecycleSleepActionZeroValue, "AllowPodLifecycleSleepActionZeroValue")
		})
	}
}

func TestHasAPIReferences(t *testing.T) {
	tests := []struct {
		name            string
		pod             *api.Pod
		expectRejection bool
		resource        string
	}{
		{
			name:            "Empty ServiceAccount in Static Pod",
			pod:             &api.Pod{Spec: api.PodSpec{}},
			expectRejection: false,
		},
		{
			name:            "Non empty ServiceAccount",
			pod:             &api.Pod{Spec: api.PodSpec{ServiceAccountName: "default"}},
			expectRejection: true,
			resource:        "serviceaccounts",
		},
		{
			name:            "Empty Volume list",
			pod:             &api.Pod{Spec: api.PodSpec{Volumes: nil}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with HostPath volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-hostpath", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with EmptyDir volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-emptydir", VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with Secret volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-secret", VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{}}},
			}}},
			expectRejection: true,
			resource:        "secrets (via secret volumes)",
		},
		{
			name: "Non empty volume list with ConfigMap volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-configmap", VolumeSource: api.VolumeSource{ConfigMap: &api.ConfigMapVolumeSource{}}},
			}}},
			expectRejection: true,
			resource:        "configmaps (via configmap volumes)",
		},
		{
			name: "Non empty volume list with GCEPersistentDisk volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-gce", VolumeSource: api.VolumeSource{GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with AWSElasticBlockStore volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-aws", VolumeSource: api.VolumeSource{AWSElasticBlockStore: &api.AWSElasticBlockStoreVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with GitRepo volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-gitrepo", VolumeSource: api.VolumeSource{GitRepo: &api.GitRepoVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with NFS volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-nfs", VolumeSource: api.VolumeSource{NFS: &api.NFSVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with ISCSI volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-iscsi", VolumeSource: api.VolumeSource{ISCSI: &api.ISCSIVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with Glusterfs volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-glusterfs", VolumeSource: api.VolumeSource{Glusterfs: &api.GlusterfsVolumeSource{}}},
			}}},
			expectRejection: true,
			resource:        "endpoints (via glusterFS volumes)",
		},
		{
			name: "Non empty volume list with PersistentVolumeClaim",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-pvc", VolumeSource: api.VolumeSource{PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{}}},
			}}},
			expectRejection: true,
			resource:        "persistentvolumeclaims",
		},
		{
			name: "Non empty volume list with RBD volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-rbd", VolumeSource: api.VolumeSource{RBD: &api.RBDVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with FlexVolume volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-flexvolume", VolumeSource: api.VolumeSource{FlexVolume: &api.FlexVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with Cinder volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-cinder", VolumeSource: api.VolumeSource{Cinder: &api.CinderVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with CephFS volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-cephfs", VolumeSource: api.VolumeSource{CephFS: &api.CephFSVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with Flocker volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-flocker", VolumeSource: api.VolumeSource{Flocker: &api.FlockerVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with DownwardAPI volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-downwardapi", VolumeSource: api.VolumeSource{DownwardAPI: &api.DownwardAPIVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with FC volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-fc", VolumeSource: api.VolumeSource{FC: &api.FCVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with AzureFile volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-azurefile", VolumeSource: api.VolumeSource{AzureFile: &api.AzureFileVolumeSource{}}},
			}}},
			expectRejection: true,
			resource:        "secrets (via azureFile volumes)",
		},
		{
			name: "Non empty volume list with VsphereVolume volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-vsphere", VolumeSource: api.VolumeSource{VsphereVolume: &api.VsphereVirtualDiskVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with Quobyte volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-quobyte", VolumeSource: api.VolumeSource{Quobyte: &api.QuobyteVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with AzureDisk volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-azuredisk", VolumeSource: api.VolumeSource{AzureDisk: &api.AzureDiskVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with PhotonPersistentDisk volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-photon", VolumeSource: api.VolumeSource{PhotonPersistentDisk: &api.PhotonPersistentDiskVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with Projected volume with clustertrustbundles",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-projected", VolumeSource: api.VolumeSource{Projected: &api.ProjectedVolumeSource{Sources: []api.VolumeProjection{{ClusterTrustBundle: &api.ClusterTrustBundleProjection{}}}}}},
			}}},
			expectRejection: true,
			resource:        "clustertrustbundles",
		},
		{
			name: "Non empty volume list with Projected volume with podcertificates",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-projected", VolumeSource: api.VolumeSource{Projected: &api.ProjectedVolumeSource{Sources: []api.VolumeProjection{{PodCertificate: &api.PodCertificateProjection{}}}}}},
			}}},
			expectRejection: true,
			resource:        "podcertificates",
		},
		{
			name: "Non empty volume list with Projected volume with secrets",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-projected", VolumeSource: api.VolumeSource{Projected: &api.ProjectedVolumeSource{Sources: []api.VolumeProjection{{Secret: &api.SecretProjection{}}}}}},
			}}},
			expectRejection: true,
			resource:        "secrets (via projected volumes)",
		},
		{
			name: "Non empty volume list with Projected volume with configmap",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-projected", VolumeSource: api.VolumeSource{Projected: &api.ProjectedVolumeSource{Sources: []api.VolumeProjection{{ConfigMap: &api.ConfigMapProjection{}}}}}},
			}}},
			expectRejection: true,
			resource:        "configmaps (via projected volumes)",
		},
		{
			name: "Non empty volume list with Projected volume with serviceaccounttoken",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-projected", VolumeSource: api.VolumeSource{Projected: &api.ProjectedVolumeSource{Sources: []api.VolumeProjection{{ServiceAccountToken: &api.ServiceAccountTokenProjection{}}}}}},
			}}},
			expectRejection: true,
			resource:        "serviceaccounts (via projected volumes)",
		},
		{
			name: "Non empty volume list with Projected volume with downwardapi",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-projected", VolumeSource: api.VolumeSource{Projected: &api.ProjectedVolumeSource{Sources: []api.VolumeProjection{{DownwardAPI: &api.DownwardAPIProjection{}}}}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with Portworx volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-portworx", VolumeSource: api.VolumeSource{PortworxVolume: &api.PortworxVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with ScaleIO volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-scaleio", VolumeSource: api.VolumeSource{ScaleIO: &api.ScaleIOVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with StorageOS volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-storageos", VolumeSource: api.VolumeSource{StorageOS: &api.StorageOSVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name: "Non empty volume list with CSI volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-csi", VolumeSource: api.VolumeSource{CSI: &api.CSIVolumeSource{}}},
			}}},
			expectRejection: true,
			resource:        "csidrivers (via CSI volumes)",
		},
		{
			name: "Non empty volume list with Ephemeral volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-ephemeral", VolumeSource: api.VolumeSource{Ephemeral: &api.EphemeralVolumeSource{}}},
			}}},
			expectRejection: true,
			resource:        "persistentvolumeclaims (via ephemeral volumes)",
		},
		{
			name: "Non empty volume list with Image volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-image", VolumeSource: api.VolumeSource{Image: &api.ImageVolumeSource{}}},
			}}},
			expectRejection: false,
		},
		{
			name:            "No envs",
			pod:             &api.Pod{Spec: api.PodSpec{}},
			expectRejection: false,
		},
		{
			name: "Non empty Env with value",
			pod: &api.Pod{Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env: []api.EnvVar{
							{
								Name:  "test-env",
								Value: "TEST_ENV_VAL",
							},
						},
					},
				},
			}},
			expectRejection: false,
		},
		{
			name: "Non empty EnvFrom with ConfigMap",
			pod: &api.Pod{Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						EnvFrom: []api.EnvFromSource{
							{ConfigMapRef: &api.ConfigMapEnvSource{LocalObjectReference: api.LocalObjectReference{Name: "test"}}},
						},
					},
				},
			}},
			expectRejection: true,
			resource:        "configmaps",
		},
		{
			name: "Non empty EnvFrom with Secret",
			pod: &api.Pod{Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						EnvFrom: []api.EnvFromSource{
							{SecretRef: &api.SecretEnvSource{LocalObjectReference: api.LocalObjectReference{Name: "test"}}},
						},
					},
				},
			}},
			expectRejection: true,
			resource:        "secrets",
		},
		{
			name: "Non empty Env with ConfigMap",
			pod: &api.Pod{Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env: []api.EnvVar{
							{
								Name:      "test-env",
								ValueFrom: &api.EnvVarSource{ConfigMapKeyRef: &api.ConfigMapKeySelector{LocalObjectReference: api.LocalObjectReference{Name: "test"}}},
							},
						},
					},
				},
			}},
			expectRejection: true,
			resource:        "configmaps",
		},
		{
			name: "Non empty Env with Secret",
			pod: &api.Pod{Spec: api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env: []api.EnvVar{
							{
								Name:      "test-env",
								ValueFrom: &api.EnvVarSource{SecretKeyRef: &api.SecretKeySelector{LocalObjectReference: api.LocalObjectReference{Name: "test"}}},
							},
						},
					},
				},
			}},
			expectRejection: true,
			resource:        "secrets",
		},
		{
			name: "Multiple volume list where invalid volume comes after valid volume source",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-portworx", VolumeSource: api.VolumeSource{PortworxVolume: &api.PortworxVolumeSource{}}},
				{Name: "test-volume-configmap", VolumeSource: api.VolumeSource{ConfigMap: &api.ConfigMapVolumeSource{}}},
			}}},
			expectRejection: true,
			resource:        "configmaps (via configmap volumes)",
		},
		{
			name: "Multiple volume list where invalid configmap volume comes after valid downwardapi projected volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-projected", VolumeSource: api.VolumeSource{Projected: &api.ProjectedVolumeSource{Sources: []api.VolumeProjection{{DownwardAPI: &api.DownwardAPIProjection{}}}}}},
				{Name: "test-volume-configmap", VolumeSource: api.VolumeSource{ConfigMap: &api.ConfigMapVolumeSource{}}},
			}}},
			expectRejection: true,
			resource:        "configmaps (via configmap volumes)",
		},
		{
			name: "Multiple volume list where invalid configmap projected volume comes after valid downwardapi projected volume",
			pod: &api.Pod{Spec: api.PodSpec{Volumes: []api.Volume{
				{Name: "test-volume-projected", VolumeSource: api.VolumeSource{Projected: &api.ProjectedVolumeSource{Sources: []api.VolumeProjection{{DownwardAPI: &api.DownwardAPIProjection{}}}}}},
				{Name: "test-volume-projected", VolumeSource: api.VolumeSource{Projected: &api.ProjectedVolumeSource{Sources: []api.VolumeProjection{{ConfigMap: &api.ConfigMapProjection{}}}}}},
			}}},
			expectRejection: true,
			resource:        "configmaps (via projected volumes)",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualResult, resource, _ := HasAPIObjectReference(test.pod)
			if test.expectRejection != actualResult || resource != test.resource {
				t.Errorf("unexpected result, expected %v but got %v, expected resource %v, but got %v", test.expectRejection, actualResult, test.resource, resource)
			}
		})
	}
}

func TestDropFileKeyRefInUse(t *testing.T) {
	testCases := []struct {
		name           string
		featureEnabled bool
		oldPodSpec     *api.PodSpec
		newPodSpec     *api.PodSpec
		expectedSpec   *api.PodSpec
	}{
		{
			name:           "feature enabled - should not drop FileKeyRef",
			featureEnabled: true,
			oldPodSpec:     nil,
			newPodSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env: []api.EnvVar{
							{
								Name: "TEST_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "test-volume",
										Path:       "/path/to/file",
										Key:        "test-key",
									},
								},
							},
						},
					},
				},
			},
			expectedSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env: []api.EnvVar{
							{
								Name: "TEST_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "test-volume",
										Path:       "/path/to/file",
										Key:        "test-key",
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name:           "feature disabled - old pod has FileKeyRef - should not drop",
			featureEnabled: false,
			oldPodSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "old-container",
						Env: []api.EnvVar{
							{
								Name: "OLD_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "old-volume",
										Path:       "/old/path",
										Key:        "old-key",
									},
								},
							},
						},
					},
				},
			},
			newPodSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "new-container",
						Env: []api.EnvVar{
							{
								Name: "NEW_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "new-volume",
										Path:       "/new/path",
										Key:        "new-key",
									},
								},
							},
						},
					},
				},
			},
			expectedSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "new-container",
						Env: []api.EnvVar{
							{
								Name: "NEW_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "new-volume",
										Path:       "/new/path",
										Key:        "new-key",
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name:           "feature disabled - old pod has no FileKeyRef - should drop",
			featureEnabled: false,
			oldPodSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "old-container",
						Env: []api.EnvVar{
							{
								Name:  "OLD_ENV",
								Value: "old-value",
							},
						},
					},
				},
			},
			newPodSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "new-container",
						Env: []api.EnvVar{
							{
								Name: "NEW_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "new-volume",
										Path:       "/new/path",
										Key:        "new-key",
									},
								},
							},
							{
								Name:  "REGULAR_ENV",
								Value: "regular-value",
							},
						},
					},
				},
			},
			expectedSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "new-container",
						Env: []api.EnvVar{
							{
								Name: "NEW_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: nil,
								},
							},
							{
								Name:  "REGULAR_ENV",
								Value: "regular-value",
							},
						},
					},
				},
			},
		},
		{
			name:           "feature disabled - old pod is nil - should drop",
			featureEnabled: false,
			oldPodSpec:     nil,
			newPodSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env: []api.EnvVar{
							{
								Name: "TEST_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "test-volume",
										Path:       "/path/to/file",
										Key:        "test-key",
									},
								},
							},
						},
					},
				},
			},
			expectedSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env: []api.EnvVar{
							{
								Name: "TEST_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: nil,
								},
							},
						},
					},
				},
			},
		},
		{
			name:           "feature disabled - multiple containers with FileKeyRef - should drop all",
			featureEnabled: false,
			oldPodSpec:     nil,
			newPodSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "container1",
						Env: []api.EnvVar{
							{
								Name: "ENV1",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "volume1",
										Path:       "/path1",
										Key:        "key1",
									},
								},
							},
						},
					},
					{
						Name: "container2",
						Env: []api.EnvVar{
							{
								Name: "ENV2",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "volume2",
										Path:       "/path2",
										Key:        "key2",
									},
								},
							},
						},
					},
				},
				InitContainers: []api.Container{
					{
						Name: "init-container",
						Env: []api.EnvVar{
							{
								Name: "INIT_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "init-volume",
										Path:       "/init/path",
										Key:        "init-key",
									},
								},
							},
						},
					},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{
						EphemeralContainerCommon: api.EphemeralContainerCommon{
							Name: "ephemeral-container",
							Env: []api.EnvVar{
								{
									Name: "EPHEMERAL_ENV",
									ValueFrom: &api.EnvVarSource{
										FileKeyRef: &api.FileKeySelector{
											VolumeName: "ephemeral-volume",
											Path:       "/ephemeral/path",
											Key:        "ephemeral-key",
										},
									},
								},
							},
						},
					},
				},
			},
			expectedSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "container1",
						Env: []api.EnvVar{
							{
								Name: "ENV1",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: nil,
								},
							},
						},
					},
					{
						Name: "container2",
						Env: []api.EnvVar{
							{
								Name: "ENV2",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: nil,
								},
							},
						},
					},
				},
				InitContainers: []api.Container{
					{
						Name: "init-container",
						Env: []api.EnvVar{
							{
								Name: "INIT_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: nil,
								},
							},
						},
					},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{
						EphemeralContainerCommon: api.EphemeralContainerCommon{
							Name: "ephemeral-container",
							Env: []api.EnvVar{
								{
									Name: "EPHEMERAL_ENV",
									ValueFrom: &api.EnvVarSource{
										FileKeyRef: nil,
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name:           "feature disabled - mixed env vars - should only drop FileKeyRef",
			featureEnabled: false,
			oldPodSpec:     nil,
			newPodSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env: []api.EnvVar{
							{
								Name:  "REGULAR_ENV",
								Value: "regular-value",
							},
							{
								Name: "FILE_KEY_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "test-volume",
										Path:       "/path/to/file",
										Key:        "test-key",
									},
								},
							},
							{
								Name: "SECRET_ENV",
								ValueFrom: &api.EnvVarSource{
									SecretKeyRef: &api.SecretKeySelector{
										LocalObjectReference: api.LocalObjectReference{
											Name: "test-secret",
										},
										Key: "secret-key",
									},
								},
							},
						},
					},
				},
			},
			expectedSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env: []api.EnvVar{
							{
								Name:  "REGULAR_ENV",
								Value: "regular-value",
							},
							{
								Name: "FILE_KEY_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: nil,
								},
							},
							{
								Name: "SECRET_ENV",
								ValueFrom: &api.EnvVarSource{
									SecretKeyRef: &api.SecretKeySelector{
										LocalObjectReference: api.LocalObjectReference{
											Name: "test-secret",
										},
										Key: "secret-key",
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name:           "feature disabled - container with nil Env - should not panic",
			featureEnabled: false,
			oldPodSpec:     nil,
			newPodSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env:  nil,
					},
				},
			},
			expectedSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env:  nil,
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EnvFiles, tc.featureEnabled)
			newPodSpecCopy := tc.newPodSpec.DeepCopy()
			dropFileKeyRefInUse(newPodSpecCopy, tc.oldPodSpec)

			if diff := cmp.Diff(tc.expectedSpec, newPodSpecCopy); diff != "" {
				t.Errorf("new pod changed (- want, + got): %s", diff)
			}

			if tc.oldPodSpec != nil {
				oldPodSpecCopy := tc.oldPodSpec.DeepCopy()
				// old pod should never be changed
				if diff := cmp.Diff(tc.oldPodSpec, oldPodSpecCopy); diff != "" {
					t.Errorf("old pod changed: %s", diff)
				}
			}
		})
	}
}

// TestPodFileKeyRefInUse tests the podFileKeyRefInUse function
func TestPodFileKeyRefInUse(t *testing.T) {
	testCases := []struct {
		name     string
		podSpec  *api.PodSpec
		expected bool
	}{
		{
			name:     "nil pod spec",
			podSpec:  nil,
			expected: false,
		},
		{
			name: "pod spec with FileKeyRef in container",
			podSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env: []api.EnvVar{
							{
								Name: "TEST_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "test-volume",
										Path:       "/path/to/file",
										Key:        "test-key",
									},
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "pod spec with FileKeyRef in init container",
			podSpec: &api.PodSpec{
				InitContainers: []api.Container{
					{
						Name: "init-container",
						Env: []api.EnvVar{
							{
								Name: "INIT_ENV",
								ValueFrom: &api.EnvVarSource{
									FileKeyRef: &api.FileKeySelector{
										VolumeName: "init-volume",
										Path:       "/init/path",
										Key:        "init-key",
									},
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "pod spec with FileKeyRef in ephemeral container",
			podSpec: &api.PodSpec{
				EphemeralContainers: []api.EphemeralContainer{
					{
						EphemeralContainerCommon: api.EphemeralContainerCommon{
							Name: "ephemeral-container",
							Env: []api.EnvVar{
								{
									Name: "EPHEMERAL_ENV",
									ValueFrom: &api.EnvVarSource{
										FileKeyRef: &api.FileKeySelector{
											VolumeName: "ephemeral-volume",
											Path:       "/ephemeral/path",
											Key:        "ephemeral-key",
										},
									},
								},
							},
						},
					},
				},
			},
			expected: true,
		},
		{
			name: "pod spec without FileKeyRef",
			podSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env: []api.EnvVar{
							{
								Name:  "REGULAR_ENV",
								Value: "regular-value",
							},
							{
								Name: "SECRET_ENV",
								ValueFrom: &api.EnvVarSource{
									SecretKeyRef: &api.SecretKeySelector{
										LocalObjectReference: api.LocalObjectReference{
											Name: "test-secret",
										},
										Key: "secret-key",
									},
								},
							},
						},
					},
				},
			},
			expected: false,
		},
		{
			name: "pod spec with nil Env in container",
			podSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env:  nil,
					},
				},
			},
			expected: false,
		},
		{
			name: "pod spec with env var without ValueFrom",
			podSpec: &api.PodSpec{
				Containers: []api.Container{
					{
						Name: "test-container",
						Env: []api.EnvVar{
							{
								Name:  "REGULAR_ENV",
								Value: "regular-value",
							},
						},
					},
				},
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := podFileKeyRefInUse(tc.podSpec)
			if result != tc.expected {
				t.Errorf("expected %v, got %v", tc.expected, result)
			}
		})
	}
}
