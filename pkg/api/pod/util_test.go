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

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/security/apparmor"
)

func TestVisitContainers(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.EphemeralContainers, true)()

	testCases := []struct {
		description string
		haveSpec    *api.PodSpec
		wantNames   []string
	}{
		{
			"empty podspec",
			&api.PodSpec{},
			[]string{},
		},
		{
			"regular containers",
			&api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
			},
			[]string{"c1", "c2"},
		},
		{
			"init containers",
			&api.PodSpec{
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2"},
				},
			},
			[]string{"i1", "i2"},
		},
		{
			"regular and init containers",
			&api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2"},
				},
			},
			[]string{"i1", "i2", "c1", "c2"},
		},
		{
			"ephemeral containers",
			&api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e1"}},
				},
			},
			[]string{"c1", "c2", "e1"},
		},
		{
			"all container types",
			&api.PodSpec{
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
			[]string{"i1", "i2", "c1", "c2", "e1", "e2"},
		},
		{
			"dropping fields",
			&api.PodSpec{
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
			[]string{"i1", "i2", "c1", "c2", "e1", "e2"},
		},
	}

	for _, tc := range testCases {
		gotNames := []string{}
		VisitContainers(tc.haveSpec, func(c *api.Container) bool {
			gotNames = append(gotNames, c.Name)
			if c.SecurityContext != nil {
				c.SecurityContext = nil
			}
			return true
		})
		if !reflect.DeepEqual(gotNames, tc.wantNames) {
			t.Errorf("VisitContainers() for test case %q visited containers %q, wanted to visit %q", tc.description, gotNames, tc.wantNames)
		}
		for _, c := range tc.haveSpec.Containers {
			if c.SecurityContext != nil {
				t.Errorf("VisitContainers() for test case %q: got SecurityContext %#v for container %v, wanted nil", tc.description, c.SecurityContext, c.Name)
			}
		}
		for _, c := range tc.haveSpec.InitContainers {
			if c.SecurityContext != nil {
				t.Errorf("VisitContainers() for test case %q: got SecurityContext %#v for init container %v, wanted nil", tc.description, c.SecurityContext, c.Name)
			}
		}
		for _, c := range tc.haveSpec.EphemeralContainers {
			if c.SecurityContext != nil {
				t.Errorf("VisitContainers() for test case %q: got SecurityContext %#v for ephemeral container %v, wanted nil", tc.description, c.SecurityContext, c.Name)
			}
		}
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
	})

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
}

// collectResourcePaths traverses the object, computing all the struct paths that lead to fields with resourcename in the name.
func collectResourcePaths(t *testing.T, resourcename string, path *field.Path, name string, tp reflect.Type) sets.String {
	resourcename = strings.ToLower(resourcename)
	resourcePaths := sets.NewString()

	if tp.Kind() == reflect.Ptr {
		resourcePaths.Insert(collectResourcePaths(t, resourcename, path, name, tp.Elem()).List()...)
		return resourcePaths
	}

	if strings.Contains(strings.ToLower(name), resourcename) {
		resourcePaths.Insert(path.String())
	}

	switch tp.Kind() {
	case reflect.Ptr:
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
	})

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
}

func TestDropAlphaVolumeDevices(t *testing.T) {
	podWithVolumeDevices := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyNever,
				Containers: []api.Container{
					{
						Name:  "container1",
						Image: "testimage",
						VolumeDevices: []api.VolumeDevice{
							{
								Name:       "myvolume",
								DevicePath: "/usr/test",
							},
						},
					},
				},
				InitContainers: []api.Container{
					{
						Name:  "container1",
						Image: "testimage",
						VolumeDevices: []api.VolumeDevice{
							{
								Name:       "myvolume",
								DevicePath: "/usr/test",
							},
						},
					},
				},
				Volumes: []api.Volume{
					{
						Name: "myvolume",
						VolumeSource: api.VolumeSource{
							HostPath: &api.HostPathVolumeSource{
								Path: "/dev/xvdc",
							},
						},
					},
				},
			},
		}
	}
	podWithoutVolumeDevices := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy: api.RestartPolicyNever,
				Containers: []api.Container{
					{
						Name:  "container1",
						Image: "testimage",
					},
				},
				InitContainers: []api.Container{
					{
						Name:  "container1",
						Image: "testimage",
					},
				},
				Volumes: []api.Volume{
					{
						Name: "myvolume",
						VolumeSource: api.VolumeSource{
							HostPath: &api.HostPathVolumeSource{
								Path: "/dev/xvdc",
							},
						},
					},
				},
			},
		}
	}

	podInfo := []struct {
		description      string
		hasVolumeDevices bool
		pod              func() *api.Pod
	}{
		{
			description:      "has VolumeDevices",
			hasVolumeDevices: true,
			pod:              podWithVolumeDevices,
		},
		{
			description:      "does not have VolumeDevices",
			hasVolumeDevices: false,
			pod:              podWithoutVolumeDevices,
		},
		{
			description:      "is nil",
			hasVolumeDevices: false,
			pod:              func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasVolumeDevices, oldPod := oldPodInfo.hasVolumeDevices, oldPodInfo.pod()
				newPodHasVolumeDevices, newPod := newPodInfo.hasVolumeDevices, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.BlockVolume, enabled)()

					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasVolumeDevices:
						// new pod should not be changed if the feature is enabled, or if the old pod had VolumeDevices
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					case newPodHasVolumeDevices:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have VolumeDevices
						if !reflect.DeepEqual(newPod, podWithoutVolumeDevices()) {
							t.Errorf("new pod had VolumeDevices: %v", diff.ObjectReflectDiff(newPod, podWithoutVolumeDevices()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}

func TestDropSubPath(t *testing.T) {
	podWithSubpaths := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:  api.RestartPolicyNever,
				Containers:     []api.Container{{Name: "container1", Image: "testimage", VolumeMounts: []api.VolumeMount{{Name: "a", SubPath: "foo"}, {Name: "a", SubPath: "foo2"}, {Name: "a", SubPath: "foo3"}}}},
				InitContainers: []api.Container{{Name: "container1", Image: "testimage", VolumeMounts: []api.VolumeMount{{Name: "a", SubPath: "foo"}, {Name: "a", SubPath: "foo2"}}}},
				Volumes:        []api.Volume{{Name: "a", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/dev/xvdc"}}}},
			},
		}
	}
	podWithoutSubpaths := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:  api.RestartPolicyNever,
				Containers:     []api.Container{{Name: "container1", Image: "testimage", VolumeMounts: []api.VolumeMount{{Name: "a", SubPath: ""}, {Name: "a", SubPath: ""}, {Name: "a", SubPath: ""}}}},
				InitContainers: []api.Container{{Name: "container1", Image: "testimage", VolumeMounts: []api.VolumeMount{{Name: "a", SubPath: ""}, {Name: "a", SubPath: ""}}}},
				Volumes:        []api.Volume{{Name: "a", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/dev/xvdc"}}}},
			},
		}
	}

	podInfo := []struct {
		description string
		hasSubpaths bool
		pod         func() *api.Pod
	}{
		{
			description: "has subpaths",
			hasSubpaths: true,
			pod:         podWithSubpaths,
		},
		{
			description: "does not have subpaths",
			hasSubpaths: false,
			pod:         podWithoutSubpaths,
		},
		{
			description: "is nil",
			hasSubpaths: false,
			pod:         func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasSubpaths, oldPod := oldPodInfo.hasSubpaths, oldPodInfo.pod()
				newPodHasSubpaths, newPod := newPodInfo.hasSubpaths, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeSubpath, enabled)()

					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasSubpaths:
						// new pod should not be changed if the feature is enabled, or if the old pod had subpaths
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					case newPodHasSubpaths:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have subpaths
						if !reflect.DeepEqual(newPod, podWithoutSubpaths()) {
							t.Errorf("new pod had subpaths: %v", diff.ObjectReflectDiff(newPod, podWithoutSubpaths()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}

func TestDropRuntimeClass(t *testing.T) {
	runtimeClassName := "some_container_engine"
	podWithoutRuntimeClass := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RuntimeClassName: nil,
			},
		}
	}
	podWithRuntimeClass := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RuntimeClassName: &runtimeClassName,
			},
		}
	}

	podInfo := []struct {
		description            string
		hasPodRuntimeClassName bool
		pod                    func() *api.Pod
	}{
		{
			description:            "pod Without RuntimeClassName",
			hasPodRuntimeClassName: false,
			pod:                    podWithoutRuntimeClass,
		},
		{
			description:            "pod With RuntimeClassName",
			hasPodRuntimeClassName: true,
			pod:                    podWithRuntimeClass,
		},
		{
			description:            "is nil",
			hasPodRuntimeClassName: false,
			pod:                    func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasRuntimeClassName, oldPod := oldPodInfo.hasPodRuntimeClassName, oldPodInfo.pod()
				newPodHasRuntimeClassName, newPod := newPodInfo.hasPodRuntimeClassName, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RuntimeClass, enabled)()

					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasRuntimeClassName:
						// new pod should not be changed if the feature is enabled, or if the old pod had RuntimeClass
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					case newPodHasRuntimeClassName:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have RuntimeClass
						if !reflect.DeepEqual(newPod, podWithoutRuntimeClass()) {
							t.Errorf("new pod had PodRuntimeClassName: %v", diff.ObjectReflectDiff(newPod, podWithoutRuntimeClass()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
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
						t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasProcMount:
						// new pod should not be changed if the feature is enabled, or if the old pod had ProcMount
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
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
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
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
						t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasEmptyDirSizeLimit:
						// new pod should not be changed if the feature is enabled, or if the old pod had EmptyDir SizeLimit
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					case newPodHasEmptyDirSizeLimit:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have EmptyDir SizeLimit
						if !reflect.DeepEqual(newPod, podWithoutEmptyDirSizeLimit()) {
							t.Errorf("new pod had EmptyDir SizeLimit: %v", diff.ObjectReflectDiff(newPod, podWithoutEmptyDirSizeLimit()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
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
			ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{"a": "1", apparmor.ContainerAnnotationKeyPrefix + "foo": "default"}},
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
						t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasAppArmor:
						// new pod should not be changed if the feature is enabled, or if the old pod had AppArmor
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					case newPodHasAppArmor:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have AppArmor
						if !reflect.DeepEqual(newPod, podWithoutAppArmor()) {
							t.Errorf("new pod had EmptyDir SizeLimit: %v", diff.ObjectReflectDiff(newPod, podWithoutAppArmor()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}

func TestDropTokenRequestProjection(t *testing.T) {
	podWithoutTRProjection := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				Volumes: []api.Volume{{
					VolumeSource: api.VolumeSource{
						Projected: &api.ProjectedVolumeSource{
							Sources: []api.VolumeProjection{{
								ServiceAccountToken: nil,
							}},
						}}},
				},
			},
		}
	}
	podWithoutProjectedVolumeSource := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				Volumes: []api.Volume{
					{VolumeSource: api.VolumeSource{
						ConfigMap: &api.ConfigMapVolumeSource{},
					}},
				},
			},
		}
	}
	podWithTRProjection := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				Volumes: []api.Volume{{
					VolumeSource: api.VolumeSource{
						Projected: &api.ProjectedVolumeSource{
							Sources: []api.VolumeProjection{{
								ServiceAccountToken: &api.ServiceAccountTokenProjection{
									Audience:          "api",
									ExpirationSeconds: 3600,
									Path:              "token",
								}},
							}},
					},
				},
				},
			}}
	}
	podInfo := []struct {
		description     string
		hasTRProjection bool
		pod             func() *api.Pod
	}{
		{
			description:     "has TokenRequestProjection",
			hasTRProjection: true,
			pod:             podWithTRProjection,
		},
		{
			description:     "does not have TokenRequestProjection",
			hasTRProjection: false,
			pod:             podWithoutTRProjection,
		},
		{
			description:     "does not have ProjectedVolumeSource",
			hasTRProjection: false,
			pod:             podWithoutProjectedVolumeSource,
		},
		{
			description:     "is nil",
			hasTRProjection: false,
			pod:             func() *api.Pod { return nil },
		},
	}
	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodhasTRProjection, oldPod := oldPodInfo.hasTRProjection, oldPodInfo.pod()
				newPodhasTRProjection, newPod := newPodInfo.hasTRProjection, newPodInfo.pod()
				if newPod == nil {
					continue
				}
				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.TokenRequestProjection, enabled)()
					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)
					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodInfo.pod()))
					}
					switch {
					case enabled || oldPodhasTRProjection:
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					case newPodhasTRProjection:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("%v", oldPod)
							t.Errorf("%v", newPod)
							t.Errorf("new pod was not changed")
						}
						if !reflect.DeepEqual(newPod, podWithoutTRProjection()) {
							t.Errorf("new pod had Tokenrequestprojection: %v", diff.ObjectReflectDiff(newPod, podWithoutTRProjection()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}

func TestDropRunAsGroup(t *testing.T) {
	group := func() *int64 {
		testGroup := int64(1000)
		return &testGroup
	}
	defaultProcMount := api.DefaultProcMount
	defaultSecurityContext := func() *api.SecurityContext {
		return &api.SecurityContext{ProcMount: &defaultProcMount}
	}
	securityContextWithRunAsGroup := func() *api.SecurityContext {
		return &api.SecurityContext{ProcMount: &defaultProcMount, RunAsGroup: group()}
	}
	podWithoutRunAsGroup := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:   api.RestartPolicyNever,
				SecurityContext: &api.PodSecurityContext{},
				Containers:      []api.Container{{Name: "container1", Image: "testimage", SecurityContext: defaultSecurityContext()}},
				InitContainers:  []api.Container{{Name: "initContainer1", Image: "testimage", SecurityContext: defaultSecurityContext()}},
			},
		}
	}
	podWithRunAsGroupInPod := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:   api.RestartPolicyNever,
				SecurityContext: &api.PodSecurityContext{RunAsGroup: group()},
				Containers:      []api.Container{{Name: "container1", Image: "testimage", SecurityContext: defaultSecurityContext()}},
				InitContainers:  []api.Container{{Name: "initContainer1", Image: "testimage", SecurityContext: defaultSecurityContext()}},
			},
		}
	}
	podWithRunAsGroupInContainers := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:   api.RestartPolicyNever,
				SecurityContext: &api.PodSecurityContext{},
				Containers:      []api.Container{{Name: "container1", Image: "testimage", SecurityContext: securityContextWithRunAsGroup()}},
				InitContainers:  []api.Container{{Name: "initContainer1", Image: "testimage", SecurityContext: defaultSecurityContext()}},
			},
		}
	}
	podWithRunAsGroupInInitContainers := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:   api.RestartPolicyNever,
				SecurityContext: &api.PodSecurityContext{},
				Containers:      []api.Container{{Name: "container1", Image: "testimage", SecurityContext: defaultSecurityContext()}},
				InitContainers:  []api.Container{{Name: "initContainer1", Image: "testimage", SecurityContext: securityContextWithRunAsGroup()}},
			},
		}
	}

	podInfo := []struct {
		description   string
		hasRunAsGroup bool
		pod           func() *api.Pod
	}{
		{
			description:   "have RunAsGroup in Pod",
			hasRunAsGroup: true,
			pod:           podWithRunAsGroupInPod,
		},
		{
			description:   "have RunAsGroup in Container",
			hasRunAsGroup: true,
			pod:           podWithRunAsGroupInContainers,
		},
		{
			description:   "have RunAsGroup in InitContainer",
			hasRunAsGroup: true,
			pod:           podWithRunAsGroupInInitContainers,
		},
		{
			description:   "does not have RunAsGroup",
			hasRunAsGroup: false,
			pod:           podWithoutRunAsGroup,
		},
		{
			description:   "is nil",
			hasRunAsGroup: false,
			pod:           func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasRunAsGroup, oldPod := oldPodInfo.hasRunAsGroup, oldPodInfo.pod()
				newPodHasRunAsGroup, newPod := newPodInfo.hasRunAsGroup, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.RunAsGroup, enabled)()

					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasRunAsGroup:
						// new pod should not be changed if the feature is enabled, or if the old pod had RunAsGroup
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					case newPodHasRunAsGroup:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("%v", oldPod)
							t.Errorf("%v", newPod)
							t.Errorf("new pod was not changed")
						}
						// new pod should not have RunAsGroup
						if !reflect.DeepEqual(newPod, podWithoutRunAsGroup()) {
							t.Errorf("new pod had RunAsGroup: %v", diff.ObjectReflectDiff(newPod, podWithoutRunAsGroup()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}

func TestDropGMSAFields(t *testing.T) {
	defaultContainerSecurityContextFactory := func() *api.SecurityContext {
		defaultProcMount := api.DefaultProcMount
		return &api.SecurityContext{ProcMount: &defaultProcMount}
	}
	podWithoutWindowsOptionsFactory := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:   api.RestartPolicyNever,
				SecurityContext: &api.PodSecurityContext{},
				Containers:      []api.Container{{Name: "container1", Image: "testimage", SecurityContext: defaultContainerSecurityContextFactory()}},
				InitContainers:  []api.Container{{Name: "initContainer1", Image: "testimage", SecurityContext: defaultContainerSecurityContextFactory()}},
			},
		}
	}

	type podFactoryInfo struct {
		description  string
		hasGMSAField bool
		// this factory should generate the input pod whose spec will be fed to dropDisabledFields
		podFactory func() *api.Pod
		// this factory should generate the expected pod after the GMSA fields have been dropped
		// we can't just use podWithoutWindowsOptionsFactory as is for this, since in some cases
		// we'll be left with a WindowsSecurityContextOptions struct with no GMSA field set, as opposed
		// to a nil pointer in the pod generated by podWithoutWindowsOptionsFactory
		// if this field is not set, it will default to the podFactory
		strippedPodFactory func() *api.Pod
	}
	podFactoryInfos := []podFactoryInfo{
		{
			description:  "does not have any GMSA field set",
			hasGMSAField: false,
			podFactory:   podWithoutWindowsOptionsFactory,
		},
		{
			description:  "has a pod-level WindowsSecurityContextOptions struct with no GMSA field set",
			hasGMSAField: false,
			podFactory: func() *api.Pod {
				pod := podWithoutWindowsOptionsFactory()
				pod.Spec.SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{}
				return pod
			},
		},
		{
			description:  "has a WindowsSecurityContextOptions struct with no GMSA field set on a container",
			hasGMSAField: false,
			podFactory: func() *api.Pod {
				pod := podWithoutWindowsOptionsFactory()
				pod.Spec.Containers[0].SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{}
				return pod
			},
		},
		{
			description:  "has a WindowsSecurityContextOptions struct with no GMSA field set on an init container",
			hasGMSAField: false,
			podFactory: func() *api.Pod {
				pod := podWithoutWindowsOptionsFactory()
				pod.Spec.InitContainers[0].SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{}
				return pod
			},
		},
		{
			description:  "is nil",
			hasGMSAField: false,
			podFactory:   func() *api.Pod { return nil },
		},
	}

	toPtr := func(s string) *string {
		return &s
	}
	addGMSACredentialSpecName := func(windowsOptions *api.WindowsSecurityContextOptions) {
		windowsOptions.GMSACredentialSpecName = toPtr("dummy-gmsa-cred-spec-name")
	}
	addGMSACredentialSpec := func(windowsOptions *api.WindowsSecurityContextOptions) {
		windowsOptions.GMSACredentialSpec = toPtr("dummy-gmsa-cred-spec-contents")
	}
	addBothGMSAFields := func(windowsOptions *api.WindowsSecurityContextOptions) {
		addGMSACredentialSpecName(windowsOptions)
		addGMSACredentialSpec(windowsOptions)
	}

	for fieldName, windowsOptionsTransformingFunc := range map[string]func(*api.WindowsSecurityContextOptions){
		"GMSACredentialSpecName field": addGMSACredentialSpecName,
		"GMSACredentialSpec field":     addGMSACredentialSpec,
		"both GMSA fields":             addBothGMSAFields,
	} {
		// yes, these variables are indeed needed for the closure to work
		// properly, please do NOT remove them
		name := fieldName
		transformingFunc := windowsOptionsTransformingFunc

		windowsOptionsWithGMSAFieldFactory := func() *api.WindowsSecurityContextOptions {
			windowsOptions := &api.WindowsSecurityContextOptions{}
			transformingFunc(windowsOptions)
			return windowsOptions
		}

		podFactoryInfos = append(podFactoryInfos,
			podFactoryInfo{
				description:  fmt.Sprintf("has %s in Pod", name),
				hasGMSAField: true,
				podFactory: func() *api.Pod {
					pod := podWithoutWindowsOptionsFactory()
					pod.Spec.SecurityContext.WindowsOptions = windowsOptionsWithGMSAFieldFactory()
					return pod
				},
				strippedPodFactory: func() *api.Pod {
					pod := podWithoutWindowsOptionsFactory()
					pod.Spec.SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{}
					return pod
				},
			},
			podFactoryInfo{
				description:  fmt.Sprintf("has %s in Container", name),
				hasGMSAField: true,
				podFactory: func() *api.Pod {
					pod := podWithoutWindowsOptionsFactory()
					pod.Spec.Containers[0].SecurityContext.WindowsOptions = windowsOptionsWithGMSAFieldFactory()
					return pod
				},
				strippedPodFactory: func() *api.Pod {
					pod := podWithoutWindowsOptionsFactory()
					pod.Spec.Containers[0].SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{}
					return pod
				},
			},
			podFactoryInfo{
				description:  fmt.Sprintf("has %s in InitContainer", name),
				hasGMSAField: true,
				podFactory: func() *api.Pod {
					pod := podWithoutWindowsOptionsFactory()
					pod.Spec.InitContainers[0].SecurityContext.WindowsOptions = windowsOptionsWithGMSAFieldFactory()
					return pod
				},
				strippedPodFactory: func() *api.Pod {
					pod := podWithoutWindowsOptionsFactory()
					pod.Spec.InitContainers[0].SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{}
					return pod
				},
			})
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodFactoryInfo := range podFactoryInfos {
			for _, newPodFactoryInfo := range podFactoryInfos {
				newPodHasGMSAField, newPod := newPodFactoryInfo.hasGMSAField, newPodFactoryInfo.podFactory()
				if newPod == nil {
					continue
				}
				oldPodHasGMSAField, oldPod := oldPodFactoryInfo.hasGMSAField, oldPodFactoryInfo.podFactory()

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %s, new pod %s", enabled, oldPodFactoryInfo.description, newPodFactoryInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WindowsGMSA, enabled)()

					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodFactoryInfo.podFactory()) {
						t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodFactoryInfo.podFactory()))
					}

					switch {
					case enabled || oldPodHasGMSAField:
						// new pod should not be changed if the feature is enabled, or if the old pod had any GMSA field set
						if !reflect.DeepEqual(newPod, newPodFactoryInfo.podFactory()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodFactoryInfo.podFactory()))
						}
					case newPodHasGMSAField:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodFactoryInfo.podFactory()) {
							t.Errorf("%v", oldPod)
							t.Errorf("%v", newPod)
							t.Errorf("new pod was not changed")
						}
						// new pod should not have any GMSA field set
						var expectedStrippedPod *api.Pod
						if newPodFactoryInfo.strippedPodFactory == nil {
							expectedStrippedPod = newPodFactoryInfo.podFactory()
						} else {
							expectedStrippedPod = newPodFactoryInfo.strippedPodFactory()
						}

						if !reflect.DeepEqual(newPod, expectedStrippedPod) {
							t.Errorf("new pod had some GMSA field set: %v", diff.ObjectReflectDiff(newPod, expectedStrippedPod))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodFactoryInfo.podFactory()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodFactoryInfo.podFactory()))
						}
					}
				})
			}
		}
	}
}

func TestDropWindowsRunAsUserNameFields(t *testing.T) {
	defaultContainerSecurityContextFactory := func() *api.SecurityContext {
		defaultProcMount := api.DefaultProcMount
		return &api.SecurityContext{ProcMount: &defaultProcMount}
	}
	podWithoutWindowsOptionsFactory := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:   api.RestartPolicyNever,
				SecurityContext: &api.PodSecurityContext{},
				Containers:      []api.Container{{Name: "container1", Image: "testimage", SecurityContext: defaultContainerSecurityContextFactory()}},
				InitContainers:  []api.Container{{Name: "initContainer1", Image: "testimage", SecurityContext: defaultContainerSecurityContextFactory()}},
			},
		}
	}

	type podFactoryInfo struct {
		description           string
		hasRunAsUserNameField bool
		// this factory should generate the input pod whose spec will be fed to dropDisabledFields
		podFactory func() *api.Pod
		// this factory should generate the expected pod after the RunAsUserName fields have been dropped
		// we can't just use podWithoutWindowsOptionsFactory as is for this, since in some cases
		// we'll be left with a WindowsSecurityContextOptions struct with no RunAsUserName field set,
		// as oposed to a nil pointer in the pod generated by podWithoutWindowsOptionsFactory
		// if this field is not set, it will default to the podFactory
		strippedPodFactory func() *api.Pod
	}

	toPtr := func(s string) *string {
		return &s
	}

	podFactoryInfos := []podFactoryInfo{
		{
			description:           "is nil",
			hasRunAsUserNameField: false,
			podFactory:            func() *api.Pod { return nil },
		},
		{
			description:           "does not have any RunAsUserName field set",
			hasRunAsUserNameField: false,
			podFactory:            podWithoutWindowsOptionsFactory,
		},
		{
			description:           "has a pod-level WindowsSecurityContextOptions struct with no RunAsUserName field set",
			hasRunAsUserNameField: false,
			podFactory: func() *api.Pod {
				pod := podWithoutWindowsOptionsFactory()
				pod.Spec.SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{}
				return pod
			},
		},
		{
			description:           "has a WindowsSecurityContextOptions struct with no RunAsUserName field set on a container",
			hasRunAsUserNameField: false,
			podFactory: func() *api.Pod {
				pod := podWithoutWindowsOptionsFactory()
				pod.Spec.Containers[0].SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{}
				return pod
			},
		},
		{
			description:           "has a WindowsSecurityContextOptions struct with no RunAsUserName field set on an init container",
			hasRunAsUserNameField: false,
			podFactory: func() *api.Pod {
				pod := podWithoutWindowsOptionsFactory()
				pod.Spec.InitContainers[0].SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{}
				return pod
			},
		},
		{
			description:           "has RunAsUserName field set in the PodSecurityContext",
			hasRunAsUserNameField: true,
			podFactory: func() *api.Pod {
				pod := podWithoutWindowsOptionsFactory()
				pod.Spec.SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{RunAsUserName: toPtr("foo-lish")}
				return pod
			},
			strippedPodFactory: func() *api.Pod {
				pod := podWithoutWindowsOptionsFactory()
				pod.Spec.SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{}
				return pod
			},
		},
		{
			description:           "has RunAsUserName field set in a container's SecurityContext",
			hasRunAsUserNameField: true,
			podFactory: func() *api.Pod {
				pod := podWithoutWindowsOptionsFactory()
				pod.Spec.Containers[0].SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{RunAsUserName: toPtr("foo-lish")}
				return pod
			},
			strippedPodFactory: func() *api.Pod {
				pod := podWithoutWindowsOptionsFactory()
				pod.Spec.Containers[0].SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{}
				return pod
			},
		},
		{
			description:           "has RunAsUserName field set in an init container's PodSecurityContext",
			hasRunAsUserNameField: true,
			podFactory: func() *api.Pod {
				pod := podWithoutWindowsOptionsFactory()
				pod.Spec.InitContainers[0].SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{RunAsUserName: toPtr("foo-lish")}
				return pod
			},
			strippedPodFactory: func() *api.Pod {
				pod := podWithoutWindowsOptionsFactory()
				pod.Spec.InitContainers[0].SecurityContext.WindowsOptions = &api.WindowsSecurityContextOptions{}
				return pod
			},
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodFactoryInfo := range podFactoryInfos {
			for _, newPodFactoryInfo := range podFactoryInfos {
				newPodHasRunAsUserNameField, newPod := newPodFactoryInfo.hasRunAsUserNameField, newPodFactoryInfo.podFactory()
				if newPod == nil {
					continue
				}
				oldPodHasRunAsUserNameField, oldPod := oldPodFactoryInfo.hasRunAsUserNameField, oldPodFactoryInfo.podFactory()

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %s, new pod %s", enabled, oldPodFactoryInfo.description, newPodFactoryInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WindowsRunAsUserName, enabled)()

					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodFactoryInfo.podFactory()) {
						t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodFactoryInfo.podFactory()))
					}

					switch {
					case enabled || oldPodHasRunAsUserNameField:
						// new pod should not be changed if the feature is enabled, or if the old pod had the RunAsUserName field set
						if !reflect.DeepEqual(newPod, newPodFactoryInfo.podFactory()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodFactoryInfo.podFactory()))
						}
					case newPodHasRunAsUserNameField:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodFactoryInfo.podFactory()) {
							t.Errorf("%v", oldPod)
							t.Errorf("%v", newPod)
							t.Errorf("new pod was not changed")
						}
						// new pod should not have the RunAsUserName field set
						var expectedStrippedPod *api.Pod
						if newPodFactoryInfo.strippedPodFactory == nil {
							expectedStrippedPod = newPodFactoryInfo.podFactory()
						} else {
							expectedStrippedPod = newPodFactoryInfo.strippedPodFactory()
						}

						if !reflect.DeepEqual(newPod, expectedStrippedPod) {
							t.Errorf("new pod had some RunAsUserName field set: %v", diff.ObjectReflectDiff(newPod, expectedStrippedPod))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodFactoryInfo.podFactory()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodFactoryInfo.podFactory()))
						}
					}
				})
			}
		}
	}
}

func TestDropPodSysctls(t *testing.T) {
	podWithSysctls := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{
					Sysctls: []api.Sysctl{{Name: "test", Value: "value"}},
				},
			},
		}
	}
	podWithoutSysctls := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				SecurityContext: &api.PodSecurityContext{},
			},
		}
	}
	podWithoutSecurityContext := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{},
		}
	}

	podInfo := []struct {
		description string
		hasSysctls  bool
		pod         func() *api.Pod
	}{
		{
			description: "has Sysctls",
			hasSysctls:  true,
			pod:         podWithSysctls,
		},
		{
			description: "does not have Sysctls",
			hasSysctls:  false,
			pod:         podWithoutSysctls,
		},
		{
			description: "does not have SecurityContext",
			hasSysctls:  false,
			pod:         podWithoutSecurityContext,
		},
		{
			description: "is nil",
			hasSysctls:  false,
			pod:         func() *api.Pod { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldPodInfo := range podInfo {
			for _, newPodInfo := range podInfo {
				oldPodHasSysctls, oldPod := oldPodInfo.hasSysctls, oldPodInfo.pod()
				newPodHasSysctls, newPod := newPodInfo.hasSysctls, newPodInfo.pod()
				if newPod == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.Sysctls, enabled)()

					var oldPodSpec *api.PodSpec
					if oldPod != nil {
						oldPodSpec = &oldPod.Spec
					}
					dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

					// old pod should never be changed
					if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
						t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasSysctls:
						// new pod should not be changed if the feature is enabled, or if the old pod had Sysctls set
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					case newPodHasSysctls:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have Sysctls
						if !reflect.DeepEqual(newPod, podWithoutSysctls()) {
							t.Errorf("new pod had Sysctls: %v", diff.ObjectReflectDiff(newPod, podWithoutSysctls()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}

func TestDropSubPathExpr(t *testing.T) {
	podWithSubpaths := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:  api.RestartPolicyNever,
				Containers:     []api.Container{{Name: "container1", Image: "testimage", VolumeMounts: []api.VolumeMount{{Name: "a", SubPathExpr: "foo"}, {Name: "a", SubPathExpr: "foo2"}, {Name: "a", SubPathExpr: "foo3"}}}},
				InitContainers: []api.Container{{Name: "container1", Image: "testimage", VolumeMounts: []api.VolumeMount{{Name: "a", SubPathExpr: "foo"}, {Name: "a", SubPathExpr: "foo2"}}}},
				Volumes:        []api.Volume{{Name: "a", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/dev/xvdc"}}}},
			},
		}
	}
	podWithoutSubpaths := func() *api.Pod {
		return &api.Pod{
			Spec: api.PodSpec{
				RestartPolicy:  api.RestartPolicyNever,
				Containers:     []api.Container{{Name: "container1", Image: "testimage", VolumeMounts: []api.VolumeMount{{Name: "a", SubPathExpr: ""}, {Name: "a", SubPathExpr: ""}, {Name: "a", SubPathExpr: ""}}}},
				InitContainers: []api.Container{{Name: "container1", Image: "testimage", VolumeMounts: []api.VolumeMount{{Name: "a", SubPathExpr: ""}, {Name: "a", SubPathExpr: ""}}}},
				Volumes:        []api.Volume{{Name: "a", VolumeSource: api.VolumeSource{HostPath: &api.HostPathVolumeSource{Path: "/dev/xvdc"}}}},
			},
		}
	}

	podInfo := []struct {
		description string
		hasSubpaths bool
		pod         func() *api.Pod
	}{
		{
			description: "has subpaths",
			hasSubpaths: true,
			pod:         podWithSubpaths,
		},
		{
			description: "does not have subpaths",
			hasSubpaths: false,
			pod:         podWithoutSubpaths,
		},
		{
			description: "is nil",
			hasSubpaths: false,
			pod:         func() *api.Pod { return nil },
		},
	}

	enabled := true
	for _, oldPodInfo := range podInfo {
		for _, newPodInfo := range podInfo {
			oldPodHasSubpaths, oldPod := oldPodInfo.hasSubpaths, oldPodInfo.pod()
			newPodHasSubpaths, newPod := newPodInfo.hasSubpaths, newPodInfo.pod()
			if newPod == nil {
				continue
			}

			t.Run(fmt.Sprintf("feature enabled=%v, old pod %v, new pod %v", enabled, oldPodInfo.description, newPodInfo.description), func(t *testing.T) {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeSubpathEnvExpansion, enabled)()

				var oldPodSpec *api.PodSpec
				if oldPod != nil {
					oldPodSpec = &oldPod.Spec
				}
				dropDisabledFields(&newPod.Spec, nil, oldPodSpec, nil)

				// old pod should never be changed
				if !reflect.DeepEqual(oldPod, oldPodInfo.pod()) {
					t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodInfo.pod()))
				}

				switch {
				case enabled || oldPodHasSubpaths:
					// new pod should not be changed if the feature is enabled, or if the old pod had subpaths
					if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
						t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
					}
				case newPodHasSubpaths:
					// new pod should be changed
					if reflect.DeepEqual(newPod, newPodInfo.pod()) {
						t.Errorf("new pod was not changed")
					}
					// new pod should not have subpaths
					if !reflect.DeepEqual(newPod, podWithoutSubpaths()) {
						t.Errorf("new pod had subpaths: %v", diff.ObjectReflectDiff(newPod, podWithoutSubpaths()))
					}
				default:
					// new pod should not need to be changed
					if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
						t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
					}
				}
			})
		}
	}
}

// helper creates a podStatus with list of PodIPs
func makePodStatus(podIPs []api.PodIP) *api.PodStatus {
	return &api.PodStatus{
		PodIPs: podIPs,
	}
}

func TestDropStatusPodIPs(t *testing.T) {
	testCases := []struct {
		name             string
		podStatus        *api.PodStatus
		oldPodStatus     *api.PodStatus
		comparePodStatus *api.PodStatus
		enableDualStack  bool
	}{
		{
			name:             "nil pod ips",
			enableDualStack:  false,
			podStatus:        makePodStatus(nil),
			oldPodStatus:     nil,
			comparePodStatus: makePodStatus(nil),
		},
		{
			name:             "empty pod ips",
			enableDualStack:  false,
			podStatus:        makePodStatus([]api.PodIP{}),
			oldPodStatus:     nil,
			comparePodStatus: makePodStatus([]api.PodIP{}),
		},
		{
			name:             "single family ipv6",
			enableDualStack:  false,
			podStatus:        makePodStatus([]api.PodIP{{IP: "::1"}}),
			comparePodStatus: makePodStatus([]api.PodIP{{IP: "::1"}}),
		},
		{
			name:             "single family ipv4",
			enableDualStack:  false,
			podStatus:        makePodStatus([]api.PodIP{{IP: "1.1.1.1"}}),
			comparePodStatus: makePodStatus([]api.PodIP{{IP: "1.1.1.1"}}),
		},
		{
			name:             "dualstack 4-6",
			enableDualStack:  true,
			podStatus:        makePodStatus([]api.PodIP{{IP: "1.1.1.1"}, {IP: "::1"}}),
			comparePodStatus: makePodStatus([]api.PodIP{{IP: "1.1.1.1"}, {IP: "::1"}}),
		},
		{
			name:             "dualstack 6-4",
			enableDualStack:  true,
			podStatus:        makePodStatus([]api.PodIP{{IP: "::1"}, {IP: "1.1.1.1"}}),
			comparePodStatus: makePodStatus([]api.PodIP{{IP: "::1"}, {IP: "1.1.1.1"}}),
		},
		{
			name:             "not dualstack 6-4=>4only",
			enableDualStack:  false,
			podStatus:        makePodStatus([]api.PodIP{{IP: "::1"}, {IP: "1.1.1.1"}}),
			oldPodStatus:     nil,
			comparePodStatus: makePodStatus([]api.PodIP{{IP: "::1"}}),
		},
		{
			name:             "not dualstack 6-4=>as is (used in old)",
			enableDualStack:  false,
			podStatus:        makePodStatus([]api.PodIP{{IP: "::1"}, {IP: "1.1.1.1"}}),
			oldPodStatus:     makePodStatus([]api.PodIP{{IP: "::1"}, {IP: "1.1.1.1"}}),
			comparePodStatus: makePodStatus([]api.PodIP{{IP: "::1"}, {IP: "1.1.1.1"}}),
		},
		{
			name:             "not dualstack 6-4=>6only",
			enableDualStack:  false,
			podStatus:        makePodStatus([]api.PodIP{{IP: "::1"}, {IP: "1.1.1.1"}}),
			oldPodStatus:     nil,
			comparePodStatus: makePodStatus([]api.PodIP{{IP: "::1"}}),
		},
		{
			name:             "not dualstack 6-4=>as is (used in old)",
			enableDualStack:  false,
			podStatus:        makePodStatus([]api.PodIP{{IP: "::1"}, {IP: "1.1.1.1"}}),
			oldPodStatus:     makePodStatus([]api.PodIP{{IP: "::1"}, {IP: "1.1.1.1"}}),
			comparePodStatus: makePodStatus([]api.PodIP{{IP: "::1"}, {IP: "1.1.1.1"}}),
		},
	}

	for _, tc := range testCases {
		func() {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.IPv6DualStack, tc.enableDualStack)()
			dropPodStatusDisabledFields(tc.podStatus, tc.oldPodStatus)

			old := tc.oldPodStatus.DeepCopy()
			// old pod status should never be changed
			if !reflect.DeepEqual(tc.oldPodStatus, old) {
				t.Errorf("%v: old pod status changed: %v", tc.name, diff.ObjectReflectDiff(tc.oldPodStatus, old))
			}

			if !reflect.DeepEqual(tc.podStatus, tc.comparePodStatus) {
				t.Errorf("%v: unexpected pod status: %v", tc.name, diff.ObjectReflectDiff(tc.podStatus, tc.comparePodStatus))
			}
		}()
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
			description:            "has subpaths",
			hasEphemeralContainers: true,
			pod:                    podWithEphemeralContainers,
		},
		{
			description:            "does not have subpaths",
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
						t.Errorf("old pod changed: %v", diff.ObjectReflectDiff(oldPod, oldPodInfo.pod()))
					}

					switch {
					case enabled || oldPodHasEphemeralContainers:
						// new pod should not be changed if the feature is enabled, or if the old pod had subpaths
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					case newPodHasEphemeralContainers:
						// new pod should be changed
						if reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod was not changed")
						}
						// new pod should not have subpaths
						if !reflect.DeepEqual(newPod, podWithoutEphemeralContainers()) {
							t.Errorf("new pod had subpaths: %v", diff.ObjectReflectDiff(newPod, podWithoutEphemeralContainers()))
						}
					default:
						// new pod should not need to be changed
						if !reflect.DeepEqual(newPod, newPodInfo.pod()) {
							t.Errorf("new pod changed: %v", diff.ObjectReflectDiff(newPod, newPodInfo.pod()))
						}
					}
				})
			}
		}
	}
}
