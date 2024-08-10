/*
Copyright 2016 The Kubernetes Authors.

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
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	volumetesting "k8s.io/kubernetes/pkg/volume/testing"
	"k8s.io/kubernetes/pkg/volume/util"
	volumetypes "k8s.io/kubernetes/pkg/volume/util/types"
)

// Calls AddPodToVolume() to add new pod to new volume
// Verifies newly added pod/volume exists via
// PodExistsInVolume() VolumeExists() and GetVolumesToMount()
func Test_AddPodToVolume_Positive_NewPodNewVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestKubeletVolumePluginMgr(t)
	seLinuxTranslator := util.NewFakeSELinuxLabelTranslator()
	dsw := NewDesiredStateOfWorld(volumePluginMgr, seLinuxTranslator)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod3",
			UID:  "pod3uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := util.GetUniquePodName(pod)

	// Act
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */, nil /* seLinuxContainerContexts */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsDsw(t, generatedVolumeName, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolumeName, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, podName, generatedVolumeName, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsWithSpecNameInVolumeDsw(t, podName, volumeSpec.Name(), dsw)
}

// Calls AddPodToVolume() twice to add the same pod to the same volume
// Verifies newly added pod/volume exists via
// PodExistsInVolume() VolumeExists() and GetVolumesToMount() and no errors.
func Test_AddPodToVolume_Positive_ExistingPodExistingVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestKubeletVolumePluginMgr(t)
	seLinuxTranslator := util.NewFakeSELinuxLabelTranslator()
	dsw := NewDesiredStateOfWorld(volumePluginMgr, seLinuxTranslator)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod3",
			UID:  "pod3uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := util.GetUniquePodName(pod)

	// Act
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */, nil /* seLinuxContainerContexts */)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsDsw(t, generatedVolumeName, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolumeName, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, podName, generatedVolumeName, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsWithSpecNameInVolumeDsw(t, podName, volumeSpec.Name(), dsw)
}

// Call AddPodToVolume() on different pods for different kinds of volumes
// Verities generated names are same for different pods if volume is device mountable or attachable
// Verities generated names are different for different pods if volume is not device mountble and attachable
func Test_AddPodToVolume_Positive_NamesForDifferentPodsAndDifferentVolumes(t *testing.T) {
	// Arrange
	fakeVolumeHost := volumetesting.NewFakeVolumeHost(t,
		"",  /* rootDir */
		nil, /* kubeClient */
		nil, /* plugins */
	)
	plugins := []volume.VolumePlugin{
		&volumetesting.FakeBasicVolumePlugin{
			Plugin: volumetesting.FakeVolumePlugin{
				PluginName: "basic",
			},
		},
		&volumetesting.FakeDeviceMountableVolumePlugin{
			FakeBasicVolumePlugin: volumetesting.FakeBasicVolumePlugin{
				Plugin: volumetesting.FakeVolumePlugin{
					PluginName: "device-mountable",
				},
			},
		},
		&volumetesting.FakeAttachableVolumePlugin{
			FakeDeviceMountableVolumePlugin: volumetesting.FakeDeviceMountableVolumePlugin{
				FakeBasicVolumePlugin: volumetesting.FakeBasicVolumePlugin{
					Plugin: volumetesting.FakeVolumePlugin{
						PluginName: "attachable",
					},
				},
			},
		},
	}
	volumePluginMgr := volume.VolumePluginMgr{}
	volumePluginMgr.InitPlugins(plugins, nil /* prober */, fakeVolumeHost)
	seLinuxTranslator := util.NewFakeSELinuxLabelTranslator()
	dsw := NewDesiredStateOfWorld(&volumePluginMgr, seLinuxTranslator)

	testcases := map[string]struct {
		pod1 *v1.Pod
		pod2 *v1.Pod
		same bool
	}{
		"basic": {
			&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					UID:  "pod1uid",
				},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name:         "basic",
							VolumeSource: v1.VolumeSource{},
						},
					},
				},
			},
			&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod2",
					UID:  "pod2uid",
				},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name:         "basic",
							VolumeSource: v1.VolumeSource{},
						},
					},
				},
			},
			false,
		},
		"device-mountable": {
			&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					UID:  "pod1uid",
				},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name:         "device-mountable",
							VolumeSource: v1.VolumeSource{},
						},
					},
				},
			},
			&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod2",
					UID:  "pod2uid",
				},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name:         "device-mountable",
							VolumeSource: v1.VolumeSource{},
						},
					},
				},
			},
			true,
		},
		"attachable": {
			&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod1",
					UID:  "pod1uid",
				},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name:         "attachable",
							VolumeSource: v1.VolumeSource{},
						},
					},
				},
			},
			&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod2",
					UID:  "pod2uid",
				},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							Name:         "attachable",
							VolumeSource: v1.VolumeSource{},
						},
					},
				},
			},
			true,
		},
	}

	// Act & Assert
	for name, v := range testcases {
		volumeSpec1 := &volume.Spec{Volume: &v.pod1.Spec.Volumes[0]}
		volumeSpec2 := &volume.Spec{Volume: &v.pod2.Spec.Volumes[0]}
		generatedVolumeName1, err1 := dsw.AddPodToVolume(util.GetUniquePodName(v.pod1), v.pod1, volumeSpec1, volumeSpec1.Name(), "", nil)
		generatedVolumeName2, err2 := dsw.AddPodToVolume(util.GetUniquePodName(v.pod2), v.pod2, volumeSpec2, volumeSpec2.Name(), "", nil)
		if err1 != nil {
			t.Fatalf("test %q: AddPodToVolume failed. Expected: <no error> Actual: <%v>", name, err1)
		}
		if err2 != nil {
			t.Fatalf("test %q: AddPodToVolume failed. Expected: <no error> Actual: <%v>", name, err2)
		}
		if v.same {
			if generatedVolumeName1 != generatedVolumeName2 {
				t.Fatalf("test %q: AddPodToVolume should generate same names, but got %q != %q", name, generatedVolumeName1, generatedVolumeName2)
			}
		} else {
			if generatedVolumeName1 == generatedVolumeName2 {
				t.Fatalf("test %q: AddPodToVolume should generate different names, but got %q == %q", name, generatedVolumeName1, generatedVolumeName2)
			}
		}
	}

}

// Populates data struct with a new volume/pod
// Calls DeletePodFromVolume() to removes the pod
// Verifies newly added pod/volume are deleted
func Test_DeletePodFromVolume_Positive_PodExistsVolumeExists(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestKubeletVolumePluginMgr(t)
	seLinuxTranslator := util.NewFakeSELinuxLabelTranslator()
	dsw := NewDesiredStateOfWorld(volumePluginMgr, seLinuxTranslator)
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod3",
			UID:  "pod3uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{Volume: &pod.Spec.Volumes[0]}
	podName := util.GetUniquePodName(pod)
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */, nil /* seLinuxContainerContexts */)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}
	verifyVolumeExistsDsw(t, generatedVolumeName, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolumeName, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, podName, generatedVolumeName, "" /* SELinuxContext */, dsw)

	// Act
	dsw.DeletePodFromVolume(podName, generatedVolumeName)

	// Assert
	verifyVolumeDoesntExist(t, generatedVolumeName, "" /* SELinuxContext */, dsw)
	verifyVolumeDoesntExistInVolumesToMount(t, generatedVolumeName, dsw)
	verifyPodDoesntExistInVolumeDsw(t, podName, generatedVolumeName, "" /* SELinuxContext */, dsw)
	verifyVolumeDoesntExistWithSpecNameInVolumeDsw(t, podName, volumeSpec.Name(), dsw)
}

// Calls AddPodToVolume() to add three new volumes to data struct
// Verifies newly added pod/volume exists via PodExistsInVolume()
// VolumeExists() and GetVolumesToMount()
// Marks only second volume as reported in use.
// Verifies only that volume is marked reported in use
// Marks only first volume as reported in use.
// Verifies only that volume is marked reported in use
func Test_MarkVolumesReportedInUse_Positive_NewPodNewVolume(t *testing.T) {
	// Arrange
	volumePluginMgr, _ := volumetesting.GetTestKubeletVolumePluginMgr(t)
	seLinuxTranslator := util.NewFakeSELinuxLabelTranslator()
	dsw := NewDesiredStateOfWorld(volumePluginMgr, seLinuxTranslator)

	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume1-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device1",
						},
					},
				},
			},
		},
	}

	volume1Spec := &volume.Spec{Volume: &pod1.Spec.Volumes[0]}
	pod1Name := util.GetUniquePodName(pod1)

	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod2",
			UID:  "pod2uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume2-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device2",
						},
					},
				},
			},
		},
	}

	volume2Spec := &volume.Spec{Volume: &pod2.Spec.Volumes[0]}
	pod2Name := util.GetUniquePodName(pod2)

	pod3 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod3",
			UID:  "pod3uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "volume3-name",
					VolumeSource: v1.VolumeSource{
						GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
							PDName: "fake-device3",
						},
					},
				},
			},
		},
	}

	volume3Spec := &volume.Spec{Volume: &pod3.Spec.Volumes[0]}
	pod3Name := util.GetUniquePodName(pod3)

	generatedVolume1Name, err := dsw.AddPodToVolume(
		pod1Name, pod1, volume1Spec, volume1Spec.Name(), "" /* volumeGidValue */, nil /* seLinuxContainerContexts */)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	generatedVolume2Name, err := dsw.AddPodToVolume(
		pod2Name, pod2, volume2Spec, volume2Spec.Name(), "" /* volumeGidValue */, nil /* seLinuxContainerContexts */)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	generatedVolume3Name, err := dsw.AddPodToVolume(
		pod3Name, pod3, volume3Spec, volume3Spec.Name(), "" /* volumeGidValue */, nil /* seLinuxContainerContexts */)
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	// Act
	volumesReportedInUse := []v1.UniqueVolumeName{generatedVolume2Name}
	dsw.MarkVolumesReportedInUse(volumesReportedInUse)

	// Assert
	verifyVolumeExistsDsw(t, generatedVolume1Name, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolume1Name, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, pod1Name, generatedVolume1Name, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsDsw(t, generatedVolume2Name, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolume2Name, true /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, pod2Name, generatedVolume2Name, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsDsw(t, generatedVolume3Name, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolume3Name, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, pod3Name, generatedVolume3Name, "" /* SELinuxContext */, dsw)

	// Act
	volumesReportedInUse = []v1.UniqueVolumeName{generatedVolume3Name}
	dsw.MarkVolumesReportedInUse(volumesReportedInUse)

	// Assert
	verifyVolumeExistsDsw(t, generatedVolume1Name, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolume1Name, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, pod1Name, generatedVolume1Name, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsDsw(t, generatedVolume2Name, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolume2Name, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, pod2Name, generatedVolume2Name, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsDsw(t, generatedVolume3Name, "" /* SELinuxContext */, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolume3Name, true /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, pod3Name, generatedVolume3Name, "" /* SELinuxContext */, dsw)
}

func Test_AddPodToVolume_WithEmptyDirSizeLimit(t *testing.T) {
	volumePluginMgr, _ := volumetesting.GetTestKubeletVolumePluginMgr(t)
	seLinuxTranslator := util.NewFakeSELinuxLabelTranslator()
	dsw := NewDesiredStateOfWorld(volumePluginMgr, seLinuxTranslator)
	quantity1Gi := resource.MustParse("1Gi")
	quantity2Gi := resource.MustParse("2Gi")
	quantity3Gi := resource.MustParse("3Gi")

	pod1 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceEphemeralStorage: quantity1Gi,
						},
					},
				},
				{
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceEphemeralStorage: quantity1Gi,
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "emptyDir1",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{
							SizeLimit: &quantity1Gi,
						},
					},
				},
				{
					Name: "emptyDir2",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{
							SizeLimit: &quantity2Gi,
						},
					},
				},
				{
					Name: "emptyDir3",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{
							SizeLimit: &quantity3Gi,
						},
					},
				},
				{
					Name: "emptyDir4",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				},
			},
		},
	}
	pod1Name := util.GetUniquePodName(pod1)
	pod1DesiredSizeLimitMap := map[string]*resource.Quantity{
		"emptyDir1": &quantity1Gi,
		"emptyDir2": &quantity2Gi,
		"emptyDir3": &quantity2Gi,
		"emptyDir4": &quantity2Gi,
	}
	pod2 := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod2",
			UID:  "pod2uid",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					Name: "emptyDir5",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{
							SizeLimit: &quantity1Gi,
						},
					},
				},
				{
					Name: "emptyDir6",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{
							SizeLimit: &quantity2Gi,
						},
					},
				},
				{
					Name: "emptyDir7",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{
							SizeLimit: &quantity3Gi,
						},
					},
				},
				{
					Name: "emptyDir8",
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				},
			},
		},
	}
	pod2Name := util.GetUniquePodName(pod2)
	pod2DesiredSizeLimitMap := map[string]*resource.Quantity{
		"emptyDir5": &quantity1Gi,
		"emptyDir6": &quantity2Gi,
		"emptyDir7": &quantity3Gi,
		"emptyDir8": resource.NewQuantity(0, resource.BinarySI),
	}
	for i := range pod1.Spec.Volumes {
		volumeSpec := &volume.Spec{Volume: &pod1.Spec.Volumes[i]}
		_, err := dsw.AddPodToVolume(pod1Name, pod1, volumeSpec, volumeSpec.Name(), "", nil /* seLinuxContainerContexts */)
		if err != nil {
			t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
		}
	}
	for i := range pod2.Spec.Volumes {
		volumeSpec := &volume.Spec{Volume: &pod2.Spec.Volumes[i]}
		_, err := dsw.AddPodToVolume(pod2Name, pod2, volumeSpec, volumeSpec.Name(), "", nil /* seLinuxContainerContexts */)
		if err != nil {
			t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
		}
	}
	verifyDesiredSizeLimitInVolumeDsw(t, pod1Name, pod1DesiredSizeLimitMap, dsw)
	verifyDesiredSizeLimitInVolumeDsw(t, pod2Name, pod2DesiredSizeLimitMap, dsw)
}

// Calls AddPodToVolume() with a volume that support SELinux, but is ReadWriteMany.
// Verifies newly added pod/volume exists via PodExistsInVolume() without SELinux context
// VolumeExists() and GetVolumesToMount() and no errors.
func Test_AddPodToVolume_Positive_SELinuxNoRWOP(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, true)
	// Arrange
	plugins := []volume.VolumePlugin{
		&volumetesting.FakeDeviceMountableVolumePlugin{
			FakeBasicVolumePlugin: volumetesting.FakeBasicVolumePlugin{
				Plugin: volumetesting.FakeVolumePlugin{
					PluginName:      "basic",
					SupportsSELinux: true,
				},
			},
		},
	}
	volumePluginMgr := volume.VolumePluginMgr{}
	fakeVolumeHost := volumetesting.NewFakeVolumeHost(t,
		"",  /* rootDir */
		nil, /* kubeClient */
		nil, /* plugins */
	)
	volumePluginMgr.InitPlugins(plugins, nil /* prober */, fakeVolumeHost)
	seLinuxTranslator := util.NewFakeSELinuxLabelTranslator()
	dsw := NewDesiredStateOfWorld(&volumePluginMgr, seLinuxTranslator)
	seLinux := v1.SELinuxOptions{
		User:  "system_u",
		Role:  "object_r",
		Type:  "container_t",
		Level: "s0:c1,c2",
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{
				SELinuxOptions: &seLinux,
			},
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "myClaim",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{
		PersistentVolume: &v1.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{
				Name: "basicPV",
			},
			Spec: v1.PersistentVolumeSpec{
				AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteMany},
			},
		},
	}
	podName := util.GetUniquePodName(pod)
	seLinuxContainerContexts := []*v1.SELinuxOptions{&seLinux}

	// Act
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */, seLinuxContainerContexts)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsDsw(t, generatedVolumeName, "" /* SELinux */, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolumeName, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, podName, generatedVolumeName, "" /* SELinux */, dsw)
	verifyVolumeExistsWithSpecNameInVolumeDsw(t, podName, volumeSpec.Name(), dsw)
}

// Calls AddPodToVolume() with a volume that does not support SELinux.
// Verifies newly added pod/volume exists via PodExistsInVolume() without SELinux context
// VolumeExists() and GetVolumesToMount() and no errors.
func Test_AddPodToVolume_Positive_NoSELinuxPlugin(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, true)
	// Arrange
	plugins := []volume.VolumePlugin{
		&volumetesting.FakeDeviceMountableVolumePlugin{
			FakeBasicVolumePlugin: volumetesting.FakeBasicVolumePlugin{
				Plugin: volumetesting.FakeVolumePlugin{
					PluginName:      "basic",
					SupportsSELinux: false,
				},
			},
		},
	}
	volumePluginMgr := volume.VolumePluginMgr{}
	fakeVolumeHost := volumetesting.NewFakeVolumeHost(t,
		"",  /* rootDir */
		nil, /* kubeClient */
		nil, /* plugins */
	)
	volumePluginMgr.InitPlugins(plugins, nil /* prober */, fakeVolumeHost)
	seLinuxTranslator := util.NewFakeSELinuxLabelTranslator()
	dsw := NewDesiredStateOfWorld(&volumePluginMgr, seLinuxTranslator)
	seLinux := v1.SELinuxOptions{
		User:  "system_u",
		Role:  "object_r",
		Type:  "container_t",
		Level: "s0:c1,c2",
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{
				SELinuxOptions: &seLinux,
			},
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "myClaim",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{
		PersistentVolume: &v1.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{
				Name: "basicPV",
			},
			Spec: v1.PersistentVolumeSpec{
				AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod},
			},
		},
	}
	podName := util.GetUniquePodName(pod)
	seLinuxContainerContexts := []*v1.SELinuxOptions{&seLinux}

	// Act
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */, seLinuxContainerContexts)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsDsw(t, generatedVolumeName, "" /* SELinux */, dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolumeName, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, podName, generatedVolumeName, "" /* SELinux */, dsw)
	verifyVolumeExistsWithSpecNameInVolumeDsw(t, podName, volumeSpec.Name(), dsw)
}

// Calls AddPodToVolume() twice to add two pods with the same SELinuxContext
// to the same ReadWriteOncePod PV.
// Verifies newly added pod/volume exists via PodExistsInVolume()
// VolumeExists() and GetVolumesToMount() and no errors.
func Test_AddPodToVolume_Positive_ExistingPodSameSELinuxRWOP(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, true)
	// Arrange
	plugins := []volume.VolumePlugin{
		&volumetesting.FakeDeviceMountableVolumePlugin{
			FakeBasicVolumePlugin: volumetesting.FakeBasicVolumePlugin{
				Plugin: volumetesting.FakeVolumePlugin{
					PluginName:      "basic",
					SupportsSELinux: true,
				},
			},
		},
	}
	volumePluginMgr := volume.VolumePluginMgr{}
	fakeVolumeHost := volumetesting.NewFakeVolumeHost(t,
		"",  /* rootDir */
		nil, /* kubeClient */
		nil, /* plugins */
	)
	volumePluginMgr.InitPlugins(plugins, nil /* prober */, fakeVolumeHost)
	seLinuxTranslator := util.NewFakeSELinuxLabelTranslator()
	dsw := NewDesiredStateOfWorld(&volumePluginMgr, seLinuxTranslator)
	seLinux := v1.SELinuxOptions{
		User:  "system_u",
		Role:  "object_r",
		Type:  "container_t",
		Level: "s0:c1,c2",
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{
				SELinuxOptions: &seLinux,
			},
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "myClaim",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{
		PersistentVolume: &v1.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{
				Name: "basicPV",
			},
			Spec: v1.PersistentVolumeSpec{
				AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod},
			},
		},
	}
	podName := util.GetUniquePodName(pod)
	seLinuxContainerContexts := []*v1.SELinuxOptions{&seLinux}

	// Act
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */, seLinuxContainerContexts)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsDsw(t, generatedVolumeName, "system_u:object_r:container_file_t:s0:c1,c2", dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolumeName, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, podName, generatedVolumeName, "system_u:object_r:container_file_t:s0:c1,c2", dsw)
	verifyVolumeExistsWithSpecNameInVolumeDsw(t, podName, volumeSpec.Name(), dsw)

	// Arrange: prepare a different pod with the same context
	pod2 := pod.DeepCopy()
	pod2.Name = "pod2"
	pod2.UID = "pod2uid"
	pod2Name := util.GetUniquePodName(pod)

	// Act
	generatedVolumeName2, err := dsw.AddPodToVolume(
		pod2Name, pod2, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */, seLinuxContainerContexts)
	// Assert
	if err != nil {
		t.Fatalf("Second AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}
	if generatedVolumeName2 != generatedVolumeName {
		t.Errorf("Expected second generatedVolumeName %s, got %s", generatedVolumeName, generatedVolumeName2)
	}

	verifyPodExistsInVolumeDsw(t, pod2Name, generatedVolumeName, "system_u:object_r:container_file_t:s0:c1,c2", dsw)
}

// Calls AddPodToVolume() twice to add two pods with different SELinuxContext
// to the same ReadWriteOncePod PV.
// Verifies newly added pod/volume exists via PodExistsInVolume()
// VolumeExists() and GetVolumesToMount() and no errors.
func Test_AddPodToVolume_Negative_ExistingPodDifferentSELinuxRWOP(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SELinuxMountReadWriteOncePod, true)
	// Arrange
	plugins := []volume.VolumePlugin{
		&volumetesting.FakeDeviceMountableVolumePlugin{
			FakeBasicVolumePlugin: volumetesting.FakeBasicVolumePlugin{
				Plugin: volumetesting.FakeVolumePlugin{
					PluginName:      "basic",
					SupportsSELinux: true,
				},
			},
		},
	}
	volumePluginMgr := volume.VolumePluginMgr{}
	fakeVolumeHost := volumetesting.NewFakeVolumeHost(t,
		"",  /* rootDir */
		nil, /* kubeClient */
		nil, /* plugins */
	)
	volumePluginMgr.InitPlugins(plugins, nil /* prober */, fakeVolumeHost)
	seLinuxTranslator := util.NewFakeSELinuxLabelTranslator()
	dsw := NewDesiredStateOfWorld(&volumePluginMgr, seLinuxTranslator)
	seLinux1 := v1.SELinuxOptions{
		User:  "system_u",
		Role:  "object_r",
		Type:  "container_t",
		Level: "s0:c1,c2",
	}
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: "pod1",
			UID:  "pod1uid",
		},
		Spec: v1.PodSpec{
			SecurityContext: &v1.PodSecurityContext{
				SELinuxOptions: &seLinux1,
			},
			Volumes: []v1.Volume{
				{
					Name: "volume-name",
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
							ClaimName: "myClaim",
						},
					},
				},
			},
		},
	}

	volumeSpec := &volume.Spec{
		PersistentVolume: &v1.PersistentVolume{
			ObjectMeta: metav1.ObjectMeta{
				Name: "basicPV",
			},
			Spec: v1.PersistentVolumeSpec{
				AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOncePod},
			},
		},
	}
	podName := util.GetUniquePodName(pod)
	seLinuxContainerContexts := []*v1.SELinuxOptions{&seLinux1}

	// Act
	generatedVolumeName, err := dsw.AddPodToVolume(
		podName, pod, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */, seLinuxContainerContexts)

	// Assert
	if err != nil {
		t.Fatalf("AddPodToVolume failed. Expected: <no error> Actual: <%v>", err)
	}

	verifyVolumeExistsDsw(t, generatedVolumeName, "system_u:object_r:container_file_t:s0:c1,c2", dsw)
	verifyVolumeExistsInVolumesToMount(
		t, generatedVolumeName, false /* expectReportedInUse */, dsw)
	verifyPodExistsInVolumeDsw(t, podName, generatedVolumeName, "system_u:object_r:container_file_t:s0:c1,c2", dsw)
	verifyVolumeExistsWithSpecNameInVolumeDsw(t, podName, volumeSpec.Name(), dsw)

	// Arrange: prepare a different pod with the same context
	seLinux2 := v1.SELinuxOptions{
		User:  "system_u",
		Role:  "object_r",
		Type:  "container_t",
		Level: "s0:c3,c4",
	}
	seLinuxContainerContexts2 := []*v1.SELinuxOptions{&seLinux2}
	pod2 := pod.DeepCopy()
	pod2.Name = "pod2"
	pod2.UID = "pod2uid"
	pod2.Spec.SecurityContext.SELinuxOptions = &seLinux2
	pod2Name := util.GetUniquePodName(pod2)

	// Act
	_, err = dsw.AddPodToVolume(
		pod2Name, pod2, volumeSpec, volumeSpec.Name(), "" /* volumeGidValue */, seLinuxContainerContexts2)
	// Assert
	if err == nil {
		t.Fatalf("Second AddPodToVolume succeeded, expected a failure")
	}
	// Verify the original SELinux context is still in DSW
	verifyPodExistsInVolumeDsw(t, podName, generatedVolumeName, "system_u:object_r:container_file_t:s0:c1,c2", dsw)
}

func verifyVolumeExistsDsw(
	t *testing.T, expectedVolumeName v1.UniqueVolumeName, expectedSELinuxContext string, dsw DesiredStateOfWorld) {
	volumeExists := dsw.VolumeExists(expectedVolumeName, expectedSELinuxContext)
	if !volumeExists {
		t.Fatalf(
			"VolumeExists(%q) failed. Expected: <true> Actual: <%v>",
			expectedVolumeName,
			volumeExists)
	}
}

func verifyVolumeDoesntExist(
	t *testing.T, expectedVolumeName v1.UniqueVolumeName, expectedSELinuxContext string, dsw DesiredStateOfWorld) {
	volumeExists := dsw.VolumeExists(expectedVolumeName, expectedSELinuxContext)
	if volumeExists {
		t.Fatalf(
			"VolumeExists(%q) returned incorrect value. Expected: <false> Actual: <%v>",
			expectedVolumeName,
			volumeExists)
	}
}

func verifyVolumeExistsInVolumesToMount(
	t *testing.T,
	expectedVolumeName v1.UniqueVolumeName,
	expectReportedInUse bool,
	dsw DesiredStateOfWorld) {
	volumesToMount := dsw.GetVolumesToMount()
	for _, volume := range volumesToMount {
		if volume.VolumeName == expectedVolumeName {
			if volume.ReportedInUse != expectReportedInUse {
				t.Fatalf(
					"Found volume %v in the list of VolumesToMount, but ReportedInUse incorrect. Expected: <%v> Actual: <%v>",
					expectedVolumeName,
					expectReportedInUse,
					volume.ReportedInUse)
			}

			return
		}
	}

	t.Fatalf(
		"Could not find volume %v in the list of desired state of world volumes to mount %+v",
		expectedVolumeName,
		volumesToMount)
}

func verifyVolumeDoesntExistInVolumesToMount(
	t *testing.T, volumeToCheck v1.UniqueVolumeName, dsw DesiredStateOfWorld) {
	volumesToMount := dsw.GetVolumesToMount()
	for _, volume := range volumesToMount {
		if volume.VolumeName == volumeToCheck {
			t.Fatalf(
				"Found volume %v in the list of desired state of world volumes to mount. Expected it not to exist.",
				volumeToCheck)
		}
	}
}

func verifyPodExistsInVolumeDsw(
	t *testing.T,
	expectedPodName volumetypes.UniquePodName,
	expectedVolumeName v1.UniqueVolumeName,
	expectedSeLinuxContext string,
	dsw DesiredStateOfWorld) {
	if podExistsInVolume := dsw.PodExistsInVolume(
		expectedPodName, expectedVolumeName, expectedSeLinuxContext); !podExistsInVolume {
		t.Fatalf(
			"DSW PodExistsInVolume returned incorrect value. Expected: <true> Actual: <%v>",
			podExistsInVolume)
	}
}

func verifyPodDoesntExistInVolumeDsw(
	t *testing.T,
	expectedPodName volumetypes.UniquePodName,
	expectedVolumeName v1.UniqueVolumeName,
	expectedSeLinuxContext string,
	dsw DesiredStateOfWorld) {
	if podExistsInVolume := dsw.PodExistsInVolume(
		expectedPodName, expectedVolumeName, expectedSeLinuxContext); podExistsInVolume {
		t.Fatalf(
			"DSW PodExistsInVolume returned incorrect value. Expected: <true> Actual: <%v>",
			podExistsInVolume)
	}
}

func verifyVolumeExistsWithSpecNameInVolumeDsw(
	t *testing.T,
	expectedPodName volumetypes.UniquePodName,
	expectedVolumeSpecName string,
	dsw DesiredStateOfWorld) {
	if podExistsInVolume := dsw.VolumeExistsWithSpecName(
		expectedPodName, expectedVolumeSpecName); !podExistsInVolume {
		t.Fatalf(
			"DSW VolumeExistsWithSpecName returned incorrect value. Expected: <true> Actual: <%v>",
			podExistsInVolume)
	}
}

func verifyVolumeDoesntExistWithSpecNameInVolumeDsw(
	t *testing.T,
	expectedPodName volumetypes.UniquePodName,
	expectedVolumeSpecName string,
	dsw DesiredStateOfWorld) {
	if podExistsInVolume := dsw.VolumeExistsWithSpecName(
		expectedPodName, expectedVolumeSpecName); podExistsInVolume {
		t.Fatalf(
			"DSW VolumeExistsWithSpecName returned incorrect value. Expected: <true> Actual: <%v>",
			podExistsInVolume)
	}
}

func verifyDesiredSizeLimitInVolumeDsw(
	t *testing.T,
	expectedPodName volumetypes.UniquePodName,
	expectedDesiredSizeMap map[string]*resource.Quantity,
	dsw DesiredStateOfWorld) {
	volumesToMount := dsw.GetVolumesToMount()
	for volumeName, expectedDesiredSize := range expectedDesiredSizeMap {
		if podExistsInVolume := dsw.VolumeExistsWithSpecName(
			expectedPodName, volumeName); !podExistsInVolume {
			t.Fatalf(
				"DSW VolumeExistsWithSpecName returned incorrect value. Expected: <true> Actual: <%v>",
				podExistsInVolume)
		}
		for _, v := range volumesToMount {
			if v.VolumeSpec.Name() == volumeName && v.PodName == expectedPodName {
				if v.DesiredSizeLimit == nil || v.DesiredSizeLimit.Value() != expectedDesiredSize.Value() {
					t.Fatalf(
						"Found volume %v in the list of VolumesToMount, but DesiredSizeLimit incorrect. Expected: <%v> Actual: <%v>",
						volumeName,
						expectedDesiredSize,
						v.DesiredSizeLimit)

				}
			}
		}
	}
}
