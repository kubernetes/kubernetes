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

package set

import (
	"errors"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/fake"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
	"k8s.io/kubernetes/pkg/kubectl/scheme"
)

func fakePodWithVol() *v1.Pod {
	fakePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "fakepod",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "fake-container",
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "fake-mount",
							MountPath: "/var/www/html",
						},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name: "fake-mount",
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{
							Path: "/var/www/html",
						},
					},
				},
			},
		},
	}
	return fakePod
}

func makeFakePod() *v1.Pod {
	fakePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "default",
			Name:      "fakepod",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name: "fake-container",
				},
			},
		},
	}
	return fakePod
}

func getFakeMapping() *meta.RESTMapping {
	fakeMapping := &meta.RESTMapping{
		Resource: "fake-mount",
		GroupVersionKind: schema.GroupVersionKind{
			Group:   "",
			Version: "v1",
		},
		ObjectConvertor: scheme.Scheme,
	}
	return fakeMapping
}

func getFakeInfo(podInfo *v1.Pod) ([]*resource.Info, *VolumeOptions) {
	f := cmdutil.NewFactory(nil)
	fakeMapping := getFakeMapping()
	info := &resource.Info{
		Client:    fake.NewSimpleClientset().Core().RESTClient(),
		Mapping:   fakeMapping,
		Namespace: "default",
		Name:      "fakepod",
		Object:    podInfo,
	}
	infos := []*resource.Info{info}
	vOptions := &VolumeOptions{}
	vOptions.Name = "fake-mount"
	vOptions.Encoder = scheme.Codecs.LegacyCodec(scheme.Registry.EnabledVersions()...)
	vOptions.Containers = "*"
	vOptions.UpdatePodSpecForObject = f.UpdatePodSpecForObject
	return infos, vOptions
}

func TestRemoveVolume(t *testing.T) {
	fakePod := fakePodWithVol()
	addOpts := &AddVolumeOptions{}
	infos, vOptions := getFakeInfo(fakePod)
	vOptions.AddOpts = addOpts
	vOptions.Remove = true
	vOptions.Confirm = true

	patches, patchError := vOptions.getVolumeUpdatePatches(infos, false)
	if len(patches) < 1 {
		t.Errorf("Expected at least 1 patch object")
	}
	updatedInfo := patches[0].Info
	podObject, ok := updatedInfo.VersionedObject.(*v1.Pod)
	if !ok {
		t.Errorf("Expected pod info to be updated")
	}

	updatedPodSpec := podObject.Spec
	if len(updatedPodSpec.Volumes) > 0 {
		t.Errorf("Expected volume to be removed")
	}

	if patchError != nil {
		t.Error(patchError)
	}
}

func TestAddVolume(t *testing.T) {
	fakePod := makeFakePod()
	addOpts := &AddVolumeOptions{}
	infos, vOptions := getFakeInfo(fakePod)
	vOptions.AddOpts = addOpts
	vOptions.Add = true
	addOpts.Type = "emptyDir"
	addOpts.MountPath = "/var/www/html"

	patches, patchError := vOptions.getVolumeUpdatePatches(infos, false)
	if len(patches) < 1 {
		t.Errorf("Expected at least 1 patch object")
	}
	updatedInfo := patches[0].Info
	podObject, ok := updatedInfo.VersionedObject.(*v1.Pod)

	if !ok {
		t.Errorf("Expected pod info to be updated")
	}

	updatedPodSpec := podObject.Spec

	if len(updatedPodSpec.Volumes) < 1 {
		t.Errorf("Expected volume to be added")
	}

	if patchError != nil {
		t.Error(patchError)
	}
}

func TestCreateClaim(t *testing.T) {
	addOpts := &AddVolumeOptions{
		Type:       "persistentVolumeClaim",
		ClaimClass: "foobar",
		ClaimName:  "foo-vol",
		ClaimSize:  "5G",
		MountPath:  "/sandbox",
	}

	pvc := addOpts.createClaim()
	if len(pvc.Annotations) == 0 {
		t.Errorf("Expected storage class annotation")
	}

	if pvc.Annotations[v1.BetaStorageClassAnnotation] != "foobar" {
		t.Errorf("Expected storage annotated class to be %s", addOpts.ClaimClass)
	}
}

func TestValidateAddOptions(t *testing.T) {
	tests := []struct {
		name          string
		addOpts       *AddVolumeOptions
		expectedError error
	}{
		{
			"using existing pvc",
			&AddVolumeOptions{Type: "persistentvolumeclaim"},
			errors.New("must provide --claim-name or --claim-size (to create a new claim) for --type=pvc"),
		},
		{
			"creating new pvc",
			&AddVolumeOptions{Type: "persistentvolumeclaim", ClaimName: "sandbox-pvc", ClaimSize: "5G"},
			nil,
		},
		{
			"error creating pvc with storage class",
			&AddVolumeOptions{Type: "persistentvolumeclaim", ClaimName: "sandbox-pvc", ClaimClass: "slow"},
			nil,
		},
		{
			"creating pvc with storage class",
			&AddVolumeOptions{Type: "persistentvolumeclaim", ClaimName: "sandbox-pvc", ClaimClass: "slow", ClaimSize: "5G"},
			nil,
		},
		{
			"creating secret with good default-mode",
			&AddVolumeOptions{Type: "secret", SecretName: "sandbox-pv", DefaultMode: "0644"},
			nil,
		},
		{
			"creating secret with good default-mode, three number variant",
			&AddVolumeOptions{Type: "secret", SecretName: "sandbox-pv", DefaultMode: "777"},
			nil,
		},
		{
			"creating secret with bad default-mode, bad bits",
			&AddVolumeOptions{Type: "secret", SecretName: "sandbox-pv", DefaultMode: "0888"},
			errors.New("--default-mode must be between 0000 and 0777"),
		},
		{
			"creating secret with bad default-mode, too long",
			&AddVolumeOptions{Type: "secret", SecretName: "sandbox-pv", DefaultMode: "07777"},
			errors.New("--default-mode must be between 0000 and 0777"),
		},
		{
			"creating configmap with good default-mode",
			&AddVolumeOptions{Type: "configmap", ConfigMapName: "sandbox-pv", DefaultMode: "0644"},
			nil,
		},
		{
			"creating configmap with good default-mode, three number variant",
			&AddVolumeOptions{Type: "configmap", ConfigMapName: "sandbox-pv", DefaultMode: "777"},
			nil,
		},
		{
			"creating configmap with bad default-mode, bad bits",
			&AddVolumeOptions{Type: "configmap", ConfigMapName: "sandbox-pv", DefaultMode: "0888"},
			errors.New("--default-mode must be between 0000 and 0777"),
		},
		{
			"creating configmap with bad default-mode, too long",
			&AddVolumeOptions{Type: "configmap", ConfigMapName: "sandbox-pv", DefaultMode: "07777"},
			errors.New("--default-mode must be between 0000 and 0777"),
		},
	}

	for _, testCase := range tests {
		addOpts := testCase.addOpts
		err := addOpts.Validate(true)
		if testCase.expectedError == nil && err != nil {
			t.Errorf("Expected nil error for %s got %s", testCase.name, err)
			continue
		}

		if testCase.expectedError != nil {
			if err == nil {
				t.Errorf("Expected %s, got nil", testCase.expectedError)
				continue
			}

			if testCase.expectedError.Error() != err.Error() {
				t.Errorf("Expected %s, got %s", testCase.expectedError, err)
			}
		}

	}
}
