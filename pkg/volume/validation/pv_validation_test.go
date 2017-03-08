package validation

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/volume"
)

func TestValidatePersistentVolumes(t *testing.T) {
	scenarios := map[string]struct {
		isExpectedFailure bool
		volume            *api.PersistentVolume
	}{
		"volume with valid mount option for nfs": {
			isExpectedFailure: false,
			volume: testVolumeWithMountOption("good-nfs-mount-volume", "", "ro,nfsvers=3", api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
				PersistentVolumeSource: api.PersistentVolumeSource{
					NFS: &api.NFSVolumeSource{Server: "localhost", Path: "/srv", ReadOnly: false},
				},
			}),
		},
		"volume with mount option for host path": {
			isExpectedFailure: true,
			volume: testVolumeWithMountOption("bad-hostpath-mount-volume", "", "ro,nfsvers=3", api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
				PersistentVolumeSource: api.PersistentVolumeSource{
					HostPath: &api.HostPathVolumeSource{Path: "/a/.."},
				},
			}),
		},
	}

	for name, scenario := range scenarios {
		errs := ValidateForMountOptions(scenario.volume)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}

func testVolumeWithMountOption(name string, namespace string, mountOptions string, spec api.PersistentVolumeSpec) *api.PersistentVolume {
	annotations := map[string]string{
		volume.MountOptionAnnotation: mountOptions,
	}
	objMeta := metav1.ObjectMeta{
		Name:        name,
		Annotations: annotations,
	}

	if namespace != "" {
		objMeta.Namespace = namespace
	}

	return &api.PersistentVolume{
		ObjectMeta: objMeta,
		Spec:       spec,
	}
}
