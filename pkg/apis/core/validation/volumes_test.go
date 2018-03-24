/*
Copyright 2014 The Kubernetes Authors.

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

package validation

import (
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	_ "k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/apis/core"
)

const (
	dnsLabelErrMsg          = "a DNS-1123 label must consist of"
	dnsSubdomainLabelErrMsg = "a DNS-1123 subdomain"
	envVarNameErrMsg        = "a valid environment variable name must consist of"
)

func TestValidatePersistentVolumes(t *testing.T) {
	validMode := core.PersistentVolumeFilesystem
	scenarios := map[string]struct {
		isExpectedFailure bool
		volume            *core.PersistentVolume
	}{
		"good-volume": {
			isExpectedFailure: false,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		"good-volume-with-capacity-unit": {
			isExpectedFailure: false,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10Gi"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		"good-volume-without-capacity-unit": {
			isExpectedFailure: false,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		"good-volume-with-storage-class": {
			isExpectedFailure: false,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
				StorageClassName: "valid",
			}),
		},
		"good-volume-with-retain-policy": {
			isExpectedFailure: false,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
				PersistentVolumeReclaimPolicy: core.PersistentVolumeReclaimRetain,
			}),
		},
		"invalid-accessmode": {
			isExpectedFailure: true,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{"fakemode"},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		"invalid-reclaimpolicy": {
			isExpectedFailure: true,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
				PersistentVolumeReclaimPolicy: "fakeReclaimPolicy",
			}),
		},
		"unexpected-namespace": {
			isExpectedFailure: true,
			volume: testVolume("foo", "unexpected-namespace", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		"missing-volume-source": {
			isExpectedFailure: true,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
			}),
		},
		"bad-name": {
			isExpectedFailure: true,
			volume: testVolume("123*Bad(Name", "unexpected-namespace", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		"missing-name": {
			isExpectedFailure: true,
			volume: testVolume("", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		"missing-capacity": {
			isExpectedFailure: true,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		"bad-volume-zero-capacity": {
			isExpectedFailure: true,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("0"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		"missing-accessmodes": {
			isExpectedFailure: true,
			volume: testVolume("goodname", "missing-accessmodes", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		"too-many-sources": {
			isExpectedFailure: true,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("5G"),
				},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
					GCEPersistentDisk: &core.GCEPersistentDiskVolumeSource{PDName: "foo", FSType: "ext4"},
				},
			}),
		},
		"host mount of / with recycle reclaim policy": {
			isExpectedFailure: true,
			volume: testVolume("bad-recycle-do-not-want", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
				PersistentVolumeReclaimPolicy: core.PersistentVolumeReclaimRecycle,
			}),
		},
		"host mount of / with recycle reclaim policy 2": {
			isExpectedFailure: true,
			volume: testVolume("bad-recycle-do-not-want", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/a/..",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
				PersistentVolumeReclaimPolicy: core.PersistentVolumeReclaimRecycle,
			}),
		},
		"invalid-storage-class-name": {
			isExpectedFailure: true,
			volume: testVolume("invalid-storage-class-name", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
				StorageClassName: "-invalid-",
			}),
		},
		// VolumeMode alpha feature disabled
		// TODO: remove when no longer alpha
		"alpha disabled valid volume mode": {
			isExpectedFailure: true,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
				StorageClassName: "valid",
				VolumeMode:       &validMode,
			}),
		},
		"bad-hostpath-volume-backsteps": {
			isExpectedFailure: true,
			volume: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo/..",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
				StorageClassName: "backstep-hostpath",
			}),
		},
		"volume-node-affinity": {
			isExpectedFailure: false,
			volume:            testVolumeWithNodeAffinity(simpleVolumeNodeAffinity("foo", "bar")),
		},
		"volume-empty-node-affinity": {
			isExpectedFailure: true,
			volume:            testVolumeWithNodeAffinity(&core.VolumeNodeAffinity{}),
		},
		"volume-bad-node-affinity": {
			isExpectedFailure: true,
			volume: testVolumeWithNodeAffinity(
				&core.VolumeNodeAffinity{
					Required: &core.NodeSelector{
						NodeSelectorTerms: []core.NodeSelectorTerm{
							{
								MatchExpressions: []core.NodeSelectorRequirement{
									{
										Operator: core.NodeSelectorOpIn,
										Values:   []string{"test-label-value"},
									},
								},
							},
						},
					},
				}),
		},
	}

	for name, scenario := range scenarios {
		errs := ValidatePersistentVolume(scenario.volume)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}

}

func TestValidatePersistentVolumeSourceUpdate(t *testing.T) {
	validVolume := testVolume("foo", "", core.PersistentVolumeSpec{
		Capacity: core.ResourceList{
			core.ResourceName(core.ResourceStorage): resource.MustParse("1G"),
		},
		AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
		PersistentVolumeSource: core.PersistentVolumeSource{
			HostPath: &core.HostPathVolumeSource{
				Path: "/foo",
				Type: newHostPathType(string(core.HostPathDirectory)),
			},
		},
		StorageClassName: "valid",
	})
	validPvSourceNoUpdate := validVolume.DeepCopy()
	invalidPvSourceUpdateType := validVolume.DeepCopy()
	invalidPvSourceUpdateType.Spec.PersistentVolumeSource = core.PersistentVolumeSource{
		FlexVolume: &core.FlexPersistentVolumeSource{
			Driver: "kubernetes.io/blue",
			FSType: "ext4",
		},
	}
	invalidPvSourceUpdateDeep := validVolume.DeepCopy()
	invalidPvSourceUpdateDeep.Spec.PersistentVolumeSource = core.PersistentVolumeSource{
		HostPath: &core.HostPathVolumeSource{
			Path: "/updated",
			Type: newHostPathType(string(core.HostPathDirectory)),
		},
	}
	scenarios := map[string]struct {
		isExpectedFailure bool
		oldVolume         *core.PersistentVolume
		newVolume         *core.PersistentVolume
	}{
		"condition-no-update": {
			isExpectedFailure: false,
			oldVolume:         validVolume,
			newVolume:         validPvSourceNoUpdate,
		},
		"condition-update-source-type": {
			isExpectedFailure: true,
			oldVolume:         validVolume,
			newVolume:         invalidPvSourceUpdateType,
		},
		"condition-update-source-deep": {
			isExpectedFailure: true,
			oldVolume:         validVolume,
			newVolume:         invalidPvSourceUpdateDeep,
		},
	}
	for name, scenario := range scenarios {
		errs := ValidatePersistentVolumeUpdate(scenario.newVolume, scenario.oldVolume)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}

func testLocalVolume(path string, affinity *core.VolumeNodeAffinity) core.PersistentVolumeSpec {
	return core.PersistentVolumeSpec{
		Capacity: core.ResourceList{
			core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
		},
		AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
		PersistentVolumeSource: core.PersistentVolumeSource{
			Local: &core.LocalVolumeSource{
				Path: path,
			},
		},
		NodeAffinity:     affinity,
		StorageClassName: "test-storage-class",
	}
}

func TestValidateLocalVolumes(t *testing.T) {
	scenarios := map[string]struct {
		isExpectedFailure bool
		volume            *core.PersistentVolume
	}{
		"alpha invalid local volume nil annotations": {
			isExpectedFailure: true,
			volume: testVolume(
				"invalid-local-volume-nil-annotations",
				"",
				testLocalVolume("/foo", nil)),
		},
		"valid local volume": {
			isExpectedFailure: false,
			volume: testVolume("valid-local-volume", "",
				testLocalVolume("/foo", simpleVolumeNodeAffinity("foo", "bar"))),
		},
		"invalid local volume no node affinity": {
			isExpectedFailure: true,
			volume: testVolume("invalid-local-volume-no-node-affinity", "",
				testLocalVolume("/foo", nil)),
		},
		"invalid local volume empty path": {
			isExpectedFailure: true,
			volume: testVolume("invalid-local-volume-empty-path", "",
				testLocalVolume("", simpleVolumeNodeAffinity("foo", "bar"))),
		},
		"invalid-local-volume-backsteps": {
			isExpectedFailure: true,
			volume: testVolume("foo", "",
				testLocalVolume("/foo/..", simpleVolumeNodeAffinity("foo", "bar"))),
		},
		"invalid-local-volume-relative-path": {
			isExpectedFailure: true,
			volume: testVolume("foo", "",
				testLocalVolume("foo", simpleVolumeNodeAffinity("foo", "bar"))),
		},
	}

	for name, scenario := range scenarios {
		errs := ValidatePersistentVolume(scenario.volume)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}

func TestValidateLocalVolumesDisabled(t *testing.T) {
	scenarios := map[string]struct {
		isExpectedFailure bool
		volume            *core.PersistentVolume
	}{
		"feature disabled valid local volume": {
			isExpectedFailure: true,
			volume: testVolume("valid-local-volume", "",
				testLocalVolume("/foo", simpleVolumeNodeAffinity("foo", "bar"))),
		},
	}

	utilfeature.DefaultFeatureGate.Set("PersistentLocalVolumes=false")
	for name, scenario := range scenarios {
		errs := ValidatePersistentVolume(scenario.volume)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
	utilfeature.DefaultFeatureGate.Set("PersistentLocalVolumes=true")

	utilfeature.DefaultFeatureGate.Set("VolumeScheduling=false")
	for name, scenario := range scenarios {
		errs := ValidatePersistentVolume(scenario.volume)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
	utilfeature.DefaultFeatureGate.Set("VolumeScheduling=true")
}

func testVolumeWithNodeAffinity(affinity *core.VolumeNodeAffinity) *core.PersistentVolume {
	return testVolume("test-affinity-volume", "",
		core.PersistentVolumeSpec{
			Capacity: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
			AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
			PersistentVolumeSource: core.PersistentVolumeSource{
				GCEPersistentDisk: &core.GCEPersistentDiskVolumeSource{
					PDName: "foo",
				},
			},
			StorageClassName: "test-storage-class",
			NodeAffinity:     affinity,
		})
}

func simpleVolumeNodeAffinity(key, value string) *core.VolumeNodeAffinity {
	return &core.VolumeNodeAffinity{
		Required: &core.NodeSelector{
			NodeSelectorTerms: []core.NodeSelectorTerm{
				{
					MatchExpressions: []core.NodeSelectorRequirement{
						{
							Key:      key,
							Operator: core.NodeSelectorOpIn,
							Values:   []string{value},
						},
					},
				},
			},
		},
	}
}

func TestValidateVolumeNodeAffinityUpdate(t *testing.T) {
	scenarios := map[string]struct {
		isExpectedFailure bool
		oldPV             *core.PersistentVolume
		newPV             *core.PersistentVolume
	}{
		"nil-nothing-changed": {
			isExpectedFailure: false,
			oldPV:             testVolumeWithNodeAffinity(nil),
			newPV:             testVolumeWithNodeAffinity(nil),
		},
		"affinity-nothing-changed": {
			isExpectedFailure: false,
			oldPV:             testVolumeWithNodeAffinity(simpleVolumeNodeAffinity("foo", "bar")),
			newPV:             testVolumeWithNodeAffinity(simpleVolumeNodeAffinity("foo", "bar")),
		},
		"affinity-changed": {
			isExpectedFailure: true,
			oldPV:             testVolumeWithNodeAffinity(simpleVolumeNodeAffinity("foo", "bar")),
			newPV:             testVolumeWithNodeAffinity(simpleVolumeNodeAffinity("foo", "bar2")),
		},
		"nil-to-obj": {
			isExpectedFailure: false,
			oldPV:             testVolumeWithNodeAffinity(nil),
			newPV:             testVolumeWithNodeAffinity(simpleVolumeNodeAffinity("foo", "bar")),
		},
		"obj-to-nil": {
			isExpectedFailure: true,
			oldPV:             testVolumeWithNodeAffinity(simpleVolumeNodeAffinity("foo", "bar")),
			newPV:             testVolumeWithNodeAffinity(nil),
		},
	}

	for name, scenario := range scenarios {
		errs := ValidatePersistentVolumeUpdate(scenario.newPV, scenario.oldPV)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}

func testVolumeClaim(name string, namespace string, spec core.PersistentVolumeClaimSpec) *core.PersistentVolumeClaim {
	return &core.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec:       spec,
	}
}

func testVolumeClaimWithStatus(
	name, namespace string,
	spec core.PersistentVolumeClaimSpec,
	status core.PersistentVolumeClaimStatus) *core.PersistentVolumeClaim {
	return &core.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec:       spec,
		Status:     status,
	}
}

func testVolumeClaimStorageClass(name string, namespace string, annval string, spec core.PersistentVolumeClaimSpec) *core.PersistentVolumeClaim {
	annotations := map[string]string{
		v1.BetaStorageClassAnnotation: annval,
	}

	return &core.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Annotations: annotations,
		},
		Spec: spec,
	}
}

func testVolumeClaimAnnotation(name string, namespace string, ann string, annval string, spec core.PersistentVolumeClaimSpec) *core.PersistentVolumeClaim {
	annotations := map[string]string{
		ann: annval,
	}

	return &core.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Annotations: annotations,
		},
		Spec: spec,
	}
}

func testVolumeClaimStorageClassInSpec(name, namespace, scName string, spec core.PersistentVolumeClaimSpec) *core.PersistentVolumeClaim {
	spec.StorageClassName = &scName
	return &core.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: spec,
	}
}

func testVolumeClaimStorageClassInAnnotationAndSpec(name, namespace, scNameInAnn, scName string, spec core.PersistentVolumeClaimSpec) *core.PersistentVolumeClaim {
	spec.StorageClassName = &scName
	return &core.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   namespace,
			Annotations: map[string]string{v1.BetaStorageClassAnnotation: scNameInAnn},
		},
		Spec: spec,
	}
}

func TestValidatePersistentVolumeClaim(t *testing.T) {
	invalidClassName := "-invalid-"
	validClassName := "valid"
	validMode := core.PersistentVolumeFilesystem
	scenarios := map[string]struct {
		isExpectedFailure bool
		claim             *core.PersistentVolumeClaim
	}{
		"good-claim": {
			isExpectedFailure: false,
			claim: testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
				Selector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "key2",
							Operator: "Exists",
						},
					},
				},
				AccessModes: []core.PersistentVolumeAccessMode{
					core.ReadWriteOnce,
					core.ReadOnlyMany,
				},
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
					},
				},
				StorageClassName: &validClassName,
			}),
		},
		"invalid-claim-zero-capacity": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
				Selector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "key2",
							Operator: "Exists",
						},
					},
				},
				AccessModes: []core.PersistentVolumeAccessMode{
					core.ReadWriteOnce,
					core.ReadOnlyMany,
				},
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceStorage): resource.MustParse("0G"),
					},
				},
				StorageClassName: &validClassName,
			}),
		},
		"invalid-label-selector": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
				Selector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "key2",
							Operator: "InvalidOp",
							Values:   []string{"value1", "value2"},
						},
					},
				},
				AccessModes: []core.PersistentVolumeAccessMode{
					core.ReadWriteOnce,
					core.ReadOnlyMany,
				},
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
					},
				},
			}),
		},
		"invalid-accessmode": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
				AccessModes: []core.PersistentVolumeAccessMode{"fakemode"},
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
					},
				},
			}),
		},
		"missing-namespace": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "", core.PersistentVolumeClaimSpec{
				AccessModes: []core.PersistentVolumeAccessMode{
					core.ReadWriteOnce,
					core.ReadOnlyMany,
				},
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
					},
				},
			}),
		},
		"no-access-modes": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
					},
				},
			}),
		},
		"no-resource-requests": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
				AccessModes: []core.PersistentVolumeAccessMode{
					core.ReadWriteOnce,
				},
			}),
		},
		"invalid-resource-requests": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
				AccessModes: []core.PersistentVolumeAccessMode{
					core.ReadWriteOnce,
				},
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					},
				},
			}),
		},
		"negative-storage-request": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
				Selector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "key2",
							Operator: "Exists",
						},
					},
				},
				AccessModes: []core.PersistentVolumeAccessMode{
					core.ReadWriteOnce,
					core.ReadOnlyMany,
				},
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceStorage): resource.MustParse("-10G"),
					},
				},
			}),
		},
		"zero-storage-request": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
				Selector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "key2",
							Operator: "Exists",
						},
					},
				},
				AccessModes: []core.PersistentVolumeAccessMode{
					core.ReadWriteOnce,
					core.ReadOnlyMany,
				},
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceStorage): resource.MustParse("0G"),
					},
				},
			}),
		},
		"invalid-storage-class-name": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
				Selector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "key2",
							Operator: "Exists",
						},
					},
				},
				AccessModes: []core.PersistentVolumeAccessMode{
					core.ReadWriteOnce,
					core.ReadOnlyMany,
				},
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
					},
				},
				StorageClassName: &invalidClassName,
			}),
		},
		// VolumeMode alpha feature disabled
		// TODO: remove when no longer alpha
		"disabled alpha valid volume mode": {
			isExpectedFailure: true,
			claim: testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
				Selector: &metav1.LabelSelector{
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "key2",
							Operator: "Exists",
						},
					},
				},
				AccessModes: []core.PersistentVolumeAccessMode{
					core.ReadWriteOnce,
					core.ReadOnlyMany,
				},
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
					},
				},
				StorageClassName: &validClassName,
				VolumeMode:       &validMode,
			}),
		},
	}

	for name, scenario := range scenarios {
		errs := ValidatePersistentVolumeClaim(scenario.claim)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}

func TestAlphaPVVolumeModeUpdate(t *testing.T) {
	block := core.PersistentVolumeBlock
	file := core.PersistentVolumeFilesystem

	scenarios := map[string]struct {
		isExpectedFailure bool
		oldPV             *core.PersistentVolume
		newPV             *core.PersistentVolume
		enableBlock       bool
	}{
		"valid-update-volume-mode-block-to-block": {
			isExpectedFailure: false,
			oldPV:             createTestVolModePV(&block),
			newPV:             createTestVolModePV(&block),
			enableBlock:       true,
		},
		"valid-update-volume-mode-file-to-file": {
			isExpectedFailure: false,
			oldPV:             createTestVolModePV(&file),
			newPV:             createTestVolModePV(&file),
			enableBlock:       true,
		},
		"invalid-update-volume-mode-to-block": {
			isExpectedFailure: true,
			oldPV:             createTestVolModePV(&file),
			newPV:             createTestVolModePV(&block),
			enableBlock:       true,
		},
		"invalid-update-volume-mode-to-file": {
			isExpectedFailure: true,
			oldPV:             createTestVolModePV(&block),
			newPV:             createTestVolModePV(&file),
			enableBlock:       true,
		},
		"invalid-update-blocksupport-disabled": {
			isExpectedFailure: true,
			oldPV:             createTestVolModePV(&block),
			newPV:             createTestVolModePV(&block),
			enableBlock:       false,
		},
		"invalid-update-volume-mode-nil-to-file": {
			isExpectedFailure: true,
			oldPV:             createTestVolModePV(nil),
			newPV:             createTestVolModePV(&file),
			enableBlock:       true,
		},
		"invalid-update-volume-mode-nil-to-block": {
			isExpectedFailure: true,
			oldPV:             createTestVolModePV(nil),
			newPV:             createTestVolModePV(&block),
			enableBlock:       true,
		},
		"invalid-update-volume-mode-file-to-nil": {
			isExpectedFailure: true,
			oldPV:             createTestVolModePV(&file),
			newPV:             createTestVolModePV(nil),
			enableBlock:       true,
		},
		"invalid-update-volume-mode-block-to-nil": {
			isExpectedFailure: true,
			oldPV:             createTestVolModePV(&block),
			newPV:             createTestVolModePV(nil),
			enableBlock:       true,
		},
		"invalid-update-volume-mode-nil-to-nil": {
			isExpectedFailure: false,
			oldPV:             createTestVolModePV(nil),
			newPV:             createTestVolModePV(nil),
			enableBlock:       true,
		},
		"invalid-update-volume-mode-empty-to-mode": {
			isExpectedFailure: true,
			oldPV:             createTestPV(),
			newPV:             createTestVolModePV(&block),
			enableBlock:       true,
		},
	}

	for name, scenario := range scenarios {
		// ensure we have a resource version specified for updates
		toggleBlockVolumeFeature(scenario.enableBlock, t)
		errs := ValidatePersistentVolumeUpdate(scenario.newPV, scenario.oldPV)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}

func TestValidatePersistentVolumeClaimUpdate(t *testing.T) {
	block := core.PersistentVolumeBlock
	file := core.PersistentVolumeFilesystem

	validClaim := testVolumeClaimWithStatus("foo", "ns", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
	}, core.PersistentVolumeClaimStatus{
		Phase: core.ClaimBound,
	})

	validClaimStorageClass := testVolumeClaimStorageClass("foo", "ns", "fast", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
	})
	validClaimAnnotation := testVolumeClaimAnnotation("foo", "ns", "description", "foo-description", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
	})
	validUpdateClaim := testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
		VolumeName: "volume",
	})
	invalidUpdateClaimResources := testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("20G"),
			},
		},
		VolumeName: "volume",
	})
	invalidUpdateClaimAccessModes := testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
		VolumeName: "volume",
	})
	validClaimVolumeModeFile := testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
		},
		VolumeMode: &file,
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
		VolumeName: "volume",
	})
	validClaimVolumeModeBlock := testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
		},
		VolumeMode: &block,
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
		VolumeName: "volume",
	})
	invalidClaimVolumeModeNil := testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
		},
		VolumeMode: nil,
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
		VolumeName: "volume",
	})
	invalidUpdateClaimStorageClass := testVolumeClaimStorageClass("foo", "ns", "fast2", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
		VolumeName: "volume",
	})
	validUpdateClaimMutableAnnotation := testVolumeClaimAnnotation("foo", "ns", "description", "updated-or-added-foo-description", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
		VolumeName: "volume",
	})
	validAddClaimAnnotation := testVolumeClaimAnnotation("foo", "ns", "description", "updated-or-added-foo-description", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
		VolumeName: "volume",
	})
	validSizeUpdate := testVolumeClaimWithStatus("foo", "ns", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("15G"),
			},
		},
	}, core.PersistentVolumeClaimStatus{
		Phase: core.ClaimBound,
	})

	invalidSizeUpdate := testVolumeClaimWithStatus("foo", "ns", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("5G"),
			},
		},
	}, core.PersistentVolumeClaimStatus{
		Phase: core.ClaimBound,
	})

	unboundSizeUpdate := testVolumeClaimWithStatus("foo", "ns", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("12G"),
			},
		},
	}, core.PersistentVolumeClaimStatus{
		Phase: core.ClaimPending,
	})

	validClaimStorageClassInSpec := testVolumeClaimStorageClassInSpec("foo", "ns", "fast", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
	})

	invalidClaimStorageClassInSpec := testVolumeClaimStorageClassInSpec("foo", "ns", "fast2", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
	})

	validClaimStorageClassInAnnotationAndSpec := testVolumeClaimStorageClassInAnnotationAndSpec(
		"foo", "ns", "fast", "fast", core.PersistentVolumeClaimSpec{
			AccessModes: []core.PersistentVolumeAccessMode{
				core.ReadOnlyMany,
			},
			Resources: core.ResourceRequirements{
				Requests: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
			},
		})

	invalidClaimStorageClassInAnnotationAndSpec := testVolumeClaimStorageClassInAnnotationAndSpec(
		"foo", "ns", "fast2", "fast", core.PersistentVolumeClaimSpec{
			AccessModes: []core.PersistentVolumeAccessMode{
				core.ReadOnlyMany,
			},
			Resources: core.ResourceRequirements{
				Requests: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
			},
		})

	scenarios := map[string]struct {
		isExpectedFailure bool
		oldClaim          *core.PersistentVolumeClaim
		newClaim          *core.PersistentVolumeClaim
		enableResize      bool
		enableBlock       bool
	}{
		"valid-update-volumeName-only": {
			isExpectedFailure: false,
			oldClaim:          validClaim,
			newClaim:          validUpdateClaim,
			enableResize:      false,
			enableBlock:       false,
		},
		"valid-no-op-update": {
			isExpectedFailure: false,
			oldClaim:          validUpdateClaim,
			newClaim:          validUpdateClaim,
			enableResize:      false,
			enableBlock:       false,
		},
		"invalid-update-change-resources-on-bound-claim": {
			isExpectedFailure: true,
			oldClaim:          validUpdateClaim,
			newClaim:          invalidUpdateClaimResources,
			enableResize:      false,
			enableBlock:       false,
		},
		"invalid-update-change-access-modes-on-bound-claim": {
			isExpectedFailure: true,
			oldClaim:          validUpdateClaim,
			newClaim:          invalidUpdateClaimAccessModes,
			enableResize:      false,
			enableBlock:       false,
		},
		"valid-update-volume-mode-block-to-block": {
			isExpectedFailure: false,
			oldClaim:          validClaimVolumeModeBlock,
			newClaim:          validClaimVolumeModeBlock,
			enableResize:      false,
			enableBlock:       true,
		},
		"valid-update-volume-mode-file-to-file": {
			isExpectedFailure: false,
			oldClaim:          validClaimVolumeModeFile,
			newClaim:          validClaimVolumeModeFile,
			enableResize:      false,
			enableBlock:       true,
		},
		"invalid-update-volume-mode-to-block": {
			isExpectedFailure: true,
			oldClaim:          validClaimVolumeModeFile,
			newClaim:          validClaimVolumeModeBlock,
			enableResize:      false,
			enableBlock:       true,
		},
		"invalid-update-volume-mode-to-file": {
			isExpectedFailure: true,
			oldClaim:          validClaimVolumeModeBlock,
			newClaim:          validClaimVolumeModeFile,
			enableResize:      false,
			enableBlock:       true,
		},
		"invalid-update-volume-mode-nil-to-file": {
			isExpectedFailure: true,
			oldClaim:          invalidClaimVolumeModeNil,
			newClaim:          validClaimVolumeModeFile,
			enableResize:      false,
			enableBlock:       true,
		},
		"invalid-update-volume-mode-nil-to-block": {
			isExpectedFailure: true,
			oldClaim:          invalidClaimVolumeModeNil,
			newClaim:          validClaimVolumeModeBlock,
			enableResize:      false,
			enableBlock:       true,
		},
		"invalid-update-volume-mode-block-to-nil": {
			isExpectedFailure: true,
			oldClaim:          validClaimVolumeModeBlock,
			newClaim:          invalidClaimVolumeModeNil,
			enableResize:      false,
			enableBlock:       true,
		},
		"invalid-update-volume-mode-file-to-nil": {
			isExpectedFailure: true,
			oldClaim:          validClaimVolumeModeFile,
			newClaim:          invalidClaimVolumeModeNil,
			enableResize:      false,
			enableBlock:       true,
		},
		"invalid-update-volume-mode-empty-to-mode": {
			isExpectedFailure: true,
			oldClaim:          validClaim,
			newClaim:          validClaimVolumeModeBlock,
			enableResize:      false,
			enableBlock:       true,
		},
		"invalid-update-volume-mode-mode-to-empty": {
			isExpectedFailure: true,
			oldClaim:          validClaimVolumeModeBlock,
			newClaim:          validClaim,
			enableResize:      false,
			enableBlock:       true,
		},
		"invalid-update-blocksupport-disabled": {
			isExpectedFailure: true,
			oldClaim:          validClaimVolumeModeFile,
			newClaim:          validClaimVolumeModeFile,
			enableResize:      false,
			enableBlock:       false,
		},
		"invalid-update-change-storage-class-annotation-after-creation": {
			isExpectedFailure: true,
			oldClaim:          validClaimStorageClass,
			newClaim:          invalidUpdateClaimStorageClass,
			enableResize:      false,
			enableBlock:       false,
		},
		"valid-update-mutable-annotation": {
			isExpectedFailure: false,
			oldClaim:          validClaimAnnotation,
			newClaim:          validUpdateClaimMutableAnnotation,
			enableResize:      false,
			enableBlock:       false,
		},
		"valid-update-add-annotation": {
			isExpectedFailure: false,
			oldClaim:          validClaim,
			newClaim:          validAddClaimAnnotation,
			enableResize:      false,
			enableBlock:       false,
		},
		"valid-size-update-resize-disabled": {
			isExpectedFailure: true,
			oldClaim:          validClaim,
			newClaim:          validSizeUpdate,
			enableResize:      false,
			enableBlock:       false,
		},
		"valid-size-update-resize-enabled": {
			isExpectedFailure: false,
			oldClaim:          validClaim,
			newClaim:          validSizeUpdate,
			enableResize:      true,
			enableBlock:       false,
		},
		"invalid-size-update-resize-enabled": {
			isExpectedFailure: true,
			oldClaim:          validClaim,
			newClaim:          invalidSizeUpdate,
			enableResize:      true,
			enableBlock:       false,
		},
		"unbound-size-update-resize-enabled": {
			isExpectedFailure: true,
			oldClaim:          validClaim,
			newClaim:          unboundSizeUpdate,
			enableResize:      true,
			enableBlock:       false,
		},
		"valid-upgrade-storage-class-annotation-to-spec": {
			isExpectedFailure: false,
			oldClaim:          validClaimStorageClass,
			newClaim:          validClaimStorageClassInSpec,
			enableResize:      false,
			enableBlock:       false,
		},
		"invalid-upgrade-storage-class-annotation-to-spec": {
			isExpectedFailure: true,
			oldClaim:          validClaimStorageClass,
			newClaim:          invalidClaimStorageClassInSpec,
			enableResize:      false,
			enableBlock:       false,
		},
		"valid-upgrade-storage-class-annotation-to-annotation-and-spec": {
			isExpectedFailure: false,
			oldClaim:          validClaimStorageClass,
			newClaim:          validClaimStorageClassInAnnotationAndSpec,
			enableResize:      false,
			enableBlock:       false,
		},
		"invalid-upgrade-storage-class-annotation-to-annotation-and-spec": {
			isExpectedFailure: true,
			oldClaim:          validClaimStorageClass,
			newClaim:          invalidClaimStorageClassInAnnotationAndSpec,
			enableResize:      false,
			enableBlock:       false,
		},
		"invalid-upgrade-storage-class-in-spec": {
			isExpectedFailure: true,
			oldClaim:          validClaimStorageClassInSpec,
			newClaim:          invalidClaimStorageClassInSpec,
			enableResize:      false,
			enableBlock:       false,
		},
		"invalid-downgrade-storage-class-spec-to-annotation": {
			isExpectedFailure: true,
			oldClaim:          validClaimStorageClassInSpec,
			newClaim:          validClaimStorageClass,
			enableResize:      false,
			enableBlock:       false,
		},
	}

	for name, scenario := range scenarios {
		// ensure we have a resource version specified for updates
		togglePVExpandFeature(scenario.enableResize, t)
		toggleBlockVolumeFeature(scenario.enableBlock, t)
		scenario.oldClaim.ResourceVersion = "1"
		scenario.newClaim.ResourceVersion = "1"
		errs := ValidatePersistentVolumeClaimUpdate(scenario.newClaim, scenario.oldClaim)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}

func toggleBlockVolumeFeature(toggleFlag bool, t *testing.T) {
	if toggleFlag {
		// Enable alpha feature BlockVolume
		err := utilfeature.DefaultFeatureGate.Set("BlockVolume=true")
		if err != nil {
			t.Errorf("Failed to enable feature gate for BlockVolume: %v", err)
			return
		}
	} else {
		err := utilfeature.DefaultFeatureGate.Set("BlockVolume=false")
		if err != nil {
			t.Errorf("Failed to disable feature gate for BlockVolume: %v", err)
			return
		}
	}
}

func togglePVExpandFeature(toggleFlag bool, t *testing.T) {
	if toggleFlag {
		// Enable alpha feature LocalStorageCapacityIsolation
		err := utilfeature.DefaultFeatureGate.Set("ExpandPersistentVolumes=true")
		if err != nil {
			t.Errorf("Failed to enable feature gate for ExpandPersistentVolumes: %v", err)
			return
		}
	} else {
		err := utilfeature.DefaultFeatureGate.Set("ExpandPersistentVolumes=false")
		if err != nil {
			t.Errorf("Failed to disable feature gate for ExpandPersistentVolumes: %v", err)
			return
		}
	}
}

func TestValidateKeyToPath(t *testing.T) {
	testCases := []struct {
		kp      core.KeyToPath
		ok      bool
		errtype field.ErrorType
	}{
		{
			kp: core.KeyToPath{Key: "k", Path: "p"},
			ok: true,
		},
		{
			kp: core.KeyToPath{Key: "k", Path: "p/p/p/p"},
			ok: true,
		},
		{
			kp: core.KeyToPath{Key: "k", Path: "p/..p/p../p..p"},
			ok: true,
		},
		{
			kp: core.KeyToPath{Key: "k", Path: "p", Mode: newInt32(0644)},
			ok: true,
		},
		{
			kp:      core.KeyToPath{Key: "", Path: "p"},
			ok:      false,
			errtype: field.ErrorTypeRequired,
		},
		{
			kp:      core.KeyToPath{Key: "k", Path: ""},
			ok:      false,
			errtype: field.ErrorTypeRequired,
		},
		{
			kp:      core.KeyToPath{Key: "k", Path: "..p"},
			ok:      false,
			errtype: field.ErrorTypeInvalid,
		},
		{
			kp:      core.KeyToPath{Key: "k", Path: "../p"},
			ok:      false,
			errtype: field.ErrorTypeInvalid,
		},
		{
			kp:      core.KeyToPath{Key: "k", Path: "p/../p"},
			ok:      false,
			errtype: field.ErrorTypeInvalid,
		},
		{
			kp:      core.KeyToPath{Key: "k", Path: "p/.."},
			ok:      false,
			errtype: field.ErrorTypeInvalid,
		},
		{
			kp:      core.KeyToPath{Key: "k", Path: "p", Mode: newInt32(01000)},
			ok:      false,
			errtype: field.ErrorTypeInvalid,
		},
		{
			kp:      core.KeyToPath{Key: "k", Path: "p", Mode: newInt32(-1)},
			ok:      false,
			errtype: field.ErrorTypeInvalid,
		},
	}

	for i, tc := range testCases {
		errs := validateKeyToPath(&tc.kp, field.NewPath("field"))
		if tc.ok && len(errs) > 0 {
			t.Errorf("[%d] unexpected errors: %v", i, errs)
		} else if !tc.ok && len(errs) == 0 {
			t.Errorf("[%d] expected error type %v", i, tc.errtype)
		} else if len(errs) > 1 {
			t.Errorf("[%d] expected only one error, got %d", i, len(errs))
		} else if !tc.ok {
			if errs[0].Type != tc.errtype {
				t.Errorf("[%d] expected error type %v, got %v", i, tc.errtype, errs[0].Type)
			}
		}
	}
}

func TestValidateNFSVolumeSource(t *testing.T) {
	testCases := []struct {
		name      string
		nfs       *core.NFSVolumeSource
		errtype   field.ErrorType
		errfield  string
		errdetail string
	}{
		{
			name:     "missing server",
			nfs:      &core.NFSVolumeSource{Server: "", Path: "/tmp"},
			errtype:  field.ErrorTypeRequired,
			errfield: "server",
		},
		{
			name:     "missing path",
			nfs:      &core.NFSVolumeSource{Server: "my-server", Path: ""},
			errtype:  field.ErrorTypeRequired,
			errfield: "path",
		},
		{
			name:      "abs path",
			nfs:       &core.NFSVolumeSource{Server: "my-server", Path: "tmp"},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "path",
			errdetail: "must be an absolute path",
		},
	}

	for i, tc := range testCases {
		errs := validateNFSVolumeSource(tc.nfs, field.NewPath("field"))

		if len(errs) > 0 && tc.errtype == "" {
			t.Errorf("[%d: %q] unexpected error(s): %v", i, tc.name, errs)
		} else if len(errs) == 0 && tc.errtype != "" {
			t.Errorf("[%d: %q] expected error type %v", i, tc.name, tc.errtype)
		} else if len(errs) >= 1 {
			if errs[0].Type != tc.errtype {
				t.Errorf("[%d: %q] expected error type %v, got %v", i, tc.name, tc.errtype, errs[0].Type)
			} else if !strings.HasSuffix(errs[0].Field, "."+tc.errfield) {
				t.Errorf("[%d: %q] expected error on field %q, got %q", i, tc.name, tc.errfield, errs[0].Field)
			} else if !strings.Contains(errs[0].Detail, tc.errdetail) {
				t.Errorf("[%d: %q] expected error detail %q, got %q", i, tc.name, tc.errdetail, errs[0].Detail)
			}
		}
	}
}

func TestValidateGlusterfs(t *testing.T) {
	testCases := []struct {
		name     string
		gfs      *core.GlusterfsVolumeSource
		errtype  field.ErrorType
		errfield string
	}{
		{
			name:     "missing endpointname",
			gfs:      &core.GlusterfsVolumeSource{EndpointsName: "", Path: "/tmp"},
			errtype:  field.ErrorTypeRequired,
			errfield: "endpoints",
		},
		{
			name:     "missing path",
			gfs:      &core.GlusterfsVolumeSource{EndpointsName: "my-endpoint", Path: ""},
			errtype:  field.ErrorTypeRequired,
			errfield: "path",
		},
		{
			name:     "missing endpintname and path",
			gfs:      &core.GlusterfsVolumeSource{EndpointsName: "", Path: ""},
			errtype:  field.ErrorTypeRequired,
			errfield: "endpoints",
		},
	}

	for i, tc := range testCases {
		errs := validateGlusterfsVolumeSource(tc.gfs, field.NewPath("field"))

		if len(errs) > 0 && tc.errtype == "" {
			t.Errorf("[%d: %q] unexpected error(s): %v", i, tc.name, errs)
		} else if len(errs) == 0 && tc.errtype != "" {
			t.Errorf("[%d: %q] expected error type %v", i, tc.name, tc.errtype)
		} else if len(errs) >= 1 {
			if errs[0].Type != tc.errtype {
				t.Errorf("[%d: %q] expected error type %v, got %v", i, tc.name, tc.errtype, errs[0].Type)
			} else if !strings.HasSuffix(errs[0].Field, "."+tc.errfield) {
				t.Errorf("[%d: %q] expected error on field %q, got %q", i, tc.name, tc.errfield, errs[0].Field)
			}
		}
	}
}

func TestValidateCSIVolumeSource(t *testing.T) {
	testCases := []struct {
		name     string
		csi      *core.CSIPersistentVolumeSource
		errtype  field.ErrorType
		errfield string
	}{
		{
			name: "all required fields ok",
			csi:  &core.CSIPersistentVolumeSource{Driver: "test-driver", VolumeHandle: "test-123", ReadOnly: true},
		},
		{
			name: "with default values ok",
			csi:  &core.CSIPersistentVolumeSource{Driver: "test-driver", VolumeHandle: "test-123"},
		},
		{
			name:     "missing driver name",
			csi:      &core.CSIPersistentVolumeSource{VolumeHandle: "test-123"},
			errtype:  field.ErrorTypeRequired,
			errfield: "driver",
		},
		{
			name:     "missing volume handle",
			csi:      &core.CSIPersistentVolumeSource{Driver: "my-driver"},
			errtype:  field.ErrorTypeRequired,
			errfield: "volumeHandle",
		},
		{
			name: "driver name: ok no punctuations",
			csi:  &core.CSIPersistentVolumeSource{Driver: "comgooglestoragecsigcepd", VolumeHandle: "test-123"},
		},
		{
			name: "driver name: ok dot only",
			csi:  &core.CSIPersistentVolumeSource{Driver: "io.kubernetes.storage.csi.flex", VolumeHandle: "test-123"},
		},
		{
			name: "driver name: ok dash only",
			csi:  &core.CSIPersistentVolumeSource{Driver: "io-kubernetes-storage-csi-flex", VolumeHandle: "test-123"},
		},
		{
			name: "driver name: ok underscore only",
			csi:  &core.CSIPersistentVolumeSource{Driver: "io_kubernetes_storage_csi_flex", VolumeHandle: "test-123"},
		},
		{
			name: "driver name: ok dot underscores",
			csi:  &core.CSIPersistentVolumeSource{Driver: "io.kubernetes.storage_csi.flex", VolumeHandle: "test-123"},
		},
		{
			name: "driver name: ok beginnin with number",
			csi:  &core.CSIPersistentVolumeSource{Driver: "2io.kubernetes.storage_csi.flex", VolumeHandle: "test-123"},
		},
		{
			name: "driver name: ok ending with number",
			csi:  &core.CSIPersistentVolumeSource{Driver: "io.kubernetes.storage_csi.flex2", VolumeHandle: "test-123"},
		},
		{
			name: "driver name: ok dot dash underscores",
			csi:  &core.CSIPersistentVolumeSource{Driver: "io.kubernetes-storage.csi_flex", VolumeHandle: "test-123"},
		},
		{
			name:     "driver name: invalid length 0",
			csi:      &core.CSIPersistentVolumeSource{Driver: "", VolumeHandle: "test-123"},
			errtype:  field.ErrorTypeRequired,
			errfield: "driver",
		},
		{
			name:     "driver name: invalid length 1",
			csi:      &core.CSIPersistentVolumeSource{Driver: "a", VolumeHandle: "test-123"},
			errtype:  field.ErrorTypeInvalid,
			errfield: "driver",
		},
		{
			name:     "driver name: invalid length > 63",
			csi:      &core.CSIPersistentVolumeSource{Driver: "comgooglestoragecsigcepdcomgooglestoragecsigcepdcomgooglestoragecsigcepdcomgooglestoragecsigcepd", VolumeHandle: "test-123"},
			errtype:  field.ErrorTypeTooLong,
			errfield: "driver",
		},
		{
			name:     "driver name: invalid start char",
			csi:      &core.CSIPersistentVolumeSource{Driver: "_comgooglestoragecsigcepd", VolumeHandle: "test-123"},
			errtype:  field.ErrorTypeInvalid,
			errfield: "driver",
		},
		{
			name:     "driver name: invalid end char",
			csi:      &core.CSIPersistentVolumeSource{Driver: "comgooglestoragecsigcepd/", VolumeHandle: "test-123"},
			errtype:  field.ErrorTypeInvalid,
			errfield: "driver",
		},
		{
			name:     "driver name: invalid separators",
			csi:      &core.CSIPersistentVolumeSource{Driver: "com/google/storage/csi~gcepd", VolumeHandle: "test-123"},
			errtype:  field.ErrorTypeInvalid,
			errfield: "driver",
		},
	}

	err := utilfeature.DefaultFeatureGate.Set("CSIPersistentVolume=true")
	if err != nil {
		t.Errorf("Failed to enable feature gate for CSIPersistentVolumes: %v", err)
		return
	}

	for i, tc := range testCases {
		errs := validateCSIPersistentVolumeSource(tc.csi, field.NewPath("field"))

		if len(errs) > 0 && tc.errtype == "" {
			t.Errorf("[%d: %q] unexpected error(s): %v", i, tc.name, errs)
		} else if len(errs) == 0 && tc.errtype != "" {
			t.Errorf("[%d: %q] expected error type %v", i, tc.name, tc.errtype)
		} else if len(errs) >= 1 {
			if errs[0].Type != tc.errtype {
				t.Errorf("[%d: %q] expected error type %v, got %v", i, tc.name, tc.errtype, errs[0].Type)
			} else if !strings.HasSuffix(errs[0].Field, "."+tc.errfield) {
				t.Errorf("[%d: %q] expected error on field %q, got %q", i, tc.name, tc.errfield, errs[0].Field)
			}
		}
	}
	err = utilfeature.DefaultFeatureGate.Set("CSIPersistentVolume=false")
	if err != nil {
		t.Errorf("Failed to disable feature gate for CSIPersistentVolumes: %v", err)
		return
	}

}

// helper
func newInt32(val int) *int32 {
	p := new(int32)
	*p = int32(val)
	return p
}

// This test is a little too top-to-bottom.  Ideally we would test each volume
// type on its own, but we want to also make sure that the logic works through
// the one-of wrapper, so we just do it all in one place.
func TestValidateVolumes(t *testing.T) {
	validInitiatorName := "iqn.2015-02.example.com:init"
	invalidInitiatorName := "2015-02.example.com:init"
	testCases := []struct {
		name      string
		vol       core.Volume
		errtype   field.ErrorType
		errfield  string
		errdetail string
	}{
		// EmptyDir and basic volume names
		{
			name: "valid alpha name",
			vol: core.Volume{
				Name: "empty",
				VolumeSource: core.VolumeSource{
					EmptyDir: &core.EmptyDirVolumeSource{},
				},
			},
		},
		{
			name: "valid num name",
			vol: core.Volume{
				Name: "123",
				VolumeSource: core.VolumeSource{
					EmptyDir: &core.EmptyDirVolumeSource{},
				},
			},
		},
		{
			name: "valid alphanum name",
			vol: core.Volume{
				Name: "empty-123",
				VolumeSource: core.VolumeSource{
					EmptyDir: &core.EmptyDirVolumeSource{},
				},
			},
		},
		{
			name: "valid numalpha name",
			vol: core.Volume{
				Name: "123-empty",
				VolumeSource: core.VolumeSource{
					EmptyDir: &core.EmptyDirVolumeSource{},
				},
			},
		},
		{
			name: "zero-length name",
			vol: core.Volume{
				Name:         "",
				VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "name",
		},
		{
			name: "name > 63 characters",
			vol: core.Volume{
				Name:         strings.Repeat("a", 64),
				VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}},
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "name",
			errdetail: "must be no more than",
		},
		{
			name: "name not a DNS label",
			vol: core.Volume{
				Name:         "a.b.c",
				VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}},
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "name",
			errdetail: dnsLabelErrMsg,
		},
		// More than one source field specified.
		{
			name: "more than one source",
			vol: core.Volume{
				Name: "dups",
				VolumeSource: core.VolumeSource{
					EmptyDir: &core.EmptyDirVolumeSource{},
					HostPath: &core.HostPathVolumeSource{
						Path: "/mnt/path",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			},
			errtype:   field.ErrorTypeForbidden,
			errfield:  "hostPath",
			errdetail: "may not specify more than 1 volume",
		},
		// HostPath Default
		{
			name: "default HostPath",
			vol: core.Volume{
				Name: "hostpath",
				VolumeSource: core.VolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/mnt/path",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			},
		},
		// HostPath Supported
		{
			name: "valid HostPath",
			vol: core.Volume{
				Name: "hostpath",
				VolumeSource: core.VolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/mnt/path",
						Type: newHostPathType(string(core.HostPathSocket)),
					},
				},
			},
		},
		// HostPath Invalid
		{
			name: "invalid HostPath",
			vol: core.Volume{
				Name: "hostpath",
				VolumeSource: core.VolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/mnt/path",
						Type: newHostPathType("invalid"),
					},
				},
			},
			errtype:  field.ErrorTypeNotSupported,
			errfield: "type",
		},
		{
			name: "invalid HostPath backsteps",
			vol: core.Volume{
				Name: "hostpath",
				VolumeSource: core.VolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/mnt/path/..",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "path",
			errdetail: "must not contain '..'",
		},
		// GcePersistentDisk
		{
			name: "valid GcePersistentDisk",
			vol: core.Volume{
				Name: "gce-pd",
				VolumeSource: core.VolumeSource{
					GCEPersistentDisk: &core.GCEPersistentDiskVolumeSource{
						PDName:    "my-PD",
						FSType:    "ext4",
						Partition: 1,
						ReadOnly:  false,
					},
				},
			},
		},
		// AWSElasticBlockStore
		{
			name: "valid AWSElasticBlockStore",
			vol: core.Volume{
				Name: "aws-ebs",
				VolumeSource: core.VolumeSource{
					AWSElasticBlockStore: &core.AWSElasticBlockStoreVolumeSource{
						VolumeID:  "my-PD",
						FSType:    "ext4",
						Partition: 1,
						ReadOnly:  false,
					},
				},
			},
		},
		// GitRepo
		{
			name: "valid GitRepo",
			vol: core.Volume{
				Name: "git-repo",
				VolumeSource: core.VolumeSource{
					GitRepo: &core.GitRepoVolumeSource{
						Repository: "my-repo",
						Revision:   "hashstring",
						Directory:  "target",
					},
				},
			},
		},
		{
			name: "valid GitRepo in .",
			vol: core.Volume{
				Name: "git-repo-dot",
				VolumeSource: core.VolumeSource{
					GitRepo: &core.GitRepoVolumeSource{
						Repository: "my-repo",
						Directory:  ".",
					},
				},
			},
		},
		{
			name: "valid GitRepo with .. in name",
			vol: core.Volume{
				Name: "git-repo-dot-dot-foo",
				VolumeSource: core.VolumeSource{
					GitRepo: &core.GitRepoVolumeSource{
						Repository: "my-repo",
						Directory:  "..foo",
					},
				},
			},
		},
		{
			name: "GitRepo starts with ../",
			vol: core.Volume{
				Name: "gitrepo",
				VolumeSource: core.VolumeSource{
					GitRepo: &core.GitRepoVolumeSource{
						Repository: "foo",
						Directory:  "../dots/bar",
					},
				},
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "gitRepo.directory",
			errdetail: `must not contain '..'`,
		},
		{
			name: "GitRepo contains ..",
			vol: core.Volume{
				Name: "gitrepo",
				VolumeSource: core.VolumeSource{
					GitRepo: &core.GitRepoVolumeSource{
						Repository: "foo",
						Directory:  "dots/../bar",
					},
				},
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "gitRepo.directory",
			errdetail: `must not contain '..'`,
		},
		{
			name: "GitRepo absolute target",
			vol: core.Volume{
				Name: "gitrepo",
				VolumeSource: core.VolumeSource{
					GitRepo: &core.GitRepoVolumeSource{
						Repository: "foo",
						Directory:  "/abstarget",
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "gitRepo.directory",
		},
		// ISCSI
		{
			name: "valid ISCSI",
			vol: core.Volume{
				Name: "iscsi",
				VolumeSource: core.VolumeSource{
					ISCSI: &core.ISCSIVolumeSource{
						TargetPortal: "127.0.0.1",
						IQN:          "iqn.2015-02.example.com:test",
						Lun:          1,
						FSType:       "ext4",
						ReadOnly:     false,
					},
				},
			},
		},
		{
			name: "valid IQN: eui format",
			vol: core.Volume{
				Name: "iscsi",
				VolumeSource: core.VolumeSource{
					ISCSI: &core.ISCSIVolumeSource{
						TargetPortal: "127.0.0.1",
						IQN:          "eui.0123456789ABCDEF",
						Lun:          1,
						FSType:       "ext4",
						ReadOnly:     false,
					},
				},
			},
		},
		{
			name: "valid IQN: naa format",
			vol: core.Volume{
				Name: "iscsi",
				VolumeSource: core.VolumeSource{
					ISCSI: &core.ISCSIVolumeSource{
						TargetPortal: "127.0.0.1",
						IQN:          "naa.62004567BA64678D0123456789ABCDEF",
						Lun:          1,
						FSType:       "ext4",
						ReadOnly:     false,
					},
				},
			},
		},
		{
			name: "empty portal",
			vol: core.Volume{
				Name: "iscsi",
				VolumeSource: core.VolumeSource{
					ISCSI: &core.ISCSIVolumeSource{
						TargetPortal: "",
						IQN:          "iqn.2015-02.example.com:test",
						Lun:          1,
						FSType:       "ext4",
						ReadOnly:     false,
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "iscsi.targetPortal",
		},
		{
			name: "empty iqn",
			vol: core.Volume{
				Name: "iscsi",
				VolumeSource: core.VolumeSource{
					ISCSI: &core.ISCSIVolumeSource{
						TargetPortal: "127.0.0.1",
						IQN:          "",
						Lun:          1,
						FSType:       "ext4",
						ReadOnly:     false,
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "iscsi.iqn",
		},
		{
			name: "invalid IQN: iqn format",
			vol: core.Volume{
				Name: "iscsi",
				VolumeSource: core.VolumeSource{
					ISCSI: &core.ISCSIVolumeSource{
						TargetPortal: "127.0.0.1",
						IQN:          "iqn.2015-02.example.com:test;ls;",
						Lun:          1,
						FSType:       "ext4",
						ReadOnly:     false,
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "iscsi.iqn",
		},
		{
			name: "invalid IQN: eui format",
			vol: core.Volume{
				Name: "iscsi",
				VolumeSource: core.VolumeSource{
					ISCSI: &core.ISCSIVolumeSource{
						TargetPortal: "127.0.0.1",
						IQN:          "eui.0123456789ABCDEFGHIJ",
						Lun:          1,
						FSType:       "ext4",
						ReadOnly:     false,
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "iscsi.iqn",
		},
		{
			name: "invalid IQN: naa format",
			vol: core.Volume{
				Name: "iscsi",
				VolumeSource: core.VolumeSource{
					ISCSI: &core.ISCSIVolumeSource{
						TargetPortal: "127.0.0.1",
						IQN:          "naa.62004567BA_4-78D.123456789ABCDEF",
						Lun:          1,
						FSType:       "ext4",
						ReadOnly:     false,
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "iscsi.iqn",
		},
		{
			name: "valid initiatorName",
			vol: core.Volume{
				Name: "iscsi",
				VolumeSource: core.VolumeSource{
					ISCSI: &core.ISCSIVolumeSource{
						TargetPortal:  "127.0.0.1",
						IQN:           "iqn.2015-02.example.com:test",
						Lun:           1,
						InitiatorName: &validInitiatorName,
						FSType:        "ext4",
						ReadOnly:      false,
					},
				},
			},
		},
		{
			name: "invalid initiatorName",
			vol: core.Volume{
				Name: "iscsi",
				VolumeSource: core.VolumeSource{
					ISCSI: &core.ISCSIVolumeSource{
						TargetPortal:  "127.0.0.1",
						IQN:           "iqn.2015-02.example.com:test",
						Lun:           1,
						InitiatorName: &invalidInitiatorName,
						FSType:        "ext4",
						ReadOnly:      false,
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "iscsi.initiatorname",
		},
		{
			name: "empty secret",
			vol: core.Volume{
				Name: "iscsi",
				VolumeSource: core.VolumeSource{
					ISCSI: &core.ISCSIVolumeSource{
						TargetPortal:      "127.0.0.1",
						IQN:               "iqn.2015-02.example.com:test",
						Lun:               1,
						FSType:            "ext4",
						ReadOnly:          false,
						DiscoveryCHAPAuth: true,
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "iscsi.secretRef",
		},
		{
			name: "empty secret",
			vol: core.Volume{
				Name: "iscsi",
				VolumeSource: core.VolumeSource{
					ISCSI: &core.ISCSIVolumeSource{
						TargetPortal:    "127.0.0.1",
						IQN:             "iqn.2015-02.example.com:test",
						Lun:             1,
						FSType:          "ext4",
						ReadOnly:        false,
						SessionCHAPAuth: true,
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "iscsi.secretRef",
		},
		// Secret
		{
			name: "valid Secret",
			vol: core.Volume{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName: "my-secret",
					},
				},
			},
		},
		{
			name: "valid Secret with defaultMode",
			vol: core.Volume{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName:  "my-secret",
						DefaultMode: newInt32(0644),
					},
				},
			},
		},
		{
			name: "valid Secret with projection and mode",
			vol: core.Volume{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName: "my-secret",
						Items: []core.KeyToPath{{
							Key:  "key",
							Path: "filename",
							Mode: newInt32(0644),
						}},
					},
				},
			},
		},
		{
			name: "valid Secret with subdir projection",
			vol: core.Volume{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName: "my-secret",
						Items: []core.KeyToPath{{
							Key:  "key",
							Path: "dir/filename",
						}},
					},
				},
			},
		},
		{
			name: "secret with missing path",
			vol: core.Volume{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName: "s",
						Items:      []core.KeyToPath{{Key: "key", Path: ""}},
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "secret.items[0].path",
		},
		{
			name: "secret with leading ..",
			vol: core.Volume{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName: "s",
						Items:      []core.KeyToPath{{Key: "key", Path: "../foo"}},
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "secret.items[0].path",
		},
		{
			name: "secret with .. inside",
			vol: core.Volume{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName: "s",
						Items:      []core.KeyToPath{{Key: "key", Path: "foo/../bar"}},
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "secret.items[0].path",
		},
		{
			name: "secret with invalid positive defaultMode",
			vol: core.Volume{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName:  "s",
						DefaultMode: newInt32(01000),
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "secret.defaultMode",
		},
		{
			name: "secret with invalid negative defaultMode",
			vol: core.Volume{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName:  "s",
						DefaultMode: newInt32(-1),
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "secret.defaultMode",
		},
		// ConfigMap
		{
			name: "valid ConfigMap",
			vol: core.Volume{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{
							Name: "my-cfgmap",
						},
					},
				},
			},
		},
		{
			name: "valid ConfigMap with defaultMode",
			vol: core.Volume{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{
							Name: "my-cfgmap",
						},
						DefaultMode: newInt32(0644),
					},
				},
			},
		},
		{
			name: "valid ConfigMap with projection and mode",
			vol: core.Volume{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{
							Name: "my-cfgmap"},
						Items: []core.KeyToPath{{
							Key:  "key",
							Path: "filename",
							Mode: newInt32(0644),
						}},
					},
				},
			},
		},
		{
			name: "valid ConfigMap with subdir projection",
			vol: core.Volume{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{
							Name: "my-cfgmap"},
						Items: []core.KeyToPath{{
							Key:  "key",
							Path: "dir/filename",
						}},
					},
				},
			},
		},
		{
			name: "configmap with missing path",
			vol: core.Volume{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{Name: "c"},
						Items:                []core.KeyToPath{{Key: "key", Path: ""}},
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "configMap.items[0].path",
		},
		{
			name: "configmap with leading ..",
			vol: core.Volume{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{Name: "c"},
						Items:                []core.KeyToPath{{Key: "key", Path: "../foo"}},
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "configMap.items[0].path",
		},
		{
			name: "configmap with .. inside",
			vol: core.Volume{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{Name: "c"},
						Items:                []core.KeyToPath{{Key: "key", Path: "foo/../bar"}},
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "configMap.items[0].path",
		},
		{
			name: "configmap with invalid positive defaultMode",
			vol: core.Volume{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{Name: "c"},
						DefaultMode:          newInt32(01000),
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "configMap.defaultMode",
		},
		{
			name: "configmap with invalid negative defaultMode",
			vol: core.Volume{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{Name: "c"},
						DefaultMode:          newInt32(-1),
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "configMap.defaultMode",
		},
		// Glusterfs
		{
			name: "valid Glusterfs",
			vol: core.Volume{
				Name: "glusterfs",
				VolumeSource: core.VolumeSource{
					Glusterfs: &core.GlusterfsVolumeSource{
						EndpointsName: "host1",
						Path:          "path",
						ReadOnly:      false,
					},
				},
			},
		},
		{
			name: "empty hosts",
			vol: core.Volume{
				Name: "glusterfs",
				VolumeSource: core.VolumeSource{
					Glusterfs: &core.GlusterfsVolumeSource{
						EndpointsName: "",
						Path:          "path",
						ReadOnly:      false,
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "glusterfs.endpoints",
		},
		{
			name: "empty path",
			vol: core.Volume{
				Name: "glusterfs",
				VolumeSource: core.VolumeSource{
					Glusterfs: &core.GlusterfsVolumeSource{
						EndpointsName: "host",
						Path:          "",
						ReadOnly:      false,
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "glusterfs.path",
		},
		// Flocker
		{
			name: "valid Flocker -- datasetUUID",
			vol: core.Volume{
				Name: "flocker",
				VolumeSource: core.VolumeSource{
					Flocker: &core.FlockerVolumeSource{
						DatasetUUID: "d846b09d-223d-43df-ab5b-d6db2206a0e4",
					},
				},
			},
		},
		{
			name: "valid Flocker -- datasetName",
			vol: core.Volume{
				Name: "flocker",
				VolumeSource: core.VolumeSource{
					Flocker: &core.FlockerVolumeSource{
						DatasetName: "datasetName",
					},
				},
			},
		},
		{
			name: "both empty",
			vol: core.Volume{
				Name: "flocker",
				VolumeSource: core.VolumeSource{
					Flocker: &core.FlockerVolumeSource{
						DatasetName: "",
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "flocker",
		},
		{
			name: "both specified",
			vol: core.Volume{
				Name: "flocker",
				VolumeSource: core.VolumeSource{
					Flocker: &core.FlockerVolumeSource{
						DatasetName: "datasetName",
						DatasetUUID: "d846b09d-223d-43df-ab5b-d6db2206a0e4",
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "flocker",
		},
		{
			name: "slash in flocker datasetName",
			vol: core.Volume{
				Name: "flocker",
				VolumeSource: core.VolumeSource{
					Flocker: &core.FlockerVolumeSource{
						DatasetName: "foo/bar",
					},
				},
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "flocker.datasetName",
			errdetail: "must not contain '/'",
		},
		// RBD
		{
			name: "valid RBD",
			vol: core.Volume{
				Name: "rbd",
				VolumeSource: core.VolumeSource{
					RBD: &core.RBDVolumeSource{
						CephMonitors: []string{"foo"},
						RBDImage:     "bar",
						FSType:       "ext4",
					},
				},
			},
		},
		{
			name: "empty rbd monitors",
			vol: core.Volume{
				Name: "rbd",
				VolumeSource: core.VolumeSource{
					RBD: &core.RBDVolumeSource{
						CephMonitors: []string{},
						RBDImage:     "bar",
						FSType:       "ext4",
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "rbd.monitors",
		},
		{
			name: "empty image",
			vol: core.Volume{
				Name: "rbd",
				VolumeSource: core.VolumeSource{
					RBD: &core.RBDVolumeSource{
						CephMonitors: []string{"foo"},
						RBDImage:     "",
						FSType:       "ext4",
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "rbd.image",
		},
		// Cinder
		{
			name: "valid Cinder",
			vol: core.Volume{
				Name: "cinder",
				VolumeSource: core.VolumeSource{
					Cinder: &core.CinderVolumeSource{
						VolumeID: "29ea5088-4f60-4757-962e-dba678767887",
						FSType:   "ext4",
						ReadOnly: false,
					},
				},
			},
		},
		// CephFS
		{
			name: "valid CephFS",
			vol: core.Volume{
				Name: "cephfs",
				VolumeSource: core.VolumeSource{
					CephFS: &core.CephFSVolumeSource{
						Monitors: []string{"foo"},
					},
				},
			},
		},
		{
			name: "empty cephfs monitors",
			vol: core.Volume{
				Name: "cephfs",
				VolumeSource: core.VolumeSource{
					CephFS: &core.CephFSVolumeSource{
						Monitors: []string{},
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "cephfs.monitors",
		},
		// DownwardAPI
		{
			name: "valid DownwardAPI",
			vol: core.Volume{
				Name: "downwardapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{
							{
								Path: "labels",
								FieldRef: &core.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.labels",
								},
							},
							{
								Path: "labels with subscript",
								FieldRef: &core.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.labels['key']",
								},
							},
							{
								Path: "labels with complex subscript",
								FieldRef: &core.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.labels['test.example.com/key']",
								},
							},
							{
								Path: "annotations",
								FieldRef: &core.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.annotations",
								},
							},
							{
								Path: "annotations with subscript",
								FieldRef: &core.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.annotations['key']",
								},
							},
							{
								Path: "annotations with complex subscript",
								FieldRef: &core.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.annotations['TEST.EXAMPLE.COM/key']",
								},
							},
							{
								Path: "namespace",
								FieldRef: &core.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.namespace",
								},
							},
							{
								Path: "name",
								FieldRef: &core.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.name",
								},
							},
							{
								Path: "path/with/subdirs",
								FieldRef: &core.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.labels",
								},
							},
							{
								Path: "path/./withdot",
								FieldRef: &core.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.labels",
								},
							},
							{
								Path: "path/with/embedded..dotdot",
								FieldRef: &core.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.labels",
								},
							},
							{
								Path: "path/with/leading/..dotdot",
								FieldRef: &core.ObjectFieldSelector{
									APIVersion: "v1",
									FieldPath:  "metadata.labels",
								},
							},
							{
								Path: "cpu_limit",
								ResourceFieldRef: &core.ResourceFieldSelector{
									ContainerName: "test-container",
									Resource:      "limits.cpu",
								},
							},
							{
								Path: "cpu_request",
								ResourceFieldRef: &core.ResourceFieldSelector{
									ContainerName: "test-container",
									Resource:      "requests.cpu",
								},
							},
							{
								Path: "memory_limit",
								ResourceFieldRef: &core.ResourceFieldSelector{
									ContainerName: "test-container",
									Resource:      "limits.memory",
								},
							},
							{
								Path: "memory_request",
								ResourceFieldRef: &core.ResourceFieldSelector{
									ContainerName: "test-container",
									Resource:      "requests.memory",
								},
							},
						},
					},
				},
			},
		},
		{
			name: "downapi valid defaultMode",
			vol: core.Volume{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						DefaultMode: newInt32(0644),
					},
				},
			},
		},
		{
			name: "downapi valid item mode",
			vol: core.Volume{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Mode: newInt32(0644),
							Path: "path",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}},
					},
				},
			},
		},
		{
			name: "downapi invalid positive item mode",
			vol: core.Volume{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Mode: newInt32(01000),
							Path: "path",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}},
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "downwardAPI.mode",
		},
		{
			name: "downapi invalid negative item mode",
			vol: core.Volume{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Mode: newInt32(-1),
							Path: "path",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}},
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "downwardAPI.mode",
		},
		{
			name: "downapi empty metatada path",
			vol: core.Volume{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Path: "",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}},
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "downwardAPI.path",
		},
		{
			name: "downapi absolute path",
			vol: core.Volume{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Path: "/absolutepath",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}},
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "downwardAPI.path",
		},
		{
			name: "downapi dot dot path",
			vol: core.Volume{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Path: "../../passwd",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}},
					},
				},
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "downwardAPI.path",
			errdetail: `must not contain '..'`,
		},
		{
			name: "downapi dot dot file name",
			vol: core.Volume{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Path: "..badFileName",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}},
					},
				},
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "downwardAPI.path",
			errdetail: `must not start with '..'`,
		},
		{
			name: "downapi dot dot first level dirent",
			vol: core.Volume{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Path: "..badDirName/goodFileName",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}},
					},
				},
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "downwardAPI.path",
			errdetail: `must not start with '..'`,
		},
		{
			name: "downapi fieldRef and ResourceFieldRef together",
			vol: core.Volume{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Path: "test",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
							ResourceFieldRef: &core.ResourceFieldSelector{
								ContainerName: "test-container",
								Resource:      "requests.memory",
							},
						}},
					},
				},
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "downwardAPI",
			errdetail: "fieldRef and resourceFieldRef can not be specified simultaneously",
		},
		{
			name: "downapi invalid positive defaultMode",
			vol: core.Volume{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						DefaultMode: newInt32(01000),
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "downwardAPI.defaultMode",
		},
		{
			name: "downapi invalid negative defaultMode",
			vol: core.Volume{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						DefaultMode: newInt32(-1),
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "downwardAPI.defaultMode",
		},
		// FC
		{
			name: "FC valid targetWWNs and lun",
			vol: core.Volume{
				Name: "fc",
				VolumeSource: core.VolumeSource{
					FC: &core.FCVolumeSource{
						TargetWWNs: []string{"some_wwn"},
						Lun:        newInt32(1),
						FSType:     "ext4",
						ReadOnly:   false,
					},
				},
			},
		},
		{
			name: "FC valid wwids",
			vol: core.Volume{
				Name: "fc",
				VolumeSource: core.VolumeSource{
					FC: &core.FCVolumeSource{
						WWIDs:    []string{"some_wwid"},
						FSType:   "ext4",
						ReadOnly: false,
					},
				},
			},
		},
		{
			name: "FC empty targetWWNs and wwids",
			vol: core.Volume{
				Name: "fc",
				VolumeSource: core.VolumeSource{
					FC: &core.FCVolumeSource{
						TargetWWNs: []string{},
						Lun:        newInt32(1),
						WWIDs:      []string{},
						FSType:     "ext4",
						ReadOnly:   false,
					},
				},
			},
			errtype:   field.ErrorTypeRequired,
			errfield:  "fc.targetWWNs",
			errdetail: "must specify either targetWWNs or wwids",
		},
		{
			name: "FC invalid: both targetWWNs and wwids simultaneously",
			vol: core.Volume{
				Name: "fc",
				VolumeSource: core.VolumeSource{
					FC: &core.FCVolumeSource{
						TargetWWNs: []string{"some_wwn"},
						Lun:        newInt32(1),
						WWIDs:      []string{"some_wwid"},
						FSType:     "ext4",
						ReadOnly:   false,
					},
				},
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "fc.targetWWNs",
			errdetail: "targetWWNs and wwids can not be specified simultaneously",
		},
		{
			name: "FC valid targetWWNs and empty lun",
			vol: core.Volume{
				Name: "fc",
				VolumeSource: core.VolumeSource{
					FC: &core.FCVolumeSource{
						TargetWWNs: []string{"wwn"},
						Lun:        nil,
						FSType:     "ext4",
						ReadOnly:   false,
					},
				},
			},
			errtype:   field.ErrorTypeRequired,
			errfield:  "fc.lun",
			errdetail: "lun is required if targetWWNs is specified",
		},
		{
			name: "FC valid targetWWNs and invalid lun",
			vol: core.Volume{
				Name: "fc",
				VolumeSource: core.VolumeSource{
					FC: &core.FCVolumeSource{
						TargetWWNs: []string{"wwn"},
						Lun:        newInt32(256),
						FSType:     "ext4",
						ReadOnly:   false,
					},
				},
			},
			errtype:   field.ErrorTypeInvalid,
			errfield:  "fc.lun",
			errdetail: validation.InclusiveRangeError(0, 255),
		},
		// FlexVolume
		{
			name: "valid FlexVolume",
			vol: core.Volume{
				Name: "flex-volume",
				VolumeSource: core.VolumeSource{
					FlexVolume: &core.FlexVolumeSource{
						Driver: "kubernetes.io/blue",
						FSType: "ext4",
					},
				},
			},
		},
		// AzureFile
		{
			name: "valid AzureFile",
			vol: core.Volume{
				Name: "azure-file",
				VolumeSource: core.VolumeSource{
					AzureFile: &core.AzureFileVolumeSource{
						SecretName: "key",
						ShareName:  "share",
						ReadOnly:   false,
					},
				},
			},
		},
		{
			name: "AzureFile empty secret",
			vol: core.Volume{
				Name: "azure-file",
				VolumeSource: core.VolumeSource{
					AzureFile: &core.AzureFileVolumeSource{
						SecretName: "",
						ShareName:  "share",
						ReadOnly:   false,
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "azureFile.secretName",
		},
		{
			name: "AzureFile empty share",
			vol: core.Volume{
				Name: "azure-file",
				VolumeSource: core.VolumeSource{
					AzureFile: &core.AzureFileVolumeSource{
						SecretName: "name",
						ShareName:  "",
						ReadOnly:   false,
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "azureFile.shareName",
		},
		// Quobyte
		{
			name: "valid Quobyte",
			vol: core.Volume{
				Name: "quobyte",
				VolumeSource: core.VolumeSource{
					Quobyte: &core.QuobyteVolumeSource{
						Registry: "registry:7861",
						Volume:   "volume",
						ReadOnly: false,
						User:     "root",
						Group:    "root",
					},
				},
			},
		},
		{
			name: "empty registry quobyte",
			vol: core.Volume{
				Name: "quobyte",
				VolumeSource: core.VolumeSource{
					Quobyte: &core.QuobyteVolumeSource{
						Volume: "/test",
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "quobyte.registry",
		},
		{
			name: "wrong format registry quobyte",
			vol: core.Volume{
				Name: "quobyte",
				VolumeSource: core.VolumeSource{
					Quobyte: &core.QuobyteVolumeSource{
						Registry: "registry7861",
						Volume:   "/test",
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "quobyte.registry",
		},
		{
			name: "wrong format multiple registries quobyte",
			vol: core.Volume{
				Name: "quobyte",
				VolumeSource: core.VolumeSource{
					Quobyte: &core.QuobyteVolumeSource{
						Registry: "registry:7861,reg2",
						Volume:   "/test",
					},
				},
			},
			errtype:  field.ErrorTypeInvalid,
			errfield: "quobyte.registry",
		},
		{
			name: "empty volume quobyte",
			vol: core.Volume{
				Name: "quobyte",
				VolumeSource: core.VolumeSource{
					Quobyte: &core.QuobyteVolumeSource{
						Registry: "registry:7861",
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "quobyte.volume",
		},
		// AzureDisk
		{
			name: "valid AzureDisk",
			vol: core.Volume{
				Name: "azure-disk",
				VolumeSource: core.VolumeSource{
					AzureDisk: &core.AzureDiskVolumeSource{
						DiskName:    "foo",
						DataDiskURI: "https://blob/vhds/bar.vhd",
					},
				},
			},
		},
		{
			name: "AzureDisk empty disk name",
			vol: core.Volume{
				Name: "azure-disk",
				VolumeSource: core.VolumeSource{
					AzureDisk: &core.AzureDiskVolumeSource{
						DiskName:    "",
						DataDiskURI: "https://blob/vhds/bar.vhd",
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "azureDisk.diskName",
		},
		{
			name: "AzureDisk empty disk uri",
			vol: core.Volume{
				Name: "azure-disk",
				VolumeSource: core.VolumeSource{
					AzureDisk: &core.AzureDiskVolumeSource{
						DiskName:    "foo",
						DataDiskURI: "",
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "azureDisk.diskURI",
		},
		// ScaleIO
		{
			name: "valid scaleio volume",
			vol: core.Volume{
				Name: "scaleio-volume",
				VolumeSource: core.VolumeSource{
					ScaleIO: &core.ScaleIOVolumeSource{
						Gateway:    "http://abcd/efg",
						System:     "test-system",
						VolumeName: "test-vol-1",
					},
				},
			},
		},
		{
			name: "ScaleIO with empty name",
			vol: core.Volume{
				Name: "scaleio-volume",
				VolumeSource: core.VolumeSource{
					ScaleIO: &core.ScaleIOVolumeSource{
						Gateway:    "http://abcd/efg",
						System:     "test-system",
						VolumeName: "",
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "scaleIO.volumeName",
		},
		{
			name: "ScaleIO with empty gateway",
			vol: core.Volume{
				Name: "scaleio-volume",
				VolumeSource: core.VolumeSource{
					ScaleIO: &core.ScaleIOVolumeSource{
						Gateway:    "",
						System:     "test-system",
						VolumeName: "test-vol-1",
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "scaleIO.gateway",
		},
		{
			name: "ScaleIO with empty system",
			vol: core.Volume{
				Name: "scaleio-volume",
				VolumeSource: core.VolumeSource{
					ScaleIO: &core.ScaleIOVolumeSource{
						Gateway:    "http://agc/efg/gateway",
						System:     "",
						VolumeName: "test-vol-1",
					},
				},
			},
			errtype:  field.ErrorTypeRequired,
			errfield: "scaleIO.system",
		},
	}

	for i, tc := range testCases {
		names, errs := ValidateVolumes([]core.Volume{tc.vol}, field.NewPath("field"))
		if len(errs) > 0 && tc.errtype == "" {
			t.Errorf("[%d: %q] unexpected error(s): %v", i, tc.name, errs)
		} else if len(errs) > 1 {
			t.Errorf("[%d: %q] expected 1 error, got %d: %v", i, tc.name, len(errs), errs)
		} else if len(errs) == 0 && tc.errtype != "" {
			t.Errorf("[%d: %q] expected error type %v", i, tc.name, tc.errtype)
		} else if len(errs) == 1 {
			if errs[0].Type != tc.errtype {
				t.Errorf("[%d: %q] expected error type %v, got %v", i, tc.name, tc.errtype, errs[0].Type)
			} else if !strings.HasSuffix(errs[0].Field, "."+tc.errfield) {
				t.Errorf("[%d: %q] expected error on field %q, got %q", i, tc.name, tc.errfield, errs[0].Field)
			} else if !strings.Contains(errs[0].Detail, tc.errdetail) {
				t.Errorf("[%d: %q] expected error detail %q, got %q", i, tc.name, tc.errdetail, errs[0].Detail)
			}
		} else {
			if len(names) != 1 || !IsMatchedVolume(tc.vol.Name, names) {
				t.Errorf("[%d: %q] wrong names result: %v", i, tc.name, names)
			}
		}
	}

	dupsCase := []core.Volume{
		{Name: "abc", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
		{Name: "abc", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
	}
	_, errs := ValidateVolumes(dupsCase, field.NewPath("field"))
	if len(errs) == 0 {
		t.Errorf("expected error")
	} else if len(errs) != 1 {
		t.Errorf("expected 1 error, got %d: %v", len(errs), errs)
	} else if errs[0].Type != field.ErrorTypeDuplicate {
		t.Errorf("expected error type %v, got %v", field.ErrorTypeDuplicate, errs[0].Type)
	}

	// Validate HugePages medium type for EmptyDir when HugePages feature is enabled/disabled
	hugePagesCase := core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{Medium: core.StorageMediumHugePages}}

	// Enable alpha feature HugePages
	err := utilfeature.DefaultFeatureGate.Set("HugePages=true")
	if err != nil {
		t.Errorf("Failed to enable feature gate for HugePages: %v", err)
	}
	if errs := validateVolumeSource(&hugePagesCase, field.NewPath("field").Index(0), "working"); len(errs) != 0 {
		t.Errorf("Unexpected error when HugePages feature is enabled.")
	}

	// Disable alpha feature HugePages
	err = utilfeature.DefaultFeatureGate.Set("HugePages=false")
	if err != nil {
		t.Errorf("Failed to disable feature gate for HugePages: %v", err)
	}
	if errs := validateVolumeSource(&hugePagesCase, field.NewPath("field").Index(0), "failing"); len(errs) == 0 {
		t.Errorf("Expected error when HugePages feature is disabled got nothing.")
	}

}

func TestAlphaPVCVolumeMode(t *testing.T) {
	// Enable alpha feature BlockVolume for PVC
	err := utilfeature.DefaultFeatureGate.Set("BlockVolume=true")
	if err != nil {
		t.Errorf("Failed to enable feature gate for BlockVolume: %v", err)
		return
	}

	block := core.PersistentVolumeBlock
	file := core.PersistentVolumeFilesystem
	fake := core.PersistentVolumeMode("fake")
	empty := core.PersistentVolumeMode("")

	// Success Cases
	successCasesPVC := map[string]*core.PersistentVolumeClaim{
		"valid block value":      createTestVolModePVC(&block),
		"valid filesystem value": createTestVolModePVC(&file),
		"valid nil value":        createTestVolModePVC(nil),
	}
	for k, v := range successCasesPVC {
		if errs := ValidatePersistentVolumeClaim(v); len(errs) != 0 {
			t.Errorf("expected success for %s", k)
		}
	}

	// Error Cases
	errorCasesPVC := map[string]*core.PersistentVolumeClaim{
		"invalid value": createTestVolModePVC(&fake),
		"empty value":   createTestVolModePVC(&empty),
	}
	for k, v := range errorCasesPVC {
		if errs := ValidatePersistentVolumeClaim(v); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestAlphaPVVolumeMode(t *testing.T) {
	// Enable alpha feature BlockVolume for PV
	err := utilfeature.DefaultFeatureGate.Set("BlockVolume=true")
	if err != nil {
		t.Errorf("Failed to enable feature gate for BlockVolume: %v", err)
		return
	}

	block := core.PersistentVolumeBlock
	file := core.PersistentVolumeFilesystem
	fake := core.PersistentVolumeMode("fake")
	empty := core.PersistentVolumeMode("")

	// Success Cases
	successCasesPV := map[string]*core.PersistentVolume{
		"valid block value":      createTestVolModePV(&block),
		"valid filesystem value": createTestVolModePV(&file),
		"valid nil value":        createTestVolModePV(nil),
	}
	for k, v := range successCasesPV {
		if errs := ValidatePersistentVolume(v); len(errs) != 0 {
			t.Errorf("expected success for %s", k)
		}
	}

	// Error Cases
	errorCasesPV := map[string]*core.PersistentVolume{
		"invalid value": createTestVolModePV(&fake),
		"empty value":   createTestVolModePV(&empty),
	}
	for k, v := range errorCasesPV {
		if errs := ValidatePersistentVolume(v); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func createTestVolModePVC(vmode *core.PersistentVolumeMode) *core.PersistentVolumeClaim {
	validName := "valid-storage-class"

	pvc := core.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
		},
		Spec: core.PersistentVolumeClaimSpec{
			Resources: core.ResourceRequirements{
				Requests: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
			},
			AccessModes:      []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
			StorageClassName: &validName,
			VolumeMode:       vmode,
		},
	}
	return &pvc
}

func createTestVolModePV(vmode *core.PersistentVolumeMode) *core.PersistentVolume {

	// PersistentVolume with VolumeMode set (valid and invalid)
	pv := core.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "",
		},
		Spec: core.PersistentVolumeSpec{
			Capacity: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
			AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
			PersistentVolumeSource: core.PersistentVolumeSource{
				HostPath: &core.HostPathVolumeSource{
					Path: "/foo",
					Type: newHostPathType(string(core.HostPathDirectory)),
				},
			},
			StorageClassName: "test-storage-class",
			VolumeMode:       vmode,
		},
	}
	return &pv
}

func createTestPV() *core.PersistentVolume {

	// PersistentVolume with VolumeMode set (valid and invalid)
	pv := core.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "",
		},
		Spec: core.PersistentVolumeSpec{
			Capacity: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
			AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
			PersistentVolumeSource: core.PersistentVolumeSource{
				HostPath: &core.HostPathVolumeSource{
					Path: "/foo",
					Type: newHostPathType(string(core.HostPathDirectory)),
				},
			},
			StorageClassName: "test-storage-class",
		},
	}
	return &pv
}

func TestAlphaLocalStorageCapacityIsolation(t *testing.T) {

	testCases := []core.VolumeSource{
		{EmptyDir: &core.EmptyDirVolumeSource{SizeLimit: resource.NewQuantity(int64(5), resource.BinarySI)}},
	}
	// Enable alpha feature LocalStorageCapacityIsolation
	err := utilfeature.DefaultFeatureGate.Set("LocalStorageCapacityIsolation=true")
	if err != nil {
		t.Errorf("Failed to enable feature gate for LocalStorageCapacityIsolation: %v", err)
		return
	}
	for _, tc := range testCases {
		if errs := validateVolumeSource(&tc, field.NewPath("spec"), "tmpvol"); len(errs) != 0 {
			t.Errorf("expected success: %v", errs)
		}
	}
	// Disable alpha feature LocalStorageCapacityIsolation
	err = utilfeature.DefaultFeatureGate.Set("LocalStorageCapacityIsolation=false")
	if err != nil {
		t.Errorf("Failed to disable feature gate for LocalStorageCapacityIsolation: %v", err)
		return
	}
	for _, tc := range testCases {
		if errs := validateVolumeSource(&tc, field.NewPath("spec"), "tmpvol"); len(errs) == 0 {
			t.Errorf("expected failure: %v", errs)
		}
	}

	containerLimitCase := core.ResourceRequirements{
		Limits: core.ResourceList{
			core.ResourceEphemeralStorage: *resource.NewMilliQuantity(
				int64(40000),
				resource.BinarySI),
		},
	}
	// Enable alpha feature LocalStorageCapacityIsolation
	err = utilfeature.DefaultFeatureGate.Set("LocalStorageCapacityIsolation=true")
	if err != nil {
		t.Errorf("Failed to enable feature gate for LocalStorageCapacityIsolation: %v", err)
		return
	}
	if errs := ValidateResourceRequirements(&containerLimitCase, field.NewPath("resources")); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}
	// Disable alpha feature LocalStorageCapacityIsolation
	err = utilfeature.DefaultFeatureGate.Set("LocalStorageCapacityIsolation=false")
	if err != nil {
		t.Errorf("Failed to disable feature gate for LocalStorageCapacityIsolation: %v", err)
		return
	}
	if errs := ValidateResourceRequirements(&containerLimitCase, field.NewPath("resources")); len(errs) == 0 {
		t.Errorf("expected failure: %v", errs)
	}

}

func TestValidateVolumeMounts(t *testing.T) {
	volumes := []core.Volume{
		{Name: "abc", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim1"}}},
		{Name: "abc-123", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim2"}}},
		{Name: "123", VolumeSource: core.VolumeSource{HostPath: &core.HostPathVolumeSource{Path: "/foo/baz", Type: newHostPathType(string(core.HostPathUnset))}}},
	}
	vols, v1err := ValidateVolumes(volumes, field.NewPath("field"))
	if len(v1err) > 0 {
		t.Errorf("Invalid test volume - expected success %v", v1err)
		return
	}
	container := core.Container{
		SecurityContext: nil,
	}
	propagation := core.MountPropagationBidirectional

	successCase := []core.VolumeMount{
		{Name: "abc", MountPath: "/foo"},
		{Name: "123", MountPath: "/bar"},
		{Name: "abc-123", MountPath: "/baz"},
		{Name: "abc-123", MountPath: "/baa", SubPath: ""},
		{Name: "abc-123", MountPath: "/bab", SubPath: "baz"},
		{Name: "abc-123", MountPath: "d:", SubPath: ""},
		{Name: "abc-123", MountPath: "F:", SubPath: ""},
		{Name: "abc-123", MountPath: "G:\\mount", SubPath: ""},
		{Name: "abc-123", MountPath: "/bac", SubPath: ".baz"},
		{Name: "abc-123", MountPath: "/bad", SubPath: "..baz"},
	}
	goodVolumeDevices := []core.VolumeDevice{
		{Name: "xyz", DevicePath: "/foofoo"},
		{Name: "uvw", DevicePath: "/foofoo/share/test"},
	}
	if errs := ValidateVolumeMounts(successCase, GetVolumeDeviceMap(goodVolumeDevices), vols, &container, field.NewPath("field")); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	errorCases := map[string][]core.VolumeMount{
		"empty name":                             {{Name: "", MountPath: "/foo"}},
		"name not found":                         {{Name: "", MountPath: "/foo"}},
		"empty mountpath":                        {{Name: "abc", MountPath: ""}},
		"mountpath collision":                    {{Name: "foo", MountPath: "/path/a"}, {Name: "bar", MountPath: "/path/a"}},
		"absolute subpath":                       {{Name: "abc", MountPath: "/bar", SubPath: "/baz"}},
		"subpath in ..":                          {{Name: "abc", MountPath: "/bar", SubPath: "../baz"}},
		"subpath contains ..":                    {{Name: "abc", MountPath: "/bar", SubPath: "baz/../bat"}},
		"subpath ends in ..":                     {{Name: "abc", MountPath: "/bar", SubPath: "./.."}},
		"disabled MountPropagation feature gate": {{Name: "abc", MountPath: "/bar", MountPropagation: &propagation}},
		"name exists in volumeDevice":            {{Name: "xyz", MountPath: "/bar"}},
		"mountpath exists in volumeDevice":       {{Name: "uvw", MountPath: "/mnt/exists"}},
		"both exist in volumeDevice":             {{Name: "xyz", MountPath: "/mnt/exists"}},
	}
	badVolumeDevice := []core.VolumeDevice{
		{Name: "xyz", DevicePath: "/mnt/exists"},
	}

	for k, v := range errorCases {
		if errs := ValidateVolumeMounts(v, GetVolumeDeviceMap(badVolumeDevice), vols, &container, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateDisabledSubpath(t *testing.T) {
	utilfeature.DefaultFeatureGate.Set("VolumeSubpath=false")
	defer utilfeature.DefaultFeatureGate.Set("VolumeSubpath=true")

	volumes := []core.Volume{
		{Name: "abc", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim1"}}},
		{Name: "abc-123", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim2"}}},
		{Name: "123", VolumeSource: core.VolumeSource{HostPath: &core.HostPathVolumeSource{Path: "/foo/baz", Type: newHostPathType(string(core.HostPathUnset))}}},
	}
	vols, v1err := ValidateVolumes(volumes, field.NewPath("field"))
	if len(v1err) > 0 {
		t.Errorf("Invalid test volume - expected success %v", v1err)
		return
	}

	container := core.Container{
		SecurityContext: nil,
	}

	goodVolumeDevices := []core.VolumeDevice{
		{Name: "xyz", DevicePath: "/foofoo"},
		{Name: "uvw", DevicePath: "/foofoo/share/test"},
	}

	cases := map[string]struct {
		mounts      []core.VolumeMount
		expectError bool
	}{
		"subpath not specified": {
			[]core.VolumeMount{
				{
					Name:      "abc-123",
					MountPath: "/bab",
				},
			},
			false,
		},
		"subpath specified": {
			[]core.VolumeMount{
				{
					Name:      "abc-123",
					MountPath: "/bab",
					SubPath:   "baz",
				},
			},
			true,
		},
	}

	for name, test := range cases {
		errs := ValidateVolumeMounts(test.mounts, GetVolumeDeviceMap(goodVolumeDevices), vols, &container, field.NewPath("field"))

		if len(errs) != 0 && !test.expectError {
			t.Errorf("test %v failed: %+v", name, errs)
		}

		if len(errs) == 0 && test.expectError {
			t.Errorf("test %v failed, expected error", name)
		}
	}
}

func TestValidateMountPropagation(t *testing.T) {
	bTrue := true
	bFalse := false
	privilegedContainer := &core.Container{
		SecurityContext: &core.SecurityContext{
			Privileged: &bTrue,
		},
	}
	nonPrivilegedContainer := &core.Container{
		SecurityContext: &core.SecurityContext{
			Privileged: &bFalse,
		},
	}
	defaultContainer := &core.Container{}

	propagationBidirectional := core.MountPropagationBidirectional
	propagationHostToContainer := core.MountPropagationHostToContainer
	propagationInvalid := core.MountPropagationMode("invalid")

	tests := []struct {
		mount       core.VolumeMount
		container   *core.Container
		expectError bool
	}{
		{
			// implicitly non-privileged container + no propagation
			core.VolumeMount{Name: "foo", MountPath: "/foo"},
			defaultContainer,
			false,
		},
		{
			// implicitly non-privileged container + HostToContainer
			core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationHostToContainer},
			defaultContainer,
			false,
		},
		{
			// error: implicitly non-privileged container + Bidirectional
			core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationBidirectional},
			defaultContainer,
			true,
		},
		{
			// explicitly non-privileged container + no propagation
			core.VolumeMount{Name: "foo", MountPath: "/foo"},
			nonPrivilegedContainer,
			false,
		},
		{
			// explicitly non-privileged container + HostToContainer
			core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationHostToContainer},
			nonPrivilegedContainer,
			false,
		},
		{
			// explicitly non-privileged container + HostToContainer
			core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationBidirectional},
			nonPrivilegedContainer,
			true,
		},
		{
			// privileged container + no propagation
			core.VolumeMount{Name: "foo", MountPath: "/foo"},
			privilegedContainer,
			false,
		},
		{
			// privileged container + HostToContainer
			core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationHostToContainer},
			privilegedContainer,
			false,
		},
		{
			// privileged container + Bidirectional
			core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationBidirectional},
			privilegedContainer,
			false,
		},
		{
			// error: privileged container + invalid mount propagation
			core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationInvalid},
			privilegedContainer,
			true,
		},
		{
			// no container + Bidirectional
			core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationBidirectional},
			nil,
			false,
		},
	}

	// Enable MountPropagation for this test
	priorityEnabled := utilfeature.DefaultFeatureGate.Enabled("MountPropagation")
	defer func() {
		var err error
		// restoring the old value
		if priorityEnabled {
			err = utilfeature.DefaultFeatureGate.Set("MountPropagation=true")
		} else {
			err = utilfeature.DefaultFeatureGate.Set("MountPropagation=false")
		}
		if err != nil {
			t.Errorf("Failed to restore feature gate for MountPropagation: %v", err)
		}
	}()
	err := utilfeature.DefaultFeatureGate.Set("MountPropagation=true")
	if err != nil {
		t.Errorf("Failed to enable feature gate for MountPropagation: %v", err)
		return
	}

	volumes := []core.Volume{
		{Name: "foo", VolumeSource: core.VolumeSource{HostPath: &core.HostPathVolumeSource{Path: "/foo/baz", Type: newHostPathType(string(core.HostPathUnset))}}},
	}
	vols2, v2err := ValidateVolumes(volumes, field.NewPath("field"))
	if len(v2err) > 0 {
		t.Errorf("Invalid test volume - expected success %v", v2err)
		return
	}
	for i, test := range tests {
		errs := ValidateVolumeMounts([]core.VolumeMount{test.mount}, nil, vols2, test.container, field.NewPath("field"))
		if test.expectError && len(errs) == 0 {
			t.Errorf("test %d expected error, got none", i)
		}
		if !test.expectError && len(errs) != 0 {
			t.Errorf("test %d expected success, got error: %v", i, errs)
		}
	}
}

func TestAlphaValidateVolumeDevices(t *testing.T) {
	volumes := []core.Volume{
		{Name: "abc", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim1"}}},
		{Name: "abc-123", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim2"}}},
		{Name: "def", VolumeSource: core.VolumeSource{HostPath: &core.HostPathVolumeSource{Path: "/foo/baz", Type: newHostPathType(string(core.HostPathUnset))}}},
	}

	vols, v1err := ValidateVolumes(volumes, field.NewPath("field"))
	if len(v1err) > 0 {
		t.Errorf("Invalid test volumes - expected success %v", v1err)
		return
	}

	disabledAlphaVolDevice := []core.VolumeDevice{
		{Name: "abc", DevicePath: "/foo"},
	}

	successCase := []core.VolumeDevice{
		{Name: "abc", DevicePath: "/foo"},
		{Name: "abc-123", DevicePath: "/usr/share/test"},
	}
	goodVolumeMounts := []core.VolumeMount{
		{Name: "xyz", MountPath: "/foofoo"},
		{Name: "ghi", MountPath: "/foo/usr/share/test"},
	}

	errorCases := map[string][]core.VolumeDevice{
		"empty name":                    {{Name: "", DevicePath: "/foo"}},
		"duplicate name":                {{Name: "abc", DevicePath: "/foo"}, {Name: "abc", DevicePath: "/foo/bar"}},
		"name not found":                {{Name: "not-found", DevicePath: "/usr/share/test"}},
		"name found but invalid source": {{Name: "def", DevicePath: "/usr/share/test"}},
		"empty devicepath":              {{Name: "abc", DevicePath: ""}},
		"relative devicepath":           {{Name: "abc-123", DevicePath: "baz"}},
		"duplicate devicepath":          {{Name: "abc", DevicePath: "/foo"}, {Name: "abc-123", DevicePath: "/foo"}},
		"no backsteps":                  {{Name: "def", DevicePath: "/baz/../"}},
		"name exists in volumemounts":   {{Name: "abc", DevicePath: "/baz/../"}},
		"path exists in volumemounts":   {{Name: "xyz", DevicePath: "/this/path/exists"}},
		"both exist in volumemounts":    {{Name: "abc", DevicePath: "/this/path/exists"}},
	}
	badVolumeMounts := []core.VolumeMount{
		{Name: "abc", MountPath: "/foo"},
		{Name: "abc-123", MountPath: "/this/path/exists"},
	}

	// enable Alpha BlockVolume
	err1 := utilfeature.DefaultFeatureGate.Set("BlockVolume=true")
	if err1 != nil {
		t.Errorf("Failed to enable feature gate for BlockVolume: %v", err1)
		return
	}
	// Success Cases:
	// Validate normal success cases - only PVC volumeSource
	if errs := ValidateVolumeDevices(successCase, GetVolumeMountMap(goodVolumeMounts), vols, field.NewPath("field")); len(errs) != 0 {
		t.Errorf("expected success: %v", errs)
	}

	// Error Cases:
	// Validate normal error cases - only PVC volumeSource
	for k, v := range errorCases {
		if errs := ValidateVolumeDevices(v, GetVolumeMountMap(badVolumeMounts), vols, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("expected failure for %s", k)
		}
	}

	// disable Alpha BlockVolume
	err2 := utilfeature.DefaultFeatureGate.Set("BlockVolume=false")
	if err2 != nil {
		t.Errorf("Failed to disable feature gate for BlockVolume: %v", err2)
		return
	}
	if errs := ValidateVolumeDevices(disabledAlphaVolDevice, GetVolumeMountMap(goodVolumeMounts), vols, field.NewPath("field")); len(errs) == 0 {
		t.Errorf("expected failure: %v", errs)
	}
}

func TestValidatePersistentVolumeClaimStatusUpdate(t *testing.T) {
	validClaim := testVolumeClaim("foo", "ns", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
	})
	validConditionUpdate := testVolumeClaimWithStatus("foo", "ns", core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
			core.ReadOnlyMany,
		},
		Resources: core.ResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
	}, core.PersistentVolumeClaimStatus{
		Phase: core.ClaimPending,
		Conditions: []core.PersistentVolumeClaimCondition{
			{Type: core.PersistentVolumeClaimResizing, Status: core.ConditionTrue},
		},
	})
	scenarios := map[string]struct {
		isExpectedFailure bool
		oldClaim          *core.PersistentVolumeClaim
		newClaim          *core.PersistentVolumeClaim
		enableResize      bool
	}{
		"condition-update-with-disabled-feature-gate": {
			isExpectedFailure: true,
			oldClaim:          validClaim,
			newClaim:          validConditionUpdate,
			enableResize:      false,
		},
		"condition-update-with-enabled-feature-gate": {
			isExpectedFailure: false,
			oldClaim:          validClaim,
			newClaim:          validConditionUpdate,
			enableResize:      true,
		},
	}
	for name, scenario := range scenarios {
		// ensure we have a resource version specified for updates
		togglePVExpandFeature(scenario.enableResize, t)
		scenario.oldClaim.ResourceVersion = "1"
		scenario.newClaim.ResourceVersion = "1"
		errs := ValidatePersistentVolumeClaimStatusUpdate(scenario.newClaim, scenario.oldClaim)
		if len(errs) == 0 && scenario.isExpectedFailure {
			t.Errorf("Unexpected success for scenario: %s", name)
		}
		if len(errs) > 0 && !scenario.isExpectedFailure {
			t.Errorf("Unexpected failure for scenario: %s - %+v", name, errs)
		}
	}
}

func TestValidateFlexVolumeSource(t *testing.T) {
	testcases := map[string]struct {
		source       *core.FlexVolumeSource
		expectedErrs map[string]string
	}{
		"valid": {
			source:       &core.FlexVolumeSource{Driver: "foo"},
			expectedErrs: map[string]string{},
		},
		"valid with options": {
			source:       &core.FlexVolumeSource{Driver: "foo", Options: map[string]string{"foo": "bar"}},
			expectedErrs: map[string]string{},
		},
		"no driver": {
			source:       &core.FlexVolumeSource{Driver: ""},
			expectedErrs: map[string]string{"driver": "Required value"},
		},
		"reserved option keys": {
			source: &core.FlexVolumeSource{
				Driver: "foo",
				Options: map[string]string{
					// valid options
					"myns.io":               "A",
					"myns.io/bar":           "A",
					"myns.io/kubernetes.io": "A",

					// invalid options
					"KUBERNETES.IO":     "A",
					"kubernetes.io":     "A",
					"kubernetes.io/":    "A",
					"kubernetes.io/foo": "A",

					"alpha.kubernetes.io":     "A",
					"alpha.kubernetes.io/":    "A",
					"alpha.kubernetes.io/foo": "A",

					"k8s.io":     "A",
					"k8s.io/":    "A",
					"k8s.io/foo": "A",

					"alpha.k8s.io":     "A",
					"alpha.k8s.io/":    "A",
					"alpha.k8s.io/foo": "A",
				},
			},
			expectedErrs: map[string]string{
				"options[KUBERNETES.IO]":           "reserved",
				"options[kubernetes.io]":           "reserved",
				"options[kubernetes.io/]":          "reserved",
				"options[kubernetes.io/foo]":       "reserved",
				"options[alpha.kubernetes.io]":     "reserved",
				"options[alpha.kubernetes.io/]":    "reserved",
				"options[alpha.kubernetes.io/foo]": "reserved",
				"options[k8s.io]":                  "reserved",
				"options[k8s.io/]":                 "reserved",
				"options[k8s.io/foo]":              "reserved",
				"options[alpha.k8s.io]":            "reserved",
				"options[alpha.k8s.io/]":           "reserved",
				"options[alpha.k8s.io/foo]":        "reserved",
			},
		},
	}

	for k, tc := range testcases {
		errs := validateFlexVolumeSource(tc.source, nil)
		for _, err := range errs {
			expectedErr, ok := tc.expectedErrs[err.Field]
			if !ok {
				t.Errorf("%s: unexpected err on field %s: %v", k, err.Field, err)
				continue
			}
			if !strings.Contains(err.Error(), expectedErr) {
				t.Errorf("%s: expected err on field %s to contain '%s', was %v", k, err.Field, expectedErr, err.Error())
				continue
			}
		}
		if len(errs) != len(tc.expectedErrs) {
			t.Errorf("%s: expected errs %#v, got %#v", k, tc.expectedErrs, errs)
			continue
		}
	}
}

func TestValidateOrSetClientIPAffinityConfig(t *testing.T) {
	successCases := map[string]*core.SessionAffinityConfig{
		"non-empty config, valid timeout: 1": {
			ClientIP: &core.ClientIPConfig{
				TimeoutSeconds: newInt32(1),
			},
		},
		"non-empty config, valid timeout: core.MaxClientIPServiceAffinitySeconds-1": {
			ClientIP: &core.ClientIPConfig{
				TimeoutSeconds: newInt32(int(core.MaxClientIPServiceAffinitySeconds - 1)),
			},
		},
		"non-empty config, valid timeout: core.MaxClientIPServiceAffinitySeconds": {
			ClientIP: &core.ClientIPConfig{
				TimeoutSeconds: newInt32(int(core.MaxClientIPServiceAffinitySeconds)),
			},
		},
	}

	for name, test := range successCases {
		if errs := validateClientIPAffinityConfig(test, field.NewPath("field")); len(errs) != 0 {
			t.Errorf("case: %s, expected success: %v", name, errs)
		}
	}

	errorCases := map[string]*core.SessionAffinityConfig{
		"empty session affinity config": nil,
		"empty client IP config": {
			ClientIP: nil,
		},
		"empty timeoutSeconds": {
			ClientIP: &core.ClientIPConfig{
				TimeoutSeconds: nil,
			},
		},
		"non-empty config, invalid timeout: core.MaxClientIPServiceAffinitySeconds+1": {
			ClientIP: &core.ClientIPConfig{
				TimeoutSeconds: newInt32(int(core.MaxClientIPServiceAffinitySeconds + 1)),
			},
		},
		"non-empty config, invalid timeout: -1": {
			ClientIP: &core.ClientIPConfig{
				TimeoutSeconds: newInt32(-1),
			},
		},
		"non-empty config, invalid timeout: 0": {
			ClientIP: &core.ClientIPConfig{
				TimeoutSeconds: newInt32(0),
			},
		},
	}

	for name, test := range errorCases {
		if errs := validateClientIPAffinityConfig(test, field.NewPath("field")); len(errs) == 0 {
			t.Errorf("case: %v, expected failures: %v", name, errs)
		}
	}
}

func newHostPathType(pathType string) *core.HostPathType {
	hostPathType := new(core.HostPathType)
	*hostPathType = core.HostPathType(pathType)
	return hostPathType
}

func testVolume(name string, namespace string, spec core.PersistentVolumeSpec) *core.PersistentVolume {
	objMeta := metav1.ObjectMeta{Name: name}
	if namespace != "" {
		objMeta.Namespace = namespace
	}

	return &core.PersistentVolume{
		ObjectMeta: objMeta,
		Spec:       spec,
	}
}
