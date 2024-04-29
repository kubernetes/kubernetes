package declarative_test

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/features"
	apivalidationtesting "k8s.io/kubernetes/test/utils/apivalidation"
	"k8s.io/utils/ptr"
)

func TestValidatePersistentVolume(t *testing.T) {
	validMode := core.PersistentVolumeFilesystem
	invalidMode := core.PersistentVolumeMode("fakeVolumeMode")

	type options struct {
		enableReadWriteOncePod      bool
		enableVolumeAttributesClass bool
	}
	cases := []apivalidationtesting.TestCase[*core.PersistentVolume, options]{
		{
			Name: "good-volume",
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
		{
			Name: "good-volume-with-capacity-unit",
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
		{
			Name: "good-volume-without-capacity-unit",
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
		{
			Name: "good-volume-with-storage-class",
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
		{
			Name: "good-volume-with-retain-policy",
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
		{
			Name: "good-volume-with-volume-mode",
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
				VolumeMode: &validMode,
			}),
		},
		{
			Name: "invalid-accessmode",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.accessModes", SchemaField: "spec.accessModes[0]", Type: field.ErrorTypeNotSupported, BadValue: core.PersistentVolumeAccessMode("fakemode")},
			},
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
		{
			Name: "invalid-reclaimpolicy",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.persistentVolumeReclaimPolicy", Type: field.ErrorTypeNotSupported, BadValue: core.PersistentVolumeReclaimPolicy("fakeReclaimPolicy")},
			},
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
		{
			Name: "invalid-volume-mode",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.volumeMode", Detail: `supported values: "Block", "Filesystem"`, Type: field.ErrorTypeNotSupported, BadValue: core.PersistentVolumeMode("fakeVolumeMode")},
			},
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
				VolumeMode: &invalidMode,
			}),
		},
		{
			Name: "with-read-write-once-pod",
			Options: options{
				enableReadWriteOncePod: true,
			},
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{"ReadWriteOncePod"},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		{
			Name: "with-read-write-once-pod-and-others",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.accessModes", Detail: "may not use ReadWriteOncePod with other access modes", Type: field.ErrorTypeForbidden, BadValue: ""},
			},
			Options: options{
				enableReadWriteOncePod: true,
			},
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{"ReadWriteOncePod", "ReadWriteMany"},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		{
			Name: "unexpected-namespace",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "metadata.namespace",
					BadValue: "",
					Detail:   "not allowed on this type",
					Type:     field.ErrorTypeForbidden,
				},
			},
			Object: testVolume("foo", "unexpected-namespace", core.PersistentVolumeSpec{
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
		{
			Name: "missing-volume-source",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec", Detail: "must specify a volume type", Type: field.ErrorTypeRequired, BadValue: ""},
			},
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
			}),
		},
		{
			Name: "bad-name",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "metadata.name", BadValue: "123*Bad(Name", Type: field.ErrorTypeInvalid},
				{Field: "metadata.namespace", Detail: "not allowed on this type", Type: field.ErrorTypeForbidden, BadValue: ""},
			},
			Object: testVolume("123*Bad(Name", "unexpected-namespace", core.PersistentVolumeSpec{
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
		{
			Name: "missing-name",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "metadata.name", Detail: "name or generateName is required", Type: field.ErrorTypeRequired, BadValue: ""},
			},
			Object: testVolume("", "", core.PersistentVolumeSpec{
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
		{
			Name: "missing-capacity",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.capacity", Type: field.ErrorTypeRequired, BadValue: ""},
				{Field: "spec.capacity", Type: field.ErrorTypeNotSupported, SchemaType: field.ErrorTypeInvalid, BadValue: core.ResourceList(nil)},
			},
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/foo",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}),
		},
		{
			Name: "bad-volume-zero-capacity",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.capacity[storage]", Detail: "must be greater than zero", Type: field.ErrorTypeInvalid, BadValue: "0"},
			},
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
		{
			Name: "missing-accessmodes",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.accessModes", Type: field.ErrorTypeRequired, BadValue: ""},
				{Field: "metadata.namespace", Detail: "not allowed on this type", Type: field.ErrorTypeForbidden, BadValue: ""},
			},
			Object: testVolume("goodname", "missing-accessmodes", core.PersistentVolumeSpec{
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
		{
			Name: "too-many-sources",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.accessModes", Type: field.ErrorTypeRequired, BadValue: ""},
				{Field: "spec.gcePersistentDisk", Detail: "may not specify more than 1 volume type", Type: field.ErrorTypeForbidden, BadValue: "", SchemaSkipReason: "Blocked by lack of CEL variables"},
			},
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
		{
			Name: "host mount of / with recycle reclaim policy",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.persistentVolumeReclaimPolicy", Detail: `may not be 'recycle' for a hostPath mount of '/'`, Type: field.ErrorTypeForbidden, BadValue: ""},
			},
			Object: testVolume("bad-recycle-do-not-want", "", core.PersistentVolumeSpec{
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
		{
			Name: "host mount of / with recycle reclaim policy 2",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.hostPath.path", Detail: `must not contain '..'`, Type: field.ErrorTypeInvalid, BadValue: "/a/.."},
				{
					Field:            "spec.persistentVolumeReclaimPolicy",
					Detail:           `may not be 'recycle' for a hostPath mount of '/'`,
					Type:             field.ErrorTypeForbidden,
					BadValue:         "",
					SchemaSkipReason: "Blocked by utils in CEL to simplify the path to `/`",
				},
			},
			Object: testVolume("bad-recycle-do-not-want", "", core.PersistentVolumeSpec{
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
		{
			Name: "invalid-storage-class-name",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.storageClassName",
					BadValue: "-invalid-",
					Type:     field.ErrorTypeInvalid,
				},
			},
			Object: testVolume("invalid-storage-class-name", "", core.PersistentVolumeSpec{
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
		{
			Name: "bad-hostpath-volume-backsteps",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.hostPath.path", Type: field.ErrorTypeInvalid, BadValue: "/foo/.."},
			},
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
		{
			Name:   "volume-node-affinity",
			Object: testVolumeWithNodeAffinity(simpleVolumeNodeAffinity("foo", "bar")),
		},
		{
			Name: "volume-empty-node-affinity",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.nodeAffinity.required", Type: field.ErrorTypeRequired, BadValue: ""},
			},
			Object: testVolumeWithNodeAffinity(&core.VolumeNodeAffinity{}),
		},
		{
			Name: "volume-bad-node-affinity",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.nodeAffinity.required.nodeSelectorTerms[0].matchExpressions[0].key", Detail: "name part must be non-empty", Type: field.ErrorTypeInvalid, BadValue: ""},
				{Field: "spec.nodeAffinity.required.nodeSelectorTerms[0].matchExpressions[0].key", Detail: "name part must consist of alphanumeric characters,", Type: field.ErrorTypeInvalid, BadValue: ""},
			},
			Object: testVolumeWithNodeAffinity(
				&core.VolumeNodeAffinity{
					Required: &core.NodeSelector{
						NodeSelectorTerms: []core.NodeSelectorTerm{{
							MatchExpressions: []core.NodeSelectorRequirement{{
								Operator: core.NodeSelectorOpIn,
								Values:   []string{"test-label-value"},
							}},
						}},
					},
				}),
		},
		{
			Name: "invalid-empty-volume-attributes-class-name",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.volumeAttributesClassName", Detail: "an empty string is disallowed", Type: field.ErrorTypeRequired, BadValue: ""},
				{Field: "spec.csi", Detail: "has to be specified when using volumeAttributesClassName", Type: field.ErrorTypeRequired, BadValue: ""},
			},
			Object: testVolume("invalid-empty-volume-attributes-class-name", "", core.PersistentVolumeSpec{
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
				StorageClassName:          "invalid",
				VolumeAttributesClassName: ptr.To(""),
			}),
			Options: options{
				enableVolumeAttributesClass: true,
			},
		},
		{
			Name: "volume-with-good-volume-attributes-class-and-matched-volume-resource-when-feature-gate-is-on",
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
				Capacity: core.ResourceList{
					core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
				},
				AccessModes: []core.PersistentVolumeAccessMode{core.ReadWriteOnce},
				PersistentVolumeSource: core.PersistentVolumeSource{
					CSI: &core.CSIPersistentVolumeSource{
						Driver:       "test-driver",
						VolumeHandle: "test-123",
					},
				},
				StorageClassName:          "valid",
				VolumeAttributesClassName: ptr.To("valid"),
			}),
			Options: options{
				enableVolumeAttributesClass: true,
			},
		},
		{
			Name: "volume-with-good-volume-attributes-class-and-mismatched-volume-resource-when-feature-gate-is-on",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.csi", Detail: "has to be specified when using volumeAttributesClassName", Type: field.ErrorTypeRequired, BadValue: ""},
			},
			Object: testVolume("foo", "", core.PersistentVolumeSpec{
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
				StorageClassName:          "valid",
				VolumeAttributesClassName: ptr.To("valid"),
			}),
			Options: options{
				enableVolumeAttributesClass: true,
			},
		},
	}

	apivalidationtesting.TestValidate(t, coreScheme, coreDefs, func(a *core.PersistentVolume, o options) field.ErrorList {
		featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeAttributesClass, o.enableVolumeAttributesClass)
		opts := validation.ValidationOptionsForPersistentVolume(a, nil)
		opts.EnableVolumeAttributesClass = o.enableVolumeAttributesClass
		return validation.ValidatePersistentVolume(a, opts)
	}, cases...)
}

func TestValidateLocalVolumes(t *testing.T) {
	testLocalVolume := func(path string, affinity *core.VolumeNodeAffinity) core.PersistentVolumeSpec {
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

	scenarios := []apivalidationtesting.TestCase[*core.PersistentVolume, struct{}]{
		{
			Name: "alpha invalid local volume nil annotations",
			Object: testVolume(
				"invalid-local-volume-nil-annotations",
				"",
				testLocalVolume("/foo", nil)),
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.nodeAffinity", Type: field.ErrorTypeRequired, Detail: `Local volume requires node affinity`},
			},
		},
		{
			Name: "valid local volume",
			Object: testVolume("valid-local-volume", "",
				testLocalVolume("/foo", simpleVolumeNodeAffinity("foo", "bar"))),
		},
		{
			Name: "invalid local volume no node affinity",
			Object: testVolume("invalid-local-volume-no-node-affinity", "",
				testLocalVolume("/foo", nil)),
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.nodeAffinity", Type: field.ErrorTypeRequired, Detail: `Local volume requires node affinity`},
			},
		},
		{
			Name: "invalid local volume empty path",
			Object: testVolume("invalid-local-volume-empty-path", "",
				testLocalVolume("", simpleVolumeNodeAffinity("foo", "bar"))),
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.local.path", Type: field.ErrorTypeRequired, SchemaType: field.ErrorTypeInvalid},
			},
		},
		{
			Name: "invalid-local-volume-backsteps",
			Object: testVolume("foo", "",
				testLocalVolume("/foo/..", simpleVolumeNodeAffinity("foo", "bar"))),
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.local.path", Type: field.ErrorTypeInvalid, BadValue: "/foo/.."},
			},
		},
		{
			Name: "valid-local-volume-relative-path",
			Object: testVolume("foo", "",
				testLocalVolume("foo", simpleVolumeNodeAffinity("foo", "bar"))),
		},
	}

	apivalidationtesting.TestValidate[core.PersistentVolume](t, coreScheme, coreDefs, func(a *core.PersistentVolume, _ struct{}) field.ErrorList {
		opts := validation.ValidationOptionsForPersistentVolume(a, nil)
		errs := validation.ValidatePersistentVolume(a, opts)
		return errs
	}, scenarios...)
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
			NodeSelectorTerms: []core.NodeSelectorTerm{{
				MatchExpressions: []core.NodeSelectorRequirement{{
					Key:      key,
					Operator: core.NodeSelectorOpIn,
					Values:   []string{value},
				}},
			}},
		},
	}
}
