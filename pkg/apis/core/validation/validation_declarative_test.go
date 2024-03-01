package validation_test

import (
	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/generated/openapi"
	apivalidationtesting "k8s.io/kubernetes/test/utils/apivalidation"

	// Ensure everything installed in schema
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

func TestValidatePersistentVolumesWithFramework(t *testing.T) {
	validMode := core.PersistentVolumeFilesystem
	invalidMode := core.PersistentVolumeMode("fakeVolumeMode")

	type options struct {
		enableReadWriteOncePod bool
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
				{Field: "spec.accessModes", Type: field.ErrorTypeNotSupported},
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
				{Field: "spec.persistentVolumeReclaimPolicy", Type: field.ErrorTypeNotSupported},
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
				{Field: "spec.volumeMode", Detail: `supported values: "Block", "Filesystem"`, Type: field.ErrorTypeNotSupported},
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
			Name: "with-read-write-once-pod-feature-gate-enabled",
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
			Name: "with-read-write-once-pod-and-others-feature-gate-enabled",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.accessModes", Detail: "may not use ReadWriteOncePod with other access modes", Type: field.ErrorTypeForbidden},
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
				{Field: "metadata.namespace", BadValue: "unexpected-namespace", Detail: "not allowed on this type", Type: field.ErrorTypeForbidden},
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
				{Field: "spec", Detail: "must specify a volume type", Type: field.ErrorTypeRequired},
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
				{Field: "metadata.namespace", BadValue: "unexpected-namespace", Detail: "not allowed on this type", Type: field.ErrorTypeForbidden},
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
				{Field: "metadata.name", Detail: "name or generateName is required", Type: field.ErrorTypeRequired},
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
				{Field: "spec.capacity", Type: field.ErrorTypeRequired},
				{Field: "spec.capacity", Type: field.ErrorTypeNotSupported, SchemaType: field.ErrorTypeInvalid},
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
				{Field: "spec.capacity[storage]", Detail: "must be greater than zero", Type: field.ErrorTypeInvalid},
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
				{Field: "spec.accessModes", Type: field.ErrorTypeRequired},
				{Field: "metadata.namespace", Detail: "not allowed on this type", Type: field.ErrorTypeForbidden},
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
				{Field: "spec.accessModes", Type: field.ErrorTypeRequired},
				{Field: "spec.gcePersistentDisk", Detail: "may not specify more than 1 volume type", Type: field.ErrorTypeForbidden},
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
				{Field: "spec.persistentVolumeReclaimPolicy", Detail: `may not be 'recycle' for a hostPath mount of '/'`, Type: field.ErrorTypeForbidden},
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
				{Field: "spec.hostPath.path", Detail: `must not contain '..'`, Type: field.ErrorTypeInvalid},
				{Field: "spec.persistentVolumeReclaimPolicy", Detail: `may not be 'recycle' for a hostPath mount of '/'`, Type: field.ErrorTypeForbidden},
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
				{Field: "spec.storageClassName", BadValue: "-invalid-", Type: field.ErrorTypeInvalid},
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
				{Field: "spec.hostPath.path", Type: field.ErrorTypeInvalid},
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
				{Field: "spec.nodeAffinity.required", Type: field.ErrorTypeRequired},
			},
			Object: testVolumeWithNodeAffinity(&core.VolumeNodeAffinity{}),
		},
		{
			Name: "volume-bad-node-affinity",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.nodeAffinity.required.nodeSelectorTerms[0].matchExpressions[0].key", Detail: "name part must be non-empty", Type: field.ErrorTypeInvalid},
				{Field: "spec.nodeAffinity.required.nodeSelectorTerms[0].matchExpressions[0].key", Detail: "name part must consist of alphanumeric characters,", Type: field.ErrorTypeInvalid},
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
	}

	defs := resolver.NewDefinitionsSchemaResolver(openapi.GetOpenAPIDefinitions, legacyscheme.Scheme)
	apivalidationtesting.TestValidate(t, legacyscheme.Scheme, defs, func(a *core.PersistentVolume, o options) field.ErrorList {
		defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ReadWriteOncePod, o.enableReadWriteOncePod)()

		opts := validation.ValidationOptionsForPersistentVolume(a, nil)
		return validation.ValidatePersistentVolume(a, opts)
	}, cases...)
}

type topologyPair struct {
	key   string
	value string
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

func multipleVolumeNodeAffinity(terms [][]topologyPair) *core.VolumeNodeAffinity {
	nodeSelectorTerms := []core.NodeSelectorTerm{}
	for _, term := range terms {
		matchExpressions := []core.NodeSelectorRequirement{}
		for _, topology := range term {
			matchExpressions = append(matchExpressions, core.NodeSelectorRequirement{
				Key:      topology.key,
				Operator: core.NodeSelectorOpIn,
				Values:   []string{topology.value},
			})
		}
		nodeSelectorTerms = append(nodeSelectorTerms, core.NodeSelectorTerm{
			MatchExpressions: matchExpressions,
		})
	}

	return &core.VolumeNodeAffinity{
		Required: &core.NodeSelector{
			NodeSelectorTerms: nodeSelectorTerms,
		},
	}
}
