package declarative_test

import (
	"fmt"
	"math"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/cel/openapi/resolver"
	"k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/capabilities"
	"k8s.io/kubernetes/pkg/generated/openapi"
	apivalidationtesting "k8s.io/kubernetes/test/utils/apivalidation"
	"k8s.io/utils/ptr"
)

var (
	coreScheme *runtime.Scheme = func() *runtime.Scheme {
		sch := runtime.NewScheme()
		_ = core.AddToScheme(sch)
		_ = corev1.AddToScheme(sch)
		return sch
	}()
	coreDefs *resolver.DefinitionsSchemaResolver = resolver.NewDefinitionsSchemaResolver(openapi.GetOpenAPIDefinitions, coreScheme)
)

func TestValidatePod(t *testing.T) {
	validPodSpec := func(affinity *core.Affinity) core.PodSpec {
		spec := core.PodSpec{
			Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			RestartPolicy: core.RestartPolicyAlways,
			DNSPolicy:     core.DNSClusterFirst,
		}
		if affinity != nil {
			spec.Affinity = affinity
		}
		return spec
	}

	validPVCSpec := core.PersistentVolumeClaimSpec{
		AccessModes: []core.PersistentVolumeAccessMode{
			core.ReadWriteOnce,
		},
		Resources: core.VolumeResourceRequirements{
			Requests: core.ResourceList{
				core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
			},
		},
	}

	validPVCTemplate := core.PersistentVolumeClaimTemplate{
		Spec: validPVCSpec,
	}

	longPodName := strings.Repeat("a", 200)
	longVolName := strings.Repeat("b", 60)

	type options struct {
	}

	cases := []apivalidationtesting.TestCase[*core.Pod, options]{
		{
			Name: "basic fields",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "just about everything",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "abc.123.do-re-mi", Namespace: "ns"},
				Spec: core.PodSpec{
					Volumes: []core.Volume{
						{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
					},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					NodeSelector: map[string]string{
						"key": "value",
					},
					NodeName: "foobar",
				},
			},
		},
		{
			Name: "serialized node affinity requirements",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(
					&core.Affinity{
						NodeAffinity: &core.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
								NodeSelectorTerms: []core.NodeSelectorTerm{{
									MatchExpressions: []core.NodeSelectorRequirement{{
										Key:      "key2",
										Operator: core.NodeSelectorOpIn,
										Values:   []string{"value1", "value2"},
									}},
									MatchFields: []core.NodeSelectorRequirement{{
										Key:      "metadata.name",
										Operator: core.NodeSelectorOpIn,
										Values:   []string{"host1"},
									}},
								}},
							},
							PreferredDuringSchedulingIgnoredDuringExecution: []core.PreferredSchedulingTerm{{
								Weight: 10,
								Preference: core.NodeSelectorTerm{
									MatchExpressions: []core.NodeSelectorRequirement{{
										Key:      "foo",
										Operator: core.NodeSelectorOpIn,
										Values:   []string{"bar"},
									}},
								},
							}},
						},
					},
				),
			},
		},
		{
			Name: "serialized node affinity requirements, II",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(
					&core.Affinity{
						NodeAffinity: &core.NodeAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
								NodeSelectorTerms: []core.NodeSelectorTerm{{
									MatchExpressions: []core.NodeSelectorRequirement{},
								}},
							},
							PreferredDuringSchedulingIgnoredDuringExecution: []core.PreferredSchedulingTerm{{
								Weight: 10,
								Preference: core.NodeSelectorTerm{
									MatchExpressions: []core.NodeSelectorRequirement{},
								},
							}},
						},
					},
				),
			},
		},
		{
			Name: "serialized pod affinity in affinity requirements in annotations",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{{
									Key:      "key2",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"value1", "value2"},
								}},
							},
							TopologyKey: "zone",
							Namespaces:  []string{"ns"},
							NamespaceSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{{
									Key:      "key",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"value1", "value2"},
								}},
							},
						}},
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{{
							Weight: 10,
							PodAffinityTerm: core.PodAffinityTerm{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{{
										Key:      "key2",
										Operator: metav1.LabelSelectorOpNotIn,
										Values:   []string{"value1", "value2"},
									}},
								},
								Namespaces:  []string{"ns"},
								TopologyKey: "region",
							},
						}},
					},
				}),
			},
		},
		{
			Name: "serialized pod anti affinity with different Label Operators in affinity requirements in annotations",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{{
									Key:      "key2",
									Operator: metav1.LabelSelectorOpExists,
								}},
							},
							TopologyKey: "zone",
							Namespaces:  []string{"ns"},
						}},
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{{
							Weight: 10,
							PodAffinityTerm: core.PodAffinityTerm{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{{
										Key:      "key2",
										Operator: metav1.LabelSelectorOpDoesNotExist,
									}},
								},
								Namespaces:  []string{"ns"},
								TopologyKey: "region",
							},
						}},
					},
				}),
			},
		},
		{Name: "populate forgiveness tolerations with exists operator in annotations.",
			Object: &core.Pod{ObjectMeta: metav1.ObjectMeta{
				Name:      "123",
				Namespace: "ns",
			},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "foo", Operator: "Exists", Value: "", Effect: "NoExecute", TolerationSeconds: &[]int64{60}[0]}}),
			},
		},
		{
			Name: "populate forgiveness tolerations with equal operator in annotations.",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{
					{Key: "foo", Operator: core.TolerationOpEqual, Value: "bar", Effect: core.TaintEffectNoExecute, TolerationSeconds: &[]int64{60}[0]},
				}),
			},
		},
		{
			Name: "populate tolerations equal operator in annotations.",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{
					{Key: "foo", Operator: core.TolerationOpEqual, Value: "bar", Effect: core.TaintEffectNoSchedule},
				}),
			},
		},
		{
			Name: "populate tolerations exists operator in annotations.",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(nil),
			},
		},
		{
			Name: "empty key with Exists operator is OK for toleration, empty toleration key means match all taint keys.",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Operator: "Exists", Effect: "NoSchedule"}}),
			},
		},
		{
			Name: "empty operator is OK for toleration, defaults to Equal.",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "foo", Value: "bar", Effect: "NoSchedule"}}),
			},
		},
		{Name: "empty effect is OK for toleration, empty toleration effect means match all taint effects.",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "foo", Operator: "Equal", Value: "bar"}}),
			},
		},
		{Name: "negative tolerationSeconds is OK for toleration.",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod-forgiveness-invalid",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "node.kubernetes.io/not-ready", Operator: "Exists", Effect: "NoExecute", TolerationSeconds: &[]int64{-2}[0]}}),
			},
		},
		{Name: "runtime default seccomp profile",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompPodAnnotationKey: core.SeccompProfileRuntimeDefault,
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{Name: "docker default seccomp profile",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompPodAnnotationKey: core.DeprecatedSeccompProfileDockerDefault,
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{Name: "unconfined seccomp profile",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompPodAnnotationKey: "unconfined",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{Name: "localhost seccomp profile",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompPodAnnotationKey: "localhost/foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{Name: "localhost seccomp profile for a container",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompContainerAnnotationKeyPrefix + "foo": "localhost/foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{Name: "runtime default seccomp profile for a pod",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						SeccompProfile: &core.SeccompProfile{
							Type: core.SeccompProfileTypeRuntimeDefault,
						},
					},
				},
			},
		},
		{
			Name: "runtime default seccomp profile for a container",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						SecurityContext: &core.SecurityContext{
							SeccompProfile: &core.SeccompProfile{
								Type: core.SeccompProfileTypeRuntimeDefault,
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
				},
			},
		},
		{Name: "unconfined seccomp profile for a pod",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						SeccompProfile: &core.SeccompProfile{
							Type: core.SeccompProfileTypeUnconfined,
						},
					},
				},
			},
		},
		{Name: "unconfined seccomp profile for a container",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						SecurityContext: &core.SecurityContext{
							SeccompProfile: &core.SeccompProfile{
								Type: core.SeccompProfileTypeUnconfined,
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
				},
			},
		},
		{Name: "localhost seccomp profile for a pod",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						SeccompProfile: &core.SeccompProfile{
							Type:             core.SeccompProfileTypeLocalhost,
							LocalhostProfile: ptr.To("filename.json"),
						},
					},
				},
			},
		},
		{Name: "localhost seccomp profile for a container, II",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						SecurityContext: &core.SecurityContext{
							SeccompProfile: &core.SeccompProfile{
								Type:             core.SeccompProfileTypeLocalhost,
								LocalhostProfile: ptr.To("filename.json"),
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
				},
			},
		},
		{Name: "default AppArmor annotation for a container",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						corev1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "ctr": corev1.DeprecatedAppArmorBetaProfileRuntimeDefault,
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{Name: "default AppArmor annotation for an init container",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						corev1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "init-ctr": corev1.DeprecatedAppArmorBetaProfileRuntimeDefault,
					},
				},
				Spec: core.PodSpec{
					InitContainers: []core.Container{{Name: "init-ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					Containers:     []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy:  core.RestartPolicyAlways,
					DNSPolicy:      core.DNSClusterFirst,
				},
			},
		},
		{Name: "localhost AppArmor annotation for a container",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						corev1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "ctr": corev1.DeprecatedAppArmorBetaProfileNamePrefix + "foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{Name: "runtime default AppArmor profile for a pod",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						AppArmorProfile: &core.AppArmorProfile{
							Type: core.AppArmorProfileTypeRuntimeDefault,
						},
					},
				},
			},
		},
		{Name: "runtime default AppArmor profile for a container",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						SecurityContext: &core.SecurityContext{
							AppArmorProfile: &core.AppArmorProfile{
								Type: core.AppArmorProfileTypeRuntimeDefault,
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
				},
			},
		},
		{Name: "unconfined AppArmor profile for a pod",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						AppArmorProfile: &core.AppArmorProfile{
							Type: core.AppArmorProfileTypeUnconfined,
						},
					},
				},
			},
		},
		{Name: "unconfined AppArmor profile for a container",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						SecurityContext: &core.SecurityContext{
							AppArmorProfile: &core.AppArmorProfile{
								Type: core.AppArmorProfileTypeUnconfined,
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
				},
			},
		},
		{Name: "localhost AppArmor profile for a pod",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						AppArmorProfile: &core.AppArmorProfile{
							Type:             core.AppArmorProfileTypeLocalhost,
							LocalhostProfile: ptr.To("example-org/application-foo"),
						},
					},
				},
			},
		},
		{Name: "localhost AppArmor profile for a container field",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						SecurityContext: &core.SecurityContext{
							AppArmorProfile: &core.AppArmorProfile{
								Type:             core.AppArmorProfileTypeLocalhost,
								LocalhostProfile: ptr.To("example-org/application-foo"),
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
				},
			},
		},
		{Name: "matching AppArmor fields and annotations",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.DeprecatedAppArmorAnnotationKeyPrefix + "ctr": core.DeprecatedAppArmorAnnotationValueLocalhostPrefix + "foo",
					},
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						SecurityContext: &core.SecurityContext{
							AppArmorProfile: &core.AppArmorProfile{
								Type:             core.AppArmorProfileTypeLocalhost,
								LocalhostProfile: ptr.To("foo"),
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
				},
			},
		},
		{Name: "matching AppArmor pod field and annotations",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.DeprecatedAppArmorAnnotationKeyPrefix + "ctr": core.DeprecatedAppArmorAnnotationValueLocalhostPrefix + "foo",
					},
				},
				Spec: core.PodSpec{
					SecurityContext: &core.PodSecurityContext{
						AppArmorProfile: &core.AppArmorProfile{
							Type:             core.AppArmorProfileTypeLocalhost,
							LocalhostProfile: ptr.To("foo"),
						},
					},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
				},
			},
		},
		{Name: "syntactically valid sysctls",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					SecurityContext: &core.PodSecurityContext{
						Sysctls: []core.Sysctl{{
							Name:  "kernel.shmmni",
							Value: "32768",
						}, {
							Name:  "kernel.shmmax",
							Value: "1000000000",
						}, {
							Name:  "knet.ipv4.route.min_pmtu",
							Value: "1000",
						}},
					},
				},
			},
		},
		{Name: "valid extended resources for init container",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "valid-extended", Namespace: "ns"},
				Spec: core.PodSpec{
					InitContainers: []core.Container{{
						Name:            "valid-extended",
						Image:           "image",
						ImagePullPolicy: "IfNotPresent",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("10"),
							},
							Limits: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("10"),
							},
						},
						TerminationMessagePolicy: "File",
					}},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{Name: "valid extended resources for regular container",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "valid-extended", Namespace: "ns"},
				Spec: core.PodSpec{
					InitContainers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					Containers: []core.Container{{
						Name:            "valid-extended",
						Image:           "image",
						ImagePullPolicy: "IfNotPresent",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("10"),
							},
							Limits: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("10"),
							},
						},
						TerminationMessagePolicy: "File",
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{Name: "valid serviceaccount token projected volume with serviceaccount name specified",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "valid-extended", Namespace: "ns"},
				Spec: core.PodSpec{
					ServiceAccountName: "some-service-account",
					Containers:         []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy:      core.RestartPolicyAlways,
					DNSPolicy:          core.DNSClusterFirst,
					Volumes: []core.Volume{{
						Name: "projected-volume",
						VolumeSource: core.VolumeSource{
							Projected: &core.ProjectedVolumeSource{
								Sources: []core.VolumeProjection{{
									ServiceAccountToken: &core.ServiceAccountTokenProjection{
										Audience:          "foo-audience",
										ExpirationSeconds: 6000,
										Path:              "foo-path",
									},
								}},
							},
						},
					}},
				},
			},
		},
		{Name: "valid ClusterTrustBundlePEM projected volume referring to a CTB by name",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "valid-extended", Namespace: "ns"},
				Spec: core.PodSpec{
					ServiceAccountName: "some-service-account",
					Containers:         []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy:      core.RestartPolicyAlways,
					DNSPolicy:          core.DNSClusterFirst,
					Volumes: []core.Volume{
						{
							Name: "projected-volume",
							VolumeSource: core.VolumeSource{
								Projected: &core.ProjectedVolumeSource{
									Sources: []core.VolumeProjection{
										{
											ClusterTrustBundle: &core.ClusterTrustBundleProjection{
												Path: "foo-path",
												Name: ptr.To("foo"),
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
		{Name: "valid ClusterTrustBundlePEM projected volume referring to a CTB by signer name",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "valid-extended", Namespace: "ns"},
				Spec: core.PodSpec{
					ServiceAccountName: "some-service-account",
					Containers:         []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy:      core.RestartPolicyAlways,
					DNSPolicy:          core.DNSClusterFirst,
					Volumes: []core.Volume{
						{
							Name: "projected-volume",
							VolumeSource: core.VolumeSource{
								Projected: &core.ProjectedVolumeSource{
									Sources: []core.VolumeProjection{
										{
											ClusterTrustBundle: &core.ClusterTrustBundleProjection{
												Path:       "foo-path",
												SignerName: ptr.To("example.com/foo"),
												LabelSelector: &metav1.LabelSelector{
													MatchLabels: map[string]string{
														"version": "live",
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
			},
		},
		{Name: "ephemeral volume + PVC, no conflict between them",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Volumes: []core.Volume{
						{Name: "pvc", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "my-pvc"}}},
						{Name: "ephemeral", VolumeSource: core.VolumeSource{Ephemeral: &core.EphemeralVolumeSource{VolumeClaimTemplate: &validPVCTemplate}}},
					},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{Name: "negative pod-deletion-cost",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns", Annotations: map[string]string{core.PodDeletionCost: "-100"}},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{Name: "positive pod-deletion-cost",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns", Annotations: map[string]string{core.PodDeletionCost: "100"}},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{Name: "MatchLabelKeys/MismatchLabelKeys in required PodAffinity",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Affinity: &core.Affinity{
						PodAffinity: &core.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1", "value2"},
											},
											{
												Key:      "key2",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"value1"},
											},
											{
												Key:      "key3",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1"},
											},
										},
									},
									TopologyKey:       "k8s.io/zone",
									MatchLabelKeys:    []string{"key2"},
									MismatchLabelKeys: []string{"key3"},
								},
							},
						},
					},
				},
			},
		},
		{Name: "MatchLabelKeys/MismatchLabelKeys in preferred PodAffinity",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Affinity: &core.Affinity{
						PodAffinity: &core.PodAffinity{
							PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
								{
									Weight: 10,
									PodAffinityTerm: core.PodAffinityTerm{
										LabelSelector: &metav1.LabelSelector{
											MatchExpressions: []metav1.LabelSelectorRequirement{
												{
													Key:      "key",
													Operator: metav1.LabelSelectorOpNotIn,
													Values:   []string{"value1", "value2"},
												},
												{
													Key:      "key2",
													Operator: metav1.LabelSelectorOpIn,
													Values:   []string{"value1"},
												},
												{
													Key:      "key3",
													Operator: metav1.LabelSelectorOpNotIn,
													Values:   []string{"value1"},
												},
											},
										},
										TopologyKey:       "k8s.io/zone",
										MatchLabelKeys:    []string{"key2"},
										MismatchLabelKeys: []string{"key3"},
									},
								},
							},
						},
					},
				},
			},
		},
		{Name: "MatchLabelKeys/MismatchLabelKeys in required PodAntiAffinity",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Affinity: &core.Affinity{
						PodAntiAffinity: &core.PodAntiAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1", "value2"},
											},
											{
												Key:      "key2",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"value1"},
											},
											{
												Key:      "key3",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1"},
											},
										},
									},
									TopologyKey:       "k8s.io/zone",
									MatchLabelKeys:    []string{"key2"},
									MismatchLabelKeys: []string{"key3"},
								},
							},
						},
					},
				},
			},
		},
		{Name: "MatchLabelKeys/MismatchLabelKeys in preferred PodAntiAffinity",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Affinity: &core.Affinity{
						PodAntiAffinity: &core.PodAntiAffinity{
							PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
								{
									Weight: 10,
									PodAffinityTerm: core.PodAffinityTerm{
										LabelSelector: &metav1.LabelSelector{
											MatchExpressions: []metav1.LabelSelectorRequirement{
												{
													Key:      "key",
													Operator: metav1.LabelSelectorOpNotIn,
													Values:   []string{"value1", "value2"},
												},
												{
													Key:      "key2",
													Operator: metav1.LabelSelectorOpIn,
													Values:   []string{"value1"},
												},
												{
													Key:      "key3",
													Operator: metav1.LabelSelectorOpNotIn,
													Values:   []string{"value1"},
												},
											},
										},
										TopologyKey:       "k8s.io/zone",
										MatchLabelKeys:    []string{"key2"},
										MismatchLabelKeys: []string{"key3"},
									},
								},
							},
						},
					},
				},
			},
		},
		{
			Name: "LabelSelector can have the same key as MismatchLabelKeys",
			Object: &core.Pod{
				// Note: On the contrary, in case of matchLabelKeys, keys in matchLabelKeys are not allowed to be specified in labelSelector by users.
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Affinity: &core.Affinity{
						PodAffinity: &core.PodAffinity{
							RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
								{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1", "value2"},
											},
											{
												// This is the same key as in MismatchLabelKeys
												// but it's allowed.
												Key:      "key2",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"value1"},
											},
											{
												Key:      "key2",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1"},
											},
										},
									},
									TopologyKey:       "k8s.io/zone",
									MismatchLabelKeys: []string{"key2"},
								},
							},
						},
					},
				},
			},
		},
		{
			Name: "bad name",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "metadata.name", Type: field.ErrorTypeRequired, Detail: "name or generateName is required", BadValue: ""},
			},
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "", Namespace: "ns"},
				Spec: core.PodSpec{
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				},
			},
		},
		{
			Name: "image whitespace",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.containers[0].image", Type: field.ErrorTypeInvalid, Detail: "must not have leading or trailing whitespace", BadValue: " "},
			},
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "ns"},
				Spec: core.PodSpec{
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Containers:    []core.Container{{Name: "ctr", Image: " ", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				},
			},
		},
		{
			Name: "image leading and trailing whitespace",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.containers[0].image", Type: field.ErrorTypeInvalid, Detail: "must not have leading or trailing whitespace", BadValue: " something "},
			},
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "ns"},
				Spec: core.PodSpec{
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Containers:    []core.Container{{Name: "ctr", Image: " something ", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				},
			},
		},
		{
			Name: "bad namespace",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "metadata.namespace", Type: field.ErrorTypeRequired, BadValue: ""},
			},
			// expectedError: "metadata.namespace",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: ""},
				Spec: core.PodSpec{
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				},
			},
		},
		{
			Name: "bad spec",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "spec.containers[0].name", Type: field.ErrorTypeRequired, SchemaType: field.ErrorTypeInvalid},
				{Field: "spec.containers[0].image", Type: field.ErrorTypeRequired},
				{Field: "spec.containers[0].imagePullPolicy", Type: field.ErrorTypeRequired, SchemaSkipReason: "Blocked by lack of Conditional Defaulting"},
				{Field: "spec.containers[0].terminationMessagePolicy", Type: field.ErrorTypeRequired, SchemaSkipReason: "Defaulted in schema"},
				{Field: "spec.dnsPolicy", Type: field.ErrorTypeRequired, SchemaSkipReason: "Defaulted in schema"},
				{Field: "spec.restartPolicy", Type: field.ErrorTypeRequired, SchemaSkipReason: "Defaulted in schema"},
			},
			// expectedError: "spec.containers[0].name",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers: []core.Container{{}},
				},
			},
		},
		{
			Name: "bad label",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "metadata.labels", Type: field.ErrorTypeInvalid, Detail: `name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')`, BadValue: `NoUppercaseOrSpecialCharsLike=Equals`},
				{Field: "metadata.labels", Type: field.ErrorTypeInvalid, Detail: `label keys must be qualified names`, NativeSkipReason: `schema-based label validation`},
			},
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "abc",
					Namespace: "ns",
					Labels: map[string]string{
						"NoUppercaseOrSpecialCharsLike=Equals": "bar",
					},
				},
				Spec: core.PodSpec{
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				},
			},
		},
		{
			Name: "invalid node selector requirement in node affinity, operator can't be null",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:        "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].operator",
					Type:         field.ErrorTypeInvalid,
					BadValue:     core.NodeSelectorOperator(""),
					Detail:       "not a valid selector operator",
					SchemaType:   field.ErrorTypeNotSupported,
					SchemaDetail: `supported values: "DoesNotExist", "Exists", "Gt", "In", "Lt", "NotIn"`,
				},
			},
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{{
								MatchExpressions: []core.NodeSelectorRequirement{{
									Key: "key1",
								}},
							}},
						},
					},
				}),
			},
		},
		{
			Name: "invalid node selector requirement in node affinity, key is invalid",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key",
					Type:     field.ErrorTypeInvalid,
					BadValue: "invalid key ___@#",
				},
			},
			// expectedError: "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchExpressions[0].key",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{{
								MatchExpressions: []core.NodeSelectorRequirement{{
									Key:      "invalid key ___@#",
									Operator: core.NodeSelectorOpExists,
								}},
							}},
						},
					},
				}),
			},
		},
		{
			Name: "invalid node field selector requirement in node affinity, more values for field selector",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchFields[0].values",
					Type:     field.ErrorTypeRequired,
					Detail:   "must be only one value when `operator` is 'In' or 'NotIn' for node field selector",
					BadValue: "",
				},
			},
			// expectedError: "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchFields[0].values",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{{
								MatchFields: []core.NodeSelectorRequirement{{
									Key:      "metadata.name",
									Operator: core.NodeSelectorOpIn,
									Values:   []string{"host1", "host2"},
								}},
							}},
						},
					},
				}),
			},
		},
		{
			Name: "invalid node field selector requirement in node affinity, invalid operator",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchFields[0].operator",
					Type:     field.ErrorTypeInvalid,
					BadValue: core.NodeSelectorOpExists,
					Detail:   "not a valid selector operator",
				},
			},

			// expectedError: "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchFields[0].operator",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{{
								MatchFields: []core.NodeSelectorRequirement{{
									Key:      "metadata.name",
									Operator: core.NodeSelectorOpExists,
								}},
							}},
						},
					},
				}),
			},
		},
		{
			Name: "invalid node field selector requirement in node affinity, invalid key",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchFields[0].key",
					Type:     field.ErrorTypeInvalid,
					BadValue: "metadata.namespace",
					Detail:   "not a valid field selector key",
				},
			},

			// expectedError: "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[0].matchFields[0].key",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{{
								MatchFields: []core.NodeSelectorRequirement{{
									Key:      "metadata.namespace",
									Operator: core.NodeSelectorOpIn,
									Values:   []string{"ns1"},
								}},
							}},
						},
					},
				}),
			},
		},
		{
			Name: "invalid preferredSchedulingTerm in node affinity, weight should be in range 1-100",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:        "spec.affinity.nodeAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].weight",
					Type:         field.ErrorTypeInvalid,
					BadValue:     int32(199),
					Detail:       "must be in the range 1-100",
					SchemaDetail: "should be less than or equal to 100",
				},
			},
			// expectedError: "must be in the range 1-100",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.PreferredSchedulingTerm{{
							Weight: 199,
							Preference: core.NodeSelectorTerm{
								MatchExpressions: []core.NodeSelectorRequirement{{
									Key:      "foo",
									Operator: core.NodeSelectorOpIn,
									Values:   []string{"bar"},
								}},
							},
						}},
					},
				}),
			},
		},
		{
			Name: "invalid requiredDuringSchedulingIgnoredDuringExecution node selector, nodeSelectorTerms must have at least one term",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:  "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms",
					Type:   field.ErrorTypeRequired,
					Detail: "must have at least one node selector term",
				},
			},
			// expectedError: "spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					NodeAffinity: &core.NodeAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: &core.NodeSelector{
							NodeSelectorTerms: []core.NodeSelectorTerm{},
						},
					},
				}),
			},
		},
		{
			Name: "invalid weight in preferredDuringSchedulingIgnoredDuringExecution in pod affinity annotations, weight should be in range 1-100",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:        "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].weight",
					Type:         field.ErrorTypeInvalid,
					BadValue:     int32(109),
					Detail:       "must be in the range 1-100",
					SchemaDetail: "should be less than or equal to 100",
				},
			},
			// expectedError: "must be in the range 1-100",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{{
							Weight: 109,
							PodAffinityTerm: core.PodAffinityTerm{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{{
										Key:      "key2",
										Operator: metav1.LabelSelectorOpNotIn,
										Values:   []string{"value1", "value2"},
									}},
								},
								Namespaces:  []string{"ns"},
								TopologyKey: "region",
							},
						}},
					},
				}),
			},
		},
		{
			Name: "invalid labelSelector in preferredDuringSchedulingIgnoredDuringExecution in podaffinity annotations, values should be empty if the operator is Exists",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.labelSelector.matchExpressions[0].values",
					Type:     field.ErrorTypeForbidden,
					Detail:   "may not be specified when `operator` is 'Exists' or 'DoesNotExist'",
					BadValue: "",
				},
			},
			// expectedError: "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.labelSelector.matchExpressions[0].values",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{{
							Weight: 10,
							PodAffinityTerm: core.PodAffinityTerm{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{{
										Key:      "key2",
										Operator: metav1.LabelSelectorOpExists,
										Values:   []string{"value1", "value2"},
									}},
								},
								Namespaces:  []string{"ns"},
								TopologyKey: "region",
							},
						}},
					},
				}),
			},
		},
		{
			Name: "invalid namespaceSelector in preferredDuringSchedulingIgnoredDuringExecution in podaffinity, In operator must include Values",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:  "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.namespaceSelector.matchExpressions[0].values",
					Type:   field.ErrorTypeRequired,
					Detail: "must be specified when `operator` is 'In' or 'NotIn'",
				},
			},
			// expectedError: "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.namespaceSelector.matchExpressions[0].values",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{{
							Weight: 10,
							PodAffinityTerm: core.PodAffinityTerm{
								NamespaceSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{{
										Key:      "key2",
										Operator: metav1.LabelSelectorOpIn,
									}},
								},
								Namespaces:  []string{"ns"},
								TopologyKey: "region",
							},
						}},
					},
				}),
			},
		},
		{
			Name: "invalid namespaceSelector in preferredDuringSchedulingIgnoredDuringExecution in podaffinity, Exists operator can not have values",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:  "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.namespaceSelector.matchExpressions[0].values",
					Type:   field.ErrorTypeForbidden,
					Detail: "may not be specified when `operator` is 'Exists' or 'DoesNotExist'",
				},
			},
			// expectedError: "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.namespaceSelector.matchExpressions[0].values",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{{
							Weight: 10,
							PodAffinityTerm: core.PodAffinityTerm{
								NamespaceSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{{
										Key:      "key2",
										Operator: metav1.LabelSelectorOpExists,
										Values:   []string{"value1", "value2"},
									}},
								},
								Namespaces:  []string{"ns"},
								TopologyKey: "region",
							},
						}},
					},
				}),
			},
		},
		{
			Name: "invalid name space in preferredDuringSchedulingIgnoredDuringExecution in podaffinity annotations, namespace should be valid",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:       "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.namespace",
					Type:        field.ErrorTypeInvalid,
					Detail:      `a lowercase RFC 1123 label must consist of lower case alphanumeric characters or '-', and must start and end with an alphanumeric character (e.g. 'my-name',  or '123-abc', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?')`,
					BadValue:    "INVALID_NAMESPACE",
					SchemaField: "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.namespaces[0]",
				},
			},
			// expectedError: "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.namespace",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{{
							Weight: 10,
							PodAffinityTerm: core.PodAffinityTerm{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{{
										Key:      "key2",
										Operator: metav1.LabelSelectorOpExists,
									}},
								},
								Namespaces:  []string{"INVALID_NAMESPACE"},
								TopologyKey: "region",
							},
						}},
					},
				}),
			},
		},
		{
			Name: "invalid hard pod affinity, empty topologyKey is not allowed for hard pod affinity",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey",
					Type:     field.ErrorTypeInvalid,
					BadValue: "",
					Detail:   "name part must be non-empty",
				},
				{
					Field:        "spec.affinity.podAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey",
					Type:         field.ErrorTypeRequired,
					Detail:       "can not be empty",
					SchemaDetail: "should be at least 1 chars long",
					SchemaType:   field.ErrorTypeInvalid,
				},
				{
					Field:    "spec.affinity.podAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey",
					Type:     field.ErrorTypeInvalid,
					BadValue: "",
					Detail:   "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')",
				},
			},
			// expectedError: "can not be empty",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{{
									Key:      "key2",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"value1", "value2"},
								}},
							},
							Namespaces: []string{"ns"},
						}},
					},
				}),
			},
		},
		{
			Name: "invalid hard pod anti-affinity, empty topologyKey is not allowed for hard pod anti-affinity",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey",
					Type:     field.ErrorTypeInvalid,
					BadValue: "",
					Detail:   "name part must be non-empty",
				},
				{
					Field:        "spec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey",
					Type:         field.ErrorTypeRequired,
					Detail:       "can not be empty",
					SchemaDetail: "should be at least 1 chars long",
					SchemaType:   field.ErrorTypeInvalid,
				},
				{
					Field:    "spec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey",
					Type:     field.ErrorTypeInvalid,
					BadValue: "",
					Detail:   "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')",
				},
			},
			// expectedError: "can not be empty",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{{
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{{
									Key:      "key2",
									Operator: metav1.LabelSelectorOpIn,
									Values:   []string{"value1", "value2"},
								}},
							},
							Namespaces: []string{"ns"},
						}},
					},
				}),
			},
		},
		{
			Name: "invalid soft pod affinity, empty topologyKey is not allowed for soft pod affinity",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.topologyKey",
					Type:     field.ErrorTypeInvalid,
					BadValue: "",
					Detail:   "name part must be non-empty",
				},
				{
					Field:        "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.topologyKey",
					Type:         field.ErrorTypeRequired,
					Detail:       "can not be empty",
					SchemaDetail: "should be at least 1 chars long",
					SchemaType:   field.ErrorTypeInvalid,
				},
				{
					Field:    "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.topologyKey",
					Type:     field.ErrorTypeInvalid,
					BadValue: "",
					Detail:   "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')",
				},
			},
			// expectedError: "can not be empty",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{{
							Weight: 10,
							PodAffinityTerm: core.PodAffinityTerm{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{{
										Key:      "key2",
										Operator: metav1.LabelSelectorOpNotIn,
										Values:   []string{"value1", "value2"},
									}},
								},
								Namespaces: []string{"ns"},
							},
						}},
					},
				}),
			},
		},
		{
			Name: "invalid soft pod anti-affinity, empty topologyKey is not allowed for soft pod anti-affinity",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.topologyKey",
					Type:     field.ErrorTypeInvalid,
					BadValue: "",
					Detail:   "name part must be non-empty",
				},
				{
					Field:        "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.topologyKey",
					Type:         field.ErrorTypeRequired,
					Detail:       "can not be empty",
					SchemaDetail: "should be at least 1 chars long",
					SchemaType:   field.ErrorTypeInvalid,
				},
				{
					Field:    "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.topologyKey",
					Type:     field.ErrorTypeInvalid,
					BadValue: "",
					Detail:   "name part must consist of alphanumeric characters, '-', '_' or '.', and must start and end with an alphanumeric character (e.g. 'MyName',  or 'my.name',  or '123-abc', regex used for validation is '([A-Za-z0-9][-A-Za-z0-9_.]*)?[A-Za-z0-9]')",
				},
			},
			// expectedError: "can not be empty",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{{
							Weight: 10,
							PodAffinityTerm: core.PodAffinityTerm{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{{
										Key:      "key2",
										Operator: metav1.LabelSelectorOpNotIn,
										Values:   []string{"value1", "value2"},
									}},
								},
								Namespaces: []string{"ns"},
							},
						}},
					},
				}),
			},
		},
		{
			Name: "invalid soft pod affinity, key in MatchLabelKeys isn't correctly defined",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.matchLabelKeys[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "prefix part must be non-empty",
					BadValue: "/simple",
				},
			},
			// expectedError: "prefix part must be non-empty",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 10,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1", "value2"},
											},
										},
									},
									TopologyKey:    "k8s.io/zone",
									MatchLabelKeys: []string{"/simple"},
								},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid hard pod affinity, key in MatchLabelKeys isn't correctly defined",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].matchLabelKeys[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "prefix part must be non-empty",
					BadValue: "/simple",
				},
			},
			// expectedError: "prefix part must be non-empty",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "key",
											Operator: metav1.LabelSelectorOpNotIn,
											Values:   []string{"value1", "value2"},
										},
									},
								},
								TopologyKey:    "k8s.io/zone",
								MatchLabelKeys: []string{"/simple"},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid soft pod anti-affinity, key in MatchLabelKeys isn't correctly defined",
			// expectedError: "prefix part must be non-empty",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.matchLabelKeys[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "prefix part must be non-empty",
					BadValue: "/simple",
				},
			},
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 10,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1", "value2"},
											},
										},
									},
									TopologyKey:    "k8s.io/zone",
									MatchLabelKeys: []string{"/simple"},
								},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid hard pod anti-affinity, key in MatchLabelKeys isn't correctly defined",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].matchLabelKeys[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "prefix part must be non-empty",
					BadValue: "/simple",
				},
			},
			// expectedError: "prefix part must be non-empty",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "key",
											Operator: metav1.LabelSelectorOpNotIn,
											Values:   []string{"value1", "value2"},
										},
									},
								},
								TopologyKey:    "k8s.io/zone",
								MatchLabelKeys: []string{"/simple"},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid soft pod affinity, key in MismatchLabelKeys isn't correctly defined",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.mismatchLabelKeys[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "prefix part must be non-empty",
					BadValue: "/simple",
				},
			},
			// expectedError: "prefix part must be non-empty",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 10,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1", "value2"},
											},
										},
									},
									TopologyKey:       "k8s.io/zone",
									MismatchLabelKeys: []string{"/simple"},
								},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid hard pod affinity, key in MismatchLabelKeys isn't correctly defined",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].mismatchLabelKeys[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "prefix part must be non-empty",
					BadValue: "/simple",
				},
			},
			// expectedError: "prefix part must be non-empty",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "key",
											Operator: metav1.LabelSelectorOpNotIn,
											Values:   []string{"value1", "value2"},
										},
									},
								},
								TopologyKey:       "k8s.io/zone",
								MismatchLabelKeys: []string{"/simple"},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid soft pod anti-affinity, key in MismatchLabelKeys isn't correctly defined",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.mismatchLabelKeys[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "prefix part must be non-empty",
					BadValue: "/simple",
				},
			},
			// expectedError: "prefix part must be non-empty",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 10,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1", "value2"},
											},
										},
									},
									TopologyKey:       "k8s.io/zone",
									MismatchLabelKeys: []string{"/simple"},
								},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid hard pod anti-affinity, key in MismatchLabelKeys isn't correctly defined",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].mismatchLabelKeys[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "prefix part must be non-empty",
					BadValue: "/simple",
				},
			},
			// expectedError: "prefix part must be non-empty",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "key",
											Operator: metav1.LabelSelectorOpNotIn,
											Values:   []string{"value1", "value2"},
										},
									},
								},
								TopologyKey:       "k8s.io/zone",
								MismatchLabelKeys: []string{"/simple"},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid soft pod affinity, key exists in both matchLabelKeys and labelSelector",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm[0]",
					Type:             field.ErrorTypeInvalid,
					Detail:           "exists in both matchLabelKeys and labelSelector",
					BadValue:         "key",
					SchemaSkipReason: `blocked by missing CEL functionality - validateMatchLabelKeysAndMismatchLabelKeys too complex to implement for CEL`,
				},
			},
			// expectedError: "exists in both matchLabelKeys and labelSelector",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Labels:    map[string]string{"key": "value1"},
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 10,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											// This one should be created from MatchLabelKeys.
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"value1"},
											},
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value2"},
											},
										},
									},
									TopologyKey:    "k8s.io/zone",
									MatchLabelKeys: []string{"key"},
								},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid hard pod affinity, key exists in both matchLabelKeys and labelSelector",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "spec.affinity.podAffinity.requiredDuringSchedulingIgnoredDuringExecution[0][0]",
					Type:             field.ErrorTypeInvalid,
					Detail:           "exists in both matchLabelKeys and labelSelector",
					BadValue:         "key",
					SchemaSkipReason: `blocked by missing CEL functionality - validateMatchLabelKeysAndMismatchLabelKeys too complex to implement for CEL`,
				},
			},
			// expectedError: "exists in both matchLabelKeys and labelSelector",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Labels:    map[string]string{"key": "value1"},
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										// This one should be created from MatchLabelKeys.
										{
											Key:      "key",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"value1"},
										},
										{
											Key:      "key",
											Operator: metav1.LabelSelectorOpNotIn,
											Values:   []string{"value2"},
										},
									},
								},
								TopologyKey:    "k8s.io/zone",
								MatchLabelKeys: []string{"key"},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid soft pod anti-affinity, key exists in both matchLabelKeys and labelSelector",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm[0]",
					Type:             field.ErrorTypeInvalid,
					Detail:           "exists in both matchLabelKeys and labelSelector",
					BadValue:         "key",
					SchemaSkipReason: `blocked by missing CEL functionality - validateMatchLabelKeysAndMismatchLabelKeys too complex to implement for CEL`,
				},
			},
			// expectedError: "exists in both matchLabelKeys and labelSelector",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Labels:    map[string]string{"key": "value1"},
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 10,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											// This one should be created from MatchLabelKeys.
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpIn,
												Values:   []string{"value1"},
											},
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value2"},
											},
										},
									},
									TopologyKey:    "k8s.io/zone",
									MatchLabelKeys: []string{"key"},
								},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid hard pod anti-affinity, key exists in both matchLabelKeys and labelSelector",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "spec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0][0]",
					Type:             field.ErrorTypeInvalid,
					Detail:           "exists in both matchLabelKeys and labelSelector",
					BadValue:         "key",
					SchemaSkipReason: `blocked by missing CEL functionality - validateMatchLabelKeysAndMismatchLabelKeys too complex to implement for CEL`,
				},
			},
			// expectedError: "exists in both matchLabelKeys and labelSelector",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Labels:    map[string]string{"key": "value1"},
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										// This one should be created from MatchLabelKeys.
										{
											Key:      "key",
											Operator: metav1.LabelSelectorOpIn,
											Values:   []string{"value1"},
										},
										{
											Key:      "key",
											Operator: metav1.LabelSelectorOpNotIn,
											Values:   []string{"value2"},
										},
									},
								},
								TopologyKey:    "k8s.io/zone",
								MatchLabelKeys: []string{"key"},
							},
						},
					},
				}),
			},
		},
		{
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.matchLabelKeys[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "exists in both matchLabelKeys and mismatchLabelKeys",
					BadValue: "samekey",

					SchemaField:  "spec.affinity.podAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm",
					SchemaDetail: "matchLabelKeys key must not be in mismatchLabelKeys",
				},
			},
			Name: "invalid soft pod affinity, key exists in both MatchLabelKeys and MismatchLabelKeys",
			// expectedError: "exists in both matchLabelKeys and mismatchLabelKeys",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 10,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1", "value2"},
											},
										},
									},
									TopologyKey:       "k8s.io/zone",
									MatchLabelKeys:    []string{"samekey"},
									MismatchLabelKeys: []string{"samekey"},
								},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid hard pod affinity, key exists in both MatchLabelKeys and MismatchLabelKeys",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].matchLabelKeys[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "exists in both matchLabelKeys and mismatchLabelKeys",
					BadValue: "samekey",

					SchemaField:  "spec.affinity.podAffinity.requiredDuringSchedulingIgnoredDuringExecution[0]",
					SchemaDetail: "matchLabelKeys key must not be in mismatchLabelKeys",
				},
			},
			// expectedError: "exists in both matchLabelKeys and mismatchLabelKeys",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAffinity: &core.PodAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "key",
											Operator: metav1.LabelSelectorOpNotIn,
											Values:   []string{"value1", "value2"},
										},
									},
								},
								TopologyKey:       "k8s.io/zone",
								MatchLabelKeys:    []string{"samekey"},
								MismatchLabelKeys: []string{"samekey"},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid soft pod anti-affinity, key exists in both MatchLabelKeys and MismatchLabelKeys",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.matchLabelKeys[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "exists in both matchLabelKeys and mismatchLabelKeys",
					BadValue: "samekey",

					SchemaField:  "spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm",
					SchemaDetail: "matchLabelKeys key must not be in mismatchLabelKeys",
				},
			},
			// expectedError: "exists in both matchLabelKeys and mismatchLabelKeys",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						PreferredDuringSchedulingIgnoredDuringExecution: []core.WeightedPodAffinityTerm{
							{
								Weight: 10,
								PodAffinityTerm: core.PodAffinityTerm{
									LabelSelector: &metav1.LabelSelector{
										MatchExpressions: []metav1.LabelSelectorRequirement{
											{
												Key:      "key",
												Operator: metav1.LabelSelectorOpNotIn,
												Values:   []string{"value1", "value2"},
											},
										},
									},
									TopologyKey:       "k8s.io/zone",
									MatchLabelKeys:    []string{"samekey"},
									MismatchLabelKeys: []string{"samekey"},
								},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid hard pod anti-affinity, key exists in both MatchLabelKeys and MismatchLabelKeys",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].matchLabelKeys[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "exists in both matchLabelKeys and mismatchLabelKeys",
					BadValue: "samekey",

					SchemaField:  "spec.affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0]",
					SchemaDetail: "matchLabelKeys key must not be in mismatchLabelKeys",
				},
			},
			// expectedError: "exists in both matchLabelKeys and mismatchLabelKeys",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: validPodSpec(&core.Affinity{
					PodAntiAffinity: &core.PodAntiAffinity{
						RequiredDuringSchedulingIgnoredDuringExecution: []core.PodAffinityTerm{
							{
								LabelSelector: &metav1.LabelSelector{
									MatchExpressions: []metav1.LabelSelectorRequirement{
										{
											Key:      "key",
											Operator: metav1.LabelSelectorOpNotIn,
											Values:   []string{"value1", "value2"},
										},
									},
								},
								TopologyKey:       "k8s.io/zone",
								MatchLabelKeys:    []string{"samekey"},
								MismatchLabelKeys: []string{"samekey"},
							},
						},
					},
				}),
			},
		},
		{
			Name: "invalid toleration key",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.tolerations[0].key",
					Type:     field.ErrorTypeInvalid,
					BadValue: "nospecialchars^=@",
					Detail:   "name part must consist of alphanumeric characters",
				},
			},
			// expectedError: "spec.tolerations[0].key",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "nospecialchars^=@", Operator: "Equal", Value: "bar", Effect: "NoSchedule"}}),
			},
		},
		{
			Name: "invalid toleration operator",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.tolerations[0].operator",
					Type:     field.ErrorTypeNotSupported,
					BadValue: core.TolerationOperator("In"),
					Detail:   `supported values: "Equal", "Exists"`,
				},
			},
			// expectedError: "spec.tolerations[0].operator",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "foo", Operator: "In", Value: "bar", Effect: "NoSchedule"}}),
			},
		},
		{
			Name: "value must be empty when `operator` is 'Exists'",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.tolerations[0].operator",
					Type:     field.ErrorTypeInvalid,
					BadValue: core.Toleration{Key: "foo", Operator: "Exists", Value: "bar", Effect: "NoSchedule"},
					Detail:   "must be empty when `operator` is 'Exists'",
				},
			},
			// expectedError: "spec.tolerations[0].operator",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "foo", Operator: "Exists", Value: "bar", Effect: "NoSchedule"}}),
			},
		},

		{
			Name: "operator must be 'Exists' when `key` is empty",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.tolerations[0].operator",
					Type:     field.ErrorTypeInvalid,
					BadValue: core.TolerationOpEqual,
					Detail:   "operator must be Exists when `key` is empty, which means \"match all values and all keys\"",
				},
			},
			// expectedError: "spec.tolerations[0].operator",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Operator: "Equal", Value: "bar", Effect: "NoSchedule"}}),
			},
		},
		{
			Name: "effect must be 'NoExecute' when `TolerationSeconds` is set",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.tolerations[0].effect",
					Type:     field.ErrorTypeInvalid,
					BadValue: core.TaintEffectNoSchedule,
					Detail:   "must be 'NoExecute' when `tolerationSeconds` is set",
				},
			},
			// expectedError: "spec.tolerations[0].effect",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "pod-forgiveness-invalid",
					Namespace: "ns",
				},
				Spec: extendPodSpecwithTolerations(validPodSpec(nil), []core.Toleration{{Key: "node.kubernetes.io/not-ready", Operator: "Exists", Effect: "NoSchedule", TolerationSeconds: &[]int64{20}[0]}}),
			},
		},
		{
			Name: "must be a valid pod seccomp profile",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "metadata.annotations.seccomp.security.alpha.kubernetes.io/pod",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must be a valid seccomp profile",
					BadValue: "foo",

					// Cant express the correct field value in schema, even
					// with propertyNames, since the value validaiton depends
					// on the key
					SchemaField:  `metadata.annotations`,
					SchemaDetail: `all seccomp profile annotations must be valid`,
				},
			},
			// expectedError: "must be a valid seccomp profile",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompPodAnnotationKey: "foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{
			Name: "must be a valid container seccomp profile",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "metadata.annotations.container.seccomp.security.alpha.kubernetes.io/foo",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must be a valid seccomp profile",
					BadValue: "foo",

					SchemaField:  `metadata.annotations`,
					SchemaDetail: `all seccomp profile annotations must be valid`,
				},
			},
			// expectedError: "must be a valid seccomp profile",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompContainerAnnotationKeyPrefix + "foo": "foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{
			Name: "must be a non-empty container name in seccomp annotation",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    `metadata.annotations`,
					Type:     field.ErrorTypeInvalid,
					Detail:   "name part must be non-empty",
					BadValue: "container.seccomp.security.alpha.kubernetes.io/",
				},
				{
					Field:    `metadata.annotations`,
					Type:     field.ErrorTypeInvalid,
					Detail:   "name part must consist of alphanumeric characters",
					BadValue: "container.seccomp.security.alpha.kubernetes.io/",
				},
				{
					Field:    "metadata.annotations.container.seccomp.security.alpha.kubernetes.io/",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must be a valid seccomp profile",
					BadValue: "foo",

					SchemaField:  `metadata.annotations`,
					SchemaDetail: `all seccomp profile annotations must be valid`,
				},
				{
					Field:  "metadata.annotations",
					Type:   field.ErrorTypeInvalid,
					Detail: "annotation keys must be qualified names",

					NativeSkipReason: `schema-based validation of annoations`,
				},
			},
			// expectedError: "name part must be non-empty",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompContainerAnnotationKeyPrefix: "foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{
			Name: "must be a non-empty container profile in seccomp annotation",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:        "metadata.annotations.container.seccomp.security.alpha.kubernetes.io/foo",
					Type:         field.ErrorTypeInvalid,
					Detail:       "must be a valid seccomp profile",
					BadValue:     "",
					SchemaField:  `metadata.annotations`,
					SchemaDetail: `all seccomp profile annotations must be valid`,
				},
			},
			// expectedError: "must be a valid seccomp profile",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompContainerAnnotationKeyPrefix + "foo": "",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{
			Name: "must match seccomp profile type and pod annotation",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:  "spec.securityContext.seccompProfile.type",
					Type:   field.ErrorTypeForbidden,
					Detail: "seccomp type in annotation and field must match",
				},
			},
			// expectedError: "seccomp type in annotation and field must match",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompPodAnnotationKey: "unconfined",
					},
				},
				Spec: core.PodSpec{
					SecurityContext: &core.PodSecurityContext{
						SeccompProfile: &core.SeccompProfile{
							Type: core.SeccompProfileTypeRuntimeDefault,
						},
					},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "must match seccomp profile type and container annotation",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:       "spec.containers[0].securityContext.seccompProfile.type",
					Type:        field.ErrorTypeForbidden,
					Detail:      "seccomp type in annotation and field must match",
					SchemaField: `metadata.annotations`,
				},
			},
			// expectedError: "seccomp type in annotation and field must match",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompContainerAnnotationKeyPrefix + "ctr": "unconfined",
					},
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						SecurityContext: &core.SecurityContext{
							SeccompProfile: &core.SeccompProfile{
								Type: core.SeccompProfileTypeRuntimeDefault,
							},
						}}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "must be a relative path in a node-local seccomp profile annotation",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:        "metadata.annotations.seccomp.security.alpha.kubernetes.io/pod",
					Type:         field.ErrorTypeInvalid,
					Detail:       "must be a relative path",
					BadValue:     "/foo",
					SchemaField:  `metadata.annotations`,
					SchemaDetail: `all seccomp profile annotations must be valid`,
				},
			},
			// expectedError: "must be a relative path",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompPodAnnotationKey: "localhost//foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{
			Name: "must not start with '../'",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:        "metadata.annotations.seccomp.security.alpha.kubernetes.io/pod",
					Type:         field.ErrorTypeInvalid,
					Detail:       "must not contain '..'",
					BadValue:     "../foo",
					SchemaField:  `metadata.annotations`,
					SchemaDetail: `all seccomp profile annotations must be valid`,
				},
			},
			// expectedError: "must not contain '..'",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.SeccompPodAnnotationKey: "localhost/../foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{
			Name: "AppArmor profile must apply to a container",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "metadata.annotations[container.apparmor.security.beta.kubernetes.io/fake-ctr]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "container not found",
					BadValue: "fake-ctr",

					SchemaSkipReason: `deprecated field`,
				},
			},
			// expectedError: "metadata.annotations[container.apparmor.security.beta.kubernetes.io/fake-ctr]",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						corev1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "ctr":      corev1.DeprecatedAppArmorBetaProfileRuntimeDefault,
						corev1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "init-ctr": corev1.DeprecatedAppArmorBetaProfileRuntimeDefault,
						corev1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "fake-ctr": corev1.DeprecatedAppArmorBetaProfileRuntimeDefault,
					},
				},
				Spec: core.PodSpec{
					InitContainers: []core.Container{{Name: "init-ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					Containers:     []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy:  core.RestartPolicyAlways,
					DNSPolicy:      core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "AppArmor profile format must be valid",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "metadata.annotations[container.apparmor.security.beta.kubernetes.io/ctr]",
					Type:             field.ErrorTypeInvalid,
					Detail:           "invalid AppArmor profile name",
					BadValue:         "bad-name",
					SchemaSkipReason: `deprecated field`,
				},
			},
			// expectedError: "invalid AppArmor profile name",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						corev1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "ctr": "bad-name",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{
			Name: "only default AppArmor profile may start with runtime/",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "metadata.annotations[container.apparmor.security.beta.kubernetes.io/ctr]",
					Type:             field.ErrorTypeInvalid,
					Detail:           "invalid AppArmor profile name",
					BadValue:         "runtime/foo",
					SchemaSkipReason: `deprecated field`,
				},
			},
			// expectedError: "invalid AppArmor profile name",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						corev1.DeprecatedAppArmorBetaContainerAnnotationKeyPrefix + "ctr": "runtime/foo",
					},
				},
				Spec: validPodSpec(nil),
			},
		},
		{
			Name: "unsupported pod AppArmor profile type",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.securityContext.appArmorProfile.type",
					Type:     field.ErrorTypeNotSupported,
					BadValue: core.AppArmorProfileType("test"),
					Detail:   `supported values: "Localhost", "RuntimeDefault", "Unconfined"`,
				},
			},
			// expectedError: `Unsupported value: "test"`,
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						AppArmorProfile: &core.AppArmorProfile{
							Type: "test",
						},
					},
				},
			},
		},
		{
			Name: "unsupported container AppArmor profile type",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.containers[0].securityContext.appArmorProfile.type",
					Type:     field.ErrorTypeNotSupported,
					BadValue: core.AppArmorProfileType("test"),
					Detail:   `supported values: "Localhost", "RuntimeDefault", "Unconfined"`,
				},
			},
			// expectedError: `Unsupported value: "test"`,
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						SecurityContext: &core.SecurityContext{
							AppArmorProfile: &core.AppArmorProfile{
								Type: "test",
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
				},
			},
		},
		{
			Name: "missing pod AppArmor profile type",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:        "spec.securityContext.appArmorProfile.type",
					Type:         field.ErrorTypeRequired,
					Detail:       "type is required when appArmorProfile is set",
					SchemaType:   field.ErrorTypeNotSupported,
					SchemaDetail: `supported values: "Localhost",`,
				},
			},
			// expectedError: "Required value: type is required when appArmorProfile is set",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						AppArmorProfile: &core.AppArmorProfile{
							Type: "",
						},
					},
				},
			},
		},
		{
			Name: "missing AppArmor localhost profile",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:  "spec.securityContext.appArmorProfile.localhostProfile",
					Type:   field.ErrorTypeRequired,
					Detail: "must be set when AppArmor type is Localhost",
				},
			},
			// expectedError: "Required value: must be set when AppArmor type is Localhost",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						AppArmorProfile: &core.AppArmorProfile{
							Type: core.AppArmorProfileTypeLocalhost,
						},
					},
				},
			},
		},
		{
			Name: "empty AppArmor localhost profile",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:  "spec.securityContext.appArmorProfile.localhostProfile",
					Type:   field.ErrorTypeRequired,
					Detail: "must be set when AppArmor type is Localhost",
				},
			},
			// expectedError: "Required value: must be set when AppArmor type is Localhost",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						AppArmorProfile: &core.AppArmorProfile{
							Type:             core.AppArmorProfileTypeLocalhost,
							LocalhostProfile: ptr.To(""),
						},
					},
				},
			},
		},
		{
			Name: "invalid AppArmor localhost profile type",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.securityContext.appArmorProfile.localhostProfile",
					Type:     field.ErrorTypeInvalid,
					Detail:   "can only be set when AppArmor type is Localhost",
					BadValue: ptr.To("foo-bar"),
				},
			},
			// expectedError: `Invalid value: "foo-bar"`,
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						AppArmorProfile: &core.AppArmorProfile{
							Type:             core.AppArmorProfileTypeRuntimeDefault,
							LocalhostProfile: ptr.To("foo-bar"),
						},
					},
				},
			},
		},
		{
			Name: "invalid AppArmor localhost profile",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.securityContext.appArmorProfile.localhostProfile",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must not be padded with whitespace",
					BadValue: "foo-bar ",
				},
			},
			// expectedError: `Invalid value: "foo-bar "`,
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						AppArmorProfile: &core.AppArmorProfile{
							Type:             core.AppArmorProfileTypeLocalhost,
							LocalhostProfile: ptr.To("foo-bar "),
						},
					},
				},
			},
		},
		{
			Name: "too long AppArmor localhost profile",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.securityContext.appArmorProfile.localhostProfile",
					Type:     field.ErrorTypeTooLong,
					Detail:   "may not be longer than 4095",
					BadValue: strings.Repeat("a", 4096),
				},
			},
			// expectedError: "Too long: may not be longer than 4095",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
				},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
					SecurityContext: &core.PodSecurityContext{
						AppArmorProfile: &core.AppArmorProfile{
							Type:             core.AppArmorProfileTypeLocalhost,
							LocalhostProfile: ptr.To[string](strings.Repeat("a", 4096)),
						},
					},
				},
			},
		},
		{
			Name: "mismatched AppArmor field and annotation types",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:  "spec.containers[0].securityContext.appArmorProfile.type",
					Type:   field.ErrorTypeForbidden,
					Detail: "apparmor type in annotation and field must match",

					SchemaSkipReason: `deprecated field`,
				},
			},
			// expectedError: "Forbidden: apparmor type in annotation and field must match",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.DeprecatedAppArmorAnnotationKeyPrefix + "ctr": core.DeprecatedAppArmorAnnotationValueRuntimeDefault,
					},
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						SecurityContext: &core.SecurityContext{
							AppArmorProfile: &core.AppArmorProfile{
								Type: core.AppArmorProfileTypeUnconfined,
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
				},
			},
		},
		{
			Name: "mismatched AppArmor pod field and annotation types",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "spec.containers[0].securityContext.appArmorProfile.type",
					Type:             field.ErrorTypeForbidden,
					Detail:           "apparmor type in annotation and field must match",
					SchemaSkipReason: `deprecated field`,
				},
			},
			// expectedError: "Forbidden: apparmor type in annotation and field must match",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.DeprecatedAppArmorAnnotationKeyPrefix + "ctr": core.DeprecatedAppArmorAnnotationValueRuntimeDefault,
					},
				},
				Spec: core.PodSpec{
					SecurityContext: &core.PodSecurityContext{
						AppArmorProfile: &core.AppArmorProfile{
							Type: core.AppArmorProfileTypeUnconfined,
						},
					},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
				},
			},
		},
		{
			Name: "mismatched AppArmor localhost profiles",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "spec.containers[0].securityContext.appArmorProfile.localhostProfile",
					Type:             field.ErrorTypeForbidden,
					Detail:           "apparmor profile in annotation and field must match",
					SchemaSkipReason: `deprecated field`,
				},
			},
			// expectedError: "Forbidden: apparmor profile in annotation and field must match",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "123",
					Namespace: "ns",
					Annotations: map[string]string{
						core.DeprecatedAppArmorAnnotationKeyPrefix + "ctr": core.DeprecatedAppArmorAnnotationValueLocalhostPrefix + "foo",
					},
				},
				Spec: core.PodSpec{
					Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						SecurityContext: &core.SecurityContext{
							AppArmorProfile: &core.AppArmorProfile{
								Type:             core.AppArmorProfileTypeLocalhost,
								LocalhostProfile: ptr.To("bar"),
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSDefault,
				},
			},
		},
		{
			Name: "invalid extended resource name in container request",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field: "spec.containers[0].terminationMessagePolicy",
					Type:  field.ErrorTypeRequired,

					SchemaSkipReason: `defaulted for schema`,
				},
				{
					Field:    "spec.containers[0].resources.requests[invalid-name]",
					Type:     field.ErrorTypeInvalid,
					BadValue: core.ResourceName("invalid-name"),
					Detail:   "must be a standard resource for containers",

					SchemaField:  `spec.containers[0].resources`,
					SchemaDetail: `limits and requests keys must be a standard resource for containers`,
				},
				{
					Field:    "spec.containers[0].resources.limits[invalid-name]",
					Type:     field.ErrorTypeInvalid,
					BadValue: core.ResourceName("invalid-name"),
					Detail:   "must be a standard resource for containers",

					SchemaSkipReason: `combined error for schema due to lack of propertyNames support`,
				},
				{
					Field:    "spec.containers[0].resources.requests[invalid-name]",
					Type:     field.ErrorTypeInvalid,
					BadValue: core.ResourceName("invalid-name"),
					Detail:   "must be a standard resource type or fully qualified",

					SchemaSkipReason: `combined error for schema due to lack of propertyNames support`,
				},
				{
					Field:    "spec.containers[0].resources.limits[invalid-name]",
					Type:     field.ErrorTypeInvalid,
					BadValue: core.ResourceName("invalid-name"),
					Detail:   "must be a standard resource type or fully qualified",

					SchemaSkipReason: `combined error for schema due to lack of propertyNames support`,
				},
			},
			// expectedError: "must be a standard resource for containers",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers: []core.Container{{
						Name:            "invalid",
						Image:           "image",
						ImagePullPolicy: "IfNotPresent",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName("invalid-name"): resource.MustParse("2"),
							},
							Limits: core.ResourceList{
								core.ResourceName("invalid-name"): resource.MustParse("2"),
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "invalid extended resource requirement: request must be == limit",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "spec.containers[0].terminationMessagePolicy",
					Type:             field.ErrorTypeRequired,
					SchemaSkipReason: `defaulted for schema`,
				},
				{
					Field:            "spec.containers[0].resources.requests",
					Type:             field.ErrorTypeInvalid,
					BadValue:         "2",
					Detail:           "must be equal to example.com/a limit of 1",
					SchemaSkipReason: `Blocked by lack of CEL variables`,
				},
			},
			// expectedError: "must be equal to example.com/a",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers: []core.Container{{
						Name:            "invalid",
						Image:           "image",
						ImagePullPolicy: "IfNotPresent",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("2"),
							},
							Limits: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("1"),
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "invalid extended resource requirement without limit",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "spec.containers[0].terminationMessagePolicy",
					Type:             field.ErrorTypeRequired,
					SchemaSkipReason: `defaulted for schema`,
				},
				{
					Field:            "spec.containers[0].resources.limits",
					Type:             field.ErrorTypeRequired,
					Detail:           "Limit must be set",
					SchemaSkipReason: `Blocked by lack of CEL variables and ability to access key when validating value`,
				},
			},
			// expectedError: "Limit must be set",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers: []core.Container{{
						Name:            "invalid",
						Image:           "image",
						ImagePullPolicy: "IfNotPresent",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("2"),
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "invalid fractional extended resource in container request",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "spec.containers[0].resources.requests[example.com/a]",
					Type:             field.ErrorTypeInvalid,
					BadValue:         resource.MustParse("500m"),
					Detail:           "must be an integer",
					SchemaSkipReason: `Blocked by lack ability to access key when validating value`,
				},
				{
					Field:            "spec.containers[0].resources.limits",
					Type:             field.ErrorTypeRequired,
					Detail:           "Limit must be set for non overcommitable resources",
					SchemaSkipReason: `Blocked by lack of CEL variables and ability to access key when validating value`,
				},
			},
			// expectedError: "must be an integer",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers: []core.Container{{
						Name:                     "invalid",
						Image:                    "image",
						ImagePullPolicy:          "IfNotPresent",
						TerminationMessagePolicy: "File",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("500m"),
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "invalid fractional extended resource in init container request",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "spec.initContainers[0].resources.requests[example.com/a]",
					Type:             field.ErrorTypeInvalid,
					BadValue:         resource.MustParse("500m"),
					Detail:           "must be an integer",
					SchemaSkipReason: `Blocked by lack ability to access key when validating value`,
				},
				{
					Field:            "spec.initContainers[0].resources.limits",
					Type:             field.ErrorTypeRequired,
					Detail:           "Limit must be set for non overcommitable resources",
					SchemaSkipReason: `Blocked by lack of CEL variables and ability to access key when validating value`,
				},
			},
			// expectedError: "must be an integer",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					InitContainers: []core.Container{{
						Name:                     "invalid",
						Image:                    "image",
						ImagePullPolicy:          "IfNotPresent",
						TerminationMessagePolicy: "File",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("500m"),
							},
						},
					}},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "invalid fractional extended resource in container limit",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "spec.containers[0].resources.limits[example.com/a]",
					Type:             field.ErrorTypeInvalid,
					BadValue:         resource.MustParse("2.5"),
					Detail:           "must be an integer",
					SchemaSkipReason: `Blocked by lack ability to access key when validating value`,
				},
				{
					Field:            "spec.containers[0].resources.requests",
					Type:             field.ErrorTypeInvalid,
					BadValue:         "5",
					Detail:           "must be equal to example.com/a limit of 2500m",
					SchemaSkipReason: `Blocked by lack CEL variables`,
				},
			},
			// expectedError: "must be an integer",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers: []core.Container{{
						Name:                     "invalid",
						Image:                    "image",
						ImagePullPolicy:          "IfNotPresent",
						TerminationMessagePolicy: "File",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("5"),
							},
							Limits: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("2.5"),
							},
						},
					}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "invalid fractional extended resource in init container limit",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:            "spec.initContainers[0].resources.limits[example.com/a]",
					Type:             field.ErrorTypeInvalid,
					BadValue:         resource.MustParse("2.5"),
					Detail:           "must be an integer",
					SchemaSkipReason: `Blocked by lack ability to access key when validating value`,
				},
				{
					Field:            "spec.initContainers[0].terminationMessagePolicy",
					Type:             field.ErrorTypeRequired,
					SchemaSkipReason: `defaulted for schema`,
				},
				{
					Field:            "spec.initContainers[0].resources.requests[example.com/a]",
					Type:             field.ErrorTypeInvalid,
					BadValue:         resource.MustParse("2.5"),
					Detail:           "must be an integer",
					SchemaSkipReason: `Blocked by lack ability to access key when validating value`,
				},
			},
			// expectedError: "must be an integer",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					InitContainers: []core.Container{{
						Name:            "invalid",
						Image:           "image",
						ImagePullPolicy: "IfNotPresent",
						Resources: core.ResourceRequirements{
							Requests: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("2.5"),
							},
							Limits: core.ResourceList{
								core.ResourceName("example.com/a"): resource.MustParse("2.5"),
							},
						},
					}},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "mirror-pod present without nodeName",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "metadata.annotations[kubernetes.io/config.mirror]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must set spec.nodeName if mirror pod annotation is set",
					BadValue: "",
				},
			},
			// expectedError: "mirror",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns", Annotations: map[string]string{core.MirrorPodAnnotationKey: ""}},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "mirror-pod populated without nodeName",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "metadata.annotations[kubernetes.io/config.mirror]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must set spec.nodeName if mirror pod annotation is set",
					BadValue: "foo",
				},
			},
			// expectedError: "mirror",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns", Annotations: map[string]string{core.MirrorPodAnnotationKey: "foo"}},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "serviceaccount token projected volume with no serviceaccount name specified",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:  "spec.volumes[0].projected.sources[0].serviceAccountToken",
					Type:   field.ErrorTypeForbidden,
					Detail: "must not be specified when serviceAccountName is not set",

					SchemaField:  `spec.serviceAccountName`,
					SchemaDetail: `may not set serviceAccountToken in a pod with serviceAccountName`,
				},
			},
			// expectedError: "must not be specified when serviceAccountName is not set",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
					Volumes: []core.Volume{{
						Name: "projected-volume",
						VolumeSource: core.VolumeSource{
							Projected: &core.ProjectedVolumeSource{
								Sources: []core.VolumeProjection{{
									ServiceAccountToken: &core.ServiceAccountTokenProjection{
										Audience:          "foo-audience",
										ExpirationSeconds: 6000,
										Path:              "foo-path",
									},
								}},
							},
						},
					}},
				},
			},
		},
		{
			Name: "ClusterTrustBundlePEM projected volume using both byName and bySigner",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:       "spec.volumes[0].projected.sources[0].clusterTrustBundlePEM",
					SchemaField: "spec.volumes[0].projected.sources[0].clusterTrustBundle",

					BadValue: &core.ClusterTrustBundleProjection{
						Path:       "foo-path",
						SignerName: ptr.To("example.com/foo"),
						LabelSelector: &metav1.LabelSelector{
							MatchLabels: map[string]string{
								"version": "live",
							},
						},
						Name: ptr.To("foo"),
					},
					Type:   field.ErrorTypeInvalid,
					Detail: "only one of name and signerName may be used",
				},
			},

			// expectedError: "only one of name and signerName may be used",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "valid-extended", Namespace: "ns"},
				Spec: core.PodSpec{
					ServiceAccountName: "some-service-account",
					Containers:         []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy:      core.RestartPolicyAlways,
					DNSPolicy:          core.DNSClusterFirst,
					Volumes: []core.Volume{
						{
							Name: "projected-volume",
							VolumeSource: core.VolumeSource{
								Projected: &core.ProjectedVolumeSource{
									Sources: []core.VolumeProjection{
										{
											ClusterTrustBundle: &core.ClusterTrustBundleProjection{
												Path:       "foo-path",
												SignerName: ptr.To("example.com/foo"),
												LabelSelector: &metav1.LabelSelector{
													MatchLabels: map[string]string{
														"version": "live",
													},
												},
												Name: ptr.To("foo"),
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
		{
			Name: "ClusterTrustBundlePEM projected volume byName with no name",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:  "spec.volumes[0].projected.sources[0].clusterTrustBundlePEM.name",
					Type:   field.ErrorTypeRequired,
					Detail: "must be a valid object name",

					SchemaField:  "spec.volumes[0].projected.sources[0].clusterTrustBundle.name",
					SchemaType:   field.ErrorTypeInvalid,
					SchemaDetail: `should be at least 1 chars long`,
				},
				{
					Field:    "spec.volumes[0].projected.sources[0].clusterTrustBundlePEM.name",
					Type:     field.ErrorTypeInvalid,
					Detail:   "a lowercase RFC 1123 subdomain must consist of lower case",
					BadValue: "",

					SchemaField: "spec.volumes[0].projected.sources[0].clusterTrustBundle.name",
				},
			},
			// expectedError: "must be a valid object name",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "valid-extended", Namespace: "ns"},
				Spec: core.PodSpec{
					ServiceAccountName: "some-service-account",
					Containers:         []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy:      core.RestartPolicyAlways,
					DNSPolicy:          core.DNSClusterFirst,
					Volumes: []core.Volume{
						{
							Name: "projected-volume",
							VolumeSource: core.VolumeSource{
								Projected: &core.ProjectedVolumeSource{
									Sources: []core.VolumeProjection{
										{
											ClusterTrustBundle: &core.ClusterTrustBundleProjection{
												Path: "foo-path",
												Name: ptr.To(""),
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
		{
			Name: "ClusterTrustBundlePEM projected volume bySigner with no signer name",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:       "spec.volumes[0].projected.sources[0].clusterTrustBundlePEM.signerName",
					SchemaField: "spec.volumes[0].projected.sources[0].clusterTrustBundle.signerName",
					Type:        field.ErrorTypeRequired,
					Detail:      "must be a valid signer name",

					SchemaType:   field.ErrorTypeInvalid,
					SchemaDetail: `should be at least 1 chars long`,
				},
				{
					Field: "spec.volumes[0].projected.sources[0].clusterTrustBundlePEM.signerName",
					Type:  field.ErrorTypeRequired,

					SchemaSkipReason: `redundant error`,
				},
			},
			// expectedError: "must be a valid signer name",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "valid-extended", Namespace: "ns"},
				Spec: core.PodSpec{
					ServiceAccountName: "some-service-account",
					Containers:         []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy:      core.RestartPolicyAlways,
					DNSPolicy:          core.DNSClusterFirst,
					Volumes: []core.Volume{
						{
							Name: "projected-volume",
							VolumeSource: core.VolumeSource{
								Projected: &core.ProjectedVolumeSource{
									Sources: []core.VolumeProjection{
										{
											ClusterTrustBundle: &core.ClusterTrustBundleProjection{
												Path:       "foo-path",
												SignerName: ptr.To(""),
												LabelSelector: &metav1.LabelSelector{
													MatchLabels: map[string]string{
														"foo": "bar",
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
			},
		},
		{
			Name: "ClusterTrustBundlePEM projected volume bySigner with invalid signer name",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:       "spec.volumes[0].projected.sources[0].clusterTrustBundlePEM.signerName",
					SchemaField: "spec.volumes[0].projected.sources[0].clusterTrustBundle.signerName",
					Type:        field.ErrorTypeInvalid,
					Detail:      "must be a fully qualified domain and path of the form",
					BadValue:    "example.com/foo/invalid",
				},
			},
			// expectedError: "must be a fully qualified domain and path of the form",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "valid-extended", Namespace: "ns"},
				Spec: core.PodSpec{
					ServiceAccountName: "some-service-account",
					Containers:         []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy:      core.RestartPolicyAlways,
					DNSPolicy:          core.DNSClusterFirst,
					Volumes: []core.Volume{
						{
							Name: "projected-volume",
							VolumeSource: core.VolumeSource{
								Projected: &core.ProjectedVolumeSource{
									Sources: []core.VolumeProjection{
										{
											ClusterTrustBundle: &core.ClusterTrustBundleProjection{
												Path:       "foo-path",
												SignerName: ptr.To("example.com/foo/invalid"),
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
		{
			Name: "final PVC name for ephemeral volume must be valid",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.volumes[1].name",
					Type:     field.ErrorTypeInvalid,
					Detail:   "PVC name \"" + longPodName + "-" + longVolName + "\": must be no more than 253 characters",
					BadValue: longVolName,

					// More detailed field path blocked by lack of CEL variables
					SchemaField:  "spec.volumes",
					SchemaDetail: "ephemeral PVC name \"" + longPodName + "-" + longVolName + "\": is invalid",
				},
			},
			// expectedError: "spec.volumes[1].name: Invalid value: \"" + longVolName + "\": PVC name \"" + longPodName + "-" + longVolName + "\": must be no more than 253 characters",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: longPodName, Namespace: "ns"},
				Spec: core.PodSpec{
					Volumes: []core.Volume{
						{Name: "pvc", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "my-pvc"}}},
						{Name: longVolName, VolumeSource: core.VolumeSource{Ephemeral: &core.EphemeralVolumeSource{VolumeClaimTemplate: &validPVCTemplate}}},
					},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "PersistentVolumeClaimVolumeSource must not reference a generated PVC",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.volumes[0].persistentVolumeClaim.claimName",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must not reference a PVC that gets created for an ephemeral volume",
					BadValue: "123-ephemeral-volume",

					SchemaSkipReason: `Blocked by lack of CEL variables`,
				},
			},
			// expectedError: "spec.volumes[0].persistentVolumeClaim.claimName: Invalid value: \"123-ephemeral-volume\": must not reference a PVC that gets created for an ephemeral volume",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns"},
				Spec: core.PodSpec{
					Volumes: []core.Volume{
						{Name: "pvc-volume", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "123-ephemeral-volume"}}},
						{Name: "ephemeral-volume", VolumeSource: core.VolumeSource{Ephemeral: &core.EphemeralVolumeSource{VolumeClaimTemplate: &validPVCTemplate}}},
					},
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "invalid pod-deletion-cost",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "metadata.annotations[controller.kubernetes.io/pod-deletion-cost]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must be a 32bit integer",
					BadValue: "text",

					// CEL Bug causes extra dot
					SchemaField: "metadata.annotations.[controller.kubernetes.io/pod-deletion-cost]",
				},
			},
			// expectedError: "metadata.annotations[controller.kubernetes.io/pod-deletion-cost]: Invalid value: \"text\": must be a 32bit integer",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns", Annotations: map[string]string{core.PodDeletionCost: "text"}},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "invalid leading zeros pod-deletion-cost",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "metadata.annotations[controller.kubernetes.io/pod-deletion-cost]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must be a 32bit integer",
					BadValue: "008",

					// CEL Bug causes extra dot
					SchemaField: "metadata.annotations.[controller.kubernetes.io/pod-deletion-cost]",
				},
			},
			// expectedError: "metadata.annotations[controller.kubernetes.io/pod-deletion-cost]: Invalid value: \"008\": must be a 32bit integer",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns", Annotations: map[string]string{core.PodDeletionCost: "008"}},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
		{
			Name: "invalid leading plus sign pod-deletion-cost",
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "metadata.annotations[controller.kubernetes.io/pod-deletion-cost]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must be a 32bit integer",
					BadValue: "+10",

					// CEL Bug causes extra dot
					SchemaField: "metadata.annotations.[controller.kubernetes.io/pod-deletion-cost]",
				},
			},
			// expectedError: "metadata.annotations[controller.kubernetes.io/pod-deletion-cost]: Invalid value: \"+10\": must be a 32bit integer",
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "123", Namespace: "ns", Annotations: map[string]string{core.PodDeletionCost: "+10"}},
				Spec: core.PodSpec{
					Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
					RestartPolicy: core.RestartPolicyAlways,
					DNSPolicy:     core.DNSClusterFirst,
				},
			},
		},
	}

	apivalidationtesting.TestValidate(t, coreScheme, coreDefs, func(a *core.Pod, o options) field.ErrorList {
		opts := pod.GetValidationOptionsFromPodSpecAndMeta(&a.Spec, nil, &a.ObjectMeta, nil)
		opts.ResourceIsPod = true
		return validation.ValidatePodCreate(a, opts)
	}, cases...)
}

func TestValidatePodSpec(t *testing.T) {
	type options validation.PodValidationOptions

	activeDeadlineSecondsMax := int64(math.MaxInt32)

	minUserID := int64(0)
	maxUserID := int64(2147483647)
	minGroupID := int64(0)
	maxGroupID := int64(2147483647)

	specCases := []apivalidationtesting.TestCase[*core.PodSpec, options]{
		{
			Name: "populate basic fields, leave defaults for most",
			Object: &core.PodSpec{
				Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		}, {
			Name: "populate all fields",
			Object: &core.PodSpec{
				Volumes: []core.Volume{
					{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
				},
				Containers:     []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				InitContainers: []core.Container{{Name: "ictr", Image: "iimage", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy:  core.RestartPolicyAlways,
				NodeSelector: map[string]string{
					"key": "value",
				},
				NodeName:              "foobar",
				DNSPolicy:             core.DNSClusterFirst,
				ActiveDeadlineSeconds: ptr.To[int64](30),
				ServiceAccountName:    "acct",
			},
		}, {
			Name: "populate all fields with larger active deadline",
			Object: &core.PodSpec{
				Volumes: []core.Volume{
					{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
				},
				Containers:     []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				InitContainers: []core.Container{{Name: "ictr", Image: "iimage", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy:  core.RestartPolicyAlways,
				NodeSelector: map[string]string{
					"key": "value",
				},
				NodeName:              "foobar",
				DNSPolicy:             core.DNSClusterFirst,
				ActiveDeadlineSeconds: &activeDeadlineSecondsMax,
				ServiceAccountName:    "acct",
			},
		}, {
			Name: "populate HostNetwork",
			Object: &core.PodSpec{
				Containers: []core.Container{
					{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File",
						Ports: []core.ContainerPort{
							{HostPort: 8080, ContainerPort: 8080, Protocol: "TCP"}},
					},
				},
				SecurityContext: &core.PodSecurityContext{
					HostNetwork: true,
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		}, {
			Name: "populate RunAsUser SupplementalGroups FSGroup with minID 0",
			Object: &core.PodSpec{
				Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				SecurityContext: &core.PodSecurityContext{
					SupplementalGroups: []int64{minGroupID},
					RunAsUser:          &minUserID,
					FSGroup:            &minGroupID,
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		}, {
			Name: "populate RunAsUser SupplementalGroups FSGroup with maxID 2147483647", Object: &core.PodSpec{
				Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				SecurityContext: &core.PodSecurityContext{
					SupplementalGroups: []int64{maxGroupID},
					RunAsUser:          &maxUserID,
					FSGroup:            &maxGroupID,
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		}, {
			Name: "populate HostIPC",
			Object: &core.PodSpec{
				SecurityContext: &core.PodSecurityContext{
					HostIPC: true,
				},
				Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		}, {
			Name: "populate HostPID",
			Object: &core.PodSpec{
				SecurityContext: &core.PodSecurityContext{
					HostPID: true,
				},
				Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		}, {
			Name: "populate Affinity",
			Object: &core.PodSpec{
				Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		}, {
			Name: "populate HostAliases",
			Object: &core.PodSpec{
				HostAliases:   []core.HostAlias{{IP: "12.34.56.78", Hostnames: []string{"host1", "host2"}}},
				Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		}, {
			Name: "populate HostAliases with `foo.bar` hostnames",
			Object: &core.PodSpec{
				HostAliases:   []core.HostAlias{{IP: "12.34.56.78", Hostnames: []string{"host1.foo", "host2.bar"}}},
				Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		}, {
			Name: "populate HostAliases with HostNetwork",
			Object: &core.PodSpec{
				HostAliases: []core.HostAlias{{IP: "12.34.56.78", Hostnames: []string{"host1.foo", "host2.bar"}}},
				Containers:  []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				SecurityContext: &core.PodSecurityContext{
					HostNetwork: true,
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		}, {
			Name: "populate PriorityClassName",
			Object: &core.PodSpec{
				Volumes:           []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
				Containers:        []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy:     core.RestartPolicyAlways,
				DNSPolicy:         core.DNSClusterFirst,
				PriorityClassName: "valid-name",
			},
		}, {
			Name: "populate ShareProcessNamespace",
			Object: &core.PodSpec{
				Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
				SecurityContext: &core.PodSecurityContext{
					ShareProcessNamespace: &[]bool{true}[0],
				},
			},
		}, {
			Name: "populate RuntimeClassName",
			Object: &core.PodSpec{
				Containers:       []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy:    core.RestartPolicyAlways,
				DNSPolicy:        core.DNSClusterFirst,
				RuntimeClassName: ptr.To("valid-sandbox"),
			},
		}, {
			Name: "populate Overhead",
			Object: &core.PodSpec{
				Containers:       []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy:    core.RestartPolicyAlways,
				DNSPolicy:        core.DNSClusterFirst,
				RuntimeClassName: ptr.To("valid-sandbox"),
				Overhead:         core.ResourceList{},
			},
		}, {
			Name: "populate DNSPolicy",
			Object: &core.PodSpec{
				Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				SecurityContext: &core.PodSecurityContext{
					FSGroupChangePolicy: ptr.To(core.FSGroupChangeAlways),
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
		}, {
			Name: "bad volume",
			Object: &core.PodSpec{
				Volumes:       []core.Volume{{}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field: "spec.volumes[0].name",
					Type:  field.ErrorTypeRequired,

					SchemaType: field.ErrorTypeInvalid,
				},
				{
					Field: "spec.volumes[0]",
					Type:  field.ErrorTypeRequired,

					SchemaSkipReason: `Redundant error`,
				},
			},
		}, {
			Name: "no containers",
			Object: &core.PodSpec{
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field: "spec.containers",
					Type:  field.ErrorTypeRequired,
				},
			},
		}, {
			Name: "bad container",
			Object: &core.PodSpec{
				Containers:    []core.Container{{}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:      "spec.containers[0].name",
					Type:       field.ErrorTypeRequired,
					SchemaType: field.ErrorTypeInvalid,
				},
				{
					Field: "spec.containers[0].image",
					Type:  field.ErrorTypeRequired,
				},
				{
					Field:            "spec.containers[0].imagePullPolicy",
					Type:             field.ErrorTypeRequired,
					SchemaSkipReason: `Defaulted`,
				},
				{
					Field:            "spec.containers[0].terminationMessagePolicy",
					Type:             field.ErrorTypeRequired,
					SchemaSkipReason: `Defaulted`,
				},
			},
		}, {
			Name: "bad init container",
			Object: &core.PodSpec{
				Containers:     []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				InitContainers: []core.Container{{}},
				RestartPolicy:  core.RestartPolicyAlways,
				DNSPolicy:      core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:      "spec.initContainers[0].name",
					Type:       field.ErrorTypeRequired,
					SchemaType: field.ErrorTypeInvalid,
				},
				{
					Field: "spec.initContainers[0].image",
					Type:  field.ErrorTypeRequired,
				},
				{
					Field:            "spec.initContainers[0].imagePullPolicy",
					Type:             field.ErrorTypeRequired,
					SchemaSkipReason: `Defaulted`,
				},
				{
					Field:            "spec.initContainers[0].terminationMessagePolicy",
					Type:             field.ErrorTypeRequired,
					SchemaSkipReason: `Defaulted`,
				},
			},
		}, {
			Name: "bad DNS policy",
			Object: &core.PodSpec{
				DNSPolicy:     core.DNSPolicy("invalid"),
				RestartPolicy: core.RestartPolicyAlways,
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.dnsPolicy",
					Type:     field.ErrorTypeNotSupported,
					Detail:   `ClusterFirstWithHostNet`,
					BadValue: ptr.To(core.DNSPolicy("invalid")),
				},
			},
		}, {
			Name: "bad service account name",
			Object: &core.PodSpec{
				Containers:         []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy:      core.RestartPolicyAlways,
				DNSPolicy:          core.DNSClusterFirst,
				ServiceAccountName: "invalidName",
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.serviceAccountName",
					Type:     field.ErrorTypeInvalid,
					Detail:   "lowercase RFC 1123 subdomain",
					BadValue: "invalidName",
				},
			},
		}, {
			Name: "bad restart policy",
			Object: &core.PodSpec{
				RestartPolicy: "UnknowPolicy",
				DNSPolicy:     core.DNSClusterFirst,
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.restartPolicy",
					Type:     field.ErrorTypeNotSupported,
					Detail:   `supported values: "Always", `,
					BadValue: core.RestartPolicy("UnknowPolicy"),
				},
			},
		}, {
			Name: "with hostNetwork hostPort unspecified",
			Object: &core.PodSpec{
				Containers: []core.Container{
					{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: core.TerminationMessageReadFile, Ports: []core.ContainerPort{
						{HostPort: 0, ContainerPort: 2600, Protocol: "TCP"}},
					},
				},
				SecurityContext: &core.PodSecurityContext{
					HostNetwork: true,
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.containers[0].ports[0].hostPort",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must match `containerPort` when `hostNetwork` is true",
					BadValue: int32(0),

					// More detailed field path blocked by lack of CEL variables
					SchemaField: `spec.containers`,
				},
			},
		}, {
			Name: "with hostNetwork hostPort not equal to containerPort",
			Object: &core.PodSpec{
				Containers: []core.Container{
					{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: core.TerminationMessageReadFile, Ports: []core.ContainerPort{
						{HostPort: 8080, ContainerPort: 2600, Protocol: "TCP"}},
					},
				},
				SecurityContext: &core.PodSecurityContext{
					HostNetwork: true,
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.containers[0].ports[0].hostPort",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must match `containerPort` when `hostNetwork` is true",
					BadValue: int32(8080),

					// More detailed field path blocked by lack of CEL variables
					SchemaField: `spec.containers`,
				},
			},
		}, {
			Name: "with hostAliases with invalid IP",
			Object: &core.PodSpec{
				SecurityContext: &core.PodSecurityContext{
					HostNetwork: false,
				},
				DNSPolicy:     core.DNSClusterFirst,
				RestartPolicy: core.RestartPolicyAlways,
				HostAliases:   []core.HostAlias{{IP: "999.999.999.999", Hostnames: []string{"host1", "host2"}}},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.hostAliases[0].ip",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must be a valid IP address",
					BadValue: "999.999.999.999",
				},
				{
					Field: "spec.containers",
					Type:  field.ErrorTypeRequired,
				},
			},
		}, {
			Name: "with hostAliases with invalid hostname",
			Object: &core.PodSpec{
				SecurityContext: &core.PodSecurityContext{
					HostNetwork: false,
				},
				DNSPolicy:     core.DNSClusterFirst,
				RestartPolicy: core.RestartPolicyAlways,
				HostAliases:   []core.HostAlias{{IP: "12.34.56.78", Hostnames: []string{"@#$^#@#$"}}},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.hostAliases[0].hostnames[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "a lowercase RFC 1123 subdomain",
					BadValue: "@#$^#@#$",
				},
				{
					Field: "spec.containers",
					Type:  field.ErrorTypeRequired,
				},
			},
		}, {
			Name: "bad supplementalGroups large than math.MaxInt32",
			Object: &core.PodSpec{
				Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				SecurityContext: &core.PodSecurityContext{
					SupplementalGroups: []int64{maxGroupID + 1, 1234},
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.securityContext.supplementalGroups[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must be between 0 and 2147483647",
					BadValue: maxGroupID + 1,

					SchemaDetail: `should be less than or equal to 2`,
				},
			},
		}, {
			Name: "bad supplementalGroups less than 0",
			Object: &core.PodSpec{
				Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				SecurityContext: &core.PodSecurityContext{
					SupplementalGroups: []int64{minGroupID - 1, 1234},
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.securityContext.supplementalGroups[0]",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must be between 0 and 2147483647",
					BadValue: minGroupID - 1,

					SchemaDetail: `should be greater than or equal to 0`,
				},
			},
		}, {
			Name: "bad runAsUser large than math.MaxInt32",
			Object: &core.PodSpec{
				Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				SecurityContext: &core.PodSecurityContext{
					RunAsUser: ptr.To[int64](maxUserID + 1),
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.securityContext.runAsUser",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must be between 0 and 2147483647, inclusive",
					BadValue: maxUserID + 1,

					SchemaDetail: `should be less than or equal to`,
				},
			},
		}, {
			Name: "bad runAsUser less than 0",
			Object: &core.PodSpec{
				Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				SecurityContext: &core.PodSecurityContext{
					RunAsUser: ptr.To[int64](minUserID - 1),
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.securityContext.runAsUser",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must be between 0 and 2147483647",
					BadValue: minUserID - 1,

					SchemaDetail: `should be greater than or equal to 0`,
				},
			},
		}, {
			Name: "bad fsGroup large than math.MaxInt32",
			Object: &core.PodSpec{
				Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				SecurityContext: &core.PodSecurityContext{
					FSGroup: ptr.To[int64](maxGroupID + 1),
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:        "spec.securityContext.fsGroup",
					Type:         field.ErrorTypeInvalid,
					Detail:       "must be between 0 and 2147483647",
					BadValue:     maxGroupID + 1,
					SchemaDetail: `should be less than or equal to`,
				},
			},
		}, {
			Name: "bad fsGroup less than 0",
			Object: &core.PodSpec{
				Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				SecurityContext: &core.PodSecurityContext{
					FSGroup: ptr.To[int64](minGroupID - 1),
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:        "spec.securityContext.fsGroup",
					Type:         field.ErrorTypeInvalid,
					Detail:       "must be between 0 and 2147483647",
					BadValue:     minGroupID - 1,
					SchemaDetail: `should be greater than or equal to 0`,
				},
			},
		}, {
			Name: "bad-active-deadline-seconds",
			Object: &core.PodSpec{
				Volumes: []core.Volume{
					{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
				},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				NodeSelector: map[string]string{
					"key": "value",
				},
				NodeName:              "foobar",
				DNSPolicy:             core.DNSClusterFirst,
				ActiveDeadlineSeconds: ptr.To[int64](0),
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:        "spec.activeDeadlineSeconds",
					Type:         field.ErrorTypeInvalid,
					Detail:       "must be between 1 and 2147483647",
					BadValue:     int64(0),
					SchemaDetail: `should be greater than or equal to`,
				},
			},
		}, {
			Name: "active-deadline-seconds-too-large",
			Object: &core.PodSpec{
				Volumes: []core.Volume{
					{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
				},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				NodeSelector: map[string]string{
					"key": "value",
				},
				NodeName:              "foobar",
				DNSPolicy:             core.DNSClusterFirst,
				ActiveDeadlineSeconds: ptr.To(activeDeadlineSecondsMax + 1),
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.activeDeadlineSeconds",
					Type:     field.ErrorTypeInvalid,
					Detail:   "must be between 1 and 2147483647",
					BadValue: activeDeadlineSecondsMax + 1,

					SchemaDetail: `should be less than or equal`,
				},
			},
		}, {
			Name: "bad nodeName",
			Object: &core.PodSpec{
				NodeName:      "node name",
				Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.nodeName",
					Type:     field.ErrorTypeInvalid,
					Detail:   "a lowercase RFC 1123 subdomain",
					BadValue: "node name",
				},
			},
		}, {
			Name: "bad PriorityClassName",
			Object: &core.PodSpec{
				Volumes:           []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
				Containers:        []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy:     core.RestartPolicyAlways,
				DNSPolicy:         core.DNSClusterFirst,
				PriorityClassName: "InvalidName",
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.priorityClassName",
					Type:     field.ErrorTypeInvalid,
					Detail:   "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters",
					BadValue: "InvalidName",
				},
			},
		}, {
			Name: "ShareProcessNamespace and HostPID both set",
			Object: &core.PodSpec{
				Volumes:       []core.Volume{{Name: "vol", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}}},
				Containers:    []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
				SecurityContext: &core.PodSecurityContext{
					HostPID:               true,
					ShareProcessNamespace: &[]bool{true}[0],
				},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.securityContext.shareProcessNamespace",
					Type:     field.ErrorTypeInvalid,
					Detail:   "ShareProcessNamespace and HostPID cannot both be enabled",
					BadValue: true,

					SchemaField: `spec.shareProcessNamespace`,
				},
			},
		}, {
			Name: "bad RuntimeClassName",
			Object: &core.PodSpec{
				Containers:       []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				RestartPolicy:    core.RestartPolicyAlways,
				DNSPolicy:        core.DNSClusterFirst,
				RuntimeClassName: ptr.To("invalid/sandbox"),
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.runtimeClassName",
					Type:     field.ErrorTypeInvalid,
					Detail:   "a lowercase RFC 1123 subdomain must",
					BadValue: "invalid/sandbox",
				},
			},
		}, {
			Name: "bad empty fsGroupchangepolicy",
			Object: &core.PodSpec{
				Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				SecurityContext: &core.PodSecurityContext{
					FSGroupChangePolicy: ptr.To(core.PodFSGroupChangePolicy("")),
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.securityContext.fsGroupChangePolicy",
					Type:     field.ErrorTypeNotSupported,
					BadValue: ptr.To(core.PodFSGroupChangePolicy("")),
					Detail:   `supported values: "Always", "OnRootMismatch"`,
				},
			},
		}, {
			Name: "bad invalid fsgroupchangepolicy",
			Object: &core.PodSpec{
				Containers: []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				SecurityContext: &core.PodSecurityContext{
					FSGroupChangePolicy: ptr.To(core.PodFSGroupChangePolicy("invalid")),
				},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    "spec.securityContext.fsGroupChangePolicy",
					Type:     field.ErrorTypeNotSupported,
					BadValue: ptr.To(core.PodFSGroupChangePolicy("invalid")),
					Detail:   `supported values: "Always", "OnRootMismatch"`,
				},
			},
		}, {
			Name: "disallowed resources resize policy for init containers",
			Object: &core.PodSpec{
				InitContainers: []core.Container{{
					Name:  "initctr",
					Image: "initimage",
					ResizePolicy: []core.ContainerResizePolicy{
						{ResourceName: "cpu", RestartPolicy: "NotRequired"},
					},
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
				}},
				Containers: []core.Container{{
					Name:  "ctr",
					Image: "image",
					ResizePolicy: []core.ContainerResizePolicy{
						{ResourceName: "cpu", RestartPolicy: "NotRequired"},
					},
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
				}},
				RestartPolicy: core.RestartPolicyAlways,
				DNSPolicy:     core.DNSClusterFirst,
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:  "spec.initContainers[0].resizePolicy",
					Type:   field.ErrorTypeInvalid,
					Detail: "must not be set for init containers",
					BadValue: []core.ContainerResizePolicy{{
						ResourceName: "cpu", RestartPolicy: "NotRequired",
					}},
				},
			},
		},
	}

	cases := []apivalidationtesting.TestCase[*core.Pod, options]{}
	for _, c := range specCases {
		c.Name = "pod spec: " + c.Name
		cases = append(cases, apivalidationtesting.TestCase[*core.Pod, options]{
			Name: c.Name,
			Object: &core.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "ns"},
				Spec:       *c.Object,
			},
			ExpectedErrors: c.ExpectedErrors,
		})
	}

	apivalidationtesting.TestValidate(t, coreScheme, coreDefs, func(a *core.Pod, o options) field.ErrorList {
		opts := pod.GetValidationOptionsFromPodSpecAndMeta(&a.Spec, nil, &a.ObjectMeta, nil)
		opts.ResourceIsPod = true
		return validation.ValidatePodSpec(&a.Spec, nil, field.NewPath("spec"), opts)
	}, cases...)
}

// validateContainersCommon is not exported, so we test it through Pod.
// However some tests are specific to template usage, so they are tested through
// other usages like Job
func TestValidateContainers(t *testing.T) {
	type options struct {
		allowPrivileged bool
	}

	cases := []apivalidationtesting.TestCase[*core.PodTemplate, options]{}
	containerCases := []apivalidationtesting.TestCase[[]core.Container, options]{
		{Object: []core.Container{{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}}},
		// backwards compatibility to ensure containers in pod template spec do not check for this
		{Object: []core.Container{{Name: "def", Image: " ", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}}},
		{Object: []core.Container{{Name: "ghi", Image: " some  ", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}}},
		{Object: []core.Container{{Name: "123", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}}},
		{Object: []core.Container{{Name: "abc-123", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}}},
		{Object: []core.Container{{
			Name:  "life-123",
			Image: "image",
			Lifecycle: &core.Lifecycle{
				PreStop: &core.LifecycleHandler{
					Exec: &core.ExecAction{Command: []string{"ls", "-l"}},
				},
			},
			ImagePullPolicy:          "IfNotPresent",
			TerminationMessagePolicy: "File",
		}}}, {
			Object: []core.Container{{
				Name:  "resources-test",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits: core.ResourceList{
						core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
						core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
						core.ResourceName("my.org/resource"):   resource.MustParse("10"),
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
			}}, {
			Object: []core.Container{{
				Name:  "resources-test-with-request-and-limit",
				Image: "image",
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
						core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					},
					Limits: core.ResourceList{
						core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
						core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
			}}, {
			Object: []core.Container{{
				Name:  "resources-request-limit-simple",
				Image: "image",
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceCPU): resource.MustParse("8"),
					},
					Limits: core.ResourceList{
						core.ResourceName(core.ResourceCPU): resource.MustParse("10"),
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
			}}, {
			Object: []core.Container{{
				Name:  "resources-request-limit-edge",
				Image: "image",
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
						core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
						core.ResourceName("my.org/resource"):   resource.MustParse("10"),
					},
					Limits: core.ResourceList{
						core.ResourceName(core.ResourceCPU):    resource.MustParse("10"),
						core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
						core.ResourceName("my.org/resource"):   resource.MustParse("10"),
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
			}}, {
			Object: []core.Container{{
				Name:  "resources-request-limit-partials",
				Image: "image",
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceCPU):    resource.MustParse("9.5"),
						core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					},
					Limits: core.ResourceList{
						core.ResourceName(core.ResourceCPU):  resource.MustParse("10"),
						core.ResourceName("my.org/resource"): resource.MustParse("10"),
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
			}}, {
			Object: []core.Container{{
				Name:  "resources-request",
				Image: "image",
				Resources: core.ResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceCPU):    resource.MustParse("9.5"),
						core.ResourceName(core.ResourceMemory): resource.MustParse("10G"),
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
			}}, {
			Object: []core.Container{{
				Name:  "resources-resize-policy",
				Image: "image",
				ResizePolicy: []core.ContainerResizePolicy{
					{ResourceName: "cpu", RestartPolicy: "NotRequired"},
					{ResourceName: "memory", RestartPolicy: "RestartContainer"},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
			}}, {
			Object: []core.Container{{
				Name:  "same-host-port-different-protocol",
				Image: "image",
				Ports: []core.ContainerPort{
					{ContainerPort: 80, HostPort: 80, Protocol: "TCP"},
					{ContainerPort: 80, HostPort: 80, Protocol: "UDP"},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
			}}, {
			Object: []core.Container{{
				Name:                     "fallback-to-logs-termination-message",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "FallbackToLogsOnError",
			},
			}}, {
			Object: []core.Container{{
				Name:                     "file-termination-message",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
			}}, {
			Object: []core.Container{{
				Name:                     "env-from-source",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				EnvFrom: []core.EnvFromSource{{
					ConfigMapRef: &core.ConfigMapEnvSource{
						LocalObjectReference: core.LocalObjectReference{
							Name: "test",
						},
					},
				}},
			},
			}}, {
			Object: []core.Container{{Name: "abc-1234", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File", SecurityContext: fakeValidSecurityContext(true)}}}, {
			Object: []core.Container{{
				Name:  "live-123",
				Image: "image",
				LivenessProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
					SuccessThreshold: 1,
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
			}}, {
			Object: []core.Container{{
				Name:  "startup-123",
				Image: "image",
				StartupProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
					SuccessThreshold: 1,
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			},
			}}, {
			Object: []core.Container{{
				Name:                     "resize-policy-cpu",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				ResizePolicy: []core.ContainerResizePolicy{
					{ResourceName: "cpu", RestartPolicy: "NotRequired"},
				},
			},
			}}, {
			Object: []core.Container{{
				Name:                     "resize-policy-mem",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				ResizePolicy: []core.ContainerResizePolicy{
					{ResourceName: "memory", RestartPolicy: "RestartContainer"},
				},
			},
			}}, {
			Object: []core.Container{{
				Name:                     "resize-policy-cpu-and-mem",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				ResizePolicy: []core.ContainerResizePolicy{
					{ResourceName: "memory", RestartPolicy: "NotRequired"},
					{ResourceName: "cpu", RestartPolicy: "RestartContainer"},
				},
			},
			}},
		{
			Name:           "zero-length name",
			Object:         []core.Container{{Name: "", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "containers[0].name", SchemaType: field.ErrorTypeInvalid}},
		}, {
			Name:           "zero-length-image",
			Object:         []core.Container{{Name: "abc", Image: "", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "containers[0].image"}},
		}, {
			Name:           "name > 63 characters",
			Object:         []core.Container{{Name: strings.Repeat("a", 64), Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].name", BadValue: strings.Repeat("a", 64)}},
		}, {
			Name:           "name not a DNS label",
			Object:         []core.Container{{Name: "a.b.c", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].name", BadValue: "a.b.c"}},
		}, {
			Name: "name not unique",
			Object: []core.Container{
				{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
				{Name: "abc", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeDuplicate, Field: "containers[1].name", BadValue: "abc", SchemaField: "containers[1]"}},
		}, {
			Name:           "zero-length image",
			Object:         []core.Container{{Name: "abc", Image: "", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "containers[0].image"}},
		}, {
			Name: "host port not unique",
			Object: []core.Container{
				{Name: "abc", Image: "image", Ports: []core.ContainerPort{{ContainerPort: 80, HostPort: 80, Protocol: "TCP"}},
					ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
				{Name: "def", Image: "image", Ports: []core.ContainerPort{{ContainerPort: 81, HostPort: 80, Protocol: "TCP"}},
					ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeDuplicate, Field: "containers[1].ports[0].hostPort", BadValue: "TCP//80", SchemaSkipReason: `Blocked by lack of CEL variables, sets, list flatten`}},
		}, {
			Name: "invalid env var name",
			Object: []core.Container{
				{Name: "abc", Image: "image", Env: []core.EnvVar{{Name: "ev!1"}}, ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].env[0].name", BadValue: "ev!1"}},
		}, {
			Name: "unknown volume name",
			Object: []core.Container{
				{Name: "abc", Image: "image", VolumeMounts: []core.VolumeMount{{Name: "anything", MountPath: "/foo"}},
					ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotFound, Field: "containers[0].volumeMounts[0].name", BadValue: "anything", SchemaField: `containers`, SchemaType: field.ErrorTypeInvalid}},
		}, {
			Name: "invalid lifecycle, no exec command.",
			Object: []core.Container{{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						Exec: &core.ExecAction{},
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "containers[0].lifecycle.preStop.exec.command"}},
		}, {
			Name: "invalid lifecycle, no http path.",
			Object: []core.Container{{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						HTTPGet: &core.HTTPGetAction{
							Port:   intstr.FromInt32(80),
							Scheme: "HTTP",
						},
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "containers[0].lifecycle.preStop.httpGet.path", SchemaSkipReason: `Defaulted for schema`}},
		}, {
			Name: "invalid lifecycle, no http port.",
			Object: []core.Container{{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						HTTPGet: &core.HTTPGetAction{
							Path:   "/",
							Scheme: "HTTP",
						},
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].lifecycle.preStop.httpGet.port", BadValue: 0}},
		}, {
			Name: "invalid lifecycle, no http scheme.",
			Object: []core.Container{{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						HTTPGet: &core.HTTPGetAction{
							Path: "/",
							Port: intstr.FromInt32(80),
						},
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotSupported, Field: "containers[0].lifecycle.preStop.httpGet.scheme", BadValue: core.URIScheme(""), SchemaSkipReason: `Defaulted in schema`}},
		}, {
			Name: "invalid lifecycle, no tcp socket port.",
			Object: []core.Container{{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						TCPSocket: &core.TCPSocketAction{},
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].lifecycle.preStop.tcpSocket.port", BadValue: 0}},
		}, {
			Name: "invalid lifecycle, zero tcp socket port.",
			Object: []core.Container{{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(0),
						},
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].lifecycle.preStop.tcpSocket.port", BadValue: 0}},
		}, {
			Name: "invalid lifecycle, no action.",
			Object: []core.Container{{
				Name:  "life-123",
				Image: "image",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "containers[0].lifecycle.preStop"}},
		}, {
			Name: "invalid readiness probe, terminationGracePeriodSeconds set.",
			Object: []core.Container{{
				Name:  "life-123",
				Image: "image",
				ReadinessProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
					TerminationGracePeriodSeconds: ptr.To[int64](10),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].readinessProbe.terminationGracePeriodSeconds", BadValue: ptr.To(int64(10))}},
		}, {
			Name: "invalid liveness probe, no tcp socket port.",
			Object: []core.Container{{
				Name:  "live-123",
				Image: "image",
				LivenessProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{},
					},
					SuccessThreshold: 1,
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].livenessProbe.tcpSocket.port", BadValue: 0}},
		}, {
			Name: "invalid liveness probe, no action.",
			Object: []core.Container{{
				Name:  "live-123",
				Image: "image",
				LivenessProbe: &core.Probe{
					ProbeHandler:     core.ProbeHandler{},
					SuccessThreshold: 1,
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "containers[0].livenessProbe"}},
		}, {
			Name: "invalid liveness probe, successThreshold != 1",
			Object: []core.Container{{
				Name:  "live-123",
				Image: "image",
				LivenessProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
					SuccessThreshold: 2,
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].livenessProbe.successThreshold", BadValue: int32(2)}},
		}, {
			Name: "invalid startup probe, successThreshold != 1",
			Object: []core.Container{{
				Name:  "startup-123",
				Image: "image",
				StartupProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
					SuccessThreshold: 2,
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].startupProbe.successThreshold", BadValue: int32(2)}},
		}, {
			Name: "invalid liveness probe, negative numbers",
			Object: []core.Container{{
				Name:  "live-123",
				Image: "image",
				LivenessProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
					InitialDelaySeconds:           -1,
					TimeoutSeconds:                -1,
					PeriodSeconds:                 -1,
					SuccessThreshold:              -1,
					FailureThreshold:              -1,
					TerminationGracePeriodSeconds: ptr.To[int64](-1),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Type: field.ErrorTypeInvalid, Field: "containers[0].livenessProbe.initialDelaySeconds", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].livenessProbe.timeoutSeconds", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].livenessProbe.periodSeconds", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].livenessProbe.successThreshold", BadValue: int32(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].livenessProbe.failureThreshold", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].livenessProbe.terminationGracePeriodSeconds", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].livenessProbe.successThreshold", BadValue: int64(-1)},
			},
		}, {
			Name: "invalid readiness probe, negative numbers",
			Object: []core.Container{{
				Name:  "ready-123",
				Image: "image",
				ReadinessProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
					InitialDelaySeconds:           -1,
					TimeoutSeconds:                -1,
					PeriodSeconds:                 -1,
					SuccessThreshold:              -1,
					FailureThreshold:              -1,
					TerminationGracePeriodSeconds: ptr.To[int64](-1),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Type: field.ErrorTypeInvalid, Field: "containers[0].readinessProbe.initialDelaySeconds", Detail: "be greater than or equal to 0", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].readinessProbe.timeoutSeconds", Detail: "be greater than or equal to 0", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].readinessProbe.periodSeconds", Detail: "be greater than or equal to 0", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].readinessProbe.successThreshold", Detail: "be greater than or equal to 0", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].readinessProbe.failureThreshold", Detail: "be greater than or equal to 0", BadValue: int64(-1)},
				// terminationGracePeriodSeconds returns multiple validation errors here:
				// containers[0].readinessProbe.terminationGracePeriodSeconds: Invalid value: -1: must be greater than 0
				{Type: field.ErrorTypeInvalid, Field: "containers[0].readinessProbe.terminationGracePeriodSeconds", Detail: "must not be set for readinessProbes", BadValue: ptr.To(int64(-1))},
				// containers[0].readinessProbe.terminationGracePeriodSeconds: Invalid value: -1: must not be set for readinessProbes
				{Type: field.ErrorTypeInvalid, Field: "containers[0].readinessProbe.terminationGracePeriodSeconds", Detail: "must be greater than 0", SchemaDetail: `should be greater than or equal to 1`, BadValue: int64(-1)},
			},
		}, {
			Name: "invalid startup probe, negative numbers",
			Object: []core.Container{{
				Name:  "startup-123",
				Image: "image",
				StartupProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
					InitialDelaySeconds:           -1,
					TimeoutSeconds:                -1,
					PeriodSeconds:                 -1,
					SuccessThreshold:              -1,
					FailureThreshold:              -1,
					TerminationGracePeriodSeconds: ptr.To[int64](-1),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Type: field.ErrorTypeInvalid, Field: "containers[0].startupProbe.initialDelaySeconds", Detail: "be greater than or equal to 0", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].startupProbe.timeoutSeconds", Detail: "be greater than or equal to 0", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].startupProbe.periodSeconds", Detail: "be greater than or equal to 0", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].startupProbe.successThreshold", Detail: "must be greater than or equal to 0", SchemaDetail: `should be greater than or equal to 0`, BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].startupProbe.failureThreshold", Detail: "be greater than or equal to 0", BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].startupProbe.terminationGracePeriodSeconds", Detail: "must be greater than 0", SchemaDetail: `should be greater than or equal to 1`, BadValue: int64(-1)},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].startupProbe.successThreshold", Detail: "must be 1", BadValue: int32(-1)},
			},
		}, {
			Name: "invalid message termination policy",
			Object: []core.Container{{
				Name:                     "life-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "Unknown",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotSupported, Field: "containers[0].terminationMessagePolicy", BadValue: core.TerminationMessagePolicy("Unknown")}},
		}, {
			Name: "empty message termination policy",
			Object: []core.Container{{
				Name:                     "life-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "containers[0].terminationMessagePolicy", SchemaSkipReason: `Defaulted in schema`}},
		}, {
			Name: "privilege disabled",
			Object: []core.Container{{
				Name:                     "abc",
				Image:                    "image",
				SecurityContext:          fakeValidSecurityContext(true),
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "containers[0].securityContext.privileged", SchemaSkipReason: `Blocked by lack of CEL variables and access to capabilities/features`}},
		}, {
			Name: "invalid compute resource",
			Object: []core.Container{{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits: core.ResourceList{
						"disk": resource.MustParse("10G"),
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Type: field.ErrorTypeInvalid, Field: "containers[0].resources.limits[disk]", BadValue: core.ResourceName("disk"), Detail: "must be a standard resource for containers", SchemaField: "containers[0].resources"},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].resources.limits[disk]", BadValue: core.ResourceName("disk"), Detail: "must be a standard resource type or fully qualified", SchemaSkipReason: "redundant error"},
			},
		}, {
			Name: "Resource CPU invalid",
			Object: []core.Container{{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits: getResourceLimits("-10", "0"),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].resources.limits[cpu]", BadValue: "-10"}},
		}, {
			Name: "Resource Requests CPU invalid",
			Object: []core.Container{{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Requests: getResourceLimits("-10", "0"),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].resources.requests[cpu]", BadValue: "-10"}},
		}, {
			Name: "Resource Memory invalid",
			Object: []core.Container{{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits: getResourceLimits("0", "-10"),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].resources.limits[memory]", BadValue: "-10"}},
		}, {
			Name: "Request limit simple invalid",
			Object: []core.Container{{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits:   getResourceLimits("5", "3"),
					Requests: getResourceLimits("6", "3"),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].resources.requests", BadValue: "6", SchemaSkipReason: `Blocked by lack of CEL variables and access to current key in value validation`}},
		}, {
			Name: "Invalid storage limit request",
			Object: []core.Container{{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits: core.ResourceList{
						core.ResourceName("attachable-volumes-aws-ebs"): *resource.NewQuantity(10, resource.DecimalSI),
					},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Type: field.ErrorTypeInvalid, Field: "containers[0].resources.limits[attachable-volumes-aws-ebs]", BadValue: core.ResourceName("attachable-volumes-aws-ebs"), Detail: "must be a standard resource for containers", SchemaField: "containers[0].resources"},
				{Type: field.ErrorTypeInvalid, Field: "containers[0].resources.limits[attachable-volumes-aws-ebs]", BadValue: core.ResourceName("attachable-volumes-aws-ebs"), Detail: "must be a standard resource type or fully qualified", SchemaSkipReason: "redundant error"},
			},
		}, {
			Name: "CPU request limit multiple invalid",
			Object: []core.Container{{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits:   getResourceLimits("5", "3"),
					Requests: getResourceLimits("6", "3"),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].resources.requests", BadValue: "6", SchemaSkipReason: `Blocked by lack of CEL variables and access to current key in value validation`}},
		}, {
			Name: "Memory request limit multiple invalid",
			Object: []core.Container{{
				Name:  "abc-123",
				Image: "image",
				Resources: core.ResourceRequirements{
					Limits:   getResourceLimits("5", "3"),
					Requests: getResourceLimits("5", "4"),
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].resources.requests", BadValue: "4", SchemaSkipReason: `Blocked by lack of CEL variables and access to current key in value validation`}},
		}, {
			Name: "Invalid env from",
			Object: []core.Container{{
				Name:                     "env-from-source",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				EnvFrom: []core.EnvFromSource{{
					ConfigMapRef: &core.ConfigMapEnvSource{
						LocalObjectReference: core.LocalObjectReference{
							Name: "$%^&*#",
						},
					},
				}},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "containers[0].envFrom[0].configMapRef.name", BadValue: "$%^&*#"}},
		}, {
			Name: "Unsupported resize policy for memory",
			Object: []core.Container{{
				Name:                     "resize-policy-mem-invalid",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				ResizePolicy: []core.ContainerResizePolicy{
					{ResourceName: "memory", RestartPolicy: core.ResourceResizeRestartPolicy("RestartContainerrrr")},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotSupported, Field: "containers[0].resizePolicy", SchemaField: `containers[0].resizePolicy[0].restartPolicy`, BadValue: core.ResourceResizeRestartPolicy("RestartContainerrrr")}},
		}, {
			Name: "Unsupported resize policy for CPU",
			Object: []core.Container{{
				Name:                     "resize-policy-cpu-invalid",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				ResizePolicy: []core.ContainerResizePolicy{
					{ResourceName: "cpu", RestartPolicy: "RestartNotRequired"},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotSupported, Field: "containers[0].resizePolicy", SchemaField: `containers[0].resizePolicy[0].restartPolicy`, BadValue: core.ResourceResizeRestartPolicy("RestartNotRequired")}},
		}, {
			Name: "Forbidden RestartPolicy: Always",
			Object: []core.Container{{
				Name:                     "foo",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "containers[0].restartPolicy", Detail: "may not be set for non-init containers"}},
		}, {
			Name: "Forbidden RestartPolicy: OnFailure",
			Object: []core.Container{{
				Name:                     "foo",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicy("OnFailure")),
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "containers[0].restartPolicy", Detail: "may not be set for non-init containers"}},
		}, {
			Name: "Forbidden RestartPolicy: Never",
			Object: []core.Container{{
				Name:                     "foo",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicy("Never")),
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "containers[0].restartPolicy", Detail: "may not be set for non-init containers"}},
		}, {
			Name: "Forbidden RestartPolicy: invalid",
			Object: []core.Container{{
				Name:                     "foo",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicy("Invalid")),
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "containers[0].restartPolicy", Detail: "may not be set for non-init containers"}},
		}, {
			Name: "Forbidden RestartPolicy: empty",
			Object: []core.Container{{
				Name:                     "foo",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicy("")),
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "containers[0].restartPolicy", Detail: "may not be set for non-init containers"}},
		},
	}

	for i, c := range containerCases {
		caseName := c.Name
		if len(caseName) == 0 && len(c.Object) == 1 {
			caseName = c.Object[0].Name
		}

		prefixedErrors := make(apivalidationtesting.ExpectedErrorList, len(c.ExpectedErrors))
		for i, e := range c.ExpectedErrors {
			prefixedErrors[i] = e
			prefixedErrors[i].Field = "template.spec." + prefixedErrors[i].Field
			if len(prefixedErrors[i].SchemaField) > 0 {
				prefixedErrors[i].SchemaField = "template.spec." + prefixedErrors[i].SchemaField
			}
		}

		cases = append(cases, apivalidationtesting.TestCase[*core.PodTemplate, options]{
			Name:           caseName,
			ExpectedErrors: prefixedErrors,
			Options:        options{allowPrivileged: len(c.ExpectedErrors) == 0},
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("container-%d", i), Namespace: "ns"},
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{
						Containers:    c.Object,
						RestartPolicy: core.RestartPolicyAlways,
						DNSPolicy:     core.DNSClusterFirst,
					},
				},
			},
		})
	}

	apivalidationtesting.TestValidate(t, coreScheme, coreDefs, func(a *core.PodTemplate, o options) field.ErrorList {
		capabilities.SetForTests(capabilities.Capabilities{
			AllowPrivileged: o.allowPrivileged,
		})
		opts := pod.GetValidationOptionsFromPodSpecAndMeta(&a.Template.Spec, nil, &a.ObjectMeta, nil)
		opts.ResourceIsPod = false
		return validation.ValidatePodTemplate(a, opts)
	}, cases...)
}

func TestValidateInitContainers(t *testing.T) {
	type options struct {
		allowPrivileged bool
	}
	containerCases := []apivalidationtesting.TestCase[[]core.Container, options]{
		{
			Name:    "successCase",
			Options: options{allowPrivileged: true},
			Object: []core.Container{{
				Name:  "container-1-same-host-port-different-protocol",
				Image: "image",
				Ports: []core.ContainerPort{
					{ContainerPort: 80, HostPort: 80, Protocol: "TCP"},
					{ContainerPort: 80, HostPort: 80, Protocol: "UDP"},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}, {
				Name:  "container-2-same-host-port-different-protocol",
				Image: "image",
				Ports: []core.ContainerPort{
					{ContainerPort: 80, HostPort: 80, Protocol: "TCP"},
					{ContainerPort: 80, HostPort: 80, Protocol: "UDP"},
				},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}, {
				Name:                     "container-3-restart-always-with-lifecycle-hook-and-probes",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
				Lifecycle: &core.Lifecycle{
					PostStart: &core.LifecycleHandler{
						Exec: &core.ExecAction{
							Command: []string{"echo", "post start"},
						},
					},
					PreStop: &core.LifecycleHandler{
						Exec: &core.ExecAction{
							Command: []string{"echo", "pre stop"},
						},
					},
				},
				LivenessProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
					SuccessThreshold: 1,
				},
				ReadinessProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
				},
				StartupProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
					SuccessThreshold: 1,
				},
			}},
		}, {
			Name: "empty name",
			Object: []core.Container{{
				Name:                     "",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "initContainers[0].name", BadValue: "", SchemaType: field.ErrorTypeInvalid}},
		}, {
			Name: "name collision with regular container",
			Object: []core.Container{{
				Name:                     "app",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeDuplicate, Field: "initContainers[0].name", BadValue: "app", SchemaField: "initContainers"}},
		}, {
			Name: "invalid termination message policy",
			Object: []core.Container{{
				Name:                     "init",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "Unknown",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotSupported, Field: "initContainers[0].terminationMessagePolicy", BadValue: core.TerminationMessagePolicy("Unknown")}},
		}, {
			Name: "duplicate names",
			Object: []core.Container{{
				Name:                     "init",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}, {
				Name:                     "init",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeDuplicate, Field: "initContainers[1].name", BadValue: "init", SchemaField: "initContainers[1]"}},
		}, {
			Name: "duplicate ports",
			Object: []core.Container{{
				Name:  "abc",
				Image: "image",
				Ports: []core.ContainerPort{{
					ContainerPort: 8080, HostPort: 8080, Protocol: "TCP",
				}, {
					ContainerPort: 8080, HostPort: 8080, Protocol: "TCP",
				}},
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeDuplicate, Field: "initContainers[0].ports[1].hostPort", BadValue: "TCP//8080", SchemaField: "initContainers[0].ports[1]"}},
		}, {
			Name: "uses disallowed field: Lifecycle",
			Object: []core.Container{{
				Name:                     "debug",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						Exec: &core.ExecAction{Command: []string{"ls", "-l"}},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "initContainers[0].lifecycle", BadValue: ""}},
		}, {
			Name: "uses disallowed field: LivenessProbe",
			Object: []core.Container{{
				Name:                     "debug",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				LivenessProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{Port: intstr.FromInt32(80)},
					},
					SuccessThreshold: 1,
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "initContainers[0].livenessProbe", BadValue: ""}},
		}, {
			Name: "uses disallowed field: ReadinessProbe",
			Object: []core.Container{{
				Name:                     "debug",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				ReadinessProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{Port: intstr.FromInt32(80)},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "initContainers[0].readinessProbe", BadValue: ""}},
		}, {
			Name: "Container uses disallowed field: StartupProbe",
			Object: []core.Container{{
				Name:                     "debug",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				StartupProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{Port: intstr.FromInt32(80)},
					},
					SuccessThreshold: 1,
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "initContainers[0].startupProbe", BadValue: ""}},
		}, {
			Name: "Disallowed field with other errors should only return a single Forbidden",
			Object: []core.Container{{
				Name:                     "debug",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				StartupProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{Port: intstr.FromInt32(80)},
					},
					InitialDelaySeconds:           -1,
					TimeoutSeconds:                -1,
					PeriodSeconds:                 -1,
					SuccessThreshold:              -1,
					FailureThreshold:              -1,
					TerminationGracePeriodSeconds: ptr.To[int64](-1),
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Type: field.ErrorTypeForbidden, Field: "initContainers[0].startupProbe", BadValue: ""},
				{Type: field.ErrorTypeInvalid, Field: "initContainers[0].startupProbe.failureThreshold", NativeSkipReason: "Can't prevent validation of subfields in schema"},
				{Type: field.ErrorTypeInvalid, Field: "initContainers[0].startupProbe.initialDelaySeconds", NativeSkipReason: "Can't prevent validation of subfields in schema"},
				{Type: field.ErrorTypeInvalid, Field: "initContainers[0].startupProbe.periodSeconds", NativeSkipReason: "Can't prevent validation of subfields in schema"},
				{Type: field.ErrorTypeInvalid, Field: "initContainers[0].startupProbe.successThreshold", NativeSkipReason: "Can't prevent validation of subfields in schema"},
				{Type: field.ErrorTypeInvalid, Field: "initContainers[0].startupProbe.successThreshold", NativeSkipReason: "Can't prevent validation of subfields in schema"},
				{Type: field.ErrorTypeInvalid, Field: "initContainers[0].startupProbe.timeoutSeconds", NativeSkipReason: "Can't prevent validation of subfields in schema"},
				{Type: field.ErrorTypeInvalid, Field: "initContainers[0].startupProbe.terminationGracePeriodSeconds", NativeSkipReason: "Can't prevent validation of subfields in schema"},
			},
		}, {
			Name: "Not supported RestartPolicy: OnFailure",
			Object: []core.Container{{
				Name:                     "init",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicy("OnFailure")),
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotSupported, Field: "initContainers[0].restartPolicy", BadValue: core.ContainerRestartPolicy("OnFailure"), SchemaType: field.ErrorTypeInvalid}},
		}, {
			Name: "Not supported RestartPolicy: Never",
			Object: []core.Container{{
				Name:                     "init",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicy("Never")),
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotSupported, Field: "initContainers[0].restartPolicy", BadValue: core.ContainerRestartPolicy("Never"), SchemaType: field.ErrorTypeInvalid}},
		}, {
			Name: "Not supported RestartPolicy: invalid",
			Object: []core.Container{{
				Name:                     "init",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicy("invalid")),
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotSupported, Field: "initContainers[0].restartPolicy", BadValue: core.ContainerRestartPolicy("invalid"), SchemaType: field.ErrorTypeInvalid}},
		}, {
			Name: "Not supported RestartPolicy: empty",
			Object: []core.Container{{
				Name:                     "init",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicy("")),
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotSupported, Field: "initContainers[0].restartPolicy", BadValue: core.ContainerRestartPolicy(""), SchemaType: field.ErrorTypeInvalid}},
		}, {
			Name: "invalid startup probe in restartable container, successThreshold != 1",
			Object: []core.Container{{
				Name:                     "restartable-init",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
				StartupProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{Port: intstr.FromInt32(80)},
					},
					SuccessThreshold: 2,
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "initContainers[0].startupProbe.successThreshold", BadValue: int32(2)}},
		}, {
			Name: "invalid readiness probe, terminationGracePeriodSeconds set.",
			Object: []core.Container{{
				Name:                     "life-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
				ReadinessProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
					TerminationGracePeriodSeconds: ptr.To[int64](10),
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "initContainers[0].readinessProbe.terminationGracePeriodSeconds", BadValue: ptr.To[int64](10)}},
		}, {
			Name: "invalid liveness probe, successThreshold != 1",
			Object: []core.Container{{
				Name:                     "live-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
				LivenessProbe: &core.Probe{
					ProbeHandler: core.ProbeHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(80),
						},
					},
					SuccessThreshold: 2,
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "initContainers[0].livenessProbe.successThreshold", BadValue: int32(2)}},
		}, {
			Name: "invalid lifecycle, no exec command.",
			Object: []core.Container{{
				Name:                     "life-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						Exec: &core.ExecAction{},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "initContainers[0].lifecycle.preStop.exec.command", BadValue: ""}},
		}, {
			Name: "invalid lifecycle, no http path.",
			Object: []core.Container{{
				Name:                     "life-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						HTTPGet: &core.HTTPGetAction{
							Port:   intstr.FromInt32(80),
							Scheme: "HTTP",
						},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "initContainers[0].lifecycle.preStop.httpGet.path", BadValue: "", SchemaSkipReason: `Defaulted`}},
		}, {
			Name: "invalid lifecycle, no http port.",
			Object: []core.Container{{
				Name:                     "life-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						HTTPGet: &core.HTTPGetAction{
							Path:   "/",
							Scheme: "HTTP",
						},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "initContainers[0].lifecycle.preStop.httpGet.port", BadValue: 0}},
		}, {
			Name: "invalid lifecycle, no http scheme.",
			Object: []core.Container{{
				Name:                     "life-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						HTTPGet: &core.HTTPGetAction{
							Path: "/",
							Port: intstr.FromInt32(80),
						},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotSupported, Field: "initContainers[0].lifecycle.preStop.httpGet.scheme", BadValue: core.URIScheme(""), SchemaSkipReason: `Defaulted`}},
		}, {
			Name: "invalid lifecycle, no tcp socket port.",
			Object: []core.Container{{
				Name:                     "life-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						TCPSocket: &core.TCPSocketAction{},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "initContainers[0].lifecycle.preStop.tcpSocket.port", BadValue: 0}},
		}, {
			Name: "invalid lifecycle, zero tcp socket port.",
			Object: []core.Container{{
				Name:                     "life-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{
						TCPSocket: &core.TCPSocketAction{
							Port: intstr.FromInt32(0),
						},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "initContainers[0].lifecycle.preStop.tcpSocket.port", BadValue: 0}},
		}, {
			Name: "invalid lifecycle, no action.",
			Object: []core.Container{{
				Name:                     "life-123",
				Image:                    "image",
				ImagePullPolicy:          "IfNotPresent",
				TerminationMessagePolicy: "File",
				RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
				Lifecycle: &core.Lifecycle{
					PreStop: &core.LifecycleHandler{},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "initContainers[0].lifecycle.preStop", BadValue: ""}},
		},
	}

	cases := []apivalidationtesting.TestCase[*core.PodTemplate, options]{}
	for i, c := range containerCases {
		caseName := c.Name
		if len(caseName) == 0 && len(c.Object) == 1 {
			caseName = c.Object[0].Name
		}
		prefxiedErrors := make(apivalidationtesting.ExpectedErrorList, len(c.ExpectedErrors))
		for i, e := range c.ExpectedErrors {
			prefxiedErrors[i] = e
			prefxiedErrors[i].Field = "template.spec." + prefxiedErrors[i].Field
			if len(prefxiedErrors[i].SchemaField) > 0 {
				prefxiedErrors[i].SchemaField = "template.spec." + prefxiedErrors[i].SchemaField
			}
		}
		cases = append(cases, apivalidationtesting.TestCase[*core.PodTemplate, options]{
			Name:           caseName,
			ExpectedErrors: prefxiedErrors,
			Options:        c.Options,
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("container-%d", i), Namespace: "ns"},
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{
						Containers: []core.Container{{
							Name:                     "app",
							Image:                    "nginx",
							ImagePullPolicy:          "IfNotPresent",
							TerminationMessagePolicy: "File",
						}},
						InitContainers: c.Object,
						RestartPolicy:  core.RestartPolicyNever,
						DNSPolicy:      core.DNSClusterFirst,
					},
				},
			},
		})
	}

	apivalidationtesting.TestValidate[core.PodTemplate, options](t, coreScheme, coreDefs, func(a *core.PodTemplate, o options) field.ErrorList {
		capabilities.SetForTests(capabilities.Capabilities{
			AllowPrivileged: o.allowPrivileged,
		})
		opts := pod.GetValidationOptionsFromPodSpecAndMeta(&a.Template.Spec, nil, &a.ObjectMeta, nil)
		opts.ResourceIsPod = false
		return validation.ValidatePodTemplate(a, opts)
	}, cases...)
}

func TestValidateEphemeralContainers(t *testing.T) {
	containers := []core.Container{{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}}
	initContainers := []core.Container{{Name: "ictr", Image: "iimage", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}}
	vols := map[string]core.VolumeSource{
		"blk": {PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "pvc"}},
		"vol": {EmptyDir: &core.EmptyDirVolumeSource{}},
	}

	// Failure Cases
	tcs := []apivalidationtesting.TestCase[[]core.EphemeralContainer, validation.PodValidationOptions]{
		{
			Name:   "Empty Ephemeral Container",
			Object: []core.EphemeralContainer{},
		},
		{
			Name: "Single Container",
			Object: []core.EphemeralContainer{
				{EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debug", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
		},
		{
			Name: "Multiple Containers",
			Object: []core.EphemeralContainer{
				{EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debug1", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				{EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debug2", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
		},
		{
			Name: "Single Container with Target",
			Object: []core.EphemeralContainer{
				{
					EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debug", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
					TargetContainerName:      "ctr",
				},
			},
		},
		{
			Name: "All allowed fields",
			Object: []core.EphemeralContainer{
				{
					EphemeralContainerCommon: core.EphemeralContainerCommon{

						Name:       "debug",
						Image:      "image",
						Command:    []string{"bash"},
						Args:       []string{"bash"},
						WorkingDir: "/",
						EnvFrom: []core.EnvFromSource{{
							ConfigMapRef: &core.ConfigMapEnvSource{
								LocalObjectReference: core.LocalObjectReference{Name: "dummy"},
								Optional:             &[]bool{true}[0],
							},
						}},
						Env: []core.EnvVar{
							{Name: "TEST", Value: "TRUE"},
						},
						VolumeMounts: []core.VolumeMount{
							{Name: "vol", MountPath: "/vol"},
						},
						VolumeDevices: []core.VolumeDevice{
							{Name: "blk", DevicePath: "/dev/block"},
						},
						TerminationMessagePath:   "/dev/termination-log",
						TerminationMessagePolicy: "File",
						ImagePullPolicy:          "IfNotPresent",
						SecurityContext: &core.SecurityContext{
							Capabilities: &core.Capabilities{
								Add: []core.Capability{"SYS_ADMIN"},
							},
						},
						Stdin:     true,
						StdinOnce: true,
						TTY:       true,
					},
				},
			},
		},
		{
			Name: "Name Collision with Container.Containers",
			Object: []core.EphemeralContainer{
				{EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "ctr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				{EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debug1", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeDuplicate, Field: "ephemeralContainers[0].name", BadValue: "ctr", SchemaField: "ephemeralContainers"}},
		}, {
			Name: "Name Collision with Container.InitContainers",
			Object: []core.EphemeralContainer{
				{EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "ictr", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				{EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debug1", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeDuplicate, Field: "ephemeralContainers[0].name", BadValue: "ictr", SchemaField: "ephemeralContainers"}},
		}, {
			Name: "Name Collision with EphemeralContainers",
			Object: []core.EphemeralContainer{
				{EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debug1", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
				{EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debug1", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeDuplicate, Field: "ephemeralContainers[1].name", BadValue: "debug1", SchemaField: "ephemeralContainers[1]"}},
		}, {
			Name: "empty Container",
			Object: []core.EphemeralContainer{
				{EphemeralContainerCommon: core.EphemeralContainerCommon{}},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Type: field.ErrorTypeRequired, Field: "ephemeralContainers[0].name", SchemaType: field.ErrorTypeInvalid},
				{Type: field.ErrorTypeRequired, Field: "ephemeralContainers[0].image"},
				{Type: field.ErrorTypeRequired, Field: "ephemeralContainers[0].terminationMessagePolicy", SchemaSkipReason: `Defaulted`},
				{Type: field.ErrorTypeRequired, Field: "ephemeralContainers[0].imagePullPolicy", SchemaSkipReason: `Blocked by lack of conditional defaults`},
			},
		}, {
			Name: "empty Container Name",
			Object: []core.EphemeralContainer{
				{EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeRequired, Field: "ephemeralContainers[0].name", SchemaType: field.ErrorTypeInvalid, BadValue: ""}},
		}, {
			Name: "whitespace padded image name",
			Object: []core.EphemeralContainer{
				{EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debug", Image: " image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeInvalid, Field: "ephemeralContainers[0].image", BadValue: " image"}},
		}, {
			Name: "invalid image pull policy",
			Object: []core.EphemeralContainer{
				{EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debug", Image: "image", ImagePullPolicy: "PullThreeTimes", TerminationMessagePolicy: "File"}},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotSupported, Field: "ephemeralContainers[0].imagePullPolicy", BadValue: core.PullPolicy("PullThreeTimes")}},
		}, {
			Name: "TargetContainerName doesn't exist",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debug", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
				TargetContainerName:      "bogus",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotFound, Field: "ephemeralContainers[0].targetContainerName", BadValue: "bogus", SchemaType: field.ErrorTypeInvalid, SchemaField: `ephemeralContainers`}},
		}, {
			Name: "Targets an ephemeral container",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debug", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
			}, {
				EphemeralContainerCommon: core.EphemeralContainerCommon{Name: "debugception", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"},
				TargetContainerName:      "debug",
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeNotFound, Field: "ephemeralContainers[1].targetContainerName", BadValue: "debug", SchemaType: field.ErrorTypeInvalid, SchemaField: `ephemeralContainers`}},
		}, {
			Name: "Container uses disallowed field: Lifecycle",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "debug",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					Lifecycle: &core.Lifecycle{
						PreStop: &core.LifecycleHandler{
							Exec: &core.ExecAction{Command: []string{"ls", "-l"}},
						},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].lifecycle"}},
		}, {
			Name: "Container uses disallowed field: LivenessProbe",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "debug",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					LivenessProbe: &core.Probe{
						ProbeHandler: core.ProbeHandler{
							TCPSocket: &core.TCPSocketAction{Port: intstr.FromInt32(80)},
						},
						SuccessThreshold: 1,
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].livenessProbe"}},
		}, {
			Name: "Container uses disallowed field: Ports",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "debug",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					Ports: []core.ContainerPort{
						{Protocol: "TCP", ContainerPort: 80},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].ports"}},
		}, {
			Name: "Container uses disallowed field: ReadinessProbe",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "debug",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					ReadinessProbe: &core.Probe{
						ProbeHandler: core.ProbeHandler{
							TCPSocket: &core.TCPSocketAction{Port: intstr.FromInt32(80)},
						},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].readinessProbe"}},
		}, {
			Name: "Container uses disallowed field: StartupProbe",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "debug",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					StartupProbe: &core.Probe{
						ProbeHandler: core.ProbeHandler{
							TCPSocket: &core.TCPSocketAction{Port: intstr.FromInt32(80)},
						},
						SuccessThreshold: 1,
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].startupProbe"}},
		}, {
			Name: "Container uses disallowed field: Resources",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "debug",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					Resources: core.ResourceRequirements{
						Limits: core.ResourceList{
							core.ResourceName(core.ResourceCPU): resource.MustParse("10"),
						},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].resources"}},
		}, {
			Name: "Container uses disallowed field: VolumeMount.SubPath",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "debug",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					VolumeMounts: []core.VolumeMount{
						{Name: "vol", MountPath: "/vol"},
						{Name: "vol", MountPath: "/volsub", SubPath: "foo"},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].volumeMounts[1].subPath"}},
		}, {
			Name: "Container uses disallowed field: VolumeMount.SubPathExpr",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "debug",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					VolumeMounts: []core.VolumeMount{
						{Name: "vol", MountPath: "/vol"},
						{Name: "vol", MountPath: "/volsub", SubPathExpr: "$(POD_NAME)"},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].volumeMounts[1].subPathExpr"}},
		}, {
			Name: "Disallowed field with other errors should only return a single Forbidden",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "debug",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					Lifecycle: &core.Lifecycle{
						PreStop: &core.LifecycleHandler{
							Exec: &core.ExecAction{Command: []string{}},
						},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].lifecycle"},
				{Type: field.ErrorTypeRequired, Field: "ephemeralContainers[0].lifecycle.preStop.exec.command", NativeSkipReason: "Schema-only error can't be short circuited"},
			},
		}, {
			Name: "Container uses disallowed field: ResizePolicy",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "resources-resize-policy",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					ResizePolicy: []core.ContainerResizePolicy{
						{ResourceName: "cpu", RestartPolicy: "NotRequired"},
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].resizePolicy"}},
		}, {
			Name: "Forbidden RestartPolicy: Always",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "foo",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					RestartPolicy:            ptr.To(core.ContainerRestartPolicyAlways),
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].restartPolicy"}},
		}, {
			Name: "Forbidden RestartPolicy: OnFailure",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "foo",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					RestartPolicy:            ptr.To(core.ContainerRestartPolicy("OnFailure")),
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].restartPolicy"}},
		}, {
			Name: "Forbidden RestartPolicy: Never",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "foo",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					RestartPolicy:            ptr.To(core.ContainerRestartPolicy("Never")),
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].restartPolicy"}},
		}, {
			Name: "Forbidden RestartPolicy: invalid",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "foo",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					RestartPolicy:            ptr.To(core.ContainerRestartPolicy("invalid")),
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].restartPolicy"}},
		}, {
			Name: "Forbidden RestartPolicy: empty",
			Object: []core.EphemeralContainer{{
				EphemeralContainerCommon: core.EphemeralContainerCommon{
					Name:                     "foo",
					Image:                    "image",
					ImagePullPolicy:          "IfNotPresent",
					TerminationMessagePolicy: "File",
					RestartPolicy:            ptr.To(core.ContainerRestartPolicy("OnFailure")),
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Type: field.ErrorTypeForbidden, Field: "ephemeralContainers[0].restartPolicy"}},
		},
	}

	cases := []apivalidationtesting.TestCase[*core.Pod, validation.PodValidationOptions]{}
	for i, c := range tcs {
		prefixedErrors := make(apivalidationtesting.ExpectedErrorList, len(c.ExpectedErrors))
		for i, e := range c.ExpectedErrors {
			prefixedErrors[i] = e
			prefixedErrors[i].Field = "spec." + prefixedErrors[i].Field
			if len(prefixedErrors[i].SchemaField) > 0 {
				prefixedErrors[i].SchemaField = "spec." + prefixedErrors[i].SchemaField
			}
		}

		for _, prc := range []core.RestartPolicy{"Always", "OnFailure", "Never"} {
			cases = append(cases, apivalidationtesting.TestCase[*core.Pod, validation.PodValidationOptions]{
				Name:           c.Name + "-" + string(prc),
				ExpectedErrors: prefixedErrors,
				Object: &core.Pod{
					ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("container-%d", i), Namespace: "ns"},
					Spec: core.PodSpec{
						Containers:          containers,
						InitContainers:      initContainers,
						Volumes:             []core.Volume{{Name: "vol", VolumeSource: vols["vol"]}, {Name: "blk", VolumeSource: vols["blk"]}},
						RestartPolicy:       prc,
						DNSPolicy:           core.DNSClusterFirst,
						EphemeralContainers: c.Object,
					},
				},
			})
		}
	}

	apivalidationtesting.TestValidate[core.Pod, validation.PodValidationOptions](t, coreScheme, coreDefs, func(a *core.Pod, o validation.PodValidationOptions) field.ErrorList {
		return validation.ValidatePodSpec(&a.Spec, &a.ObjectMeta, field.NewPath("spec"), o)
	}, cases...)
}

func TestValidatePodDNSConfig(t *testing.T) {
	generateTestSearchPathFunc := func(numChars int) string {
		res := ""
		for i := 0; i < numChars; i++ {
			res = res + "a"
		}
		return res
	}
	generateTestSearchPathFuncs := func(numPaths int, numChars int) []string {
		res := make([]string, numPaths)
		for i := 0; i < numPaths; i++ {
			res[i] = generateTestSearchPathFunc(numChars)
		}
		return res
	}
	testOptionValue := "2"
	testDNSNone := core.DNSNone
	testDNSClusterFirst := core.DNSClusterFirst

	testCases := []struct {
		desc          string
		dnsConfig     *core.PodDNSConfig
		opts          validation.PodValidationOptions
		dnsPolicy     *core.DNSPolicy
		expectedError apivalidationtesting.ExpectedErrorList
	}{{
		desc:      "valid: empty DNSConfig",
		dnsConfig: &core.PodDNSConfig{},
	}, {
		desc: "valid: 1 option",
		dnsConfig: &core.PodDNSConfig{
			Options: []core.PodDNSConfigOption{
				{Name: "ndots", Value: &testOptionValue},
			},
		},
	}, {
		desc: "valid: 1 nameserver",
		dnsConfig: &core.PodDNSConfig{
			Nameservers: []string{"127.0.0.1"},
		},
	}, {
		desc: "valid: DNSNone with 1 nameserver",
		dnsConfig: &core.PodDNSConfig{
			Nameservers: []string{"127.0.0.1"},
		},
		dnsPolicy: &testDNSNone,
	}, {
		desc: "valid: 1 search path",
		dnsConfig: &core.PodDNSConfig{
			Searches: []string{"custom"},
		},
	}, {
		desc: "valid: 1 search path with trailing period",
		dnsConfig: &core.PodDNSConfig{
			Searches: []string{"custom."},
		},
	}, {
		desc: "valid: 3 nameservers and 6 search paths(legacy)",
		dnsConfig: &core.PodDNSConfig{
			Nameservers: []string{"127.0.0.1", "10.0.0.10", "8.8.8.8"},
			Searches:    []string{"custom", "mydomain.com", "local", "cluster.local", "svc.cluster.local", "default.svc.cluster.local."},
		},
	}, {
		desc: "valid: 3 nameservers and 32 search paths",
		dnsConfig: &core.PodDNSConfig{
			Nameservers: []string{"127.0.0.1", "10.0.0.10", "8.8.8.8"},
			Searches:    []string{"custom", "mydomain.com", "local", "cluster.local", "svc.cluster.local", "default.svc.cluster.local.", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32"},
		},
	}, {
		desc: "valid: 256 characters in search path list(legacy)",
		dnsConfig: &core.PodDNSConfig{
			// We can have 256 - (6 - 1) = 251 characters in total for 6 search paths.
			Searches: []string{
				generateTestSearchPathFunc(1),
				generateTestSearchPathFunc(50),
				generateTestSearchPathFunc(50),
				generateTestSearchPathFunc(50),
				generateTestSearchPathFunc(50),
				generateTestSearchPathFunc(50),
			},
		},
	}, {
		desc: "valid: 2048 characters in search path list",
		dnsConfig: &core.PodDNSConfig{
			// We can have 2048 - (32 - 1) = 2017 characters in total for 32 search paths.
			Searches: []string{
				generateTestSearchPathFunc(64),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
			},
		},
	}, {
		desc: "valid: ipv6 nameserver",
		dnsConfig: &core.PodDNSConfig{
			Nameservers: []string{"FE80::0202:B3FF:FE1E:8329"},
		},
	}, {
		desc: "invalid: 4 nameservers",
		dnsConfig: &core.PodDNSConfig{
			Nameservers: []string{"127.0.0.1", "10.0.0.10", "8.8.8.8", "1.2.3.4"},
		},
		expectedError: apivalidationtesting.ExpectedErrorList{
			{
				Type: field.ErrorTypeInvalid, Field: "nameservers", Detail: "must not have more than 3 nameservers", BadValue: []string{"127.0.0.1", "10.0.0.10", "8.8.8.8", "1.2.3.4"},
				SchemaType: field.ErrorTypeTooMany, SchemaDetail: "must have at most 3 items",
			},
		},
	}, {
		desc: "valid: 7 search paths",
		dnsConfig: &core.PodDNSConfig{
			Searches: []string{"custom", "mydomain.com", "local", "cluster.local", "svc.cluster.local", "default.svc.cluster.local", "exceeded"},
		},
	}, {
		desc: "invalid: 33 search paths",
		dnsConfig: &core.PodDNSConfig{
			Searches: []string{"custom", "mydomain.com", "local", "cluster.local", "svc.cluster.local", "default.svc.cluster.local.", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33"},
		},
		expectedError: apivalidationtesting.ExpectedErrorList{
			{
				Type: field.ErrorTypeInvalid, Field: "searches", Detail: "must not have more than 32 search paths", BadValue: []string{"custom", "mydomain.com", "local", "cluster.local", "svc.cluster.local", "default.svc.cluster.local.", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33"},
				SchemaType: field.ErrorTypeTooMany, SchemaDetail: "must have at most 32 items",
			},
		},
	}, {
		desc: "valid: 257 characters in search path list",
		dnsConfig: &core.PodDNSConfig{
			// We can have 256 - (6 - 1) = 251 characters in total for 6 search paths.
			Searches: []string{
				generateTestSearchPathFunc(2),
				generateTestSearchPathFunc(50),
				generateTestSearchPathFunc(50),
				generateTestSearchPathFunc(50),
				generateTestSearchPathFunc(50),
				generateTestSearchPathFunc(50),
			},
		},
	}, {
		desc: "invalid: 2049 characters in search path list",
		dnsConfig: &core.PodDNSConfig{
			// We can have 2048 - (32 - 1) = 2017 characters in total for 32 search paths.
			Searches: []string{
				generateTestSearchPathFunc(65),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
				generateTestSearchPathFunc(63),
			},
		},
		expectedError: apivalidationtesting.ExpectedErrorList{
			{Type: field.ErrorTypeInvalid, Field: "searches", Detail: `must not have more than 2048 characters`, BadValue: append([]string{generateTestSearchPathFunc(65)}, generateTestSearchPathFuncs(31, 63)...)},
		},
	}, {
		desc: "invalid search path",
		dnsConfig: &core.PodDNSConfig{
			Searches: []string{"custom?"},
		},
		expectedError: apivalidationtesting.ExpectedErrorList{
			{Type: field.ErrorTypeInvalid, Field: "searches[0]", BadValue: "custom?"},
		},
	}, {
		desc: "invalid nameserver",
		dnsConfig: &core.PodDNSConfig{
			Nameservers: []string{"invalid"},
		},
		expectedError: apivalidationtesting.ExpectedErrorList{
			{Type: field.ErrorTypeInvalid, Field: "nameservers[0]", BadValue: "invalid"},
		},
	}, {
		desc: "invalid empty option name",
		dnsConfig: &core.PodDNSConfig{
			Options: []core.PodDNSConfigOption{
				{Value: &testOptionValue},
			},
		},
		expectedError: apivalidationtesting.ExpectedErrorList{
			{Type: field.ErrorTypeRequired, Field: "options[0]", SchemaField: "options[0].name"},
		},
	}, {
		desc: "invalid: DNSNone with 0 nameserver",
		dnsConfig: &core.PodDNSConfig{
			Searches: []string{"custom"},
		},
		dnsPolicy: &testDNSNone,
		expectedError: apivalidationtesting.ExpectedErrorList{
			{Type: field.ErrorTypeRequired, Field: "nameservers"}},
	},
	}

	cases := make([]apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions], 0, len(testCases))
	for _, tc := range testCases {
		if tc.dnsPolicy == nil {
			tc.dnsPolicy = &testDNSClusterFirst
		}

		prefixedErrors := make(apivalidationtesting.ExpectedErrorList, len(tc.expectedError))
		for i, e := range tc.expectedError {
			prefixedErrors[i] = e
			prefixedErrors[i].Field = "template.spec.dnsConfig." + prefixedErrors[i].Field
			if len(prefixedErrors[i].SchemaField) > 0 {
				prefixedErrors[i].SchemaField = "template.spec.dnsConfig." + prefixedErrors[i].SchemaField
			}
		}

		cases = append(cases, apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{
			Name:           tc.desc,
			ExpectedErrors: prefixedErrors,
			Options:        tc.opts,
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "pod", Namespace: "ns"},
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{
						Containers:    []core.Container{{Name: "container", Image: "image", ImagePullPolicy: "IfNotPresent", TerminationMessagePolicy: "File"}},
						DNSConfig:     tc.dnsConfig,
						DNSPolicy:     *tc.dnsPolicy,
						RestartPolicy: core.RestartPolicyAlways,
					},
				},
			},
		})
	}

	apivalidationtesting.TestValidate(t, coreScheme, coreDefs, validation.ValidatePodTemplate, cases...)
}

func extendPodSpecwithTolerations(in core.PodSpec, tolerations []core.Toleration) core.PodSpec {
	var out core.PodSpec
	out.Containers = in.Containers
	out.RestartPolicy = in.RestartPolicy
	out.DNSPolicy = in.DNSPolicy
	out.Tolerations = tolerations
	return out
}

func fakeValidSecurityContext(priv bool) *core.SecurityContext {
	return &core.SecurityContext{
		Privileged: &priv,
	}
}

func getResourceLimits(cpu, memory string) core.ResourceList {
	res := core.ResourceList{}
	res[core.ResourceCPU] = resource.MustParse(cpu)
	res[core.ResourceMemory] = resource.MustParse(memory)
	return res
}

// This test is a little too top-to-bottom.  Ideally we would test each volume
// type on its own, but we want to also make sure that the logic works through
// the one-of wrapper, so we just do it all in one place.
func TestValidateVolumes(t *testing.T) {
	validInitiatorName := "iqn.2015-02.example.com:init"
	invalidInitiatorName := "2015-02.example.com:init"

	testCases := []apivalidationtesting.TestCase[[]core.Volume, validation.PodValidationOptions]{
		// EmptyDir and basic volume names
		{
			Name: "valid alpha name",
			Object: []core.Volume{{
				Name: "empty",
				VolumeSource: core.VolumeSource{
					EmptyDir: &core.EmptyDirVolumeSource{},
				},
			}},
		}, {
			Name: "valid num name",
			Object: []core.Volume{{
				Name: "123",
				VolumeSource: core.VolumeSource{
					EmptyDir: &core.EmptyDirVolumeSource{},
				},
			}},
		}, {
			Name: "valid alphanum name",
			Object: []core.Volume{{
				Name: "empty-123",
				VolumeSource: core.VolumeSource{
					EmptyDir: &core.EmptyDirVolumeSource{},
				},
			}},
		}, {
			Name: "valid numalpha name",
			Object: []core.Volume{{
				Name: "123-empty",
				VolumeSource: core.VolumeSource{
					EmptyDir: &core.EmptyDirVolumeSource{},
				},
			}},
		}, {
			Name: "zero-length name",
			Object: []core.Volume{{
				Name:         "",
				VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:       field.ErrorTypeRequired,
				Field:      "name",
				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "name > 63 characters",
			Object: []core.Volume{{
				Name:         strings.Repeat("a", 64),
				VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "name",
				Detail:   "must be no more than",
				BadValue: strings.Repeat("a", 64),
			}},
		}, {
			Name: "name has dots",
			Object: []core.Volume{{
				Name:         "a.b.c",
				VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "name",
				Detail:   "must not contain dots",
				BadValue: "a.b.c",
			}},
		}, {
			Name: "name not a DNS label",
			Object: []core.Volume{{
				Name:         "Not a DNS label!",
				VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "name",
				Detail:   `a lowercase RFC 1123 label must consist of`,
				BadValue: "Not a DNS label!",
			}},
		},
		// More than one source field specified.
		{
			Name: "more than one source",
			Object: []core.Volume{{
				Name: "dups",
				VolumeSource: core.VolumeSource{
					EmptyDir: &core.EmptyDirVolumeSource{},
					HostPath: &core.HostPathVolumeSource{
						Path: "/mnt/path",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:   field.ErrorTypeForbidden,
				Field:  "hostPath",
				Detail: "may not specify more than 1 volume",

				SchemaSkipReason: `Blocked by lack of CEL variables`,
			}},
		},
		// HostPath Default
		{
			Name: "default HostPath",
			Object: []core.Volume{{
				Name: "hostpath",
				VolumeSource: core.VolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/mnt/path",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}},
		},
		// HostPath Supported
		{
			Name: "valid HostPath",
			Object: []core.Volume{{
				Name: "hostpath",
				VolumeSource: core.VolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/mnt/path",
						Type: newHostPathType(string(core.HostPathSocket)),
					},
				},
			}},
		},
		// HostPath Invalid
		{
			Name: "invalid HostPath",
			Object: []core.Volume{{
				Name: "hostpath",
				VolumeSource: core.VolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/mnt/path",
						Type: newHostPathType("invalid"),
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeNotSupported,
				Field:    "hostPath.type",
				BadValue: ptr.To(core.HostPathType("invalid")),
			}},
		}, {
			Name: "invalid HostPath backsteps",
			Object: []core.Volume{{
				Name: "hostpath",
				VolumeSource: core.VolumeSource{
					HostPath: &core.HostPathVolumeSource{
						Path: "/mnt/path/..",
						Type: newHostPathType(string(core.HostPathDirectory)),
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "hostPath.path",
				Detail:   "must not contain '..'",
				BadValue: "/mnt/path/..",
			}},
		},
		// GcePersistentDisk
		{
			Name: "valid GcePersistentDisk",
			Object: []core.Volume{{
				Name: "gce-pd",
				VolumeSource: core.VolumeSource{
					GCEPersistentDisk: &core.GCEPersistentDiskVolumeSource{
						PDName:    "my-PD",
						FSType:    "ext4",
						Partition: 1,
						ReadOnly:  false,
					},
				},
			}},
		},
		// AWSElasticBlockStore
		{
			Name: "valid AWSElasticBlockStore",
			Object: []core.Volume{{
				Name: "aws-ebs",
				VolumeSource: core.VolumeSource{
					AWSElasticBlockStore: &core.AWSElasticBlockStoreVolumeSource{
						VolumeID:  "my-PD",
						FSType:    "ext4",
						Partition: 1,
						ReadOnly:  false,
					},
				},
			}},
		},
		// GitRepo
		{
			Name: "valid GitRepo",
			Object: []core.Volume{{
				Name: "git-repo",
				VolumeSource: core.VolumeSource{
					GitRepo: &core.GitRepoVolumeSource{
						Repository: "my-repo",
						Revision:   "hashstring",
						Directory:  "target",
					},
				},
			}},
		}, {
			Name: "valid GitRepo in .",
			Object: []core.Volume{{
				Name: "git-repo-dot",
				VolumeSource: core.VolumeSource{
					GitRepo: &core.GitRepoVolumeSource{
						Repository: "my-repo",
						Directory:  ".",
					},
				},
			}},
		}, {
			Name: "valid GitRepo with .. in name",
			Object: []core.Volume{{
				Name: "git-repo-dot-dot-foo",
				VolumeSource: core.VolumeSource{
					GitRepo: &core.GitRepoVolumeSource{
						Repository: "my-repo",
						Directory:  "..foo",
					},
				},
			}},
		}, {
			Name: "GitRepo starts with ../",
			Object: []core.Volume{{
				Name: "gitrepo",
				VolumeSource: core.VolumeSource{
					GitRepo: &core.GitRepoVolumeSource{
						Repository: "foo",
						Directory:  "../dots/bar",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "gitRepo.directory",
				Detail:   `must not contain '..'`,
				BadValue: "../dots/bar",
			}},
		}, {
			Name: "GitRepo contains ..",
			Object: []core.Volume{{
				Name: "gitrepo",
				VolumeSource: core.VolumeSource{
					GitRepo: &core.GitRepoVolumeSource{
						Repository: "foo",
						Directory:  "dots/../bar",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "gitRepo.directory",
				Detail:   `must not contain '..'`,
				BadValue: "dots/../bar",
			}},
		}, {
			Name: "GitRepo absolute target",
			Object: []core.Volume{{
				Name: "gitrepo",
				VolumeSource: core.VolumeSource{
					GitRepo: &core.GitRepoVolumeSource{
						Repository: "foo",
						Directory:  "/abstarget",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "gitRepo.directory",
				BadValue: "/abstarget",
			}},
		},
		// ISCSI
		{
			Name: "valid ISCSI",
			Object: []core.Volume{{
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
			}},
		}, {
			Name: "valid IQN: eui format",
			Object: []core.Volume{{
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
			}},
		}, {
			Name: "valid IQN: naa format",
			Object: []core.Volume{{
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
			}},
		}, {
			Name: "empty portal",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "iscsi.targetPortal",

				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "empty iqn",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "iscsi.iqn",

				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "invalid IQN: iqn format",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "iscsi.iqn",
				BadValue: "iqn.2015-02.example.com:test;ls;",
			}},
		}, {
			Name: "invalid IQN: eui format",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "iscsi.iqn",
				BadValue: "eui.0123456789ABCDEFGHIJ",
			}},
		}, {
			Name: "invalid IQN: naa format",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "iscsi.iqn",
				BadValue: "naa.62004567BA_4-78D.123456789ABCDEF",
			}},
		}, {
			Name: "valid initiatorName",
			Object: []core.Volume{{
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
			}},
		}, {
			Name: "invalid initiatorName",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "iscsi.initiatorname",
				BadValue: invalidInitiatorName,

				SchemaField: "iscsi.initiatorName",
			}},
		}, {
			Name: "empty secret",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "iscsi.secretRef",
			}},
		},
		// Secret
		{
			Name: "valid Secret",
			Object: []core.Volume{{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName: "my-secret",
					},
				},
			}},
		}, {
			Name: "valid Secret with defaultMode",
			Object: []core.Volume{{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName:  "my-secret",
						DefaultMode: ptr.To[int32](0644),
					},
				},
			}},
		}, {
			Name: "valid Secret with projection and mode",
			Object: []core.Volume{{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName: "my-secret",
						Items: []core.KeyToPath{{
							Key:  "key",
							Path: "filename",
							Mode: ptr.To[int32](0644),
						}},
					},
				},
			}},
		}, {
			Name: "valid Secret with subdir projection",
			Object: []core.Volume{{
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
			}},
		}, {
			Name: "secret with missing path",
			Object: []core.Volume{{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName: "s",
						Items:      []core.KeyToPath{{Key: "key", Path: ""}},
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "secret.items[0].path",

				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "secret with leading ..",
			Object: []core.Volume{{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName: "s",
						Items:      []core.KeyToPath{{Key: "key", Path: "../foo"}},
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "secret.items[0].path",
				BadValue: "../foo",
			}},
		}, {
			Name: "secret with .. inside",
			Object: []core.Volume{{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName: "s",
						Items:      []core.KeyToPath{{Key: "key", Path: "foo/../bar"}},
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "secret.items[0].path",
				BadValue: "foo/../bar",
			}},
		}, {
			Name: "secret with invalid positive defaultMode",
			Object: []core.Volume{{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName:  "s",
						DefaultMode: ptr.To[int32](01000),
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "secret.defaultMode",
				BadValue: int32(01000),
			}},
		}, {
			Name: "secret with invalid negative defaultMode",
			Object: []core.Volume{{
				Name: "secret",
				VolumeSource: core.VolumeSource{
					Secret: &core.SecretVolumeSource{
						SecretName:  "s",
						DefaultMode: ptr.To[int32](-1),
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "secret.defaultMode",
				BadValue: int32(-1),
			}},
		},
		// ConfigMap
		{
			Name: "valid ConfigMap",
			Object: []core.Volume{{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{
							Name: "my-cfgmap",
						},
					},
				},
			}},
		}, {
			Name: "valid ConfigMap with defaultMode",
			Object: []core.Volume{{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{
							Name: "my-cfgmap",
						},
						DefaultMode: ptr.To[int32](0644),
					},
				},
			}},
		}, {
			Name: "valid ConfigMap with projection and mode",
			Object: []core.Volume{{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{
							Name: "my-cfgmap"},
						Items: []core.KeyToPath{{
							Key:  "key",
							Path: "filename",
							Mode: ptr.To[int32](0644),
						}},
					},
				},
			}},
		}, {
			Name: "valid ConfigMap with subdir projection",
			Object: []core.Volume{{
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
			}},
		}, {
			Name: "configmap with missing path",
			Object: []core.Volume{{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{Name: "c"},
						Items:                []core.KeyToPath{{Key: "key", Path: ""}},
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:       field.ErrorTypeRequired,
				Field:      "configMap.items[0].path",
				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "configmap with leading ..",
			Object: []core.Volume{{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{Name: "c"},
						Items:                []core.KeyToPath{{Key: "key", Path: "../foo"}},
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "configMap.items[0].path",
				BadValue: "../foo",
			}},
		}, {
			Name: "configmap with .. inside",
			Object: []core.Volume{{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{Name: "c"},
						Items:                []core.KeyToPath{{Key: "key", Path: "foo/../bar"}},
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "configMap.items[0].path",
				BadValue: "foo/../bar",
			}},
		}, {
			Name: "configmap with invalid positive defaultMode",
			Object: []core.Volume{{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{Name: "c"},
						DefaultMode:          ptr.To[int32](01000),
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "configMap.defaultMode",
				BadValue: int32(01000),
			}},
		}, {
			Name: "configmap with invalid negative defaultMode",
			Object: []core.Volume{{
				Name: "cfgmap",
				VolumeSource: core.VolumeSource{
					ConfigMap: &core.ConfigMapVolumeSource{
						LocalObjectReference: core.LocalObjectReference{Name: "c"},
						DefaultMode:          ptr.To[int32](-1),
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "configMap.defaultMode",
				BadValue: int32(-1),
			}},
		},
		// Glusterfs
		{
			Name: "valid Glusterfs",
			Object: []core.Volume{{
				Name: "glusterfs",
				VolumeSource: core.VolumeSource{
					Glusterfs: &core.GlusterfsVolumeSource{
						EndpointsName: "host1",
						Path:          "path",
						ReadOnly:      false,
					},
				},
			}},
		}, {
			Name: "empty hosts",
			Object: []core.Volume{{
				Name: "glusterfs",
				VolumeSource: core.VolumeSource{
					Glusterfs: &core.GlusterfsVolumeSource{
						EndpointsName: "",
						Path:          "path",
						ReadOnly:      false,
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "glusterfs.endpoints",

				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "empty path",
			Object: []core.Volume{{
				Name: "glusterfs",
				VolumeSource: core.VolumeSource{
					Glusterfs: &core.GlusterfsVolumeSource{
						EndpointsName: "host",
						Path:          "",
						ReadOnly:      false,
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "glusterfs.path",

				SchemaType: field.ErrorTypeInvalid,
			}},
		},
		// Flocker
		{
			Name: "valid Flocker -- datasetUUID",
			Object: []core.Volume{{
				Name: "flocker",
				VolumeSource: core.VolumeSource{
					Flocker: &core.FlockerVolumeSource{
						DatasetUUID: "d846b09d-223d-43df-ab5b-d6db2206a0e4",
					},
				},
			}},
		}, {
			Name: "valid Flocker -- datasetName",
			Object: []core.Volume{{
				Name: "flocker",
				VolumeSource: core.VolumeSource{
					Flocker: &core.FlockerVolumeSource{
						DatasetName: "datasetName",
					},
				},
			}},
		}, {
			Name: "both empty",
			Object: []core.Volume{{
				Name: "flocker",
				VolumeSource: core.VolumeSource{
					Flocker: &core.FlockerVolumeSource{
						DatasetName: "",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "flocker",
			}},
		}, {
			Name: "both specified",
			Object: []core.Volume{{
				Name: "flocker",
				VolumeSource: core.VolumeSource{
					Flocker: &core.FlockerVolumeSource{
						DatasetName: "datasetName",
						DatasetUUID: "d846b09d-223d-43df-ab5b-d6db2206a0e4",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "flocker",
				BadValue: "resource",
				Detail:   "can not be specified simultaneously",
			}},
		}, {
			Name: "slash in flocker datasetName",
			Object: []core.Volume{{
				Name: "flocker",
				VolumeSource: core.VolumeSource{
					Flocker: &core.FlockerVolumeSource{
						DatasetName: "foo/bar",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "flocker.datasetName",
				Detail:   "must not contain '/'",
				BadValue: "foo/bar",
			}},
		},
		// RBD
		{
			Name: "valid RBD",
			Object: []core.Volume{{
				Name: "rbd",
				VolumeSource: core.VolumeSource{
					RBD: &core.RBDVolumeSource{
						CephMonitors: []string{"foo"},
						RBDImage:     "bar",
						FSType:       "ext4",
					},
				},
			}},
		}, {
			Name: "empty rbd monitors",
			Object: []core.Volume{{
				Name: "rbd",
				VolumeSource: core.VolumeSource{
					RBD: &core.RBDVolumeSource{
						CephMonitors: []string{},
						RBDImage:     "bar",
						FSType:       "ext4",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:       field.ErrorTypeRequired,
				Field:      "rbd.monitors",
				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "empty image",
			Object: []core.Volume{{
				Name: "rbd",
				VolumeSource: core.VolumeSource{
					RBD: &core.RBDVolumeSource{
						CephMonitors: []string{"foo"},
						RBDImage:     "",
						FSType:       "ext4",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "rbd.image",

				SchemaType: field.ErrorTypeInvalid,
			}},
		},
		// Cinder
		{
			Name: "valid Cinder",
			Object: []core.Volume{{
				Name: "cinder",
				VolumeSource: core.VolumeSource{
					Cinder: &core.CinderVolumeSource{
						VolumeID: "29ea5088-4f60-4757-962e-dba678767887",
						FSType:   "ext4",
						ReadOnly: false,
					},
				},
			}},
		},
		// CephFS
		{
			Name: "valid CephFS",
			Object: []core.Volume{{
				Name: "cephfs",
				VolumeSource: core.VolumeSource{
					CephFS: &core.CephFSVolumeSource{
						Monitors: []string{"foo"},
					},
				},
			}},
		}, {
			Name: "empty cephfs monitors",
			Object: []core.Volume{{
				Name: "cephfs",
				VolumeSource: core.VolumeSource{
					CephFS: &core.CephFSVolumeSource{
						Monitors: []string{},
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:       field.ErrorTypeRequired,
				Field:      "cephfs.monitors",
				SchemaType: field.ErrorTypeInvalid,
			}},
		},
		// DownwardAPI
		{
			Name: "valid DownwardAPI",
			Object: []core.Volume{{
				Name: "downwardapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Path: "labels",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}, {
							Path: "labels with subscript",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels['key']",
							},
						}, {
							Path: "labels with complex subscript",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels['test.example.com/key']",
							},
						}, {
							Path: "annotations",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.annotations",
							},
						}, {
							Path: "annotations with subscript",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.annotations['key']",
							},
						}, {
							Path: "annotations with complex subscript",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.annotations['TEST.EXAMPLE.COM/key']",
							},
						}, {
							Path: "namespace",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.namespace",
							},
						}, {
							Path: "name",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.name",
							},
						}, {
							Path: "path/with/subdirs",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}, {
							Path: "path/./withdot",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}, {
							Path: "path/with/embedded..dotdot",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}, {
							Path: "path/with/leading/..dotdot",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}, {
							Path: "cpu_limit",
							ResourceFieldRef: &core.ResourceFieldSelector{
								ContainerName: "test-container",
								Resource:      "limits.cpu",
							},
						}, {
							Path: "cpu_request",
							ResourceFieldRef: &core.ResourceFieldSelector{
								ContainerName: "test-container",
								Resource:      "requests.cpu",
							},
						}, {
							Path: "memory_limit",
							ResourceFieldRef: &core.ResourceFieldSelector{
								ContainerName: "test-container",
								Resource:      "limits.memory",
							},
						}, {
							Path: "memory_request",
							ResourceFieldRef: &core.ResourceFieldSelector{
								ContainerName: "test-container",
								Resource:      "requests.memory",
							},
						}},
					},
				},
			}},
		}, {
			Name: "hugepages-downwardAPI-enabled",
			Object: []core.Volume{{
				Name: "downwardapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Path: "hugepages_request",
							ResourceFieldRef: &core.ResourceFieldSelector{
								ContainerName: "test-container",
								Resource:      "requests.hugepages-2Mi",
							},
						}, {
							Path: "hugepages_limit",
							ResourceFieldRef: &core.ResourceFieldSelector{
								ContainerName: "test-container",
								Resource:      "limits.hugepages-2Mi",
							},
						}},
					},
				},
			}},
		}, {
			Name: "downapi valid defaultMode",
			Object: []core.Volume{{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						DefaultMode: ptr.To[int32](0644),
					},
				},
			}},
		}, {
			Name: "downapi valid item mode",
			Object: []core.Volume{{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Mode: ptr.To[int32](0644),
							Path: "path",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}},
					},
				},
			}},
		}, {
			Name: "downapi invalid positive item mode",
			Object: []core.Volume{{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Mode: ptr.To[int32](01000),
							Path: "path",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}},
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "downwardAPI.mode",
				BadValue: int32(01000),

				SchemaField: "downwardAPI.items[0].mode",
			}},
		}, {
			Name: "downapi invalid negative item mode",
			Object: []core.Volume{{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						Items: []core.DownwardAPIVolumeFile{{
							Mode: ptr.To[int32](-1),
							Path: "path",
							FieldRef: &core.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.labels",
							},
						}},
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:        field.ErrorTypeInvalid,
				Field:       "downwardAPI.mode",
				BadValue:    int32(-1),
				SchemaField: "downwardAPI.items[0].mode",
			}},
		}, {
			Name: "downapi empty metatada path",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:        field.ErrorTypeRequired,
				Field:       "downwardAPI.path",
				SchemaField: "downwardAPI.items[0].path",
				SchemaType:  field.ErrorTypeInvalid,
			}},
		}, {
			Name: "downapi absolute path",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:        field.ErrorTypeInvalid,
				Field:       "downwardAPI.path",
				BadValue:    "/absolutepath",
				SchemaField: "downwardAPI.items[0].path",
			}},
		}, {
			Name: "downapi dot dot path",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:        field.ErrorTypeInvalid,
				Field:       "downwardAPI.path",
				Detail:      `must not contain '..'`,
				BadValue:    "../../passwd",
				SchemaField: "downwardAPI.items[0].path",
			}},
		}, {
			Name: "downapi dot dot file name",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:        field.ErrorTypeInvalid,
				Field:       "downwardAPI.path",
				Detail:      `must not start with '..'`,
				BadValue:    "..badFileName",
				SchemaField: "downwardAPI.items[0].path",
			}},
		}, {
			Name: "downapi dot dot first level dirent",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:        field.ErrorTypeInvalid,
				Field:       "downwardAPI.path",
				Detail:      `must not start with '..'`,
				BadValue:    "..badDirName/goodFileName",
				SchemaField: "downwardAPI.items[0].path",
			}},
		}, {
			Name: "downapi fieldRef and ResourceFieldRef together",
			Object: []core.Volume{{
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
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "downwardAPI",
				Detail:   "fieldRef and resourceFieldRef can not be specified simultaneously",
				BadValue: "resource",

				SchemaField: "downwardAPI.items[0]",
			}},
		}, {
			Name: "downapi invalid positive defaultMode",
			Object: []core.Volume{{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						DefaultMode: ptr.To[int32](01000),
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "downwardAPI.defaultMode",
				BadValue: int32(01000),
			}},
		}, {
			Name: "downapi invalid negative defaultMode",
			Object: []core.Volume{{
				Name: "downapi",
				VolumeSource: core.VolumeSource{
					DownwardAPI: &core.DownwardAPIVolumeSource{
						DefaultMode: ptr.To[int32](-1),
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "downwardAPI.defaultMode",
				BadValue: int32(-1),
			}},
		},
		// FC
		{
			Name: "FC valid targetWWNs and lun",
			Object: []core.Volume{{
				Name: "fc",
				VolumeSource: core.VolumeSource{
					FC: &core.FCVolumeSource{
						TargetWWNs: []string{"some_wwn"},
						Lun:        ptr.To[int32](1),
						FSType:     "ext4",
						ReadOnly:   false,
					},
				},
			}},
		}, {
			Name: "FC valid wwids",
			Object: []core.Volume{{
				Name: "fc",
				VolumeSource: core.VolumeSource{
					FC: &core.FCVolumeSource{
						WWIDs:    []string{"some_wwid"},
						FSType:   "ext4",
						ReadOnly: false,
					},
				},
			}},
		}, {
			Name: "FC empty targetWWNs and wwids",
			Object: []core.Volume{{
				Name: "fc",
				VolumeSource: core.VolumeSource{
					FC: &core.FCVolumeSource{
						TargetWWNs: []string{},
						Lun:        ptr.To[int32](1),
						WWIDs:      []string{},
						FSType:     "ext4",
						ReadOnly:   false,
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:   field.ErrorTypeRequired,
				Field:  "fc.targetWWNs",
				Detail: "must specify either targetWWNs or wwids",
			}},
		}, {
			Name: "FC invalid: both targetWWNs and wwids simultaneously",
			Object: []core.Volume{{
				Name: "fc",
				VolumeSource: core.VolumeSource{
					FC: &core.FCVolumeSource{
						TargetWWNs: []string{"some_wwn"},
						Lun:        ptr.To[int32](1),
						WWIDs:      []string{"some_wwid"},
						FSType:     "ext4",
						ReadOnly:   false,
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "fc.targetWWNs",
				Detail:   "targetWWNs and wwids can not be specified simultaneously",
				BadValue: []string{"some_wwn"},
			}},
		}, {
			Name: "FC valid targetWWNs and empty lun",
			Object: []core.Volume{{
				Name: "fc",
				VolumeSource: core.VolumeSource{
					FC: &core.FCVolumeSource{
						TargetWWNs: []string{"wwn"},
						Lun:        nil,
						FSType:     "ext4",
						ReadOnly:   false,
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:   field.ErrorTypeRequired,
				Field:  "fc.lun",
				Detail: "lun is required if targetWWNs is specified",
			}},
		}, {
			Name: "FC valid targetWWNs and invalid lun",
			Object: []core.Volume{{
				Name: "fc",
				VolumeSource: core.VolumeSource{
					FC: &core.FCVolumeSource{
						TargetWWNs: []string{"wwn"},
						Lun:        ptr.To[int32](256),
						FSType:     "ext4",
						ReadOnly:   false,
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:         field.ErrorTypeInvalid,
				Field:        "fc.lun",
				Detail:       `must be between 0 and 255`,
				BadValue:     ptr.To(int32(256)),
				SchemaDetail: `should be less than or equal to 255`,
			}},
		},
		// FlexVolume
		{
			Name: "valid FlexVolume",
			Object: []core.Volume{{
				Name: "flex-volume",
				VolumeSource: core.VolumeSource{
					FlexVolume: &core.FlexVolumeSource{
						Driver: "kubernetes.io/blue",
						FSType: "ext4",
					},
				},
			}},
		},
		// AzureFile
		{
			Name: "valid AzureFile",
			Object: []core.Volume{{
				Name: "azure-file",
				VolumeSource: core.VolumeSource{
					AzureFile: &core.AzureFileVolumeSource{
						SecretName: "key",
						ShareName:  "share",
						ReadOnly:   false,
					},
				},
			}},
		}, {
			Name: "AzureFile empty secret",
			Object: []core.Volume{{
				Name: "azure-file",
				VolumeSource: core.VolumeSource{
					AzureFile: &core.AzureFileVolumeSource{
						SecretName: "",
						ShareName:  "share",
						ReadOnly:   false,
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:       field.ErrorTypeRequired,
				Field:      "azureFile.secretName",
				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "AzureFile empty share",
			Object: []core.Volume{{
				Name: "azure-file",
				VolumeSource: core.VolumeSource{
					AzureFile: &core.AzureFileVolumeSource{
						SecretName: "name",
						ShareName:  "",
						ReadOnly:   false,
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "azureFile.shareName",

				SchemaType: field.ErrorTypeInvalid,
			}},
		},
		// Quobyte
		{
			Name: "valid Quobyte",
			Object: []core.Volume{{
				Name: "quobyte",
				VolumeSource: core.VolumeSource{
					Quobyte: &core.QuobyteVolumeSource{
						Registry: "registry:7861",
						Volume:   "volume",
						ReadOnly: false,
						User:     "root",
						Group:    "root",
						Tenant:   "ThisIsSomeTenantUUID",
					},
				},
			}},
		}, {
			Name: "empty registry quobyte",
			Object: []core.Volume{{
				Name: "quobyte",
				VolumeSource: core.VolumeSource{
					Quobyte: &core.QuobyteVolumeSource{
						Volume: "/test",
						Tenant: "ThisIsSomeTenantUUID",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:       field.ErrorTypeRequired,
				Field:      "quobyte.registry",
				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "wrong format registry quobyte",
			Object: []core.Volume{{
				Name: "quobyte",
				VolumeSource: core.VolumeSource{
					Quobyte: &core.QuobyteVolumeSource{
						Registry: "registry7861",
						Volume:   "/test",
						Tenant:   "ThisIsSomeTenantUUID",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "quobyte.registry",
				BadValue: "registry7861",
			}},
		}, {
			Name: "wrong format multiple registries quobyte",
			Object: []core.Volume{{
				Name: "quobyte",
				VolumeSource: core.VolumeSource{
					Quobyte: &core.QuobyteVolumeSource{
						Registry: "registry:7861,reg2",
						Volume:   "/test",
						Tenant:   "ThisIsSomeTenantUUID",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:     field.ErrorTypeInvalid,
				Field:    "quobyte.registry",
				BadValue: "registry:7861,reg2",
			}},
		}, {
			Name: "empty volume quobyte",
			Object: []core.Volume{{
				Name: "quobyte",
				VolumeSource: core.VolumeSource{
					Quobyte: &core.QuobyteVolumeSource{
						Registry: "registry:7861",
						Tenant:   "ThisIsSomeTenantUUID",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:       field.ErrorTypeRequired,
				Field:      "quobyte.volume",
				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "empty tenant quobyte",
			Object: []core.Volume{{
				Name: "quobyte",
				VolumeSource: core.VolumeSource{
					Quobyte: &core.QuobyteVolumeSource{
						Registry: "registry:7861",
						Volume:   "/test",
						Tenant:   "",
					},
				},
			}},
		}, {
			Name: "too long tenant quobyte",
			Object: []core.Volume{{
				Name: "quobyte",
				VolumeSource: core.VolumeSource{
					Quobyte: &core.QuobyteVolumeSource{
						Registry: "registry:7861",
						Volume:   "/test",
						Tenant:   "this is too long to be a valid uuid so this test has to fail on the maximum length validation of the tenant.",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "quobyte.tenant",

				SchemaType: field.ErrorTypeTooLong,
			}},
		},
		// AzureDisk
		{
			Name: "valid AzureDisk",
			Object: []core.Volume{{
				Name: "azure-disk",
				VolumeSource: core.VolumeSource{
					AzureDisk: &core.AzureDiskVolumeSource{
						DiskName:    "foo",
						DataDiskURI: "https://blob/vhds/bar.vhd",
					},
				},
			}},
		}, {
			Name: "AzureDisk empty disk name",
			Object: []core.Volume{{
				Name: "azure-disk",
				VolumeSource: core.VolumeSource{
					AzureDisk: &core.AzureDiskVolumeSource{
						DiskName:    "",
						DataDiskURI: "https://blob/vhds/bar.vhd",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:       field.ErrorTypeRequired,
				Field:      "azureDisk.diskName",
				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "AzureDisk empty disk uri",
			Object: []core.Volume{{
				Name: "azure-disk",
				VolumeSource: core.VolumeSource{
					AzureDisk: &core.AzureDiskVolumeSource{
						DiskName:    "foo",
						DataDiskURI: "",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:       field.ErrorTypeRequired,
				Field:      "azureDisk.diskURI",
				SchemaType: field.ErrorTypeInvalid,
			}},
		},
		// ScaleIO
		{
			Name: "valid scaleio volume",
			Object: []core.Volume{{
				Name: "scaleio-volume",
				VolumeSource: core.VolumeSource{
					ScaleIO: &core.ScaleIOVolumeSource{
						Gateway:    "http://abcd/efg",
						System:     "test-system",
						VolumeName: "test-vol-1",
					},
				},
			}},
		}, {
			Name: "ScaleIO with empty name",
			Object: []core.Volume{{
				Name: "scaleio-volume",
				VolumeSource: core.VolumeSource{
					ScaleIO: &core.ScaleIOVolumeSource{
						Gateway:    "http://abcd/efg",
						System:     "test-system",
						VolumeName: "",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "scaleIO.volumeName",

				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "ScaleIO with empty gateway",
			Object: []core.Volume{{
				Name: "scaleio-volume",
				VolumeSource: core.VolumeSource{
					ScaleIO: &core.ScaleIOVolumeSource{
						Gateway:    "",
						System:     "test-system",
						VolumeName: "test-vol-1",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "scaleIO.gateway",

				SchemaType: field.ErrorTypeInvalid,
			}},
		}, {
			Name: "ScaleIO with empty system",
			Object: []core.Volume{{
				Name: "scaleio-volume",
				VolumeSource: core.VolumeSource{
					ScaleIO: &core.ScaleIOVolumeSource{
						Gateway:    "http://agc/efg/gateway",
						System:     "",
						VolumeName: "test-vol-1",
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "scaleIO.system",

				SchemaType: field.ErrorTypeInvalid,
			}},
		},
		// ProjectedVolumeSource
		{
			Name: "ProjectedVolumeSource more than one projection in a source",
			Object: []core.Volume{{
				Name: "projected-volume",
				VolumeSource: core.VolumeSource{
					Projected: &core.ProjectedVolumeSource{
						Sources: []core.VolumeProjection{{
							Secret: &core.SecretProjection{
								LocalObjectReference: core.LocalObjectReference{
									Name: "foo",
								},
							},
						}, {
							Secret: &core.SecretProjection{
								LocalObjectReference: core.LocalObjectReference{
									Name: "foo",
								},
							},
							DownwardAPI: &core.DownwardAPIProjection{},
						}},
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeForbidden,
				Field: "projected.sources[1]",
			}},
		}, {
			Name: "ProjectedVolumeSource more than one projection in a source",
			Object: []core.Volume{{
				Name: "projected-volume",
				VolumeSource: core.VolumeSource{
					Projected: &core.ProjectedVolumeSource{
						Sources: []core.VolumeProjection{{
							Secret: &core.SecretProjection{},
						}, {
							Secret:      &core.SecretProjection{},
							DownwardAPI: &core.DownwardAPIProjection{},
						}},
					},
				},
			}},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{{
				Type:  field.ErrorTypeRequired,
				Field: "projected.sources[0].secret.name",
			}, {
				Type:  field.ErrorTypeRequired,
				Field: "projected.sources[1].secret.name",
			}, {
				Type:  field.ErrorTypeForbidden,
				Field: "projected.sources[1]",
			}},
		},
		{
			Name: "Duplicates case",
			Object: []core.Volume{
				{Name: "abc", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
				{Name: "abc", VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{}}},
			},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{
				{
					Type:        field.ErrorTypeDuplicate,
					Field:       "volumes[1].name",
					BadValue:    "abc",
					SchemaField: "volumes[1]",
				},
			},
		},
		{
			Name: "HugePages",
			Object: []core.Volume{
				{
					Name:         "hugepages",
					VolumeSource: core.VolumeSource{EmptyDir: &core.EmptyDirVolumeSource{Medium: core.StorageMediumHugePages}},
				},
			},
		},
	}

	cases := []apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{}
	for _, tc := range testCases {

		prefixedErrors := make([]apivalidationtesting.ExpectedFieldError, len(tc.ExpectedErrors))

		for i, e := range tc.ExpectedErrors {
			prefix := "template.spec.volumes[0]."
			if strings.HasPrefix(e.Field, "volumes") {
				prefix = "template.spec."
			}

			prefixedErrors[i] = e
			prefixedErrors[i].Field = prefix + e.Field
			if len(e.SchemaField) > 0 {
				prefixedErrors[i].SchemaField = prefix + e.SchemaField
			}
		}

		cases = append(cases, apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{
			Name: tc.Name,
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"},
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{
						Containers: []core.Container{{Name: "foo", Image: "bar"}},
						Volumes:    tc.Object,
					},
				},
			},
			ExpectedErrors: prefixedErrors,
		})
	}

	apivalidationtesting.TestValidate[core.PodTemplate, validation.PodValidationOptions](t, coreScheme, coreDefs, func(pt *core.PodTemplate, pvo validation.PodValidationOptions) field.ErrorList {
		_, errs := validation.ValidateVolumes(pt.Template.Spec.Volumes, nil, field.NewPath("template.spec.volumes"), pvo)
		return errs
	}, cases...)
}

func TestValidatePorts(t *testing.T) {

	portCases := []apivalidationtesting.TestCase[[]core.ContainerPort, struct{}]{

		{
			Name:   `all fields populated`,
			Object: []core.ContainerPort{{Name: "abc", ContainerPort: 80, HostPort: 80, Protocol: "TCP"}},
		},
		{
			Name:   `missing hostport tcp`,
			Object: []core.ContainerPort{{Name: "easy", ContainerPort: 82, Protocol: "TCP"}},
		},
		{
			Name:   `missing hostport udp`,
			Object: []core.ContainerPort{{Name: "as", ContainerPort: 83, Protocol: "UDP"}},
		},
		{
			Name:   `missing hostport - sctp`,
			Object: []core.ContainerPort{{Name: "do-re-me", ContainerPort: 84, Protocol: "SCTP"}},
		},
		{
			Name:   `missing name`,
			Object: []core.ContainerPort{{ContainerPort: 85, Protocol: "TCP"}},
		},
		{
			Name: `non-canonical`,
			Object: []core.ContainerPort{
				{ContainerPort: 80, Protocol: "TCP"},
			},
		},
		{
			Name:   "name > 15 characters",
			Object: []core.ContainerPort{{Name: strings.Repeat("a", 16), ContainerPort: 80, Protocol: "TCP"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{
				Type:     field.ErrorTypeInvalid,
				Field:    "ports[0].name",
				Detail:   "15",
				BadValue: strings.Repeat("a", 16),
			}},
		},
		{
			Name:   "name contains invalid characters",
			Object: []core.ContainerPort{{Name: "a.b.c", ContainerPort: 80, Protocol: "TCP"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{
				Type:     field.ErrorTypeInvalid,
				Field:    "ports[0].name",
				Detail:   "alpha-numeric",
				BadValue: "a.b.c",
			}},
		},
		{
			Name:   "name is a number",
			Object: []core.ContainerPort{{Name: "80", ContainerPort: 80, Protocol: "TCP"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{
				Type:     field.ErrorTypeInvalid,
				Field:    "ports[0].name",
				Detail:   "at least one letter",
				BadValue: "80",
			}},
		},
		{
			Name: "name not unique",
			Object: []core.ContainerPort{
				{Name: "abc", ContainerPort: 80, Protocol: "TCP"},
				{Name: "abc", ContainerPort: 81, Protocol: "TCP"},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{
				Type:     field.ErrorTypeDuplicate,
				Field:    "ports[1].name",
				Detail:   "",
				BadValue: "abc",

				SchemaField: "ports",
			}},
		},
		{
			Name:   "zero container port",
			Object: []core.ContainerPort{{ContainerPort: 0, Protocol: "TCP"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{
				Type:       field.ErrorTypeRequired,
				Field:      "ports[0].containerPort",
				Detail:     "",
				SchemaType: field.ErrorTypeInvalid,
			}},
		},
		{
			Name:   "invalid container port",
			Object: []core.ContainerPort{{ContainerPort: 65536, Protocol: "TCP"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{
				Type:     field.ErrorTypeInvalid,
				Field:    "ports[0].containerPort",
				Detail:   "between",
				BadValue: int32(65536),

				SchemaDetail: `less than or equal to`,
			}},
		},
		{
			Name:   "invalid host port",
			Object: []core.ContainerPort{{ContainerPort: 80, HostPort: 65536, Protocol: "TCP"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{
				Type:     field.ErrorTypeInvalid,
				Field:    "ports[0].hostPort",
				Detail:   "between",
				BadValue: int32(65536),

				SchemaDetail: `less than or equal to`,
			}},
		},
		{
			Name:   "invalid protocol case",
			Object: []core.ContainerPort{{ContainerPort: 80, Protocol: "tcp"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{
				Type:     field.ErrorTypeNotSupported,
				Field:    "ports[0].protocol",
				Detail:   `supported values: "SCTP", "TCP", "UDP"`,
				BadValue: core.Protocol("tcp"),
			}},
		},
		{
			Name:   "invalid protocol",
			Object: []core.ContainerPort{{ContainerPort: 80, Protocol: "ICMP"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{
				Type:     field.ErrorTypeNotSupported,
				Field:    "ports[0].protocol",
				Detail:   `supported values: "SCTP", "TCP", "UDP"`,
				BadValue: core.Protocol("ICMP"),
			}},
		},
		{
			Name:   "protocol required",
			Object: []core.ContainerPort{{Name: "abc", ContainerPort: 80}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{
				Type:             field.ErrorTypeRequired,
				Field:            "ports[0].protocol",
				Detail:           "",
				SchemaSkipReason: `Defaulted`,
			}},
		},
	}

	cases := []apivalidationtesting.TestCase[*core.PodTemplate, struct{}]{}
	for _, tc := range portCases {
		prefixedErrors := make([]apivalidationtesting.ExpectedFieldError, len(tc.ExpectedErrors))
		for i, e := range tc.ExpectedErrors {
			prefix := "template.spec.containers[0]."
			prefixedErrors[i] = e
			prefixedErrors[i].Field = prefix + e.Field
			if len(e.SchemaField) > 0 {
				prefixedErrors[i].SchemaField = prefix + e.SchemaField
			}
		}

		cases = append(cases, apivalidationtesting.TestCase[*core.PodTemplate, struct{}]{
			Name: tc.Name,
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "foos", Namespace: "ns"},
				Template: core.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Name: "foos", Namespace: "ns"},
					Spec: core.PodSpec{
						DNSPolicy: core.DNSClusterFirst, RestartPolicy: core.RestartPolicyAlways,
						Containers: []core.Container{
							{Name: "foo", Image: "bar", ImagePullPolicy: core.PullIfNotPresent, Ports: tc.Object, TerminationMessagePolicy: core.TerminationMessageReadFile}},
					}}},
			ExpectedErrors: prefixedErrors,
		})
	}

	apivalidationtesting.TestValidate[core.PodTemplate, struct{}](t, coreScheme, coreDefs, func(pt *core.PodTemplate, _ struct{}) field.ErrorList {
		errs := validation.ValidatePodSpec(&pt.Template.Spec, &pt.Template.ObjectMeta, field.NewPath("template.spec"), validation.PodValidationOptions{})
		return errs
	}, cases...)
}

func TestValidateEnv(t *testing.T) {
	envCases := []apivalidationtesting.TestCase[[]core.EnvVar, validation.PodValidationOptions]{
		{
			Name: "success cases",
			Object: []core.EnvVar{
				{Name: "abc", Value: "value"},
				{Name: "ABC", Value: "value"},
				{Name: "AbC_123", Value: "value"},
				{Name: "abc", Value: ""},
				{Name: "a.b.c", Value: "value"},
				{Name: "a-b-c", Value: "value"}, {
					Name: "abc",
					ValueFrom: &core.EnvVarSource{
						FieldRef: &core.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  "metadata.annotations['key']",
						},
					},
				}, {
					Name: "abc",
					ValueFrom: &core.EnvVarSource{
						FieldRef: &core.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  "metadata.labels['key']",
						},
					},
				}, {
					Name: "abc",
					ValueFrom: &core.EnvVarSource{
						FieldRef: &core.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  "metadata.name",
						},
					},
				}, {
					Name: "abc",
					ValueFrom: &core.EnvVarSource{
						FieldRef: &core.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  "metadata.namespace",
						},
					},
				}, {
					Name: "abc",
					ValueFrom: &core.EnvVarSource{
						FieldRef: &core.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  "metadata.uid",
						},
					},
				}, {
					Name: "abc",
					ValueFrom: &core.EnvVarSource{
						FieldRef: &core.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  "spec.nodeName",
						},
					},
				}, {
					Name: "abc",
					ValueFrom: &core.EnvVarSource{
						FieldRef: &core.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  "spec.serviceAccountName",
						},
					},
				}, {
					Name: "abc",
					ValueFrom: &core.EnvVarSource{
						FieldRef: &core.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  "status.hostIP",
						},
					},
				}, {
					Name: "abc",
					ValueFrom: &core.EnvVarSource{
						FieldRef: &core.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  "status.podIP",
						},
					},
				}, {
					Name: "abc",
					ValueFrom: &core.EnvVarSource{
						FieldRef: &core.ObjectFieldSelector{
							APIVersion: "v1",
							FieldPath:  "status.podIPs",
						},
					},
				}, {
					Name: "secret_value",
					ValueFrom: &core.EnvVarSource{
						SecretKeyRef: &core.SecretKeySelector{
							LocalObjectReference: core.LocalObjectReference{
								Name: "some-secret",
							},
							Key: "secret-key",
						},
					},
				}, {
					Name: "ENV_VAR_1",
					ValueFrom: &core.EnvVarSource{
						ConfigMapKeyRef: &core.ConfigMapKeySelector{
							LocalObjectReference: core.LocalObjectReference{
								Name: "some-config-map",
							},
							Key: "some-key",
						},
					},
				},
			},
			ExpectedErrors: []apivalidationtesting.ExpectedFieldError{
				{
					Type:  field.ErrorTypeDuplicate,
					Field: "[3]",
					// https://github.com/kubernetes/kubernetes/issues/113482
					NativeSkipReason: "Duplicate env name allowed on native but not expressible in schema due to SSA mapkeys",
				},
			},
		},
		{
			Name:           "zero-length name",
			Object:         []core.EnvVar{{Name: ""}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: "[0].name", Type: field.ErrorTypeRequired, SchemaType: field.ErrorTypeInvalid}},
		}, {

			Name: "value and valueFrom specified",
			Object: []core.EnvVar{{
				Name:  "abc",
				Value: "foo",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "metadata.name",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: "[0].valueFrom", Type: field.ErrorTypeInvalid, BadValue: "", Detail: "may not be specified when `value` is not empty"}},
		}, {

			Name: "valueFrom without a source",
			Object: []core.EnvVar{{
				Name:      "abc",
				ValueFrom: &core.EnvVarSource{},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: "[0].valueFrom", Type: field.ErrorTypeInvalid, BadValue: "", Detail: "must specify one of: `fieldRef`, `resourceFieldRef`, `configMapKeyRef` or `secretKeyRef`"}},
		}, {

			Name: "valueFrom.fieldRef and valueFrom.secretKeyRef specified",
			Object: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "metadata.name",
					},
					SecretKeyRef: &core.SecretKeySelector{
						LocalObjectReference: core.LocalObjectReference{
							Name: "a-secret",
						},
						Key: "a-key",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: "[0].valueFrom", Type: field.ErrorTypeInvalid, BadValue: "", Detail: "may not have more than one field specified at a time"}},
		}, {

			Name: "valueFrom.fieldRef and valueFrom.configMapKeyRef set",
			Object: []core.EnvVar{{
				Name: "some_var_name",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "metadata.name",
					},
					ConfigMapKeyRef: &core.ConfigMapKeySelector{
						LocalObjectReference: core.LocalObjectReference{
							Name: "some-config-map",
						},
						Key: "some-key",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom`, Type: field.ErrorTypeInvalid, BadValue: "", Detail: `may not have more than one field specified at a time`}},
		}, {

			Name: "valueFrom.fieldRef and valueFrom.secretKeyRef specified",
			Object: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						APIVersion: "v1",
						FieldPath:  "metadata.name",
					},
					SecretKeyRef: &core.SecretKeySelector{
						LocalObjectReference: core.LocalObjectReference{
							Name: "a-secret",
						},
						Key: "a-key",
					},
					ConfigMapKeyRef: &core.ConfigMapKeySelector{
						LocalObjectReference: core.LocalObjectReference{
							Name: "some-config-map",
						},
						Key: "some-key",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom`, Type: field.ErrorTypeInvalid, BadValue: ``, Detail: `may not have more than one field specified at a time`}},
		}, {

			Name: "valueFrom.secretKeyRef.name invalid",
			Object: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					SecretKeyRef: &core.SecretKeySelector{
						LocalObjectReference: core.LocalObjectReference{
							Name: "$%^&*#",
						},
						Key: "a-key",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom.secretKeyRef.name`, BadValue: `$%^&*#`, Type: field.ErrorTypeInvalid}},
		}, {

			Name: "valueFrom.configMapKeyRef.name invalid",
			Object: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					ConfigMapKeyRef: &core.ConfigMapKeySelector{
						LocalObjectReference: core.LocalObjectReference{
							Name: "$%^&*#",
						},
						Key: "some-key",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom.configMapKeyRef.name`, BadValue: `$%^&*#`, Type: field.ErrorTypeInvalid}},
		}, {

			Name: "missing FieldPath on ObjectFieldSelector",
			Object: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						APIVersion: "v1",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom.fieldRef.fieldPath`, Type: field.ErrorTypeRequired, SchemaType: field.ErrorTypeInvalid}},
		}, {

			Name: "missing APIVersion on ObjectFieldSelector",
			Object: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath: "metadata.name",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom.fieldRef.apiVersion`, Type: field.ErrorTypeRequired}},
		}, {

			Name: "invalid fieldPath",
			Object: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "metadata.whoops",
						APIVersion: "v1",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom.fieldRef.fieldPath`, Type: field.ErrorTypeInvalid, BadValue: `metadata.whoops`, Detail: `error converting fieldPath`}},
		}, {

			Name: "metadata.name with subscript",
			Object: []core.EnvVar{{
				Name: "labels",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "metadata.name['key']",
						APIVersion: "v1",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom.fieldRef.fieldPath`, Type: field.ErrorTypeInvalid, BadValue: `metadata.name['key']`, Detail: `error converting fieldPath: field label does not support subscript`}},
		}, {

			Name: "metadata.labels without subscript",
			Object: []core.EnvVar{{
				Name: "labels",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "metadata.labels",
						APIVersion: "v1",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom.fieldRef.fieldPath`, Type: field.ErrorTypeNotSupported, BadValue: `metadata.labels`, Detail: `"metadata.name", "metadata.namespace", "metadata.uid", "spec.nodeName", "spec.serviceAccountName", "status.hostIP", "status.hostIPs", "status.podIP", "status.podIPs"`, SchemaType: field.ErrorTypeInvalid}},
		}, {

			Name: "metadata.annotations without subscript",
			Object: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "metadata.annotations",
						APIVersion: "v1",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom.fieldRef.fieldPath`, Type: field.ErrorTypeNotSupported, BadValue: `metadata.annotations`, Detail: `"metadata.name", "metadata.namespace", "metadata.uid", "spec.nodeName", "spec.serviceAccountName", "status.hostIP", "status.hostIPs", "status.podIP", "status.podIPs"`, SchemaType: field.ErrorTypeInvalid}},
		}, {

			Name: "metadata.annotations with invalid key",
			Object: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "metadata.annotations['invalid~key']",
						APIVersion: "v1",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom.fieldRef`, SchemaField: `[0].valueFrom.fieldRef.fieldPath`, Type: field.ErrorTypeInvalid, BadValue: `invalid~key`}},
		}, {

			Name: "metadata.labels with invalid key",
			Object: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "metadata.labels['Www.k8s.io/test']",
						APIVersion: "v1",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom.fieldRef`, SchemaField: `[0].valueFrom.fieldRef.fieldPath`, Type: field.ErrorTypeInvalid, BadValue: `Www.k8s.io/test`}},
		}, {

			Name: "unsupported fieldPath",
			Object: []core.EnvVar{{
				Name: "abc",
				ValueFrom: &core.EnvVarSource{
					FieldRef: &core.ObjectFieldSelector{
						FieldPath:  "status.phase",
						APIVersion: "v1",
					},
				},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: `[0].valueFrom.fieldRef.fieldPath`, Type: field.ErrorTypeNotSupported, BadValue: "status.phase", Detail: `supported values: "metadata.name", "metadata.namespace", "metadata.uid", "spec.nodeName", "spec.serviceAccountName", "status.hostIP", "status.hostIPs", "status.podIP", "status.podIPs"`, SchemaType: field.ErrorTypeInvalid}},
		},
	}
	cases := []apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{}
	for _, tc := range envCases {
		prefixedErrors := make([]apivalidationtesting.ExpectedFieldError, len(tc.ExpectedErrors))
		for i, e := range tc.ExpectedErrors {
			prefix := "template.spec.containers[0].env"
			prefixedErrors[i] = e
			prefixedErrors[i].Field = prefix + e.Field
			if len(e.SchemaField) > 0 {
				prefixedErrors[i].SchemaField = prefix + e.SchemaField
			}
		}
		cases = append(cases, apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{
			Name: tc.Name,
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"},
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{
						Containers: []core.Container{{Name: "foo", Image: "bar", Env: tc.Object}},
					},
				},
			},
			ExpectedErrors: prefixedErrors,
		})
	}

	apivalidationtesting.TestValidate[core.PodTemplate, validation.PodValidationOptions](t, coreScheme, coreDefs, func(pt *core.PodTemplate, pvo validation.PodValidationOptions) field.ErrorList {
		return validation.ValidateEnv(pt.Template.Spec.Containers[0].Env, field.NewPath("template.spec.containers[0].env"), pvo)
	}, cases...)

	// 	updateSuccessCase := []core.EnvVar{
	// 		{Name: "!\"#$%&'()", Value: "value"},
	// 		{Name: "* +,-./0123456789", Value: "value"},
	// 		{Name: ":;<>?@", Value: "value"},
	// 		{Name: "ABCDEFG", Value: "value"},
	// 		{Name: "abcdefghijklmn", Value: "value"},
	// 		{Name: "[\\]^_`{}|~", Value: "value"},
	// 	}

	// 	if errs := ValidateEnv(updateSuccessCase, field.NewPath("field"), PodValidationOptions{AllowRelaxedEnvironmentVariableValidation: true}); len(errs) != 0 {
	// 		t.Errorf("expected success, got: %v", errs)
	// 	}

	// 	updateErrorCase := []struct {
	// 		name          string
	// 		envs          []core.EnvVar
	// 		expectedError string
	// 	}{
	// 		{
	// 			name: "invalid name a",
	// 			envs: []core.EnvVar{
	// 				{Name: "!\"#$%&'()", Value: "value"},
	// 			},
	// 			expectedError: `field[0].name: Invalid value: ` + "\"!\\\"#$%&'()\": " + envVarNameErrMsg,
	// 		},
	// 		{
	// 			name: "invalid name b",
	// 			envs: []core.EnvVar{
	// 				{Name: "* +,-./0123456789", Value: "value"},
	// 			},
	// 			expectedError: `field[0].name: Invalid value: ` + "\"* +,-./0123456789\": " + envVarNameErrMsg,
	// 		},
	// 		{
	// 			name: "invalid name c",
	// 			envs: []core.EnvVar{
	// 				{Name: ":;<>?@", Value: "value"},
	// 			},
	// 			expectedError: `field[0].name: Invalid value: ` + "\":;<>?@\": " + envVarNameErrMsg,
	// 		},
	// 		{
	// 			name: "invalid name d",
	// 			envs: []core.EnvVar{
	// 				{Name: "[\\]^_{}|~", Value: "value"},
	// 			},
	// 			expectedError: `field[0].name: Invalid value: ` + "\"[\\\\]^_{}|~\": " + envVarNameErrMsg,
	// 		},
	// 	}

	// 	for _, tc := range updateErrorCase {
	// 		if errs := ValidateEnv(tc.envs, field.NewPath("field"), PodValidationOptions{}); len(errs) == 0 {
	// 			t.Errorf("expected failure for %s", tc.name)
	// 		} else {
	// 			for i := range errs {
	// 				str := errs[i].Error()
	// 				if str != "" && !strings.Contains(str, tc.expectedError) {
	// 					t.Errorf("%s: expected error detail either empty or %q, got %q", tc.name, tc.expectedError, str)
	// 				}
	// 			}
	// 		}
	// 	}
}

func TestValidateEnvFrom(t *testing.T) {
	envVarNameErrMsg := "a valid environment variable name must consist of"
	dnsSubdomainLabelErrMsg := "a lowercase RFC 1123 subdomain"

	fromCases := []apivalidationtesting.TestCase[[]core.EnvFromSource, validation.PodValidationOptions]{
		{
			Name: "success cases",
			Object: []core.EnvFromSource{{
				ConfigMapRef: &core.ConfigMapEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: "abc"},
				},
			}, {
				Prefix: "pre_",
				ConfigMapRef: &core.ConfigMapEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: "abc"},
				},
			}, {
				Prefix: "a.b",
				ConfigMapRef: &core.ConfigMapEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: "abc"},
				},
			}, {
				SecretRef: &core.SecretEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: "abc"},
				},
			}, {
				Prefix: "pre_",
				SecretRef: &core.SecretEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: "abc"},
				},
			}, {
				Prefix: "a.b",
				SecretRef: &core.SecretEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: "abc"},
				},
			},
			},
		},
		{
			Name: "invalid name a",
			Object: []core.EnvFromSource{
				{
					Prefix: "!\"#$%&'()",
					SecretRef: &core.SecretEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: "abc"},
					},
				},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    `[0].prefix`,
					Type:     field.ErrorTypeInvalid,
					Detail:   envVarNameErrMsg,
					BadValue: "!\"#$%&'()",

					SchemaDetail: `should match`,
				},
			},
		},
		{
			Name: "invalid name b",
			Object: []core.EnvFromSource{
				{
					Prefix: "* +,-./0123456789",
					SecretRef: &core.SecretEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: "abc"},
					},
				},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    `[0].prefix`,
					Type:     field.ErrorTypeInvalid,
					Detail:   envVarNameErrMsg,
					BadValue: "* +,-./0123456789",

					SchemaDetail: `should match`,
				},
			},
		},
		{
			Name: "invalid name c",
			Object: []core.EnvFromSource{
				{
					Prefix: ":;<>?@",
					SecretRef: &core.SecretEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: "abc"},
					},
				},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:        `[0].prefix`,
					Type:         field.ErrorTypeInvalid,
					Detail:       envVarNameErrMsg,
					BadValue:     ":;<>?@",
					SchemaDetail: `should match`,
				},
			},
		},
		{
			Name: "invalid name d",
			Object: []core.EnvFromSource{
				{
					Prefix: "[\\]^_{}|~",
					SecretRef: &core.SecretEnvSource{
						LocalObjectReference: core.LocalObjectReference{Name: "abc"},
					},
				},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:        `[0].prefix`,
					Type:         field.ErrorTypeInvalid,
					Detail:       envVarNameErrMsg,
					BadValue:     "[\\]^_{}|~",
					SchemaDetail: `should match`,
				},
			},
		},
		{

			Name: "zero-length name",
			Object: []core.EnvFromSource{{
				ConfigMapRef: &core.ConfigMapEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: ""}},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field: `[0].configMapRef.name`,
					Type:  field.ErrorTypeRequired,
				},
			},
		}, {

			Name: "invalid name",
			Object: []core.EnvFromSource{{
				ConfigMapRef: &core.ConfigMapEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: "$"}},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    `[0].configMapRef.name`,
					Type:     field.ErrorTypeInvalid,
					BadValue: "$",
				},
			},
		}, {

			Name: "zero-length name",
			Object: []core.EnvFromSource{{
				SecretRef: &core.SecretEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: ""}},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field: `[0].secretRef.name`,
					Type:  field.ErrorTypeRequired,
				},
			},
		}, {

			Name: "invalid name",
			Object: []core.EnvFromSource{{
				SecretRef: &core.SecretEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: "&"}},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    `[0].secretRef.name`,
					Type:     field.ErrorTypeInvalid,
					BadValue: "&",
				},
			},
		}, {

			Name: "no refs",
			Object: []core.EnvFromSource{
				{},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Type:        field.ErrorTypeInvalid,
					BadValue:    "",
					Detail:      "must specify one of: `configMapRef` or `secretRef`",
					SchemaField: `[0]`,
				},
			},
		}, {

			Name: "multiple refs",
			Object: []core.EnvFromSource{{
				SecretRef: &core.SecretEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: "abc"}},
				ConfigMapRef: &core.ConfigMapEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: "abc"}},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:       ``,
					Type:        field.ErrorTypeInvalid,
					BadValue:    "",
					Detail:      "may not have more than one field specified at a time",
					SchemaField: `[0]`,
				},
			},
		}, {

			Name: "invalid secret ref name",
			Object: []core.EnvFromSource{{
				SecretRef: &core.SecretEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: "$%^&*#"}},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    `[0].secretRef.name`,
					Type:     field.ErrorTypeInvalid,
					BadValue: "$%^&*#",
					Detail:   dnsSubdomainLabelErrMsg,
				},
			},
		}, {

			Name: "invalid config ref name",
			Object: []core.EnvFromSource{{
				ConfigMapRef: &core.ConfigMapEnvSource{
					LocalObjectReference: core.LocalObjectReference{Name: "$%^&*#"}},
			}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{
					Field:    `[0].configMapRef.name`,
					Type:     field.ErrorTypeInvalid,
					Detail:   dnsSubdomainLabelErrMsg,
					BadValue: "$%^&*#",
				},
			},
		},
	}

	cases := []apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{}
	for _, tc := range fromCases {
		prefixedErrors := make([]apivalidationtesting.ExpectedFieldError, len(tc.ExpectedErrors))
		for i, e := range tc.ExpectedErrors {
			prefix := "template.spec.containers[0].envFrom"
			prefixedErrors[i] = e
			prefixedErrors[i].Field = prefix + e.Field
			if len(e.SchemaField) > 0 {
				prefixedErrors[i].SchemaField = prefix + e.SchemaField
			}
		}
		cases = append(cases, apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{
			Name: tc.Name,
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"},
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{
						Containers: []core.Container{{Name: "foo", Image: "bar", EnvFrom: tc.Object}},
					},
				},
			},
			ExpectedErrors: prefixedErrors,
		})
	}

	apivalidationtesting.TestValidate[core.PodTemplate, validation.PodValidationOptions](t, coreScheme, coreDefs, func(pt *core.PodTemplate, pvo validation.PodValidationOptions) field.ErrorList {
		return validation.ValidateEnvFrom(pt.Template.Spec.Containers[0].EnvFrom, field.NewPath("template.spec.containers[0].envFrom"), pvo)
	}, cases...)

	// 	updateSuccessCase := []core.EnvFromSource{{
	// 		ConfigMapRef: &core.ConfigMapEnvSource{
	// 			LocalObjectReference: core.LocalObjectReference{Name: "abc"},
	// 		},
	// 	}, {
	// 		Prefix: "* +,-./0123456789",
	// 		ConfigMapRef: &core.ConfigMapEnvSource{
	// 			LocalObjectReference: core.LocalObjectReference{Name: "abc"},
	// 		},
	// 	}, {
	// 		Prefix: ":;<>?@",
	// 		ConfigMapRef: &core.ConfigMapEnvSource{
	// 			LocalObjectReference: core.LocalObjectReference{Name: "abc"},
	// 		},
	// 	}, {
	// 		SecretRef: &core.SecretEnvSource{
	// 			LocalObjectReference: core.LocalObjectReference{Name: "abc"},
	// 		},
	// 	}, {
	// 		Prefix: "abcdefghijklmn",
	// 		SecretRef: &core.SecretEnvSource{
	// 			LocalObjectReference: core.LocalObjectReference{Name: "abc"},
	// 		},
	// 	}, {
	// 		Prefix: "[\\]^_`{}|~",
	// 		SecretRef: &core.SecretEnvSource{
	// 			LocalObjectReference: core.LocalObjectReference{Name: "abc"},
	// 		},
	// 	}}

	// 	if errs := ValidateEnvFrom(updateSuccessCase, field.NewPath("field"), PodValidationOptions{AllowRelaxedEnvironmentVariableValidation: true}); len(errs) != 0 {
	// 		t.Errorf("expected success, got: %v", errs)
	// 	}

}

func TestValidateVolumeMounts(t *testing.T) {
	type options struct {
		devices []core.VolumeDevice
	}

	volumes := []core.Volume{
		{Name: "abc", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim1"}}},
		{Name: "abc-123", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim2"}}},
		{Name: "xyz", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim3"}}},
		{Name: "uvw", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim3"}}},
		{Name: "123", VolumeSource: core.VolumeSource{HostPath: &core.HostPathVolumeSource{Path: "/foo/baz", Type: newHostPathType(string(core.HostPathUnset))}}},
		{Name: "ephemeral", VolumeSource: core.VolumeSource{Ephemeral: &core.EphemeralVolumeSource{VolumeClaimTemplate: &core.PersistentVolumeClaimTemplate{
			Spec: core.PersistentVolumeClaimSpec{
				AccessModes: []core.PersistentVolumeAccessMode{
					core.ReadWriteOnce,
				},
				Resources: core.VolumeResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
					},
				},
			},
		}}}},
	}

	mountCases := []apivalidationtesting.TestCase[[]core.VolumeMount, options]{
		{
			Name: "success cases",
			Object: []core.VolumeMount{
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
				{Name: "ephemeral", MountPath: "/foobar"},
				{Name: "123", MountPath: "/rro-nil", ReadOnly: true, RecursiveReadOnly: nil},
				{Name: "123", MountPath: "/rro-disabled", ReadOnly: true, RecursiveReadOnly: ptr.To(core.RecursiveReadOnlyDisabled)},
				{Name: "123", MountPath: "/rro-disabled-2", ReadOnly: false, RecursiveReadOnly: ptr.To(core.RecursiveReadOnlyDisabled)},
				{Name: "123", MountPath: "/rro-ifpossible", ReadOnly: true, RecursiveReadOnly: ptr.To(core.RecursiveReadOnlyIfPossible)},
				{Name: "123", MountPath: "/rro-enabled", ReadOnly: true, RecursiveReadOnly: ptr.To(core.RecursiveReadOnlyEnabled)},
				{Name: "123", MountPath: "/rro-enabled-2", ReadOnly: true, RecursiveReadOnly: ptr.To(core.RecursiveReadOnlyEnabled), MountPropagation: ptr.To(core.MountPropagationNone)},
			},
			Options: options{
				devices: []core.VolumeDevice{
					{Name: "xyz", DevicePath: "/foofoo"},
					{Name: "uvw", DevicePath: "/foofoo/share/test"},
				},
			},
		},
		{Name: "empty name", Object: []core.VolumeMount{{Name: "", MountPath: "/foo"}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: "[0].name", Type: field.ErrorTypeRequired, SchemaType: field.ErrorTypeInvalid}, {Field: "[0].name", Type: field.ErrorTypeNotFound, BadValue: "", SchemaField: "containers", SchemaType: field.ErrorTypeInvalid}}},
		{Name: "name not found", Object: []core.VolumeMount{{Name: "", MountPath: "/foo"}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: "[0].name", Type: field.ErrorTypeRequired, SchemaType: field.ErrorTypeInvalid}, {Field: "[0].name", Type: field.ErrorTypeNotFound, BadValue: "", SchemaField: "containers", SchemaType: field.ErrorTypeInvalid}}},
		{Name: "empty mountpath", Object: []core.VolumeMount{{Name: "abc", MountPath: ""}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: "[0].mountPath", Type: field.ErrorTypeRequired, SchemaType: field.ErrorTypeInvalid}}},
		{Name: "mountpath collision", Object: []core.VolumeMount{{Name: "abc", MountPath: "/path/a"}, {Name: "abc-123", MountPath: "/path/a"}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: "[1].mountPath", Type: field.ErrorTypeInvalid, BadValue: "/path/a", SchemaField: `[1]`, SchemaType: field.ErrorTypeDuplicate}}},
		{Name: "absolute subpath", Object: []core.VolumeMount{{Name: "abc", MountPath: "/bar", SubPath: "/baz"}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: ".subPath", SchemaField: `[0].subPath`, Type: field.ErrorTypeInvalid, Detail: "must be a relative path", BadValue: "/baz"}}},
		{Name: "subpath in ..", Object: []core.VolumeMount{{Name: "abc", MountPath: "/bar", SubPath: "../baz"}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: ".subPath", SchemaField: `[0].subPath`, Type: field.ErrorTypeInvalid, Detail: "must not contain '..'", BadValue: "../baz"}}},
		{Name: "subpath contains ..", Object: []core.VolumeMount{{Name: "abc", MountPath: "/bar", SubPath: "baz/../bat"}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: ".subPath", SchemaField: "[0].subPath", Type: field.ErrorTypeInvalid, Detail: "must not contain '..'", BadValue: "baz/../bat"}}},
		{Name: "subpath ends in ..", Object: []core.VolumeMount{{Name: "abc", MountPath: "/bar", SubPath: "./.."}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: ".subPath", SchemaField: "[0].subPath", Type: field.ErrorTypeInvalid, Detail: "must not contain '..'", BadValue: "./.."}}},
		{Name: "disabled MountPropagation feature gate", Object: []core.VolumeMount{{Name: "abc", MountPath: "/bar", MountPropagation: ptr.To(core.MountPropagationBidirectional)}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: ".mountPropagation", Type: field.ErrorTypeForbidden, Detail: `available only to privileged`, SchemaField: `containers[0].volumeMounts`}}},
		{Name: "name exists in volumeDevice", Object: []core.VolumeMount{{Name: "xyz", MountPath: "/bar"}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: "[0].name", Type: field.ErrorTypeInvalid, BadValue: "xyz", SchemaField: `containers[0].volumeMounts`}}},
		{Name: "mountpath exists in volumeDevice", Object: []core.VolumeMount{{Name: "uvw", MountPath: "/mnt/exists"}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: "[0].mountPath", Type: field.ErrorTypeInvalid, BadValue: "/mnt/exists", SchemaField: `containers[0].volumeMounts`}}},
		{Name: "both exist in volumeDevice", Object: []core.VolumeMount{{Name: "xyz", MountPath: "/mnt/exists"}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: "[0].name", Type: field.ErrorTypeInvalid, BadValue: "xyz", SchemaField: `containers[0].volumeMounts`}, {Field: "[0].mountPath", Type: field.ErrorTypeInvalid, BadValue: "/mnt/exists", SchemaField: `containers[0].volumeMounts`}}},
		{Name: "rro but not ro", Object: []core.VolumeMount{{Name: "123", MountPath: "/rro-bad1", ReadOnly: false, RecursiveReadOnly: ptr.To(core.RecursiveReadOnlyEnabled)}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: ".recursiveReadOnly", SchemaField: "[0].recursiveReadOnly", Type: field.ErrorTypeForbidden}}},
		{Name: "rro with incompatible propagation", Object: []core.VolumeMount{{Name: "123", MountPath: "/rro-bad2", ReadOnly: true, RecursiveReadOnly: ptr.To(core.RecursiveReadOnlyEnabled), MountPropagation: ptr.To(core.MountPropagationHostToContainer)}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: ".recursiveReadOnly", SchemaField: `[0].recursiveReadOnly`, Type: field.ErrorTypeForbidden}}},
		{Name: "rro-if-possible but not ro", Object: []core.VolumeMount{{Name: "123", MountPath: "/rro-bad1", ReadOnly: false, RecursiveReadOnly: ptr.To(core.RecursiveReadOnlyIfPossible)}}, ExpectedErrors: apivalidationtesting.ExpectedErrorList{{Field: ".recursiveReadOnly", SchemaField: "[0].recursiveReadOnly", Type: field.ErrorTypeForbidden, Detail: `may only be specified when readOnly is true`}}},
	}

	badVolumeDevice := []core.VolumeDevice{
		{Name: "xyz", DevicePath: "/mnt/exists"},
	}

	cases := []apivalidationtesting.TestCase[*core.PodTemplate, options]{}
	for _, tc := range mountCases {
		if tc.Options.devices == nil {
			tc.Options.devices = badVolumeDevice
		}

		prefixedErrors := make([]apivalidationtesting.ExpectedFieldError, len(tc.ExpectedErrors))
		for i, e := range tc.ExpectedErrors {
			prefix := "template.spec.containers[0].volumeMounts"
			prefixedErrors[i] = e
			prefixedErrors[i].Field = prefix + e.Field
			if len(e.SchemaField) > 0 {
				if strings.HasPrefix(e.SchemaField, "containers") {
					prefixedErrors[i].SchemaField = "template.spec." + e.SchemaField
				} else {
					prefixedErrors[i].SchemaField = prefix + e.SchemaField
				}
			}
		}
		cases = append(cases, apivalidationtesting.TestCase[*core.PodTemplate, options]{
			Name: tc.Name,
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"},
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{
						Volumes:    volumes,
						Containers: []core.Container{{Name: "foo", Image: "bar", VolumeMounts: tc.Object, VolumeDevices: tc.Options.devices, SecurityContext: nil}},
					},
				},
			},
			Options:        tc.Options,
			ExpectedErrors: prefixedErrors,
		})
	}

	vols, v1err := validation.ValidateVolumes(volumes, nil, field.NewPath("field"), validation.PodValidationOptions{})
	if len(v1err) > 0 {
		t.Errorf("Invalid test volume - expected success %v", v1err)
		return
	}

	apivalidationtesting.TestValidate[core.PodTemplate, options](t, coreScheme, coreDefs, func(pt *core.PodTemplate, o options) field.ErrorList {
		return validation.ValidateVolumeMounts(pt.Template.Spec.Containers[0].VolumeMounts, validation.GetVolumeDeviceMap(pt.Template.Spec.Containers[0].VolumeDevices), vols, &pt.Template.Spec.Containers[0], field.NewPath("template.spec.containers[0].volumeMounts"))
	}, cases...)
}

func TestValidateSubpathMutuallyExclusive(t *testing.T) {
	volumes := []core.Volume{
		{Name: "abc", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim1"}}},
		{Name: "abc-123", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim2"}}},
		{Name: "xyz", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim3"}}},
		{Name: "uvw", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim4"}}},
		{Name: "123", VolumeSource: core.VolumeSource{HostPath: &core.HostPathVolumeSource{Path: "/foo/baz", Type: newHostPathType(string(core.HostPathUnset))}}},
	}
	vols, v1err := validation.ValidateVolumes(volumes, nil, field.NewPath("field"), validation.PodValidationOptions{})
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
		mounts       []core.VolumeMount
		expectErrors apivalidationtesting.ExpectedErrorList
	}{
		"subpath and subpathexpr not specified": {
			[]core.VolumeMount{{
				Name:      "abc-123",
				MountPath: "/bab",
			}},
			nil,
		},
		"subpath expr specified": {
			[]core.VolumeMount{{
				Name:        "abc-123",
				MountPath:   "/bab",
				SubPathExpr: "$(POD_NAME)",
			}},
			nil,
		},
		"subpath specified": {
			[]core.VolumeMount{{
				Name:      "abc-123",
				MountPath: "/bab",
				SubPath:   "baz",
			}},
			nil,
		},
		"subpath and subpathexpr specified": {
			[]core.VolumeMount{{
				Name:        "abc-123",
				MountPath:   "/bab",
				SubPath:     "baz",
				SubPathExpr: "$(POD_NAME)",
			}},
			apivalidationtesting.ExpectedErrorList{{Field: "template.spec.containers[0].volumeMounts[0].subPathExpr", BadValue: `$(POD_NAME)`, Type: field.ErrorTypeInvalid, Detail: "subPathExpr and subPath are mutually exclusive"}},
		},
	}

	cases2 := []apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{}
	for name, test := range cases {
		c := container
		c.Name = "foo"
		c.Image = "bar"
		c.VolumeMounts = test.mounts
		c.VolumeDevices = goodVolumeDevices

		cases2 = append(cases2, apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{
			Name: name,
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"},
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{
						Volumes:    volumes,
						Containers: []core.Container{c},
					},
				},
			},
			ExpectedErrors: test.expectErrors,
		})
	}

	apivalidationtesting.TestValidate[core.PodTemplate, validation.PodValidationOptions](t, coreScheme, coreDefs, func(pt *core.PodTemplate, pvo validation.PodValidationOptions) field.ErrorList {
		return validation.ValidateVolumeMounts(pt.Template.Spec.Containers[0].VolumeMounts, validation.GetVolumeDeviceMap(pt.Template.Spec.Containers[0].VolumeDevices), vols, &pt.Template.Spec.Containers[0], field.NewPath("template.spec.containers[0].volumeMounts"))
	}, cases2...)
}

func TestValidateDisabledSubpathExpr(t *testing.T) {

	volumes := []core.Volume{
		{Name: "abc", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim1"}}},
		{Name: "abc-123", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim2"}}},
		{Name: "xyz", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim3"}}},
		{Name: "uvw", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim4"}}},
		{Name: "123", VolumeSource: core.VolumeSource{HostPath: &core.HostPathVolumeSource{Path: "/foo/baz", Type: newHostPathType(string(core.HostPathUnset))}}},
	}
	vols, v1err := validation.ValidateVolumes(volumes, nil, field.NewPath("field"), validation.PodValidationOptions{})
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
		mounts []core.VolumeMount
	}{
		"subpath expr not specified": {
			[]core.VolumeMount{{
				Name:      "abc-123",
				MountPath: "/bab",
			}},
		},
		"subpath expr specified": {
			[]core.VolumeMount{{
				Name:        "abc-123",
				MountPath:   "/bab",
				SubPathExpr: "$(POD_NAME)",
			}},
		},
	}

	cases2 := []apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{}
	for name, test := range cases {
		c := container
		c.Name = "foo"
		c.Image = "bar"
		c.VolumeMounts = test.mounts
		c.VolumeDevices = goodVolumeDevices

		cases2 = append(cases2, apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{
			Name: name,
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"},
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{
						Volumes:    volumes,
						Containers: []core.Container{c},
					},
				},
			},
			ExpectedErrors: nil,
		})
	}

	apivalidationtesting.TestValidate[core.PodTemplate, validation.PodValidationOptions](t, coreScheme, coreDefs, func(pt *core.PodTemplate, pvo validation.PodValidationOptions) field.ErrorList {
		return validation.ValidateVolumeMounts(pt.Template.Spec.Containers[0].VolumeMounts, validation.GetVolumeDeviceMap(pt.Template.Spec.Containers[0].VolumeDevices), vols, &pt.Template.Spec.Containers[0], field.NewPath("template.spec.containers[0].volumeMounts"))
	}, cases2...)
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
	propagationNone := core.MountPropagationNone
	propagationInvalid := core.MountPropagationMode("invalid")

	tests := []struct {
		mount       core.VolumeMount
		container   *core.Container
		expectError bool
	}{{
		// implicitly non-privileged container + no propagation
		core.VolumeMount{Name: "foo", MountPath: "/foo"},
		defaultContainer,
		false,
	}, {
		// implicitly non-privileged container + HostToContainer
		core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationHostToContainer},
		defaultContainer,
		false,
	}, {
		// non-privileged container + None
		core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationNone},
		defaultContainer,
		false,
	}, {
		// error: implicitly non-privileged container + Bidirectional
		core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationBidirectional},
		defaultContainer,
		true,
	}, {
		// explicitly non-privileged container + no propagation
		core.VolumeMount{Name: "foo", MountPath: "/foo"},
		nonPrivilegedContainer,
		false,
	}, {
		// explicitly non-privileged container + HostToContainer
		core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationHostToContainer},
		nonPrivilegedContainer,
		false,
	}, {
		// explicitly non-privileged container + HostToContainer
		core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationBidirectional},
		nonPrivilegedContainer,
		true,
	}, {
		// privileged container + no propagation
		core.VolumeMount{Name: "foo", MountPath: "/foo"},
		privilegedContainer,
		false,
	}, {
		// privileged container + HostToContainer
		core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationHostToContainer},
		privilegedContainer,
		false,
	}, {
		// privileged container + Bidirectional
		core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationBidirectional},
		privilegedContainer,
		false,
	}, {
		// error: privileged container + invalid mount propagation
		core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationInvalid},
		privilegedContainer,
		true,
	}, {
		// no container + Bidirectional
		core.VolumeMount{Name: "foo", MountPath: "/foo", MountPropagation: &propagationBidirectional},
		nil,
		false,
	},
	}

	volumes := []core.Volume{
		{Name: "foo", VolumeSource: core.VolumeSource{HostPath: &core.HostPathVolumeSource{Path: "/foo/baz", Type: newHostPathType(string(core.HostPathUnset))}}},
	}

	vols2, v2err := validation.ValidateVolumes(volumes, nil, field.NewPath("field"), validation.PodValidationOptions{})
	if len(v2err) > 0 {
		t.Errorf("Invalid test volume - expected success %v", v2err)
		return
	}

	cases := []apivalidationtesting.TestCase[*core.PodTemplate, *core.Volume]{}
	for i, test := range tests {
		prefixedErrors := make([]apivalidationtesting.ExpectedFieldError, 0)
		if test.expectError {
			err := apivalidationtesting.ExpectedFieldError{
				SchemaField: "template.spec.containers[0].volumeMounts[0].mountPropagation",
				Field:       "template.spec.containers[0].volumeMounts.mountPropagation",
				Type:        field.ErrorTypeForbidden,
			}

			if test.mount.MountPropagation == &propagationBidirectional {
				err.SchemaField = "template.spec.containers[0].volumeMounts"
			}

			if test.mount.MountPropagation == &propagationInvalid {
				err.Type = field.ErrorTypeNotSupported
				err.BadValue = propagationInvalid
			}

			prefixedErrors = append(prefixedErrors, err)
		}

		var c core.Container
		if test.container != nil {
			c = *test.container
		} else {
			c = *privilegedContainer
		}
		c.Name = "foo"
		c.Image = "bar"
		c.VolumeMounts = []core.VolumeMount{test.mount}

		cases = append(cases, apivalidationtesting.TestCase[*core.PodTemplate, *core.Volume]{
			Name: fmt.Sprintf("test %d", i),
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"},
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{
						Volumes:    volumes,
						Containers: []core.Container{c},
					},
				},
			},
			ExpectedErrors: prefixedErrors,
		})
	}

	apivalidationtesting.TestValidate[core.PodTemplate, *core.Volume](t, coreScheme, coreDefs, func(pt *core.PodTemplate, v *core.Volume) field.ErrorList {
		return validation.ValidateVolumeMounts(pt.Template.Spec.Containers[0].VolumeMounts, nil, vols2, &pt.Template.Spec.Containers[0], field.NewPath("template.spec.containers[0].volumeMounts"))
	}, cases...)
}

func TestAlphaValidateVolumeDevices(t *testing.T) {
	volumes := []core.Volume{
		{Name: "abc", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim1"}}},
		{Name: "abc-123", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim2"}}},
		{Name: "xyz", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim3"}}},
		{Name: "ghi", VolumeSource: core.VolumeSource{PersistentVolumeClaim: &core.PersistentVolumeClaimVolumeSource{ClaimName: "testclaim4"}}},
		{Name: "def", VolumeSource: core.VolumeSource{HostPath: &core.HostPathVolumeSource{Path: "/foo/baz", Type: newHostPathType(string(core.HostPathUnset))}}},
		{Name: "ephemeral", VolumeSource: core.VolumeSource{Ephemeral: &core.EphemeralVolumeSource{VolumeClaimTemplate: &core.PersistentVolumeClaimTemplate{
			Spec: core.PersistentVolumeClaimSpec{
				AccessModes: []core.PersistentVolumeAccessMode{
					core.ReadWriteOnce,
				},
				Resources: core.VolumeResourceRequirements{
					Requests: core.ResourceList{
						core.ResourceName(core.ResourceStorage): resource.MustParse("10G"),
					},
				},
			},
		}}}},
	}

	vols, v1err := validation.ValidateVolumes(volumes, nil, field.NewPath("field"), validation.PodValidationOptions{})
	if len(v1err) > 0 {
		t.Errorf("Invalid test volumes - expected success %v", v1err)
		return
	}

	cases := []apivalidationtesting.TestCase[[]core.VolumeDevice, validation.PodValidationOptions]{
		{
			Name: "success cases",
			Object: []core.VolumeDevice{
				{Name: "abc", DevicePath: "/foo"},
				{Name: "abc-123", DevicePath: "/usr/share/test"},
				{Name: "ephemeral", DevicePath: "/disk"},
			},
		},
		{
			Name:   "empty name",
			Object: []core.VolumeDevice{{Name: "", DevicePath: "/bar"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "[0].name", Type: field.ErrorTypeRequired, SchemaType: field.ErrorTypeInvalid},
				{Field: "[0].name", Type: field.ErrorTypeNotFound, BadValue: "", SchemaField: "containers", SchemaType: field.ErrorTypeInvalid},
			},
		}, {
			Name:   "duplicate name",
			Object: []core.VolumeDevice{{Name: "xyz", DevicePath: "/bar"}, {Name: "xyz", DevicePath: "/foo/bar"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "[1].name", Type: field.ErrorTypeInvalid, BadValue: "xyz", SchemaField: "containers", SchemaType: field.ErrorTypeDuplicate, Detail: `must be unique`, SchemaSkipReason: `Blocked by lack of sets and variables`},
			},
		}, {
			Name:   "name not found",
			Object: []core.VolumeDevice{{Name: "not-found", DevicePath: "/usr/share/test"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "[0].name", Type: field.ErrorTypeNotFound, BadValue: "not-found", SchemaField: "containers", SchemaType: field.ErrorTypeInvalid},
			},
		}, {
			Name:   "name found but invalid source",
			Object: []core.VolumeDevice{{Name: "def", DevicePath: "/usr/share/test"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "[0].name", Type: field.ErrorTypeInvalid, BadValue: "def", SchemaField: "containers", SchemaType: field.ErrorTypeInvalid},
			},
		}, {
			Name:   "empty devicepath",
			Object: []core.VolumeDevice{{Name: "xyz", DevicePath: ""}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "[0].devicePath", Type: field.ErrorTypeRequired, SchemaType: field.ErrorTypeInvalid},
			},
		},
		// Test in native suite did not test what description says, and relative
		// paths are not blocked by volume device validation
		// {
		// 	Name:   "relative devicepath",
		// 	Object: []core.VolumeDevice{{Name: "abc-123", DevicePath: "baz"}},
		// 	ExpectedErrors: apivalidationtesting.ExpectedErrorList{
		// 		{Field: "[0].devicePath", Type: field.ErrorTypeInvalid, BadValue: "baz"},
		// 	},
		// },
		{
			Name:   "duplicate devicepath",
			Object: []core.VolumeDevice{{Name: "xyz", DevicePath: "/bar"}, {Name: "ghi", DevicePath: "/bar"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "[1].devicePath", Type: field.ErrorTypeInvalid, BadValue: "/bar", SchemaField: "[1]", SchemaType: field.ErrorTypeDuplicate},
			},
		}, {
			Name:   "no backsteps",
			Object: []core.VolumeDevice{{Name: "xyz", DevicePath: "/baz/../"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "[0].devicePath", Type: field.ErrorTypeInvalid, BadValue: "/baz/../"},
			},
		}, {
			Name:   "name exists in volumemounts",
			Object: []core.VolumeDevice{{Name: "abc", DevicePath: "/baz"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "[0].name", Type: field.ErrorTypeInvalid, BadValue: "abc", SchemaField: "containers[0].volumeMounts", SchemaType: field.ErrorTypeInvalid},
			},
		}, {
			Name:   "path exists in volumemounts",
			Object: []core.VolumeDevice{{Name: "xyz", DevicePath: "/this/path/exists"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "[0].devicePath", Type: field.ErrorTypeInvalid, BadValue: "/this/path/exists", SchemaField: "containers[0].volumeMounts", SchemaType: field.ErrorTypeInvalid},
			},
		}, {
			Name:   "both exist in volumemounts",
			Object: []core.VolumeDevice{{Name: "abc", DevicePath: "/this/path/exists"}},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "[0].name", Type: field.ErrorTypeInvalid, BadValue: "abc", SchemaField: "containers[0].volumeMounts", SchemaType: field.ErrorTypeInvalid},
				{Field: "[0].devicePath", Type: field.ErrorTypeInvalid, BadValue: "/this/path/exists", SchemaField: "containers[0].volumeMounts", SchemaType: field.ErrorTypeInvalid},
			},
		},
	}
	goodVolumeMounts := []core.VolumeMount{
		{Name: "xyz", MountPath: "/foofoo"},
		{Name: "ghi", MountPath: "/foo/usr/share/test"},
	}

	badVolumeMounts := []core.VolumeMount{
		{Name: "abc", MountPath: "/foo"},
		{Name: "abc-123", MountPath: "/this/path/exists"},
	}

	cases2 := []apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{}
	for _, tc := range cases {
		prefixedErrors := make([]apivalidationtesting.ExpectedFieldError, len(tc.ExpectedErrors))
		for i, e := range tc.ExpectedErrors {
			prefix := "template.spec.containers[0].volumeDevices"
			prefixedErrors[i] = e
			prefixedErrors[i].Field = prefix + e.Field
			if len(e.SchemaField) > 0 {
				if strings.HasPrefix(e.SchemaField, "containers") {
					prefixedErrors[i].SchemaField = "template.spec." + e.SchemaField
				} else {
					prefixedErrors[i].SchemaField = prefix + e.SchemaField
				}
			}
		}
		vms := goodVolumeMounts
		if len(tc.ExpectedErrors) > 0 {
			vms = badVolumeMounts
		}

		cases2 = append(cases2, apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{
			Name: tc.Name,
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"},
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{
						Volumes:    volumes,
						Containers: []core.Container{{Name: "foo", Image: "bar", VolumeDevices: tc.Object, VolumeMounts: vms}},
					},
				},
			},
			ExpectedErrors: prefixedErrors,
		})
	}

	apivalidationtesting.TestValidate[core.PodTemplate, validation.PodValidationOptions](t, coreScheme, coreDefs, func(pt *core.PodTemplate, pvo validation.PodValidationOptions) field.ErrorList {
		return validation.ValidateVolumeDevices(pt.Template.Spec.Containers[0].VolumeDevices, validation.GetVolumeMountMap(pt.Template.Spec.Containers[0].VolumeMounts), vols, field.NewPath("template.spec.containers[0].volumeDevices"))
	}, cases2...)
}

func TestValidatePodTemplateSpecSeccomp(t *testing.T) {
}

func TestValidateResourceRequirements(t *testing.T) {

	cases := []apivalidationtesting.TestCase[*core.ResourceRequirements, validation.PodValidationOptions]{
		{
			Name: "limits and requests of hugepage resource are equal",
			Object: &core.ResourceRequirements{
				Limits: core.ResourceList{
					core.ResourceCPU: resource.MustParse("10"),
					core.ResourceName(core.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("2Mi"),
				},
				Requests: core.ResourceList{
					core.ResourceCPU: resource.MustParse("10"),
					core.ResourceName(core.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("2Mi"),
				},
			},
		}, {
			Name: "limits and requests of memory resource are equal",
			Object: &core.ResourceRequirements{
				Limits: core.ResourceList{
					core.ResourceMemory: resource.MustParse("2Mi"),
				},
				Requests: core.ResourceList{
					core.ResourceMemory: resource.MustParse("2Mi"),
				},
			},
		}, {
			Name: "limits and requests of cpu resource are equal",
			Object: &core.ResourceRequirements{
				Limits: core.ResourceList{
					core.ResourceCPU: resource.MustParse("10"),
				},
				Requests: core.ResourceList{
					core.ResourceCPU: resource.MustParse("10"),
				},
			},
		},
		{
			Name: "hugepage resource without cpu or memory",
			Object: &core.ResourceRequirements{
				Limits: core.ResourceList{
					core.ResourceName(core.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("2Mi"),
				},
				Requests: core.ResourceList{
					core.ResourceName(core.ResourceHugePagesPrefix + "2Mi"): resource.MustParse("2Mi"),
				},
			},
			ExpectedErrors: apivalidationtesting.ExpectedErrorList{
				{Field: "resources", Type: field.ErrorTypeForbidden, Detail: "HugePages require cpu or memory"},
			},
		},
	}

	cases2 := []apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{}
	for _, tc := range cases {
		prefixedErrors := make([]apivalidationtesting.ExpectedFieldError, len(tc.ExpectedErrors))
		for i, e := range tc.ExpectedErrors {
			prefix := "template.spec.containers[0]."
			prefixedErrors[i] = e
			prefixedErrors[i].Field = prefix + e.Field
			if len(e.SchemaField) > 0 {
				prefixedErrors[i].SchemaField = prefix + e.SchemaField
			}
		}

		cases2 = append(cases2, apivalidationtesting.TestCase[*core.PodTemplate, validation.PodValidationOptions]{
			Name: tc.Name,
			Object: &core.PodTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "ns"},
				Template: core.PodTemplateSpec{
					Spec: core.PodSpec{
						Containers: []core.Container{{Name: "foo", Image: "bar", Resources: *tc.Object}},
					},
				},
			},
			ExpectedErrors: prefixedErrors,
		})
	}

	apivalidationtesting.TestValidate[core.PodTemplate, validation.PodValidationOptions](t, coreScheme, coreDefs, func(pt *core.PodTemplate, pvo validation.PodValidationOptions) field.ErrorList {
		return validation.ValidateResourceRequirements(&pt.Template.Spec.Containers[0].Resources, nil, field.NewPath("template.spec.containers[0].resources"), pvo)
	}, cases2...)
}
