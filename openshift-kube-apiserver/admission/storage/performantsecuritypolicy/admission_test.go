package performantsecuritypolicy

import (
	"context"
	"testing"

	openshiftfeatures "github.com/openshift/api/features"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kapi "k8s.io/kubernetes/pkg/apis/core"
	_ "k8s.io/kubernetes/pkg/features"
)

func TestAdmit(t *testing.T) {
	type testCase struct {
		name               string
		pod                *kapi.Pod
		ns                 *corev1.Namespace
		expectedPod        *kapi.Pod
		expectError        bool
		featureGateEnabled bool
	}

	onRootMismatchPolicy := kapi.FSGroupChangeOnRootMismatch
	alwaysFSGroupChangePolicy := kapi.FSGroupChangeAlways
	selinuxRecursive := kapi.SELinuxChangePolicyRecursive
	selinuxMountOption := kapi.SELinuxChangePolicyMountOption

	testCases := []testCase{
		{
			name:               "when feature gate is disabled, no changes are made",
			featureGateEnabled: false,
			pod:                getPod(nil, nil),
			expectedPod:        getPod(nil, nil),

			ns: getNamespace(map[string]string{
				fsGroupChangePolicyLabel: "OnRootMismatch",
			}),
		},
		{
			name:               "when feature gate is enabled, FSGroupChangePolicy is set to OnRootMismatch",
			featureGateEnabled: true,
			pod:                getPod(nil, nil),
			expectedPod:        getPod(&onRootMismatchPolicy, nil),
			expectError:        false,
			ns: getNamespace(map[string]string{
				fsGroupChangePolicyLabel: "OnRootMismatch",
			}),
		},
		{
			name:               "when feature is enabled, but namespace label for fsgroupchangepolicy has invalid value, no changes are made",
			pod:                getPod(nil, nil),
			featureGateEnabled: true,
			expectedPod:        getPod(nil, nil),
			expectError:        false,
			ns: getNamespace(map[string]string{
				fsGroupChangePolicyLabel: "InvalidValue",
			}),
		},
		{
			name:               "when feature is enabled, but pod already specifies different fsgroupchangepolicy",
			pod:                getPod(&alwaysFSGroupChangePolicy, nil),
			featureGateEnabled: true,
			expectedPod:        getPod(&alwaysFSGroupChangePolicy, nil),
			expectError:        false,
			ns: getNamespace(map[string]string{
				fsGroupChangePolicyLabel: "OnRootMismatch",
			}),
		},
		{
			name:               "when feature is enabled and selinuxchangepolicy is set to Recursive",
			featureGateEnabled: true,
			pod:                getPod(nil, nil),
			expectedPod:        getPod(nil, &selinuxRecursive),
			expectError:        false,
			ns: getNamespace(map[string]string{
				selinuxChangePolicyLabel: "Recursive",
			}),
		},
		{
			name:               "when feature is enabled and selinuxchangepolicy is set to MountOption",
			featureGateEnabled: true,
			pod:                getPod(nil, nil),
			expectedPod:        getPod(nil, &selinuxMountOption),
			expectError:        false,
			ns: getNamespace(map[string]string{
				selinuxChangePolicyLabel: "MountOption",
			}),
		},
		{
			name:               "when feature is enabled, but pod already specifies different selinuxchangepolicy",
			pod:                getPod(nil, &selinuxRecursive),
			featureGateEnabled: true,
			expectedPod:        getPod(nil, &selinuxRecursive),
			expectError:        false,
			ns: getNamespace(map[string]string{
				selinuxChangePolicyLabel: "MountOption",
			}),
		},
		{
			name:               "when feature is enabled and both fsgroupchangepolicy and selinuxchangepolicy are set",
			featureGateEnabled: true,
			pod:                getPod(nil, nil),
			expectedPod:        getPod(&onRootMismatchPolicy, &selinuxMountOption),
			expectError:        false,
			ns: getNamespace(map[string]string{
				fsGroupChangePolicyLabel: "OnRootMismatch",
				selinuxChangePolicyLabel: "MountOption",
			}),
		},
		{
			name:               "when feature is enabled and both fsgroupchangepolicy and selinuxchangepolicy are set, but pod already specifies different policies",
			pod:                getPod(&alwaysFSGroupChangePolicy, &selinuxRecursive),
			featureGateEnabled: true,
			expectedPod:        getPod(&alwaysFSGroupChangePolicy, &selinuxRecursive),
			expectError:        false,
			ns: getNamespace(map[string]string{
				fsGroupChangePolicyLabel: "OnRootMismatch",
				selinuxChangePolicyLabel: "MountOption",
			}),
		},
		{
			name:               "when feature is enabled and both fsgroupchangepolicy and selinuxchangepolicy are set, but selinux lable has invalid value",
			pod:                getPod(nil, nil),
			featureGateEnabled: true,
			expectedPod:        getPod(&onRootMismatchPolicy, nil),
			expectError:        false,
			ns: getNamespace(map[string]string{
				fsGroupChangePolicyLabel: "OnRootMismatch",
				selinuxChangePolicyLabel: "InvalidValue",
			}),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, featuregate.Feature(openshiftfeatures.FeatureGateStoragePerformantSecurityPolicy), tc.featureGateEnabled)

			psp := &performantSecurityPolicy{}
			psp.nsLister = fakeNamespaceLister(tc.ns)
			psp.Handler = admission.NewHandler(admission.Create)

			psp.InspectFeatureGates(utilfeature.DefaultFeatureGate)

			if err := psp.ValidateInitialization(); err != nil {
				t.Fatalf("failed to validate initialization: %v", err)
			}

			namespaceName := tc.ns.Name
			podName := tc.pod.Name
			gvr := kapi.Resource("pods").WithVersion("version")
			attrs := admission.NewAttributesRecord(tc.pod, nil, schema.GroupVersionKind{}, namespaceName, podName, gvr, "", admission.Create, nil, false, nil)

			err := psp.Admit(context.Background(), attrs, nil)
			if (err != nil) != tc.expectError {
				t.Errorf("expected error: %v, got: %v", tc.expectError, err)
			}
			if !tc.expectError {
				currentFSGroupChangePolicy := getPodFSGroupChangePolicy(tc.pod)
				expectedFSGroupChangePolicy := getPodFSGroupChangePolicy(tc.expectedPod)

				if currentFSGroupChangePolicy != expectedFSGroupChangePolicy {
					t.Errorf("expected FSGroupChangePolicy %s, got %s", expectedFSGroupChangePolicy, currentFSGroupChangePolicy)
				}

				currentSELinuxChangePolicy := getPodSELinuxChangePolicy(tc.pod)
				expectedSELinuxChangePolicy := getPodSELinuxChangePolicy(tc.expectedPod)
				if currentSELinuxChangePolicy != expectedSELinuxChangePolicy {
					t.Errorf("expected SELinuxChangePolicy %s, got %s", expectedSELinuxChangePolicy, currentSELinuxChangePolicy)
				}
			}
		})
	}
}

func getPod(fsGroupChangePolicy *kapi.PodFSGroupChangePolicy, selinuxChangePolicy *kapi.PodSELinuxChangePolicy) *kapi.Pod {
	return &kapi.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "test-namespace",
		},
		Spec: kapi.PodSpec{
			SecurityContext: &kapi.PodSecurityContext{
				FSGroupChangePolicy: fsGroupChangePolicy,
				SELinuxChangePolicy: selinuxChangePolicy,
			},
		},
	}
}

func fakeNamespaceLister(ns *corev1.Namespace) corev1listers.NamespaceLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	_ = indexer.Add(ns)
	return corev1listers.NewNamespaceLister(indexer)
}

func getNamespace(labels map[string]string) *corev1.Namespace {
	return &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "test-namespace",
			Labels: labels,
		},
	}
}

func getPodFSGroupChangePolicy(pod *kapi.Pod) kapi.PodFSGroupChangePolicy {
	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.FSGroupChangePolicy != nil {
		return *pod.Spec.SecurityContext.FSGroupChangePolicy
	}
	return ""
}

func getPodSELinuxChangePolicy(pod *kapi.Pod) kapi.PodSELinuxChangePolicy {
	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.SELinuxChangePolicy != nil {
		return *pod.Spec.SecurityContext.SELinuxChangePolicy
	}
	return ""
}
