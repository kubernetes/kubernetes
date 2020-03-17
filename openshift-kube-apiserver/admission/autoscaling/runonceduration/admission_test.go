package runonceduration

import (
	"bytes"
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apiserver/pkg/admission"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	kapi "k8s.io/kubernetes/pkg/apis/core"

	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/runonceduration"
)

func fakeNamespaceLister(projectAnnotations map[string]string) corev1listers.NamespaceLister {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	ns := &corev1.Namespace{}
	ns.Name = "default"
	ns.Annotations = projectAnnotations
	indexer.Add(ns)
	return corev1listers.NewNamespaceLister(indexer)
}

func testConfig(n *int64) *runonceduration.RunOnceDurationConfig {
	return &runonceduration.RunOnceDurationConfig{
		ActiveDeadlineSecondsLimit: n,
	}
}

func testRunOncePod() *kapi.Pod {
	pod := &kapi.Pod{}
	pod.Spec.RestartPolicy = kapi.RestartPolicyNever
	return pod
}

func testRestartOnFailurePod() *kapi.Pod {
	pod := &kapi.Pod{}
	pod.Spec.RestartPolicy = kapi.RestartPolicyOnFailure
	return pod
}

func testRunOncePodWithDuration(n int64) *kapi.Pod {
	pod := testRunOncePod()
	pod.Spec.ActiveDeadlineSeconds = &n
	return pod
}

func testRestartAlwaysPod() *kapi.Pod {
	pod := &kapi.Pod{}
	pod.Spec.RestartPolicy = kapi.RestartPolicyAlways
	return pod
}

func int64p(n int64) *int64 {
	return &n
}

func TestRunOnceDurationAdmit(t *testing.T) {
	tests := []struct {
		name                          string
		config                        *runonceduration.RunOnceDurationConfig
		pod                           *kapi.Pod
		projectAnnotations            map[string]string
		expectedActiveDeadlineSeconds *int64
	}{
		{
			name:                          "expect globally configured duration to be set",
			config:                        testConfig(int64p(10)),
			pod:                           testRunOncePod(),
			expectedActiveDeadlineSeconds: int64p(10),
		},
		{
			name:                          "empty config, no duration to be set",
			config:                        testConfig(nil),
			pod:                           testRunOncePod(),
			expectedActiveDeadlineSeconds: nil,
		},
		{
			name:                          "expect configured duration to not limit lower existing duration",
			config:                        testConfig(int64p(10)),
			pod:                           testRunOncePodWithDuration(5),
			expectedActiveDeadlineSeconds: int64p(5),
		},
		{
			name:                          "expect empty config to not limit existing duration",
			config:                        testConfig(nil),
			pod:                           testRunOncePodWithDuration(5),
			expectedActiveDeadlineSeconds: int64p(5),
		},
		{
			name:   "expect project limit to be used with nil global value",
			config: testConfig(nil),
			pod:    testRunOncePodWithDuration(2000),
			projectAnnotations: map[string]string{
				runonceduration.ActiveDeadlineSecondsLimitAnnotation: "1000",
			},
			expectedActiveDeadlineSeconds: int64p(1000),
		},
		{
			name:   "expect project limit to not limit a smaller set value",
			config: testConfig(nil),
			pod:    testRunOncePodWithDuration(10),
			projectAnnotations: map[string]string{
				runonceduration.ActiveDeadlineSecondsLimitAnnotation: "1000",
			},
			expectedActiveDeadlineSeconds: int64p(10),
		},
		{
			name:   "expect project limit to have priority over global config value",
			config: testConfig(int64p(10)),
			pod:    testRunOncePodWithDuration(2000),
			projectAnnotations: map[string]string{
				runonceduration.ActiveDeadlineSecondsLimitAnnotation: "1000",
			},
			expectedActiveDeadlineSeconds: int64p(1000),
		},
		{
			name:                          "make no change to a pod that is not a run-once pod",
			config:                        testConfig(int64p(10)),
			pod:                           testRestartAlwaysPod(),
			expectedActiveDeadlineSeconds: nil,
		},
		{
			name:                          "update a pod that has a RestartOnFailure policy",
			config:                        testConfig(int64p(10)),
			pod:                           testRestartOnFailurePod(),
			expectedActiveDeadlineSeconds: int64p(10),
		},
	}

	for _, tc := range tests {
		admissionPlugin := NewRunOnceDuration(tc.config)
		admissionPlugin.(*runOnceDuration).nsLister = fakeNamespaceLister(tc.projectAnnotations)
		pod := tc.pod
		attrs := admission.NewAttributesRecord(pod, nil, kapi.Kind("Pod").WithVersion("version"), "default", "test", kapi.Resource("pods").WithVersion("version"), "", admission.Create, nil, false, nil)
		if err := admissionPlugin.(admission.MutationInterface).Admit(context.TODO(), attrs, nil); err != nil {
			t.Errorf("%s: unexpected mutating admission error: %v", tc.name, err)
			continue
		}

		switch {
		case tc.expectedActiveDeadlineSeconds == nil && pod.Spec.ActiveDeadlineSeconds == nil:
			// continue
		case tc.expectedActiveDeadlineSeconds == nil && pod.Spec.ActiveDeadlineSeconds != nil:
			t.Errorf("%s: expected nil ActiveDeadlineSeconds. Got: %d", tc.name, *pod.Spec.ActiveDeadlineSeconds)
		case tc.expectedActiveDeadlineSeconds != nil && pod.Spec.ActiveDeadlineSeconds == nil:
			t.Errorf("%s: unexpected nil ActiveDeadlineSeconds.", tc.name)
		case *pod.Spec.ActiveDeadlineSeconds != *tc.expectedActiveDeadlineSeconds:
			t.Errorf("%s: unexpected active deadline seconds: %d", tc.name, *pod.Spec.ActiveDeadlineSeconds)
		}
	}
}

func TestReadConfig(t *testing.T) {
	configStr := `apiVersion: autoscaling.openshift.io/v1
kind: RunOnceDurationConfig
activeDeadlineSecondsOverride: 3600
`
	buf := bytes.NewBufferString(configStr)
	config, err := readConfig(buf)
	if err != nil {
		t.Fatalf("unexpected error reading config: %v", err)
	}
	if config.ActiveDeadlineSecondsLimit == nil {
		t.Fatalf("nil value for ActiveDeadlineSecondsLimit")
	}
	if *config.ActiveDeadlineSecondsLimit != 3600 {
		t.Errorf("unexpected value for ActiveDeadlineSecondsLimit: %d", config.ActiveDeadlineSecondsLimit)
	}
}

func TestInt64MinP(t *testing.T) {
	ten := int64(10)
	twenty := int64(20)
	tests := []struct {
		a, b, expected *int64
	}{
		{
			a:        &ten,
			b:        nil,
			expected: &ten,
		},
		{
			a:        nil,
			b:        &ten,
			expected: &ten,
		},
		{
			a:        &ten,
			b:        &twenty,
			expected: &ten,
		},
		{
			a:        nil,
			b:        nil,
			expected: nil,
		},
	}

	for _, test := range tests {
		actual := int64MinP(test.a, test.b)
		switch {
		case actual == nil && test.expected != nil,
			test.expected == nil && actual != nil:
			t.Errorf("unexpected %v for %#v", actual, test)
			continue
		case actual == nil:
			continue
		case *actual != *test.expected:
			t.Errorf("unexpected: %v for %#v", actual, test)
		}
	}
}
