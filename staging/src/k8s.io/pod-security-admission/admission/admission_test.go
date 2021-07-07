/*
Copyright 2021 The Kubernetes Authors.

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

package admission

import (
	"context"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	admissionv1 "k8s.io/api/admission/v1"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	admissionapi "k8s.io/pod-security-admission/admission/api"
	"k8s.io/pod-security-admission/api"
	"k8s.io/pod-security-admission/policy"
	"k8s.io/utils/pointer"
)

func TestDefaultExtractPodSpec(t *testing.T) {
	metadata := metav1.ObjectMeta{
		Name: "foo-pod",
	}
	spec := corev1.PodSpec{
		Containers: []corev1.Container{{
			Name: "foo-container",
		}},
	}
	objects := []runtime.Object{
		&corev1.Pod{
			ObjectMeta: metadata,
			Spec:       spec,
		},
		&corev1.PodTemplate{
			ObjectMeta: metav1.ObjectMeta{Name: "foo-template"},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metadata,
				Spec:       spec,
			},
		},
		&corev1.ReplicationController{
			ObjectMeta: metav1.ObjectMeta{Name: "foo-rc"},
			Spec: corev1.ReplicationControllerSpec{
				Template: &corev1.PodTemplateSpec{
					ObjectMeta: metadata,
					Spec:       spec,
				},
			},
		},
		&appsv1.ReplicaSet{
			ObjectMeta: metav1.ObjectMeta{Name: "foo-rs"},
			Spec: appsv1.ReplicaSetSpec{
				Template: corev1.PodTemplateSpec{
					ObjectMeta: metadata,
					Spec:       spec,
				},
			},
		},
		&appsv1.Deployment{
			ObjectMeta: metav1.ObjectMeta{Name: "foo-deployment"},
			Spec: appsv1.DeploymentSpec{
				Template: corev1.PodTemplateSpec{
					ObjectMeta: metadata,
					Spec:       spec,
				},
			},
		},
		&appsv1.StatefulSet{
			ObjectMeta: metav1.ObjectMeta{Name: "foo-ss"},
			Spec: appsv1.StatefulSetSpec{
				Template: corev1.PodTemplateSpec{
					ObjectMeta: metadata,
					Spec:       spec,
				},
			},
		},
		&appsv1.DaemonSet{
			ObjectMeta: metav1.ObjectMeta{Name: "foo-ds"},
			Spec: appsv1.DaemonSetSpec{
				Template: corev1.PodTemplateSpec{
					ObjectMeta: metadata,
					Spec:       spec,
				},
			},
		},
		&batchv1.Job{
			ObjectMeta: metav1.ObjectMeta{Name: "foo-job"},
			Spec: batchv1.JobSpec{
				Template: corev1.PodTemplateSpec{
					ObjectMeta: metadata,
					Spec:       spec,
				},
			},
		},
		&batchv1.CronJob{
			ObjectMeta: metav1.ObjectMeta{Name: "foo-cronjob"},
			Spec: batchv1.CronJobSpec{
				JobTemplate: batchv1.JobTemplateSpec{
					Spec: batchv1.JobSpec{
						Template: corev1.PodTemplateSpec{
							ObjectMeta: metadata,
							Spec:       spec,
						},
					},
				},
			},
		},
	}
	extractor := &DefaultPodSpecExtractor{}
	for _, obj := range objects {
		name := obj.(metav1.Object).GetName()
		actualMetadata, actualSpec, err := extractor.ExtractPodSpec(obj)
		assert.NoError(t, err, name)
		assert.Equal(t, &metadata, actualMetadata, "%s: Metadata mismatch", name)
		assert.Equal(t, &spec, actualSpec, "%s: PodSpec mismatch", name)
	}

	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo-svc",
		},
	}
	_, _, err := extractor.ExtractPodSpec(service)
	assert.Error(t, err, "service should not have an extractable pod spec")
}

func TestDefaultHasPodSpec(t *testing.T) {
	podLikeResources := []schema.GroupResource{
		corev1.Resource("pods"),
		corev1.Resource("replicationcontrollers"),
		corev1.Resource("podtemplates"),
		appsv1.Resource("replicasets"),
		appsv1.Resource("deployments"),
		appsv1.Resource("statefulsets"),
		appsv1.Resource("daemonsets"),
		batchv1.Resource("jobs"),
		batchv1.Resource("cronjobs"),
	}
	extractor := &DefaultPodSpecExtractor{}
	for _, gr := range podLikeResources {
		assert.True(t, extractor.HasPodSpec(gr), gr.String())
	}

	nonPodResources := []schema.GroupResource{
		corev1.Resource("services"),
		admissionv1.Resource("admissionreviews"),
		appsv1.Resource("foobars"),
	}
	for _, gr := range nonPodResources {
		assert.False(t, extractor.HasPodSpec(gr), gr.String())
	}
}

type testEvaluator struct {
	lv api.LevelVersion
}

func (t *testEvaluator) EvaluatePod(lv api.LevelVersion, meta *metav1.ObjectMeta, spec *corev1.PodSpec) []policy.CheckResult {
	t.lv = lv
	if meta.Annotations["error"] != "" {
		return []policy.CheckResult{{Allowed: false, ForbiddenReason: meta.Annotations["error"]}}
	} else {
		return []policy.CheckResult{{Allowed: true}}
	}
}

type testPodLister struct {
	called bool
	pods   []*corev1.Pod
}

func (t *testPodLister) ListPods(ctx context.Context, namespace string) ([]*corev1.Pod, error) {
	t.called = true
	return t.pods, nil
}

func TestValidateNamespace(t *testing.T) {
	testcases := []struct {
		name                 string
		exemptNamespaces     []string
		exemptRuntimeClasses []string
		// override default policy
		defaultPolicy *api.Policy
		// request subresource
		subresource string
		// labels for the new namespace
		newLabels map[string]string
		// labels for the old namespace (only used if update=true)
		oldLabels map[string]string
		// list of pods to return
		pods []*corev1.Pod

		expectAllowed  bool
		expectError    string
		expectListPods bool
		expectEvaluate api.LevelVersion
		expectWarnings []string
	}{
		// creation tests, just validate labels
		{
			name:           "create privileged",
			newLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelPrivileged), api.EnforceVersionLabel: "v1.0"},
			expectAllowed:  true,
			expectListPods: false,
		},
		{
			name:           "create baseline",
			newLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelBaseline)},
			expectAllowed:  true,
			expectListPods: false,
		},
		{
			name:           "create restricted",
			newLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelBaseline)},
			expectAllowed:  true,
			expectListPods: false,
		},
		{
			name:           "create malformed level",
			newLabels:      map[string]string{api.EnforceLevelLabel: "unknown"},
			expectAllowed:  false,
			expectError:    `must be one of privileged, baseline, restricted`,
			expectListPods: false,
		},
		{
			name:           "create malformed version",
			newLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelPrivileged), api.EnforceVersionLabel: "unknown"},
			expectAllowed:  false,
			expectError:    `must be "latest" or "v1.x"`,
			expectListPods: false,
		},

		// update tests that don't tighten effective policy, no pod list/evaluate
		{
			name:           "update no-op",
			newLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelRestricted), api.EnforceVersionLabel: "v1.0"},
			oldLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelRestricted), api.EnforceVersionLabel: "v1.0"},
			expectAllowed:  true,
			expectListPods: false,
		},
		{
			name:           "update no-op malformed level",
			newLabels:      map[string]string{api.EnforceLevelLabel: "unknown"},
			oldLabels:      map[string]string{api.EnforceLevelLabel: "unknown"},
			expectAllowed:  true,
			expectListPods: false,
		},
		{
			name:           "update no-op malformed version",
			newLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelBaseline), api.EnforceVersionLabel: "unknown"},
			oldLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelBaseline), api.EnforceVersionLabel: "unknown"},
			expectAllowed:  true,
			expectListPods: false,
		},
		{
			name:           "update relax level identical version",
			newLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelBaseline), api.EnforceVersionLabel: "v1.0"},
			oldLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelRestricted), api.EnforceVersionLabel: "v1.0"},
			expectAllowed:  true,
			expectListPods: false,
		},
		{
			name:           "update relax level explicit latest",
			newLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelBaseline), api.EnforceVersionLabel: "latest"},
			oldLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelRestricted), api.EnforceVersionLabel: "latest"},
			expectAllowed:  true,
			expectListPods: false,
		},
		{
			name:           "update relax level implicit latest",
			newLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelBaseline)},
			oldLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelRestricted)},
			expectAllowed:  true,
			expectListPods: false,
		},
		{
			name:           "update to explicit privileged",
			newLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelPrivileged)},
			oldLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelRestricted), api.EnforceVersionLabel: "v1.0"},
			expectAllowed:  true,
			expectListPods: false,
		},
		{
			name:           "update to implicit privileged",
			newLabels:      map[string]string{},
			oldLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelRestricted), api.EnforceVersionLabel: "v1.0"},
			expectAllowed:  true,
			expectListPods: false,
		},
		{
			name:             "update exempt to restricted",
			exemptNamespaces: []string{"test"},
			newLabels:        map[string]string{api.EnforceLevelLabel: string(api.LevelRestricted), api.EnforceVersionLabel: "v1.0"},
			oldLabels:        map[string]string{},
			expectAllowed:    true,
			expectListPods:   false,
		},

		// update tests that introduce labels errors
		{
			name:           "update malformed level",
			newLabels:      map[string]string{api.EnforceLevelLabel: "unknown"},
			oldLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelRestricted), api.EnforceVersionLabel: "v1.0"},
			expectAllowed:  false,
			expectError:    `must be one of privileged, baseline, restricted`,
			expectListPods: false,
		},
		{
			name:           "update malformed version",
			newLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelPrivileged), api.EnforceVersionLabel: "unknown"},
			oldLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelRestricted), api.EnforceVersionLabel: "v1.0"},
			expectAllowed:  false,
			expectError:    `must be "latest" or "v1.x"`,
			expectListPods: false,
		},

		// update tests that tighten effective policy
		{
			name:           "update to implicit restricted",
			newLabels:      map[string]string{},
			oldLabels:      map[string]string{api.EnforceLevelLabel: string(api.LevelBaseline), api.EnforceVersionLabel: "v1.0"},
			defaultPolicy:  &api.Policy{Enforce: api.LevelVersion{Level: api.LevelRestricted, Version: api.LatestVersion()}},
			expectAllowed:  true,
			expectListPods: true,
			expectEvaluate: api.LevelVersion{Level: api.LevelRestricted, Version: api.LatestVersion()},
			expectWarnings: []string{"noruntimeclasspod: message", "runtimeclass1pod: message", "runtimeclass2pod: message"},
		},
		{
			name:                 "update with runtimeclass exempt pods",
			exemptRuntimeClasses: []string{"runtimeclass1"},
			newLabels:            map[string]string{api.EnforceLevelLabel: string(api.LevelRestricted)},
			oldLabels:            map[string]string{api.EnforceLevelLabel: string(api.LevelBaseline)},
			expectAllowed:        true,
			expectListPods:       true,
			expectEvaluate:       api.LevelVersion{Level: api.LevelRestricted, Version: api.LatestVersion()},
			expectWarnings:       []string{"noruntimeclasspod: message", "runtimeclass2pod: message"},
		},

		// TODO: test for aggregating pods with identical warnings
		// TODO: test for bounding evalution time with a warning
		// TODO: test for bounding pod count with a warning
		// TODO: test for prioritizing evaluating pods from unique controllers
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			newObject := &corev1.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "test",
					Labels: tc.newLabels,
				},
			}
			var operation = admissionv1.Create
			var oldObject runtime.Object
			if tc.oldLabels != nil {
				operation = admissionv1.Update
				oldObject = &corev1.Namespace{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "test",
						Labels: tc.oldLabels,
					},
				}
			}

			attrs := &AttributesRecord{
				Object:      newObject,
				OldObject:   oldObject,
				Namespace:   newObject.Name,
				Resource:    schema.GroupVersionResource{Group: "", Version: "v1", Resource: "namespaces"},
				Subresource: tc.subresource,
				Operation:   operation,
			}

			defaultPolicy := api.Policy{
				Enforce: api.LevelVersion{
					Level:   api.LevelPrivileged,
					Version: api.LatestVersion(),
				},
			}
			if tc.defaultPolicy != nil {
				defaultPolicy = *tc.defaultPolicy
			}

			pods := tc.pods
			if pods == nil {
				pods = []*corev1.Pod{
					{
						ObjectMeta: metav1.ObjectMeta{Name: "noruntimeclasspod", Annotations: map[string]string{"error": "message"}},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "runtimeclass1pod", Annotations: map[string]string{"error": "message"}},
						Spec:       corev1.PodSpec{RuntimeClassName: pointer.String("runtimeclass1")},
					},
					{
						ObjectMeta: metav1.ObjectMeta{Name: "runtimeclass2pod", Annotations: map[string]string{"error": "message"}},
						Spec:       corev1.PodSpec{RuntimeClassName: pointer.String("runtimeclass2")},
					},
				}
			}
			podLister := &testPodLister{pods: pods}
			evaluator := &testEvaluator{}
			a := &Admission{
				PodLister: podLister,
				Evaluator: evaluator,
				Configuration: &admissionapi.PodSecurityConfiguration{
					Exemptions: admissionapi.PodSecurityExemptions{
						Namespaces:     tc.exemptNamespaces,
						RuntimeClasses: tc.exemptRuntimeClasses,
					},
				},
				defaultPolicy: defaultPolicy,
			}
			result := a.ValidateNamespace(context.TODO(), attrs)
			if result.Allowed != tc.expectAllowed {
				t.Errorf("expected allowed=%v, got %v", tc.expectAllowed, result.Allowed)
			}

			resultError := ""
			if result.Result != nil {
				resultError = result.Result.Message
			}
			if (len(resultError) > 0) != (len(tc.expectError) > 0) {
				t.Errorf("expected error=%v, got %v", tc.expectError, resultError)
			}
			if len(tc.expectError) > 0 && !strings.Contains(resultError, tc.expectError) {
				t.Errorf("expected error containing '%s', got %s", tc.expectError, resultError)
			}
			if podLister.called != tc.expectListPods {
				t.Errorf("expected getPods=%v, got %v", tc.expectListPods, podLister.called)
			}
			if evaluator.lv != tc.expectEvaluate {
				t.Errorf("expected to evaluate %v, got %v", tc.expectEvaluate, evaluator.lv)
			}
			if !reflect.DeepEqual(result.Warnings, tc.expectWarnings) {
				t.Errorf("expected warnings:\n%v\ngot\n%v", tc.expectWarnings, result.Warnings)
			}
		})
	}
}
