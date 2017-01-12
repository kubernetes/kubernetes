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

package api_test

import (
	"math/rand"
	"reflect"
	"sort"
	"testing"

	"github.com/google/gofuzz"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	batchv2alpha1 "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

type orderedGroupVersionKinds []schema.GroupVersionKind

func (o orderedGroupVersionKinds) Len() int      { return len(o) }
func (o orderedGroupVersionKinds) Swap(i, j int) { o[i], o[j] = o[j], o[i] }
func (o orderedGroupVersionKinds) Less(i, j int) bool {
	return o[i].String() < o[j].String()
}

func TestVerifyDefaulting(t *testing.T) {
	job := &batchv2alpha1.JobTemplate{}
	batchv2alpha1.SetObjectDefaults_JobTemplate(job)
	if job.Template.Spec.Template.Spec.DNSPolicy != apiv1.DNSClusterFirst {
		t.Errorf("unexpected defaulting: %#v", job)
	}
}

// TODO: add a reflexive test that verifies that all SetDefaults functions are registered
func TestDefaulting(t *testing.T) {
	// these are the known types with defaulters - you must add to this list if you add a top level defaulter
	typesWithDefaulting := map[schema.GroupVersionKind]struct{}{
		{Group: "", Version: "v1", Kind: "ConfigMap"}:                                              {},
		{Group: "", Version: "v1", Kind: "ConfigMapList"}:                                          {},
		{Group: "", Version: "v1", Kind: "Endpoints"}:                                              {},
		{Group: "", Version: "v1", Kind: "EndpointsList"}:                                          {},
		{Group: "", Version: "v1", Kind: "Namespace"}:                                              {},
		{Group: "", Version: "v1", Kind: "NamespaceList"}:                                          {},
		{Group: "", Version: "v1", Kind: "Node"}:                                                   {},
		{Group: "", Version: "v1", Kind: "NodeList"}:                                               {},
		{Group: "", Version: "v1", Kind: "PersistentVolume"}:                                       {},
		{Group: "", Version: "v1", Kind: "PersistentVolumeList"}:                                   {},
		{Group: "", Version: "v1", Kind: "PersistentVolumeClaim"}:                                  {},
		{Group: "", Version: "v1", Kind: "PersistentVolumeClaimList"}:                              {},
		{Group: "", Version: "v1", Kind: "PodAttachOptions"}:                                       {},
		{Group: "", Version: "v1", Kind: "PodExecOptions"}:                                         {},
		{Group: "", Version: "v1", Kind: "Pod"}:                                                    {},
		{Group: "", Version: "v1", Kind: "PodList"}:                                                {},
		{Group: "", Version: "v1", Kind: "PodTemplate"}:                                            {},
		{Group: "", Version: "v1", Kind: "PodTemplateList"}:                                        {},
		{Group: "", Version: "v1", Kind: "ReplicationController"}:                                  {},
		{Group: "", Version: "v1", Kind: "ReplicationControllerList"}:                              {},
		{Group: "", Version: "v1", Kind: "Secret"}:                                                 {},
		{Group: "", Version: "v1", Kind: "SecretList"}:                                             {},
		{Group: "", Version: "v1", Kind: "Service"}:                                                {},
		{Group: "", Version: "v1", Kind: "ServiceList"}:                                            {},
		{Group: "apps", Version: "v1beta1", Kind: "StatefulSet"}:                                   {},
		{Group: "apps", Version: "v1beta1", Kind: "StatefulSetList"}:                               {},
		{Group: "autoscaling", Version: "v1", Kind: "HorizontalPodAutoscaler"}:                     {},
		{Group: "autoscaling", Version: "v1", Kind: "HorizontalPodAutoscalerList"}:                 {},
		{Group: "batch", Version: "v1", Kind: "Job"}:                                               {},
		{Group: "batch", Version: "v1", Kind: "JobList"}:                                           {},
		{Group: "batch", Version: "v2alpha1", Kind: "CronJob"}:                                     {},
		{Group: "batch", Version: "v2alpha1", Kind: "CronJobList"}:                                 {},
		{Group: "batch", Version: "v2alpha1", Kind: "Job"}:                                         {},
		{Group: "batch", Version: "v2alpha1", Kind: "JobList"}:                                     {},
		{Group: "batch", Version: "v2alpha1", Kind: "JobTemplate"}:                                 {},
		{Group: "batch", Version: "v2alpha1", Kind: "ScheduledJob"}:                                {},
		{Group: "batch", Version: "v2alpha1", Kind: "ScheduledJobList"}:                            {},
		{Group: "certificates.k8s.io", Version: "v1alpha1", Kind: "CertificateSigningRequest"}:     {},
		{Group: "certificates.k8s.io", Version: "v1alpha1", Kind: "CertificateSigningRequestList"}: {},
		{Group: "componentconfig", Version: "v1alpha1", Kind: "KubeProxyConfiguration"}:            {},
		{Group: "componentconfig", Version: "v1alpha1", Kind: "KubeSchedulerConfiguration"}:        {},
		{Group: "componentconfig", Version: "v1alpha1", Kind: "KubeletConfiguration"}:              {},
		{Group: "kubeadm.k8s.io", Version: "v1alpha1", Kind: "MasterConfiguration"}:                {},
		// This object contains only int fields which currently breaks the defaulting test because
		// it's pretty stupid. Once we add non integer fields, we should uncomment this.
		// {Group: "kubeadm.k8s.io", Version: "v1alpha1", Kind: "NodeConfiguration"}:                 {},
		{Group: "extensions", Version: "v1beta1", Kind: "DaemonSet"}:                              {},
		{Group: "extensions", Version: "v1beta1", Kind: "DaemonSetList"}:                          {},
		{Group: "extensions", Version: "v1beta1", Kind: "Deployment"}:                             {},
		{Group: "extensions", Version: "v1beta1", Kind: "DeploymentList"}:                         {},
		{Group: "extensions", Version: "v1beta1", Kind: "HorizontalPodAutoscaler"}:                {},
		{Group: "extensions", Version: "v1beta1", Kind: "HorizontalPodAutoscalerList"}:            {},
		{Group: "extensions", Version: "v1beta1", Kind: "ReplicaSet"}:                             {},
		{Group: "extensions", Version: "v1beta1", Kind: "ReplicaSetList"}:                         {},
		{Group: "rbac.authorization.k8s.io", Version: "v1alpha1", Kind: "ClusterRoleBinding"}:     {},
		{Group: "rbac.authorization.k8s.io", Version: "v1alpha1", Kind: "ClusterRoleBindingList"}: {},
		{Group: "rbac.authorization.k8s.io", Version: "v1alpha1", Kind: "RoleBinding"}:            {},
		{Group: "rbac.authorization.k8s.io", Version: "v1alpha1", Kind: "RoleBindingList"}:        {},
	}

	f := fuzz.New().NilChance(.5).NumElements(1, 1).RandSource(rand.NewSource(1))
	f.Funcs(
		func(s *runtime.RawExtension, c fuzz.Continue) {},
		func(s *metav1.LabelSelector, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			s.MatchExpressions = nil // need to fuzz this specially
		},
		func(s *apiv1.ListOptions, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			s.LabelSelector = "" // need to fuzz requirement strings specially
			s.FieldSelector = "" // need to fuzz requirement strings specially
		},
		func(s *extensionsv1beta1.ScaleStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s)
			s.TargetSelector = "" // need to fuzz requirement strings specially
		},
	)

	scheme := api.Scheme
	var testTypes orderedGroupVersionKinds
	for gvk := range scheme.AllKnownTypes() {
		if gvk.Version == runtime.APIVersionInternal {
			continue
		}
		testTypes = append(testTypes, gvk)
	}
	sort.Sort(testTypes)

	for _, gvk := range testTypes {
		_, expectedChanged := typesWithDefaulting[gvk]
		iter := 0
		changedOnce := false
		for {
			if iter > *fuzzIters {
				if !expectedChanged || changedOnce {
					break
				}
				if iter > 300 {
					t.Errorf("expected %s to trigger defaulting due to fuzzing", gvk)
					break
				}
				// if we expected defaulting, continue looping until the fuzzer gives us one
				// at worst, we will timeout
			}
			iter++

			src, err := scheme.New(gvk)
			if err != nil {
				t.Fatal(err)
			}
			f.Fuzz(src)

			src.GetObjectKind().SetGroupVersionKind(schema.GroupVersionKind{})

			original, err := scheme.DeepCopy(src)
			if err != nil {
				t.Fatal(err)
			}

			// get internal
			withDefaults, _ := scheme.DeepCopy(src)
			scheme.Default(withDefaults.(runtime.Object))

			if !reflect.DeepEqual(original, withDefaults) {
				changedOnce = true
				if !expectedChanged {
					t.Errorf("{Group: \"%s\", Version: \"%s\", Kind: \"%s\"} did not expect defaults to be set - update expected or check defaulter registering: %s", gvk.Group, gvk.Version, gvk.Kind, diff.ObjectReflectDiff(original, withDefaults))
				}
			}
		}
	}
}

func BenchmarkPodDefaulting(b *testing.B) {
	f := fuzz.New().NilChance(.5).NumElements(1, 1).RandSource(rand.NewSource(1))
	items := make([]apiv1.Pod, 100)
	for i := range items {
		f.Fuzz(&items[i])
	}

	scheme := api.Scheme
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		pod := &items[i%len(items)]

		scheme.Default(pod)
	}
	b.StopTimer()
}
