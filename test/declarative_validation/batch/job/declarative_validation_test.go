/*
Copyright 2025 The Kubernetes Authors.

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

package job

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	batch "k8s.io/kubernetes/pkg/apis/batch"
	_ "k8s.io/kubernetes/pkg/apis/batch/install"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/registry/batch/job"
	"k8s.io/kubernetes/test/declarative_validation/meta"
	"k8s.io/utils/ptr"
	"testing"
)

// TODO: remove this apiVersions variable once coverage tests are generated for this package.
var apiVersions = []string{"v1"}

// Helper function to create a baseline valid Job with optional tweaks
func mkJob(tweaks ...func(*batch.Job)) batch.Job {
	obj := func() batch.Job {
		labels := map[string]string{"foo": "bar"}
		gracePeriod := int64(30)
		return batch.Job{
			ObjectMeta: metav1.ObjectMeta{
				Name: "valid-resource-name",
			},
			Spec: batch.JobSpec{
				ManualSelector: ptr.To(true),
				Selector: &metav1.LabelSelector{
					MatchLabels: labels,
				},
				Template: core.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: labels,
					},
					Spec: core.PodSpec{
						Containers: []core.Container{
							{
								Name:                     "nginx",
								Image:                    "nginx",
								TerminationMessagePolicy: core.TerminationMessagePolicy("File"),
								ImagePullPolicy:          core.PullPolicy("Always"),
							},
						},
						RestartPolicy:                 core.RestartPolicy("Never"),
						DNSPolicy:                     core.DNSPolicy("Default"),
						TerminationGracePeriodSeconds: &gracePeriod,
					},
				},
			},
		}
	}()
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := job.Strategy
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "batch",
				APIVersion:        apiVersion,
				Resource:          "jobs",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "create",
			})
			obj := mkJob(func(o *batch.Job) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaTestCases(t, ctx, &obj, strategy)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := job.Strategy
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "batch",
				APIVersion:        apiVersion,
				Resource:          "jobs",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "update",
			})
			obj := mkJob(func(o *batch.Job) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaUpdateTestCases(t, ctx, &obj, strategy)
		})
	}
}
