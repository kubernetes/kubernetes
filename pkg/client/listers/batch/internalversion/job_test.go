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

package internalversion

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/client/cache"
)

func TestJobLister(t *testing.T) {
	indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	lister := NewJobLister(indexer)
	testCases := []struct {
		inJobs      []*batch.Job
		list        func() ([]*batch.Job, error)
		outJobNames sets.String
		expectErr   bool
		msg         string
	}{
		// Basic listing
		{
			inJobs: []*batch.Job{
				{ObjectMeta: api.ObjectMeta{Name: "basic"}},
			},
			list: func() ([]*batch.Job, error) {
				list, err := lister.List(labels.Everything())
				return list, err
			},
			outJobNames: sets.NewString("basic"),
			msg:         "basic listing failed",
		},
		// Listing multiple jobs
		{
			inJobs: []*batch.Job{
				{ObjectMeta: api.ObjectMeta{Name: "basic"}},
				{ObjectMeta: api.ObjectMeta{Name: "complex"}},
				{ObjectMeta: api.ObjectMeta{Name: "complex2"}},
			},
			list: func() ([]*batch.Job, error) {
				list, err := lister.List(labels.Everything())
				return list, err
			},
			outJobNames: sets.NewString("basic", "complex", "complex2"),
			msg:         "listing multiple jobs failed",
		},
		// No pod labels
		{
			inJobs: []*batch.Job{
				{
					ObjectMeta: api.ObjectMeta{Name: "basic", Namespace: "ns"},
					Spec: batch.JobSpec{
						Selector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"foo": "baz"},
						},
					},
				},
			},
			list: func() ([]*batch.Job, error) {
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{Name: "pod", Namespace: "ns"},
				}
				podJobs, err := lister.GetPodJobs(pod)
				jobs := make([]*batch.Job, 0, len(podJobs))
				for i := range podJobs {
					jobs = append(jobs, &podJobs[i])
				}
				return jobs, err
			},
			outJobNames: sets.NewString(),
			expectErr:   true,
			msg:         "listing jobs failed when pod has no labels: expected error, got none",
		},
		// No Job selectors
		{
			inJobs: []*batch.Job{
				{
					ObjectMeta: api.ObjectMeta{Name: "basic", Namespace: "ns"},
				},
			},
			list: func() ([]*batch.Job, error) {
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      "pod",
						Namespace: "ns",
						Labels:    map[string]string{"foo": "bar"},
					},
				}
				podJobs, err := lister.GetPodJobs(pod)
				jobs := make([]*batch.Job, 0, len(podJobs))
				for i := range podJobs {
					jobs = append(jobs, &podJobs[i])
				}
				return jobs, err
			},
			outJobNames: sets.NewString(),
			expectErr:   true,
			msg:         "listing jobs failed when job has no selector: expected error, got none",
		},
		// Matching labels to selectors and namespace
		{
			inJobs: []*batch.Job{
				{
					ObjectMeta: api.ObjectMeta{Name: "foo"},
					Spec: batch.JobSpec{
						Selector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"foo": "bar"},
						},
					},
				},
				{
					ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: "ns"},
					Spec: batch.JobSpec{
						Selector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"foo": "bar"},
						},
					},
				},
			},
			list: func() ([]*batch.Job, error) {
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      "pod",
						Labels:    map[string]string{"foo": "bar"},
						Namespace: "ns",
					},
				}
				podJobs, err := lister.GetPodJobs(pod)
				jobs := make([]*batch.Job, 0, len(podJobs))
				for i := range podJobs {
					jobs = append(jobs, &podJobs[i])
				}
				return jobs, err
			},
			outJobNames: sets.NewString("bar"),
			msg:         "listing jobs with namespace and selector failed",
		},
		// Matching labels to selectors and namespace, error case
		{
			inJobs: []*batch.Job{
				{
					ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "foo"},
					Spec: batch.JobSpec{
						Selector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"foo": "bar"},
						},
					},
				},
				{
					ObjectMeta: api.ObjectMeta{Name: "bar", Namespace: "bar"},
					Spec: batch.JobSpec{
						Selector: &metav1.LabelSelector{
							MatchLabels: map[string]string{"foo": "bar"},
						},
					},
				},
			},
			list: func() ([]*batch.Job, error) {
				pod := &api.Pod{
					ObjectMeta: api.ObjectMeta{
						Name:      "pod",
						Labels:    map[string]string{"foo": "bar"},
						Namespace: "baz",
					},
				}
				podJobs, err := lister.GetPodJobs(pod)
				jobs := make([]*batch.Job, 0, len(podJobs))
				for i := range podJobs {
					jobs = append(jobs, &podJobs[i])
				}
				return jobs, err
			},
			expectErr: true,
			msg:       "listing jobs with namespace and selector failed: expected error, got none",
		},
	}
	for _, c := range testCases {
		for _, r := range c.inJobs {
			indexer.Add(r)
		}

		Jobs, err := c.list()
		if err != nil && c.expectErr {
			continue
		} else if c.expectErr {
			t.Errorf("%v", c.msg)
			continue
		} else if err != nil {
			t.Errorf("Unexpected error %#v", err)
			continue
		}
		JobNames := make([]string, len(Jobs))
		for ix := range Jobs {
			JobNames[ix] = Jobs[ix].Name
		}
		if !c.outJobNames.HasAll(JobNames...) || len(JobNames) != len(c.outJobNames) {
			t.Errorf("%v : expected %v, got %v", c.msg, JobNames, c.outJobNames)
		}
	}
}
