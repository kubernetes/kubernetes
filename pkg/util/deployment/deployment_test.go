/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package deployment

import (
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
	"k8s.io/kubernetes/pkg/runtime"
)

func newPod(now time.Time, ready bool, beforeSec int) api.Pod {
	conditionStatus := api.ConditionFalse
	if ready {
		conditionStatus = api.ConditionTrue
	}
	return api.Pod{
		Status: api.PodStatus{
			Conditions: []api.PodCondition{
				{
					Type:               api.PodReady,
					LastTransitionTime: unversioned.NewTime(now.Add(-1 * time.Duration(beforeSec) * time.Second)),
					Status:             conditionStatus,
				},
			},
		},
	}
}

func TestGetReadyPodsCount(t *testing.T) {
	now := time.Now()
	tests := []struct {
		pods            []api.Pod
		minReadySeconds int
		expected        int
	}{
		{
			[]api.Pod{
				newPod(now, true, 0),
				newPod(now, true, 2),
				newPod(now, false, 1),
			},
			1,
			1,
		},
		{
			[]api.Pod{
				newPod(now, true, 2),
				newPod(now, true, 11),
				newPod(now, true, 5),
			},
			10,
			1,
		},
	}

	for _, test := range tests {
		if count := getReadyPodsCount(test.pods, test.minReadySeconds); count != test.expected {
			t.Errorf("Pods = %#v, minReadySeconds = %d, expected %d, got %d", test.pods, test.minReadySeconds, test.expected, count)
		}
	}
}

// generatePodFromRS creates a pod, with the input ReplicaSet's selector and its template
func generatePodFromRS(rs extensions.ReplicaSet) api.Pod {
	return api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels: rs.Labels,
		},
		Spec: rs.Spec.Template.Spec,
	}
}

func generatePod(labels map[string]string, image string) api.Pod {
	return api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels: labels,
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:                   image,
					Image:                  image,
					ImagePullPolicy:        api.PullAlways,
					TerminationMessagePath: api.TerminationMessagePathDefault,
				},
			},
		},
	}
}

func generateRSWithLabel(labels map[string]string, image string) extensions.ReplicaSet {
	return extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name:   api.SimpleNameGenerator.GenerateName("replicaset"),
			Labels: labels,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: 1,
			Selector: &unversioned.LabelSelector{MatchLabels: labels},
			Template: &api.PodTemplateSpec{
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:                   image,
							Image:                  image,
							ImagePullPolicy:        api.PullAlways,
							TerminationMessagePath: api.TerminationMessagePathDefault,
						},
					},
				},
			},
		},
	}
}

// generateRS creates a replica set, with the input deployment's template as its template
func generateRS(deployment extensions.Deployment) extensions.ReplicaSet {
	template := GetNewReplicaSetTemplate(deployment)
	return extensions.ReplicaSet{
		ObjectMeta: api.ObjectMeta{
			Name:   api.SimpleNameGenerator.GenerateName("replicaset"),
			Labels: template.Labels,
		},
		Spec: extensions.ReplicaSetSpec{
			Template: &template,
			Selector: &unversioned.LabelSelector{MatchLabels: template.Labels},
		},
	}
}

// generateDeployment creates a deployment, with the input image as its template
func generateDeployment(image string) extensions.Deployment {
	podLabels := map[string]string{"name": image}
	terminationSec := int64(30)
	return extensions.Deployment{
		ObjectMeta: api.ObjectMeta{
			Name: image,
		},
		Spec: extensions.DeploymentSpec{
			Replicas: 1,
			Selector: &unversioned.LabelSelector{MatchLabels: podLabels},
			Template: api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: podLabels,
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:                   image,
							Image:                  image,
							ImagePullPolicy:        api.PullAlways,
							TerminationMessagePath: api.TerminationMessagePathDefault,
						},
					},
					DNSPolicy:                     api.DNSClusterFirst,
					TerminationGracePeriodSeconds: &terminationSec,
					RestartPolicy:                 api.RestartPolicyAlways,
					SecurityContext:               &api.PodSecurityContext{},
				},
			},
		},
	}
}

func TestGetNewRC(t *testing.T) {
	newDeployment := generateDeployment("nginx")
	newRC := generateRS(newDeployment)

	tests := []struct {
		test     string
		rsList   extensions.ReplicaSetList
		expected *extensions.ReplicaSet
	}{
		{
			"No new ReplicaSet",
			extensions.ReplicaSetList{
				Items: []extensions.ReplicaSet{
					generateRS(generateDeployment("foo")),
					generateRS(generateDeployment("bar")),
				},
			},
			nil,
		},
		{
			"Has new ReplicaSet",
			extensions.ReplicaSetList{
				Items: []extensions.ReplicaSet{
					generateRS(generateDeployment("foo")),
					generateRS(generateDeployment("bar")),
					generateRS(generateDeployment("abc")),
					newRC,
					generateRS(generateDeployment("xyz")),
				},
			},
			&newRC,
		},
	}

	ns := api.NamespaceDefault
	for _, test := range tests {
		c := &simple.Client{
			Request: simple.Request{
				Method: "GET",
				Path:   testapi.Default.ResourcePath("replicaSets", ns, ""),
			},
			Response: simple.Response{
				StatusCode: 200,
				Body:       &test.rsList,
			},
		}
		rs, err := GetNewReplicaSet(newDeployment, c.Setup(t).Clientset)
		if err != nil {
			t.Errorf("In test case %s, got unexpected error %v", test.test, err)
		}
		if !api.Semantic.DeepEqual(rs, test.expected) {
			t.Errorf("In test case %s, expected %+v, got %+v", test.test, test.expected, rs)
		}
	}
}

func TestGetOldRCs(t *testing.T) {
	newDeployment := generateDeployment("nginx")
	newRS := generateRS(newDeployment)
	newPod := generatePodFromRS(newRS)

	// create 2 old deployments and related replica sets/pods, with the same labels but different template
	oldDeployment := generateDeployment("nginx")
	oldDeployment.Spec.Template.Spec.Containers[0].Name = "nginx-old-1"
	oldRS := generateRS(oldDeployment)
	oldPod := generatePodFromRS(oldRS)
	oldDeployment2 := generateDeployment("nginx")
	oldDeployment2.Spec.Template.Spec.Containers[0].Name = "nginx-old-2"
	oldRS2 := generateRS(oldDeployment2)
	oldPod2 := generatePodFromRS(oldRS2)

	// create 1 ReplicaSet that existed before the deployment, with the same labels as the deployment
	existedPod := generatePod(newDeployment.Spec.Template.Labels, "foo")
	existedRS := generateRSWithLabel(newDeployment.Spec.Template.Labels, "foo")

	tests := []struct {
		test     string
		objs     []runtime.Object
		expected []*extensions.ReplicaSet
	}{
		{
			"No old ReplicaSets",
			[]runtime.Object{
				&api.PodList{
					Items: []api.Pod{
						generatePod(newDeployment.Spec.Template.Labels, "foo"),
						generatePod(newDeployment.Spec.Template.Labels, "bar"),
						newPod,
					},
				},
				&extensions.ReplicaSetList{
					Items: []extensions.ReplicaSet{
						generateRS(generateDeployment("foo")),
						newRS,
						generateRS(generateDeployment("bar")),
					},
				},
			},
			[]*extensions.ReplicaSet{},
		},
		{
			"Has old ReplicaSet",
			[]runtime.Object{
				&api.PodList{
					Items: []api.Pod{
						oldPod,
						oldPod2,
						generatePod(map[string]string{"name": "bar"}, "bar"),
						generatePod(map[string]string{"name": "xyz"}, "xyz"),
						existedPod,
						generatePod(newDeployment.Spec.Template.Labels, "abc"),
					},
				},
				&extensions.ReplicaSetList{
					Items: []extensions.ReplicaSet{
						oldRS2,
						oldRS,
						existedRS,
						newRS,
						generateRSWithLabel(map[string]string{"name": "xyz"}, "xyz"),
						generateRSWithLabel(map[string]string{"name": "bar"}, "bar"),
					},
				},
			},
			[]*extensions.ReplicaSet{&oldRS, &oldRS2, &existedRS},
		},
	}

	for _, test := range tests {
		rss, _, err := GetOldReplicaSets(newDeployment, fake.NewSimpleClientset(test.objs...))
		if err != nil {
			t.Errorf("In test case %s, got unexpected error %v", test.test, err)
		}
		if !equal(rss, test.expected) {
			t.Errorf("In test case %q, expected %v, got %v", test.test, test.expected, rss)
		}
	}
}

// equal compares the equality of two ReplicaSet slices regardless of their ordering
func equal(rss1, rss2 []*extensions.ReplicaSet) bool {
	if reflect.DeepEqual(rss1, rss2) {
		return true
	}
	if rss1 == nil || rss2 == nil || len(rss1) != len(rss2) {
		return false
	}
	count := 0
	for _, rs1 := range rss1 {
		for _, rs2 := range rss2 {
			if reflect.DeepEqual(rs1, rs2) {
				count++
				break
			}
		}
	}
	return count == len(rss1)
}
