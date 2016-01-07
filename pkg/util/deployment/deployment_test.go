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
	"k8s.io/kubernetes/pkg/client/unversioned/testclient"
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

// generatePodFromRC creates a pod, with the input rc's selector and its template
func generatePodFromRC(rc api.ReplicationController) api.Pod {
	return api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels: rc.Spec.Selector,
		},
		Spec: rc.Spec.Template.Spec,
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

func generateRCWithLabel(labels map[string]string, image string) api.ReplicationController {
	return api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:   api.SimpleNameGenerator.GenerateName("rc"),
			Labels: labels,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 1,
			Selector: labels,
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

// generateRC creates a replication controller, with the input deployment's template as its template
func generateRC(deployment extensions.Deployment) api.ReplicationController {
	template := GetNewRCTemplate(deployment)
	return api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:   api.SimpleNameGenerator.GenerateName("rc"),
			Labels: template.Labels,
		},
		Spec: api.ReplicationControllerSpec{
			Template: &template,
			Selector: template.Labels,
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
			Replicas:       1,
			Selector:       podLabels,
			UniqueLabelKey: "deployment.kubernetes.io/podTemplateHash",
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
	newRC := generateRC(newDeployment)

	tests := []struct {
		test     string
		rcList   api.ReplicationControllerList
		expected *api.ReplicationController
	}{
		{
			"No new RC",
			api.ReplicationControllerList{
				Items: []api.ReplicationController{
					generateRC(generateDeployment("foo")),
					generateRC(generateDeployment("bar")),
				},
			},
			nil,
		},
		{
			"Has new RC",
			api.ReplicationControllerList{
				Items: []api.ReplicationController{
					generateRC(generateDeployment("foo")),
					generateRC(generateDeployment("bar")),
					generateRC(generateDeployment("abc")),
					newRC,
					generateRC(generateDeployment("xyz")),
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
				Path:   testapi.Default.ResourcePath("replicationControllers", ns, ""),
			},
			Response: simple.Response{
				StatusCode: 200,
				Body:       &test.rcList,
			},
		}
		rc, err := GetNewRC(newDeployment, c.Setup(t))
		if err != nil {
			t.Errorf("In test case %s, got unexpected error %v", test.test, err)
		}
		if !api.Semantic.DeepEqual(rc, test.expected) {
			t.Errorf("In test case %s, expected %+v, got %+v", test.test, test.expected, rc)
		}
	}
}

func TestGetOldRCs(t *testing.T) {
	newDeployment := generateDeployment("nginx")
	newRC := generateRC(newDeployment)
	newPod := generatePodFromRC(newRC)

	// create 2 old deployments and related rcs/pods, with the same labels but different template
	oldDeployment := generateDeployment("nginx")
	oldDeployment.Spec.Template.Spec.Containers[0].Name = "nginx-old-1"
	oldRC := generateRC(oldDeployment)
	oldPod := generatePodFromRC(oldRC)
	oldDeployment2 := generateDeployment("nginx")
	oldDeployment2.Spec.Template.Spec.Containers[0].Name = "nginx-old-2"
	oldRC2 := generateRC(oldDeployment2)
	oldPod2 := generatePodFromRC(oldRC2)

	// create 1 rc that existed before the deployment, with the same labels as the deployment
	existedPod := generatePod(newDeployment.Spec.Selector, "foo")
	existedRC := generateRCWithLabel(newDeployment.Spec.Selector, "foo")

	tests := []struct {
		test     string
		objs     []runtime.Object
		expected []*api.ReplicationController
	}{
		{
			"No old RCs",
			[]runtime.Object{
				&api.PodList{
					Items: []api.Pod{
						generatePod(newDeployment.Spec.Selector, "foo"),
						generatePod(newDeployment.Spec.Selector, "bar"),
						newPod,
					},
				},
				&api.ReplicationControllerList{
					Items: []api.ReplicationController{
						generateRC(generateDeployment("foo")),
						newRC,
						generateRC(generateDeployment("bar")),
					},
				},
			},
			[]*api.ReplicationController{},
		},
		{
			"Has old RC",
			[]runtime.Object{
				&api.PodList{
					Items: []api.Pod{
						oldPod,
						oldPod2,
						generatePod(map[string]string{"name": "bar"}, "bar"),
						generatePod(map[string]string{"name": "xyz"}, "xyz"),
						existedPod,
						generatePod(newDeployment.Spec.Selector, "abc"),
					},
				},
				&api.ReplicationControllerList{
					Items: []api.ReplicationController{
						oldRC2,
						oldRC,
						existedRC,
						newRC,
						generateRCWithLabel(map[string]string{"name": "xyz"}, "xyz"),
						generateRCWithLabel(map[string]string{"name": "bar"}, "bar"),
					},
				},
			},
			[]*api.ReplicationController{&oldRC, &oldRC2, &existedRC},
		},
	}

	for _, test := range tests {
		rcs, err := GetOldRCs(newDeployment, testclient.NewSimpleFake(test.objs...))
		if err != nil {
			t.Errorf("In test case %s, got unexpected error %v", test.test, err)
		}
		if !equal(rcs, test.expected) {
			t.Errorf("In test case %q, expected %v, got %v", test.test, test.expected, rcs)
		}
	}
}

// equal compares the equality of two rc slices regardless of their ordering
func equal(rcs1, rcs2 []*api.ReplicationController) bool {
	if reflect.DeepEqual(rcs1, rcs2) {
		return true
	}
	if rcs1 == nil || rcs2 == nil || len(rcs1) != len(rcs2) {
		return false
	}
	count := 0
	for _, rc1 := range rcs1 {
		for _, rc2 := range rcs2 {
			if reflect.DeepEqual(rc1, rc2) {
				count++
				break
			}
		}
	}
	return count == len(rcs1)
}
