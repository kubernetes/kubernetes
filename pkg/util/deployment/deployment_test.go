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
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/unversioned/testclient/simple"
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

func generateRC(deployment extensions.Deployment) api.ReplicationController {
	template := GetNewRCTemplate(deployment)
	return api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Labels: template.Labels,
		},
		Spec: api.ReplicationControllerSpec{
			Template: &template,
			Selector: template.Labels,
		},
	}
}

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
