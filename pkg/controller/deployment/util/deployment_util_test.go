/*
Copyright 2015 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"math/rand"
	"reflect"
	"sort"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	"k8s.io/kubernetes/pkg/controller"
)

func addListRSReactor(fakeClient *fake.Clientset, obj runtime.Object) *fake.Clientset {
	fakeClient.AddReactor("list", "replicasets", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, obj, nil
	})
	return fakeClient
}

func addListPodsReactor(fakeClient *fake.Clientset, obj runtime.Object) *fake.Clientset {
	fakeClient.AddReactor("list", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		return true, obj, nil
	})
	return fakeClient
}

func addGetRSReactor(fakeClient *fake.Clientset, obj runtime.Object) *fake.Clientset {
	rsList, ok := obj.(*extensions.ReplicaSetList)
	fakeClient.AddReactor("get", "replicasets", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		name := action.(core.GetAction).GetName()
		if ok {
			for _, rs := range rsList.Items {
				if rs.Name == name {
					return true, &rs, nil
				}
			}
		}
		return false, nil, fmt.Errorf("could not find the requested replica set: %s", name)

	})
	return fakeClient
}

func addUpdateRSReactor(fakeClient *fake.Clientset) *fake.Clientset {
	fakeClient.AddReactor("update", "replicasets", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := action.(core.UpdateAction).GetObject().(*extensions.ReplicaSet)
		return true, obj, nil
	})
	return fakeClient
}

func addUpdatePodsReactor(fakeClient *fake.Clientset) *fake.Clientset {
	fakeClient.AddReactor("update", "pods", func(action core.Action) (handled bool, ret runtime.Object, err error) {
		obj := action.(core.UpdateAction).GetObject().(*v1.Pod)
		return true, obj, nil
	})
	return fakeClient
}

func newPod(now time.Time, ready bool, beforeSec int) v1.Pod {
	conditionStatus := v1.ConditionFalse
	if ready {
		conditionStatus = v1.ConditionTrue
	}
	return v1.Pod{
		Status: v1.PodStatus{
			Conditions: []v1.PodCondition{
				{
					Type:               v1.PodReady,
					LastTransitionTime: metav1.NewTime(now.Add(-1 * time.Duration(beforeSec) * time.Second)),
					Status:             conditionStatus,
				},
			},
		},
	}
}

func newRSControllerRef(rs *extensions.ReplicaSet) *metav1.OwnerReference {
	isController := true
	return &metav1.OwnerReference{
		APIVersion: "extensions/v1beta1",
		Kind:       "ReplicaSet",
		Name:       rs.GetName(),
		UID:        rs.GetUID(),
		Controller: &isController,
	}
}

// generatePodFromRS creates a pod, with the input ReplicaSet's selector and its template
func generatePodFromRS(rs extensions.ReplicaSet) v1.Pod {
	return v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels:          rs.Labels,
			OwnerReferences: []metav1.OwnerReference{*newRSControllerRef(&rs)},
		},
		Spec: rs.Spec.Template.Spec,
	}
}

func generatePod(labels map[string]string, image string) v1.Pod {
	return v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: labels,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:                   image,
					Image:                  image,
					ImagePullPolicy:        v1.PullAlways,
					TerminationMessagePath: v1.TerminationMessagePathDefault,
				},
			},
		},
	}
}

func generateRSWithLabel(labels map[string]string, image string) extensions.ReplicaSet {
	return extensions.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:   v1.SimpleNameGenerator.GenerateName("replicaset"),
			Labels: labels,
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: func(i int32) *int32 { return &i }(1),
			Selector: &metav1.LabelSelector{MatchLabels: labels},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:                   image,
							Image:                  image,
							ImagePullPolicy:        v1.PullAlways,
							TerminationMessagePath: v1.TerminationMessagePathDefault,
						},
					},
				},
			},
		},
	}
}

func newDControllerRef(d *extensions.Deployment) *metav1.OwnerReference {
	isController := true
	return &metav1.OwnerReference{
		APIVersion: "extensions/v1beta1",
		Kind:       "Deployment",
		Name:       d.GetName(),
		UID:        d.GetUID(),
		Controller: &isController,
	}
}

// generateRS creates a replica set, with the input deployment's template as its template
func generateRS(deployment extensions.Deployment) extensions.ReplicaSet {
	template := GetNewReplicaSetTemplate(&deployment)
	return extensions.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			UID:             randomUID(),
			Name:            v1.SimpleNameGenerator.GenerateName("replicaset"),
			Labels:          template.Labels,
			OwnerReferences: []metav1.OwnerReference{*newDControllerRef(&deployment)},
		},
		Spec: extensions.ReplicaSetSpec{
			Replicas: func() *int32 { i := int32(0); return &i }(),
			Template: template,
			Selector: &metav1.LabelSelector{MatchLabels: template.Labels},
		},
	}
}

func randomUID() types.UID {
	return types.UID(strconv.FormatInt(rand.Int63(), 10))
}

// generateDeployment creates a deployment, with the input image as its template
func generateDeployment(image string) extensions.Deployment {
	podLabels := map[string]string{"name": image}
	terminationSec := int64(30)
	return extensions.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:        image,
			Annotations: make(map[string]string),
		},
		Spec: extensions.DeploymentSpec{
			Replicas: func(i int32) *int32 { return &i }(1),
			Selector: &metav1.LabelSelector{MatchLabels: podLabels},
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: podLabels,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:                   image,
							Image:                  image,
							ImagePullPolicy:        v1.PullAlways,
							TerminationMessagePath: v1.TerminationMessagePathDefault,
						},
					},
					DNSPolicy:                     v1.DNSClusterFirst,
					TerminationGracePeriodSeconds: &terminationSec,
					RestartPolicy:                 v1.RestartPolicyAlways,
					SecurityContext:               &v1.PodSecurityContext{},
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
		objs     []runtime.Object
		expected *extensions.ReplicaSet
	}{
		{
			"No new ReplicaSet",
			[]runtime.Object{
				&v1.PodList{},
				&extensions.ReplicaSetList{
					Items: []extensions.ReplicaSet{
						generateRS(generateDeployment("foo")),
						generateRS(generateDeployment("bar")),
					},
				},
			},
			nil,
		},
		{
			"Has new ReplicaSet",
			[]runtime.Object{
				&v1.PodList{},
				&extensions.ReplicaSetList{
					Items: []extensions.ReplicaSet{
						generateRS(generateDeployment("foo")),
						generateRS(generateDeployment("bar")),
						generateRS(generateDeployment("abc")),
						newRC,
						generateRS(generateDeployment("xyz")),
					},
				},
			},
			&newRC,
		},
	}

	for _, test := range tests {
		fakeClient := &fake.Clientset{}
		fakeClient = addListPodsReactor(fakeClient, test.objs[0])
		fakeClient = addListRSReactor(fakeClient, test.objs[1])
		fakeClient = addUpdatePodsReactor(fakeClient)
		fakeClient = addUpdateRSReactor(fakeClient)
		rs, err := GetNewReplicaSet(&newDeployment, fakeClient)
		if err != nil {
			t.Errorf("In test case %s, got unexpected error %v", test.test, err)
		}
		if !apiequality.Semantic.DeepEqual(rs, test.expected) {
			t.Errorf("In test case %s, expected %#v, got %#v", test.test, test.expected, rs)
		}
	}
}

func TestGetOldRCs(t *testing.T) {
	newDeployment := generateDeployment("nginx")
	newRS := generateRS(newDeployment)
	newRS.Status.FullyLabeledReplicas = *(newRS.Spec.Replicas)
	newPod := generatePodFromRS(newRS)

	// create 2 old deployments and related replica sets/pods, with the same labels but different template
	oldDeployment := generateDeployment("nginx")
	oldDeployment.Spec.Template.Spec.Containers[0].Name = "nginx-old-1"
	oldRS := generateRS(oldDeployment)
	oldRS.Status.FullyLabeledReplicas = *(oldRS.Spec.Replicas)
	oldPod := generatePodFromRS(oldRS)
	oldDeployment2 := generateDeployment("nginx")
	oldDeployment2.Spec.Template.Spec.Containers[0].Name = "nginx-old-2"
	oldRS2 := generateRS(oldDeployment2)
	oldRS2.Status.FullyLabeledReplicas = *(oldRS2.Spec.Replicas)
	oldPod2 := generatePodFromRS(oldRS2)

	// create 1 ReplicaSet that existed before the deployment,
	// with the same labels as the deployment, but no ControllerRef.
	existedPod := generatePod(newDeployment.Spec.Template.Labels, "foo")
	existedRS := generateRSWithLabel(newDeployment.Spec.Template.Labels, "foo")
	existedRS.Status.FullyLabeledReplicas = *(existedRS.Spec.Replicas)

	tests := []struct {
		test     string
		objs     []runtime.Object
		expected []*extensions.ReplicaSet
	}{
		{
			"No old ReplicaSets",
			[]runtime.Object{
				&v1.PodList{
					Items: []v1.Pod{
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
				&v1.PodList{
					Items: []v1.Pod{
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
			[]*extensions.ReplicaSet{&oldRS, &oldRS2},
		},
	}

	for _, test := range tests {
		fakeClient := &fake.Clientset{}
		fakeClient = addListPodsReactor(fakeClient, test.objs[0])
		fakeClient = addListRSReactor(fakeClient, test.objs[1])
		fakeClient = addGetRSReactor(fakeClient, test.objs[1])
		fakeClient = addUpdatePodsReactor(fakeClient)
		fakeClient = addUpdateRSReactor(fakeClient)
		rss, _, err := GetOldReplicaSets(&newDeployment, fakeClient)
		if err != nil {
			t.Errorf("In test case %s, got unexpected error %v", test.test, err)
		}
		if !equal(rss, test.expected) {
			t.Errorf("In test case %q, expected:", test.test)
			for _, rs := range test.expected {
				t.Errorf("rs = %#v", rs)
			}
			t.Errorf("In test case %q, got:", test.test)
			for _, rs := range rss {
				t.Errorf("rs = %#v", rs)
			}
		}
	}
}

func generatePodTemplateSpec(name, nodeName string, annotations, labels map[string]string) v1.PodTemplateSpec {
	return v1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Annotations: annotations,
			Labels:      labels,
		},
		Spec: v1.PodSpec{
			NodeName: nodeName,
		},
	}
}

func TestEqualIgnoreHash(t *testing.T) {
	tests := []struct {
		test           string
		former, latter v1.PodTemplateSpec
		expected       bool
	}{
		{
			"Same spec, same labels",
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{extensions.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{extensions.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			true,
		},
		{
			"Same spec, only pod-template-hash label value is different",
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{extensions.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{extensions.DefaultDeploymentUniqueLabelKey: "value-2", "something": "else"}),
			true,
		},
		{
			"Same spec, the former doesn't have pod-template-hash label",
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{"something": "else"}),
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{extensions.DefaultDeploymentUniqueLabelKey: "value-2", "something": "else"}),
			true,
		},
		{
			"Same spec, the label is different, and the pod-template-hash label value is the same",
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{extensions.DefaultDeploymentUniqueLabelKey: "value-1"}),
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{extensions.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			false,
		},
		{
			"Different spec, same labels",
			generatePodTemplateSpec("foo", "foo-node", map[string]string{"former": "value"}, map[string]string{extensions.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			generatePodTemplateSpec("foo", "foo-node", map[string]string{"latter": "value"}, map[string]string{extensions.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			false,
		},
		{
			"Different spec, different pod-template-hash label value",
			generatePodTemplateSpec("foo-1", "foo-node", map[string]string{}, map[string]string{extensions.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			generatePodTemplateSpec("foo-2", "foo-node", map[string]string{}, map[string]string{extensions.DefaultDeploymentUniqueLabelKey: "value-2", "something": "else"}),
			false,
		},
		{
			"Different spec, the former doesn't have pod-template-hash label",
			generatePodTemplateSpec("foo-1", "foo-node-1", map[string]string{}, map[string]string{"something": "else"}),
			generatePodTemplateSpec("foo-2", "foo-node-2", map[string]string{}, map[string]string{extensions.DefaultDeploymentUniqueLabelKey: "value-2", "something": "else"}),
			false,
		},
		{
			"Different spec, different labels",
			generatePodTemplateSpec("foo", "foo-node-1", map[string]string{}, map[string]string{"something": "else"}),
			generatePodTemplateSpec("foo", "foo-node-2", map[string]string{}, map[string]string{"nothing": "else"}),
			false,
		},
	}

	for _, test := range tests {
		runTest := func(t1, t2 *v1.PodTemplateSpec, reversed bool) {
			// Set up
			t1Copy, err := api.Scheme.DeepCopy(t1)
			if err != nil {
				t.Errorf("Failed setting up the test: %v", err)
			}
			t2Copy, err := api.Scheme.DeepCopy(t2)
			if err != nil {
				t.Errorf("Failed setting up the test: %v", err)
			}
			reverseString := ""
			if reversed {
				reverseString = " (reverse order)"
			}
			// Run
			equal := EqualIgnoreHash(*t1, *t2)
			if equal != test.expected {
				t.Errorf("In test case %q%s, expected %v", test.test, reverseString, test.expected)
			}
			if t1.Labels == nil || t2.Labels == nil {
				t.Errorf("In test case %q%s, unexpected labels becomes nil", test.test, reverseString)
			}
			if !reflect.DeepEqual(t1, t1Copy) || !reflect.DeepEqual(t2, t2Copy) {
				t.Errorf("In test case %q%s, unexpected input template modified", test.test, reverseString)
			}
		}
		runTest(&test.former, &test.latter, false)
		// Test the same case in reverse order
		runTest(&test.latter, &test.former, true)
	}
}

func TestFindNewReplicaSet(t *testing.T) {
	now := metav1.Now()
	later := metav1.Time{Time: now.Add(time.Minute)}

	deployment := generateDeployment("nginx")
	newRS := generateRS(deployment)
	newRS.Labels[extensions.DefaultDeploymentUniqueLabelKey] = "hash"
	newRS.CreationTimestamp = later

	newRSDup := generateRS(deployment)
	newRSDup.Labels[extensions.DefaultDeploymentUniqueLabelKey] = "different-hash"
	newRSDup.CreationTimestamp = now

	oldDeployment := generateDeployment("nginx")
	oldDeployment.Spec.Template.Spec.Containers[0].Name = "nginx-old-1"
	oldRS := generateRS(oldDeployment)
	oldRS.Status.FullyLabeledReplicas = *(oldRS.Spec.Replicas)

	tests := []struct {
		test       string
		deployment extensions.Deployment
		rsList     []*extensions.ReplicaSet
		expected   *extensions.ReplicaSet
	}{
		{
			test:       "Get new ReplicaSet with the same template as Deployment spec but different pod-template-hash value",
			deployment: deployment,
			rsList:     []*extensions.ReplicaSet{&newRS, &oldRS},
			expected:   &newRS,
		},
		{
			test:       "Get the oldest new ReplicaSet when there are more than one ReplicaSet with the same template",
			deployment: deployment,
			rsList:     []*extensions.ReplicaSet{&newRS, &oldRS, &newRSDup},
			expected:   &newRSDup,
		},
		{
			test:       "Get nil new ReplicaSet",
			deployment: deployment,
			rsList:     []*extensions.ReplicaSet{&oldRS},
			expected:   nil,
		},
	}

	for _, test := range tests {
		if rs, err := FindNewReplicaSet(&test.deployment, test.rsList); !reflect.DeepEqual(rs, test.expected) || err != nil {
			t.Errorf("In test case %q, expected %#v, got %#v: %v", test.test, test.expected, rs, err)
		}
	}
}

func TestFindOldReplicaSets(t *testing.T) {
	now := metav1.Now()
	later := metav1.Time{Time: now.Add(time.Minute)}
	before := metav1.Time{Time: now.Add(-time.Minute)}

	deployment := generateDeployment("nginx")
	newRS := generateRS(deployment)
	newRS.Labels[extensions.DefaultDeploymentUniqueLabelKey] = "hash"
	newRS.CreationTimestamp = later

	newRSDup := generateRS(deployment)
	newRSDup.Labels[extensions.DefaultDeploymentUniqueLabelKey] = "different-hash"
	newRSDup.CreationTimestamp = now

	oldDeployment := generateDeployment("nginx")
	oldDeployment.Spec.Template.Spec.Containers[0].Name = "nginx-old-1"
	oldRS := generateRS(oldDeployment)
	oldRS.Status.FullyLabeledReplicas = *(oldRS.Spec.Replicas)
	oldRS.CreationTimestamp = before

	newPod := generatePodFromRS(newRS)
	oldPod := generatePodFromRS(oldRS)

	tests := []struct {
		test       string
		deployment extensions.Deployment
		rsList     []*extensions.ReplicaSet
		podList    *v1.PodList
		expected   []*extensions.ReplicaSet
	}{
		{
			test:       "Get old ReplicaSets",
			deployment: deployment,
			rsList:     []*extensions.ReplicaSet{&newRS, &oldRS},
			podList: &v1.PodList{
				Items: []v1.Pod{
					newPod,
					oldPod,
				},
			},
			expected: []*extensions.ReplicaSet{&oldRS},
		},
		{
			test:       "Get old ReplicaSets with no new ReplicaSet",
			deployment: deployment,
			rsList:     []*extensions.ReplicaSet{&oldRS},
			podList: &v1.PodList{
				Items: []v1.Pod{
					oldPod,
				},
			},
			expected: []*extensions.ReplicaSet{&oldRS},
		},
		{
			test:       "Get old ReplicaSets with two new ReplicaSets, only the oldest new ReplicaSet is seen as new ReplicaSet",
			deployment: deployment,
			rsList:     []*extensions.ReplicaSet{&oldRS, &newRS, &newRSDup},
			podList: &v1.PodList{
				Items: []v1.Pod{
					newPod,
					oldPod,
				},
			},
			expected: []*extensions.ReplicaSet{&oldRS, &newRS},
		},
		{
			test:       "Get empty old ReplicaSets",
			deployment: deployment,
			rsList:     []*extensions.ReplicaSet{&newRS},
			podList: &v1.PodList{
				Items: []v1.Pod{
					newPod,
				},
			},
			expected: []*extensions.ReplicaSet{},
		},
	}

	for _, test := range tests {
		old, _, err := FindOldReplicaSets(&test.deployment, test.rsList, test.podList)
		sort.Sort(controller.ReplicaSetsByCreationTimestamp(old))
		sort.Sort(controller.ReplicaSetsByCreationTimestamp(test.expected))
		if !reflect.DeepEqual(old, test.expected) || err != nil {
			t.Errorf("In test case %q, expected %#v, got %#v: %v", test.test, test.expected, old, err)
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

func TestGetReplicaCountForReplicaSets(t *testing.T) {
	rs1 := generateRS(generateDeployment("foo"))
	*(rs1.Spec.Replicas) = 1
	rs1.Status.Replicas = 2
	rs2 := generateRS(generateDeployment("bar"))
	*(rs2.Spec.Replicas) = 2
	rs2.Status.Replicas = 3

	tests := []struct {
		test           string
		sets           []*extensions.ReplicaSet
		expectedCount  int32
		expectedActual int32
	}{
		{
			"1:2 Replicas",
			[]*extensions.ReplicaSet{&rs1},
			1,
			2,
		},
		{
			"3:5 Replicas",
			[]*extensions.ReplicaSet{&rs1, &rs2},
			3,
			5,
		},
	}

	for _, test := range tests {
		rs := GetReplicaCountForReplicaSets(test.sets)
		if rs != test.expectedCount {
			t.Errorf("In test case %s, expectedCount %+v, got %+v", test.test, test.expectedCount, rs)
		}
		rs = GetActualReplicaCountForReplicaSets(test.sets)
		if rs != test.expectedActual {
			t.Errorf("In test case %s, expectedActual %+v, got %+v", test.test, test.expectedActual, rs)
		}
	}
}

func TestResolveFenceposts(t *testing.T) {
	tests := []struct {
		maxSurge          string
		maxUnavailable    string
		desired           int32
		expectSurge       int32
		expectUnavailable int32
		expectError       string
	}{
		{
			maxSurge:          "0%",
			maxUnavailable:    "0%",
			desired:           0,
			expectSurge:       0,
			expectUnavailable: 1,
			expectError:       "",
		},
		{
			maxSurge:          "39%",
			maxUnavailable:    "39%",
			desired:           10,
			expectSurge:       4,
			expectUnavailable: 3,
			expectError:       "",
		},
		{
			maxSurge:          "oops",
			maxUnavailable:    "39%",
			desired:           10,
			expectSurge:       0,
			expectUnavailable: 0,
			expectError:       "invalid value for IntOrString: invalid value \"oops\": strconv.ParseInt: parsing \"oops\": invalid syntax",
		},
		{
			maxSurge:          "55%",
			maxUnavailable:    "urg",
			desired:           10,
			expectSurge:       0,
			expectUnavailable: 0,
			expectError:       "invalid value for IntOrString: invalid value \"urg\": strconv.ParseInt: parsing \"urg\": invalid syntax",
		},
	}

	for num, test := range tests {
		maxSurge := intstr.FromString(test.maxSurge)
		maxUnavail := intstr.FromString(test.maxUnavailable)
		surge, unavail, err := ResolveFenceposts(&maxSurge, &maxUnavail, test.desired)
		if err != nil {
			if test.expectError == "" {
				t.Errorf("unexpected error %v", err)
			} else {
				assert := assert.New(t)
				assert.EqualError(err, test.expectError)
			}
		}
		if err == nil && test.expectError != "" {
			t.Errorf("missing error %v", test.expectError)
		}
		if surge != test.expectSurge || unavail != test.expectUnavailable {
			t.Errorf("#%v got %v:%v, want %v:%v", num, surge, unavail, test.expectSurge, test.expectUnavailable)
		}
	}
}

func TestNewRSNewReplicas(t *testing.T) {
	tests := []struct {
		test          string
		strategyType  extensions.DeploymentStrategyType
		depReplicas   int32
		newRSReplicas int32
		maxSurge      int
		expected      int32
	}{
		{
			"can not scale up - to newRSReplicas",
			extensions.RollingUpdateDeploymentStrategyType,
			1, 5, 1, 5,
		},
		{
			"scale up - to depReplicas",
			extensions.RollingUpdateDeploymentStrategyType,
			6, 2, 10, 6,
		},
		{
			"recreate - to depReplicas",
			extensions.RecreateDeploymentStrategyType,
			3, 1, 1, 3,
		},
	}
	newDeployment := generateDeployment("nginx")
	newRC := generateRS(newDeployment)
	rs5 := generateRS(newDeployment)
	*(rs5.Spec.Replicas) = 5

	for _, test := range tests {
		*(newDeployment.Spec.Replicas) = test.depReplicas
		newDeployment.Spec.Strategy = extensions.DeploymentStrategy{Type: test.strategyType}
		newDeployment.Spec.Strategy.RollingUpdate = &extensions.RollingUpdateDeployment{
			MaxUnavailable: func(i int) *intstr.IntOrString { x := intstr.FromInt(i); return &x }(1),
			MaxSurge:       func(i int) *intstr.IntOrString { x := intstr.FromInt(i); return &x }(test.maxSurge),
		}
		*(newRC.Spec.Replicas) = test.newRSReplicas
		rs, err := NewRSNewReplicas(&newDeployment, []*extensions.ReplicaSet{&rs5}, &newRC)
		if err != nil {
			t.Errorf("In test case %s, got unexpected error %v", test.test, err)
		}
		if rs != test.expected {
			t.Errorf("In test case %s, expected %+v, got %+v", test.test, test.expected, rs)
		}
	}
}

var (
	condProgressing = func() extensions.DeploymentCondition {
		return extensions.DeploymentCondition{
			Type:   extensions.DeploymentProgressing,
			Status: v1.ConditionFalse,
			Reason: "ForSomeReason",
		}
	}

	condProgressing2 = func() extensions.DeploymentCondition {
		return extensions.DeploymentCondition{
			Type:   extensions.DeploymentProgressing,
			Status: v1.ConditionTrue,
			Reason: "BecauseItIs",
		}
	}

	condAvailable = func() extensions.DeploymentCondition {
		return extensions.DeploymentCondition{
			Type:   extensions.DeploymentAvailable,
			Status: v1.ConditionTrue,
			Reason: "AwesomeController",
		}
	}

	status = func() *extensions.DeploymentStatus {
		return &extensions.DeploymentStatus{
			Conditions: []extensions.DeploymentCondition{condProgressing(), condAvailable()},
		}
	}
)

func TestGetCondition(t *testing.T) {
	exampleStatus := status()

	tests := []struct {
		name string

		status     extensions.DeploymentStatus
		condType   extensions.DeploymentConditionType
		condStatus v1.ConditionStatus
		condReason string

		expected bool
	}{
		{
			name: "condition exists",

			status:   *exampleStatus,
			condType: extensions.DeploymentAvailable,

			expected: true,
		},
		{
			name: "condition does not exist",

			status:   *exampleStatus,
			condType: extensions.DeploymentReplicaFailure,

			expected: false,
		},
	}

	for _, test := range tests {
		cond := GetDeploymentCondition(test.status, test.condType)
		exists := cond != nil
		if exists != test.expected {
			t.Errorf("%s: expected condition to exist: %t, got: %t", test.name, test.expected, exists)
		}
	}
}

func TestSetCondition(t *testing.T) {
	tests := []struct {
		name string

		status *extensions.DeploymentStatus
		cond   extensions.DeploymentCondition

		expectedStatus *extensions.DeploymentStatus
	}{
		{
			name: "set for the first time",

			status: &extensions.DeploymentStatus{},
			cond:   condAvailable(),

			expectedStatus: &extensions.DeploymentStatus{Conditions: []extensions.DeploymentCondition{condAvailable()}},
		},
		{
			name: "simple set",

			status: &extensions.DeploymentStatus{Conditions: []extensions.DeploymentCondition{condProgressing()}},
			cond:   condAvailable(),

			expectedStatus: status(),
		},
		{
			name: "overwrite",

			status: &extensions.DeploymentStatus{Conditions: []extensions.DeploymentCondition{condProgressing()}},
			cond:   condProgressing2(),

			expectedStatus: &extensions.DeploymentStatus{Conditions: []extensions.DeploymentCondition{condProgressing2()}},
		},
	}

	for _, test := range tests {
		SetDeploymentCondition(test.status, test.cond)
		if !reflect.DeepEqual(test.status, test.expectedStatus) {
			t.Errorf("%s: expected status: %v, got: %v", test.name, test.expectedStatus, test.status)
		}
	}
}

func TestRemoveCondition(t *testing.T) {
	tests := []struct {
		name string

		status   *extensions.DeploymentStatus
		condType extensions.DeploymentConditionType

		expectedStatus *extensions.DeploymentStatus
	}{
		{
			name: "remove from empty status",

			status:   &extensions.DeploymentStatus{},
			condType: extensions.DeploymentProgressing,

			expectedStatus: &extensions.DeploymentStatus{},
		},
		{
			name: "simple remove",

			status:   &extensions.DeploymentStatus{Conditions: []extensions.DeploymentCondition{condProgressing()}},
			condType: extensions.DeploymentProgressing,

			expectedStatus: &extensions.DeploymentStatus{},
		},
		{
			name: "doesn't remove anything",

			status:   status(),
			condType: extensions.DeploymentReplicaFailure,

			expectedStatus: status(),
		},
	}

	for _, test := range tests {
		RemoveDeploymentCondition(test.status, test.condType)
		if !reflect.DeepEqual(test.status, test.expectedStatus) {
			t.Errorf("%s: expected status: %v, got: %v", test.name, test.expectedStatus, test.status)
		}
	}
}

func TestDeploymentComplete(t *testing.T) {
	deployment := func(desired, current, updated, available, maxUnavailable, maxSurge int32) *extensions.Deployment {
		return &extensions.Deployment{
			Spec: extensions.DeploymentSpec{
				Replicas: &desired,
				Strategy: extensions.DeploymentStrategy{
					RollingUpdate: &extensions.RollingUpdateDeployment{
						MaxUnavailable: func(i int) *intstr.IntOrString { x := intstr.FromInt(i); return &x }(int(maxUnavailable)),
						MaxSurge:       func(i int) *intstr.IntOrString { x := intstr.FromInt(i); return &x }(int(maxSurge)),
					},
					Type: extensions.RollingUpdateDeploymentStrategyType,
				},
			},
			Status: extensions.DeploymentStatus{
				Replicas:          current,
				UpdatedReplicas:   updated,
				AvailableReplicas: available,
			},
		}
	}

	tests := []struct {
		name string

		d *extensions.Deployment

		expected bool
	}{
		{
			name: "complete",

			d:        deployment(5, 5, 5, 4, 1, 0),
			expected: true,
		},
		{
			name: "not complete",

			d:        deployment(5, 5, 5, 3, 1, 0),
			expected: false,
		},
		{
			name: "complete #2",

			d:        deployment(5, 5, 5, 5, 0, 0),
			expected: true,
		},
		{
			name: "not complete #2",

			d:        deployment(5, 5, 4, 5, 0, 0),
			expected: false,
		},
		{
			name: "not complete #3",

			// old replica set: spec.replicas=1, status.replicas=1, status.availableReplicas=1
			// new replica set: spec.replicas=1, status.replicas=1, status.availableReplicas=0
			d:        deployment(1, 2, 1, 1, 0, 1),
			expected: false,
		},
	}

	for _, test := range tests {
		t.Log(test.name)

		if got, exp := DeploymentComplete(test.d, &test.d.Status), test.expected; got != exp {
			t.Errorf("expected complete: %t, got: %t", exp, got)
		}
	}
}

func TestDeploymentProgressing(t *testing.T) {
	deployment := func(current, updated int32) *extensions.Deployment {
		return &extensions.Deployment{
			Status: extensions.DeploymentStatus{
				Replicas:        current,
				UpdatedReplicas: updated,
			},
		}
	}
	newStatus := func(current, updated int32) extensions.DeploymentStatus {
		return extensions.DeploymentStatus{
			Replicas:        current,
			UpdatedReplicas: updated,
		}
	}

	tests := []struct {
		name string

		d         *extensions.Deployment
		newStatus extensions.DeploymentStatus

		expected bool
	}{
		{
			name: "progressing",

			d:         deployment(10, 4),
			newStatus: newStatus(10, 6),

			expected: true,
		},
		{
			name: "not progressing",

			d:         deployment(10, 4),
			newStatus: newStatus(10, 4),

			expected: false,
		},
		{
			name: "progressing #2",

			d:         deployment(10, 4),
			newStatus: newStatus(8, 4),

			expected: true,
		},
		{
			name: "not progressing #2",

			d:         deployment(10, 7),
			newStatus: newStatus(10, 6),

			expected: false,
		},
		{
			name: "progressing #3",

			d:         deployment(10, 4),
			newStatus: newStatus(8, 8),

			expected: true,
		},
		{
			name: "not progressing #2",

			d:         deployment(10, 7),
			newStatus: newStatus(10, 7),

			expected: false,
		},
	}

	for _, test := range tests {
		t.Log(test.name)

		if got, exp := DeploymentProgressing(test.d, &test.newStatus), test.expected; got != exp {
			t.Errorf("expected progressing: %t, got: %t", exp, got)
		}
	}
}

func TestDeploymentTimedOut(t *testing.T) {
	var (
		null *int32
		ten  = int32(10)
	)

	timeFn := func(min, sec int) time.Time {
		return time.Date(2016, 1, 1, 0, min, sec, 0, time.UTC)
	}
	deployment := func(condType extensions.DeploymentConditionType, status v1.ConditionStatus, pds *int32, from time.Time) extensions.Deployment {
		return extensions.Deployment{
			Spec: extensions.DeploymentSpec{
				ProgressDeadlineSeconds: pds,
			},
			Status: extensions.DeploymentStatus{
				Conditions: []extensions.DeploymentCondition{
					{
						Type:           condType,
						Status:         status,
						LastUpdateTime: metav1.Time{Time: from},
					},
				},
			},
		}
	}

	tests := []struct {
		name string

		d     extensions.Deployment
		nowFn func() time.Time

		expected bool
	}{
		{
			name: "no progressDeadlineSeconds specified - no timeout",

			d:        deployment(extensions.DeploymentProgressing, v1.ConditionTrue, null, timeFn(1, 9)),
			nowFn:    func() time.Time { return timeFn(1, 20) },
			expected: false,
		},
		{
			name: "progressDeadlineSeconds: 10s, now - started => 00:01:20 - 00:01:09 => 11s",

			d:        deployment(extensions.DeploymentProgressing, v1.ConditionTrue, &ten, timeFn(1, 9)),
			nowFn:    func() time.Time { return timeFn(1, 20) },
			expected: true,
		},
		{
			name: "progressDeadlineSeconds: 10s, now - started => 00:01:20 - 00:01:11 => 9s",

			d:        deployment(extensions.DeploymentProgressing, v1.ConditionTrue, &ten, timeFn(1, 11)),
			nowFn:    func() time.Time { return timeFn(1, 20) },
			expected: false,
		},
	}

	for _, test := range tests {
		t.Log(test.name)

		nowFn = test.nowFn
		if got, exp := DeploymentTimedOut(&test.d, &test.d.Status), test.expected; got != exp {
			t.Errorf("expected timeout: %t, got: %t", exp, got)
		}
	}
}
