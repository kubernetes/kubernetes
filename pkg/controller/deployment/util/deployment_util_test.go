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
	"math"
	"math/rand"
	"reflect"
	"sort"
	"strconv"
	"testing"
	"time"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
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
	rsList, ok := obj.(*apps.ReplicaSetList)
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
		obj := action.(core.UpdateAction).GetObject().(*apps.ReplicaSet)
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

func generateRSWithLabel(labels map[string]string, image string) apps.ReplicaSet {
	return apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:   names.SimpleNameGenerator.GenerateName("replicaset"),
			Labels: labels,
		},
		Spec: apps.ReplicaSetSpec{
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

func newDControllerRef(d *apps.Deployment) *metav1.OwnerReference {
	isController := true
	return &metav1.OwnerReference{
		APIVersion: "apps/v1",
		Kind:       "Deployment",
		Name:       d.GetName(),
		UID:        d.GetUID(),
		Controller: &isController,
	}
}

// generateRS creates a replica set, with the input deployment's template as its template
func generateRS(deployment apps.Deployment) apps.ReplicaSet {
	template := deployment.Spec.Template.DeepCopy()
	return apps.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			UID:             randomUID(),
			Name:            names.SimpleNameGenerator.GenerateName("replicaset"),
			Labels:          template.Labels,
			OwnerReferences: []metav1.OwnerReference{*newDControllerRef(&deployment)},
		},
		Spec: apps.ReplicaSetSpec{
			Replicas: new(int32),
			Template: *template,
			Selector: &metav1.LabelSelector{MatchLabels: template.Labels},
		},
	}
}

func randomUID() types.UID {
	return types.UID(strconv.FormatInt(rand.Int63(), 10))
}

// generateDeployment creates a deployment, with the input image as its template
func generateDeployment(image string) apps.Deployment {
	podLabels := map[string]string{"name": image}
	terminationSec := int64(30)
	enableServiceLinks := v1.DefaultEnableServiceLinks
	return apps.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:        image,
			Annotations: make(map[string]string),
		},
		Spec: apps.DeploymentSpec{
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
					EnableServiceLinks:            &enableServiceLinks,
				},
			},
		},
	}
}

func TestGetNewRS(t *testing.T) {
	newDeployment := generateDeployment("nginx")
	newRC := generateRS(newDeployment)

	tests := []struct {
		Name     string
		objs     []runtime.Object
		expected *apps.ReplicaSet
	}{
		{
			"No new ReplicaSet",
			[]runtime.Object{
				&v1.PodList{},
				&apps.ReplicaSetList{
					Items: []apps.ReplicaSet{
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
				&apps.ReplicaSetList{
					Items: []apps.ReplicaSet{
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
		t.Run(test.Name, func(t *testing.T) {
			fakeClient := &fake.Clientset{}
			fakeClient = addListPodsReactor(fakeClient, test.objs[0])
			fakeClient = addListRSReactor(fakeClient, test.objs[1])
			fakeClient = addUpdatePodsReactor(fakeClient)
			fakeClient = addUpdateRSReactor(fakeClient)
			rs, err := GetNewReplicaSet(&newDeployment, fakeClient.AppsV1())
			if err != nil {
				t.Errorf("In test case %s, got unexpected error %v", test.Name, err)
			}
			if !apiequality.Semantic.DeepEqual(rs, test.expected) {
				t.Errorf("In test case %s, expected %#v, got %#v", test.Name, test.expected, rs)
			}
		})
	}
}

func TestGetOldRSs(t *testing.T) {
	newDeployment := generateDeployment("nginx")
	newRS := generateRS(newDeployment)
	newRS.Status.FullyLabeledReplicas = *(newRS.Spec.Replicas)

	// create 2 old deployments and related replica sets/pods, with the same labels but different template
	oldDeployment := generateDeployment("nginx")
	oldDeployment.Spec.Template.Spec.Containers[0].Name = "nginx-old-1"
	oldRS := generateRS(oldDeployment)
	oldRS.Status.FullyLabeledReplicas = *(oldRS.Spec.Replicas)
	oldDeployment2 := generateDeployment("nginx")
	oldDeployment2.Spec.Template.Spec.Containers[0].Name = "nginx-old-2"
	oldRS2 := generateRS(oldDeployment2)
	oldRS2.Status.FullyLabeledReplicas = *(oldRS2.Spec.Replicas)

	// create 1 ReplicaSet that existed before the deployment,
	// with the same labels as the deployment, but no ControllerRef.
	existedRS := generateRSWithLabel(newDeployment.Spec.Template.Labels, "foo")
	existedRS.Status.FullyLabeledReplicas = *(existedRS.Spec.Replicas)

	tests := []struct {
		Name     string
		objs     []runtime.Object
		expected []*apps.ReplicaSet
	}{
		{
			"No old ReplicaSets",
			[]runtime.Object{
				&apps.ReplicaSetList{
					Items: []apps.ReplicaSet{
						generateRS(generateDeployment("foo")),
						newRS,
						generateRS(generateDeployment("bar")),
					},
				},
			},
			nil,
		},
		{
			"Has old ReplicaSet",
			[]runtime.Object{
				&apps.ReplicaSetList{
					Items: []apps.ReplicaSet{
						oldRS2,
						oldRS,
						existedRS,
						newRS,
						generateRSWithLabel(map[string]string{"name": "xyz"}, "xyz"),
						generateRSWithLabel(map[string]string{"name": "bar"}, "bar"),
					},
				},
			},
			[]*apps.ReplicaSet{&oldRS, &oldRS2},
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			fakeClient := &fake.Clientset{}
			fakeClient = addListRSReactor(fakeClient, test.objs[0])
			fakeClient = addGetRSReactor(fakeClient, test.objs[0])
			fakeClient = addUpdateRSReactor(fakeClient)
			_, rss, err := GetOldReplicaSets(&newDeployment, fakeClient.AppsV1())
			if err != nil {
				t.Errorf("In test case %s, got unexpected error %v", test.Name, err)
			}
			if !equal(rss, test.expected) {
				t.Errorf("In test case %q, expected:", test.Name)
				for _, rs := range test.expected {
					t.Errorf("rs = %#v", rs)
				}
				t.Errorf("In test case %q, got:", test.Name)
				for _, rs := range rss {
					t.Errorf("rs = %#v", rs)
				}
			}
		})
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
		Name           string
		former, latter v1.PodTemplateSpec
		expected       bool
	}{
		{
			"Same spec, same labels",
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			true,
		},
		{
			"Same spec, only pod-template-hash label value is different",
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-2", "something": "else"}),
			true,
		},
		{
			"Same spec, the former doesn't have pod-template-hash label",
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{"something": "else"}),
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-2", "something": "else"}),
			true,
		},
		{
			"Same spec, the label is different, the former doesn't have pod-template-hash label, same number of labels",
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{"something": "else"}),
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-2"}),
			false,
		},
		{
			"Same spec, the label is different, the latter doesn't have pod-template-hash label, same number of labels",
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-1"}),
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{"something": "else"}),
			false,
		},
		{
			"Same spec, the label is different, and the pod-template-hash label value is the same",
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-1"}),
			generatePodTemplateSpec("foo", "foo-node", map[string]string{}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			false,
		},
		{
			"Different spec, same labels",
			generatePodTemplateSpec("foo", "foo-node", map[string]string{"former": "value"}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			generatePodTemplateSpec("foo", "foo-node", map[string]string{"latter": "value"}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			false,
		},
		{
			"Different spec, different pod-template-hash label value",
			generatePodTemplateSpec("foo-1", "foo-node", map[string]string{}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-1", "something": "else"}),
			generatePodTemplateSpec("foo-2", "foo-node", map[string]string{}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-2", "something": "else"}),
			false,
		},
		{
			"Different spec, the former doesn't have pod-template-hash label",
			generatePodTemplateSpec("foo-1", "foo-node-1", map[string]string{}, map[string]string{"something": "else"}),
			generatePodTemplateSpec("foo-2", "foo-node-2", map[string]string{}, map[string]string{apps.DefaultDeploymentUniqueLabelKey: "value-2", "something": "else"}),
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
		t.Run(test.Name, func(t *testing.T) {
			runTest := func(t1, t2 *v1.PodTemplateSpec, reversed bool) {
				reverseString := ""
				if reversed {
					reverseString = " (reverse order)"
				}
				// Run
				equal := EqualIgnoreHash(t1, t2)
				if equal != test.expected {
					t.Errorf("%q%s: expected %v", test.Name, reverseString, test.expected)
					return
				}
				if t1.Labels == nil || t2.Labels == nil {
					t.Errorf("%q%s: unexpected labels becomes nil", test.Name, reverseString)
				}
			}

			runTest(&test.former, &test.latter, false)
			// Test the same case in reverse order
			runTest(&test.latter, &test.former, true)
		})
	}
}

func TestFindNewReplicaSet(t *testing.T) {
	now := metav1.Now()
	later := metav1.Time{Time: now.Add(time.Minute)}

	deployment := generateDeployment("nginx")
	newRS := generateRS(deployment)
	newRS.Labels[apps.DefaultDeploymentUniqueLabelKey] = "hash"
	newRS.CreationTimestamp = later

	newRSDup := generateRS(deployment)
	newRSDup.Labels[apps.DefaultDeploymentUniqueLabelKey] = "different-hash"
	newRSDup.CreationTimestamp = now

	oldDeployment := generateDeployment("nginx")
	oldDeployment.Spec.Template.Spec.Containers[0].Name = "nginx-old-1"
	oldRS := generateRS(oldDeployment)
	oldRS.Status.FullyLabeledReplicas = *(oldRS.Spec.Replicas)

	tests := []struct {
		Name       string
		deployment apps.Deployment
		rsList     []*apps.ReplicaSet
		expected   *apps.ReplicaSet
	}{
		{
			Name:       "Get new ReplicaSet with the same template as Deployment spec but different pod-template-hash value",
			deployment: deployment,
			rsList:     []*apps.ReplicaSet{&newRS, &oldRS},
			expected:   &newRS,
		},
		{
			Name:       "Get the oldest new ReplicaSet when there are more than one ReplicaSet with the same template",
			deployment: deployment,
			rsList:     []*apps.ReplicaSet{&newRS, &oldRS, &newRSDup},
			expected:   &newRSDup,
		},
		{
			Name:       "Get nil new ReplicaSet",
			deployment: deployment,
			rsList:     []*apps.ReplicaSet{&oldRS},
			expected:   nil,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			if rs := FindNewReplicaSet(&test.deployment, test.rsList); !reflect.DeepEqual(rs, test.expected) {
				t.Errorf("In test case %q, expected %#v, got %#v", test.Name, test.expected, rs)
			}
		})
	}
}

func TestFindOldReplicaSets(t *testing.T) {
	now := metav1.Now()
	later := metav1.Time{Time: now.Add(time.Minute)}
	before := metav1.Time{Time: now.Add(-time.Minute)}

	deployment := generateDeployment("nginx")
	newRS := generateRS(deployment)
	*(newRS.Spec.Replicas) = 1
	newRS.Labels[apps.DefaultDeploymentUniqueLabelKey] = "hash"
	newRS.CreationTimestamp = later

	newRSDup := generateRS(deployment)
	newRSDup.Labels[apps.DefaultDeploymentUniqueLabelKey] = "different-hash"
	newRSDup.CreationTimestamp = now

	oldDeployment := generateDeployment("nginx")
	oldDeployment.Spec.Template.Spec.Containers[0].Name = "nginx-old-1"
	oldRS := generateRS(oldDeployment)
	oldRS.Status.FullyLabeledReplicas = *(oldRS.Spec.Replicas)
	oldRS.CreationTimestamp = before

	tests := []struct {
		Name            string
		deployment      apps.Deployment
		rsList          []*apps.ReplicaSet
		podList         *v1.PodList
		expected        []*apps.ReplicaSet
		expectedRequire []*apps.ReplicaSet
	}{
		{
			Name:            "Get old ReplicaSets",
			deployment:      deployment,
			rsList:          []*apps.ReplicaSet{&newRS, &oldRS},
			expected:        []*apps.ReplicaSet{&oldRS},
			expectedRequire: nil,
		},
		{
			Name:            "Get old ReplicaSets with no new ReplicaSet",
			deployment:      deployment,
			rsList:          []*apps.ReplicaSet{&oldRS},
			expected:        []*apps.ReplicaSet{&oldRS},
			expectedRequire: nil,
		},
		{
			Name:            "Get old ReplicaSets with two new ReplicaSets, only the oldest new ReplicaSet is seen as new ReplicaSet",
			deployment:      deployment,
			rsList:          []*apps.ReplicaSet{&oldRS, &newRS, &newRSDup},
			expected:        []*apps.ReplicaSet{&oldRS, &newRS},
			expectedRequire: []*apps.ReplicaSet{&newRS},
		},
		{
			Name:            "Get empty old ReplicaSets",
			deployment:      deployment,
			rsList:          []*apps.ReplicaSet{&newRS},
			expected:        nil,
			expectedRequire: nil,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			requireRS, allRS := FindOldReplicaSets(&test.deployment, test.rsList)
			sort.Sort(controller.ReplicaSetsByCreationTimestamp(allRS))
			sort.Sort(controller.ReplicaSetsByCreationTimestamp(test.expected))
			if !reflect.DeepEqual(allRS, test.expected) {
				t.Errorf("In test case %q, expected %#v, got %#v", test.Name, test.expected, allRS)
			}
			// RSs are getting filtered correctly by rs.spec.replicas
			if !reflect.DeepEqual(requireRS, test.expectedRequire) {
				t.Errorf("In test case %q, expected %#v, got %#v", test.Name, test.expectedRequire, requireRS)
			}
		})
	}
}

// equal compares the equality of two ReplicaSet slices regardless of their ordering
func equal(rss1, rss2 []*apps.ReplicaSet) bool {
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
		Name           string
		sets           []*apps.ReplicaSet
		expectedCount  int32
		expectedActual int32
	}{
		{
			"1:2 Replicas",
			[]*apps.ReplicaSet{&rs1},
			1,
			2,
		},
		{
			"3:5 Replicas",
			[]*apps.ReplicaSet{&rs1, &rs2},
			3,
			5,
		},
	}

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			rs := GetReplicaCountForReplicaSets(test.sets)
			if rs != test.expectedCount {
				t.Errorf("In test case %s, expectedCount %+v, got %+v", test.Name, test.expectedCount, rs)
			}
			rs = GetActualReplicaCountForReplicaSets(test.sets)
			if rs != test.expectedActual {
				t.Errorf("In test case %s, expectedActual %+v, got %+v", test.Name, test.expectedActual, rs)
			}
		})
	}
}

func TestResolveFenceposts(t *testing.T) {
	tests := []struct {
		maxSurge          *string
		maxUnavailable    *string
		desired           int32
		expectSurge       int32
		expectUnavailable int32
		expectError       bool
	}{
		{
			maxSurge:          newString("0%"),
			maxUnavailable:    newString("0%"),
			desired:           0,
			expectSurge:       0,
			expectUnavailable: 1,
			expectError:       false,
		},
		{
			maxSurge:          newString("39%"),
			maxUnavailable:    newString("39%"),
			desired:           10,
			expectSurge:       4,
			expectUnavailable: 3,
			expectError:       false,
		},
		{
			maxSurge:          newString("oops"),
			maxUnavailable:    newString("39%"),
			desired:           10,
			expectSurge:       0,
			expectUnavailable: 0,
			expectError:       true,
		},
		{
			maxSurge:          newString("55%"),
			maxUnavailable:    newString("urg"),
			desired:           10,
			expectSurge:       0,
			expectUnavailable: 0,
			expectError:       true,
		},
		{
			maxSurge:          nil,
			maxUnavailable:    newString("39%"),
			desired:           10,
			expectSurge:       0,
			expectUnavailable: 3,
			expectError:       false,
		},
		{
			maxSurge:          newString("39%"),
			maxUnavailable:    nil,
			desired:           10,
			expectSurge:       4,
			expectUnavailable: 0,
			expectError:       false,
		},
		{
			maxSurge:          nil,
			maxUnavailable:    nil,
			desired:           10,
			expectSurge:       0,
			expectUnavailable: 1,
			expectError:       false,
		},
	}

	for num, test := range tests {
		t.Run(fmt.Sprintf("%d", num), func(t *testing.T) {
			var maxSurge, maxUnavail *intstr.IntOrString
			if test.maxSurge != nil {
				surge := intstr.FromString(*test.maxSurge)
				maxSurge = &surge
			}
			if test.maxUnavailable != nil {
				unavail := intstr.FromString(*test.maxUnavailable)
				maxUnavail = &unavail
			}
			surge, unavail, err := ResolveFenceposts(maxSurge, maxUnavail, test.desired)
			if err != nil && !test.expectError {
				t.Errorf("unexpected error %v", err)
			}
			if err == nil && test.expectError {
				t.Error("expected error")
			}
			if surge != test.expectSurge || unavail != test.expectUnavailable {
				t.Errorf("#%v got %v:%v, want %v:%v", num, surge, unavail, test.expectSurge, test.expectUnavailable)
			}
		})
	}
}

func newString(s string) *string {
	return &s
}

func TestNewRSNewReplicas(t *testing.T) {
	tests := []struct {
		Name          string
		strategyType  apps.DeploymentStrategyType
		depReplicas   int32
		newRSReplicas int32
		maxSurge      int
		expected      int32
	}{
		{
			"can not scale up - to newRSReplicas",
			apps.RollingUpdateDeploymentStrategyType,
			1, 5, 1, 5,
		},
		{
			"scale up - to depReplicas",
			apps.RollingUpdateDeploymentStrategyType,
			6, 2, 10, 6,
		},
		{
			"recreate - to depReplicas",
			apps.RecreateDeploymentStrategyType,
			3, 1, 1, 3,
		},
	}
	newDeployment := generateDeployment("nginx")
	newRC := generateRS(newDeployment)
	rs5 := generateRS(newDeployment)
	*(rs5.Spec.Replicas) = 5

	for _, test := range tests {
		t.Run(test.Name, func(t *testing.T) {
			*(newDeployment.Spec.Replicas) = test.depReplicas
			newDeployment.Spec.Strategy = apps.DeploymentStrategy{Type: test.strategyType}
			newDeployment.Spec.Strategy.RollingUpdate = &apps.RollingUpdateDeployment{
				MaxUnavailable: func(i int) *intstr.IntOrString {
					x := intstr.FromInt(i)
					return &x
				}(1),
				MaxSurge: func(i int) *intstr.IntOrString {
					x := intstr.FromInt(i)
					return &x
				}(test.maxSurge),
			}
			*(newRC.Spec.Replicas) = test.newRSReplicas
			rs, err := NewRSNewReplicas(&newDeployment, []*apps.ReplicaSet{&rs5}, &newRC)
			if err != nil {
				t.Errorf("In test case %s, got unexpected error %v", test.Name, err)
			}
			if rs != test.expected {
				t.Errorf("In test case %s, expected %+v, got %+v", test.Name, test.expected, rs)
			}
		})
	}
}

var (
	condProgressing = func() apps.DeploymentCondition {
		return apps.DeploymentCondition{
			Type:   apps.DeploymentProgressing,
			Status: v1.ConditionFalse,
			Reason: "ForSomeReason",
		}
	}

	condProgressing2 = func() apps.DeploymentCondition {
		return apps.DeploymentCondition{
			Type:   apps.DeploymentProgressing,
			Status: v1.ConditionTrue,
			Reason: "BecauseItIs",
		}
	}

	condAvailable = func() apps.DeploymentCondition {
		return apps.DeploymentCondition{
			Type:   apps.DeploymentAvailable,
			Status: v1.ConditionTrue,
			Reason: "AwesomeController",
		}
	}

	status = func() *apps.DeploymentStatus {
		return &apps.DeploymentStatus{
			Conditions: []apps.DeploymentCondition{condProgressing(), condAvailable()},
		}
	}
)

func TestGetCondition(t *testing.T) {
	exampleStatus := status()

	tests := []struct {
		name string

		status   apps.DeploymentStatus
		condType apps.DeploymentConditionType

		expected bool
	}{
		{
			name: "condition exists",

			status:   *exampleStatus,
			condType: apps.DeploymentAvailable,

			expected: true,
		},
		{
			name: "condition does not exist",

			status:   *exampleStatus,
			condType: apps.DeploymentReplicaFailure,

			expected: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cond := GetDeploymentCondition(test.status, test.condType)
			exists := cond != nil
			if exists != test.expected {
				t.Errorf("%s: expected condition to exist: %t, got: %t", test.name, test.expected, exists)
			}
		})
	}
}

func TestSetCondition(t *testing.T) {
	tests := []struct {
		name string

		status *apps.DeploymentStatus
		cond   apps.DeploymentCondition

		expectedStatus *apps.DeploymentStatus
	}{
		{
			name: "set for the first time",

			status: &apps.DeploymentStatus{},
			cond:   condAvailable(),

			expectedStatus: &apps.DeploymentStatus{Conditions: []apps.DeploymentCondition{condAvailable()}},
		},
		{
			name: "simple set",

			status: &apps.DeploymentStatus{Conditions: []apps.DeploymentCondition{condProgressing()}},
			cond:   condAvailable(),

			expectedStatus: status(),
		},
		{
			name: "overwrite",

			status: &apps.DeploymentStatus{Conditions: []apps.DeploymentCondition{condProgressing()}},
			cond:   condProgressing2(),

			expectedStatus: &apps.DeploymentStatus{Conditions: []apps.DeploymentCondition{condProgressing2()}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			SetDeploymentCondition(test.status, test.cond)
			if !reflect.DeepEqual(test.status, test.expectedStatus) {
				t.Errorf("%s: expected status: %v, got: %v", test.name, test.expectedStatus, test.status)
			}
		})
	}
}

func TestRemoveCondition(t *testing.T) {
	tests := []struct {
		name string

		status   *apps.DeploymentStatus
		condType apps.DeploymentConditionType

		expectedStatus *apps.DeploymentStatus
	}{
		{
			name: "remove from empty status",

			status:   &apps.DeploymentStatus{},
			condType: apps.DeploymentProgressing,

			expectedStatus: &apps.DeploymentStatus{},
		},
		{
			name: "simple remove",

			status:   &apps.DeploymentStatus{Conditions: []apps.DeploymentCondition{condProgressing()}},
			condType: apps.DeploymentProgressing,

			expectedStatus: &apps.DeploymentStatus{},
		},
		{
			name: "doesn't remove anything",

			status:   status(),
			condType: apps.DeploymentReplicaFailure,

			expectedStatus: status(),
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			RemoveDeploymentCondition(test.status, test.condType)
			if !reflect.DeepEqual(test.status, test.expectedStatus) {
				t.Errorf("%s: expected status: %v, got: %v", test.name, test.expectedStatus, test.status)
			}
		})
	}
}

func TestDeploymentComplete(t *testing.T) {
	deployment := func(desired, current, updated, available, maxUnavailable, maxSurge int32) *apps.Deployment {
		return &apps.Deployment{
			Spec: apps.DeploymentSpec{
				Replicas: &desired,
				Strategy: apps.DeploymentStrategy{
					RollingUpdate: &apps.RollingUpdateDeployment{
						MaxUnavailable: func(i int) *intstr.IntOrString { x := intstr.FromInt(i); return &x }(int(maxUnavailable)),
						MaxSurge:       func(i int) *intstr.IntOrString { x := intstr.FromInt(i); return &x }(int(maxSurge)),
					},
					Type: apps.RollingUpdateDeploymentStrategyType,
				},
			},
			Status: apps.DeploymentStatus{
				Replicas:          current,
				UpdatedReplicas:   updated,
				AvailableReplicas: available,
			},
		}
	}

	tests := []struct {
		name string

		d *apps.Deployment

		expected bool
	}{
		{
			name: "not complete: min but not all pods become available",

			d:        deployment(5, 5, 5, 4, 1, 0),
			expected: false,
		},
		{
			name: "not complete: min availability is not honored",

			d:        deployment(5, 5, 5, 3, 1, 0),
			expected: false,
		},
		{
			name: "complete",

			d:        deployment(5, 5, 5, 5, 0, 0),
			expected: true,
		},
		{
			name: "not complete: all pods are available but not updated",

			d:        deployment(5, 5, 4, 5, 0, 0),
			expected: false,
		},
		{
			name: "not complete: still running old pods",

			// old replica set: spec.replicas=1, status.replicas=1, status.availableReplicas=1
			// new replica set: spec.replicas=1, status.replicas=1, status.availableReplicas=0
			d:        deployment(1, 2, 1, 1, 0, 1),
			expected: false,
		},
		{
			name: "not complete: one replica deployment never comes up",

			d:        deployment(1, 1, 1, 0, 1, 1),
			expected: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got, exp := DeploymentComplete(test.d, &test.d.Status), test.expected; got != exp {
				t.Errorf("expected complete: %t, got: %t", exp, got)
			}
		})
	}
}

func TestDeploymentProgressing(t *testing.T) {
	deployment := func(current, updated, ready, available int32) *apps.Deployment {
		return &apps.Deployment{
			Status: apps.DeploymentStatus{
				Replicas:          current,
				UpdatedReplicas:   updated,
				ReadyReplicas:     ready,
				AvailableReplicas: available,
			},
		}
	}
	newStatus := func(current, updated, ready, available int32) apps.DeploymentStatus {
		return apps.DeploymentStatus{
			Replicas:          current,
			UpdatedReplicas:   updated,
			ReadyReplicas:     ready,
			AvailableReplicas: available,
		}
	}

	tests := []struct {
		name string

		d         *apps.Deployment
		newStatus apps.DeploymentStatus

		expected bool
	}{
		{
			name: "progressing: updated pods",

			d:         deployment(10, 4, 4, 4),
			newStatus: newStatus(10, 6, 4, 4),

			expected: true,
		},
		{
			name: "not progressing",

			d:         deployment(10, 4, 4, 4),
			newStatus: newStatus(10, 4, 4, 4),

			expected: false,
		},
		{
			name: "progressing: old pods removed",

			d:         deployment(10, 4, 6, 6),
			newStatus: newStatus(8, 4, 6, 6),

			expected: true,
		},
		{
			name: "not progressing: less new pods",

			d:         deployment(10, 7, 3, 3),
			newStatus: newStatus(10, 6, 3, 3),

			expected: false,
		},
		{
			name: "progressing: less overall but more new pods",

			d:         deployment(10, 4, 7, 7),
			newStatus: newStatus(8, 8, 5, 5),

			expected: true,
		},
		{
			name: "progressing: more ready pods",

			d:         deployment(10, 10, 9, 8),
			newStatus: newStatus(10, 10, 10, 8),

			expected: true,
		},
		{
			name: "progressing: more available pods",

			d:         deployment(10, 10, 10, 9),
			newStatus: newStatus(10, 10, 10, 10),

			expected: true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got, exp := DeploymentProgressing(test.d, &test.newStatus), test.expected; got != exp {
				t.Errorf("expected progressing: %t, got: %t", exp, got)
			}
		})
	}
}

func TestDeploymentTimedOut(t *testing.T) {
	var (
		null     *int32
		ten      = int32(10)
		infinite = int32(math.MaxInt32)
	)

	timeFn := func(min, sec int) time.Time {
		return time.Date(2016, 1, 1, 0, min, sec, 0, time.UTC)
	}
	deployment := func(condType apps.DeploymentConditionType, status v1.ConditionStatus, reason string, pds *int32, from time.Time) apps.Deployment {
		return apps.Deployment{
			Spec: apps.DeploymentSpec{
				ProgressDeadlineSeconds: pds,
			},
			Status: apps.DeploymentStatus{
				Conditions: []apps.DeploymentCondition{
					{
						Type:           condType,
						Status:         status,
						Reason:         reason,
						LastUpdateTime: metav1.Time{Time: from},
					},
				},
			},
		}
	}

	tests := []struct {
		name string

		d     apps.Deployment
		nowFn func() time.Time

		expected bool
	}{
		{
			name: "nil progressDeadlineSeconds specified - no timeout",

			d:        deployment(apps.DeploymentProgressing, v1.ConditionTrue, "", null, timeFn(1, 9)),
			nowFn:    func() time.Time { return timeFn(1, 20) },
			expected: false,
		},
		{
			name: "infinite progressDeadlineSeconds specified - no timeout",

			d:        deployment(apps.DeploymentProgressing, v1.ConditionTrue, "", &infinite, timeFn(1, 9)),
			nowFn:    func() time.Time { return timeFn(1, 20) },
			expected: false,
		},
		{
			name: "progressDeadlineSeconds: 10s, now - started => 00:01:20 - 00:01:09 => 11s",

			d:        deployment(apps.DeploymentProgressing, v1.ConditionTrue, "", &ten, timeFn(1, 9)),
			nowFn:    func() time.Time { return timeFn(1, 20) },
			expected: true,
		},
		{
			name: "progressDeadlineSeconds: 10s, now - started => 00:01:20 - 00:01:11 => 9s",

			d:        deployment(apps.DeploymentProgressing, v1.ConditionTrue, "", &ten, timeFn(1, 11)),
			nowFn:    func() time.Time { return timeFn(1, 20) },
			expected: false,
		},
		{
			name: "previous status was a complete deployment",

			d:        deployment(apps.DeploymentProgressing, v1.ConditionTrue, NewRSAvailableReason, nil, time.Time{}),
			expected: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nowFn = test.nowFn
			if got, exp := DeploymentTimedOut(&test.d, &test.d.Status), test.expected; got != exp {
				t.Errorf("expected timeout: %t, got: %t", exp, got)
			}
		})
	}
}

func TestMaxUnavailable(t *testing.T) {
	deployment := func(replicas int32, maxUnavailable intstr.IntOrString) apps.Deployment {
		return apps.Deployment{
			Spec: apps.DeploymentSpec{
				Replicas: func(i int32) *int32 { return &i }(replicas),
				Strategy: apps.DeploymentStrategy{
					RollingUpdate: &apps.RollingUpdateDeployment{
						MaxSurge:       func(i int) *intstr.IntOrString { x := intstr.FromInt(i); return &x }(int(1)),
						MaxUnavailable: &maxUnavailable,
					},
					Type: apps.RollingUpdateDeploymentStrategyType,
				},
			},
		}
	}
	tests := []struct {
		name       string
		deployment apps.Deployment
		expected   int32
	}{
		{
			name:       "maxUnavailable less than replicas",
			deployment: deployment(10, intstr.FromInt(5)),
			expected:   int32(5),
		},
		{
			name:       "maxUnavailable equal replicas",
			deployment: deployment(10, intstr.FromInt(10)),
			expected:   int32(10),
		},
		{
			name:       "maxUnavailable greater than replicas",
			deployment: deployment(5, intstr.FromInt(10)),
			expected:   int32(5),
		},
		{
			name:       "maxUnavailable with replicas is 0",
			deployment: deployment(0, intstr.FromInt(10)),
			expected:   int32(0),
		},
		{
			name: "maxUnavailable with Recreate deployment strategy",
			deployment: apps.Deployment{
				Spec: apps.DeploymentSpec{
					Strategy: apps.DeploymentStrategy{
						Type: apps.RecreateDeploymentStrategyType,
					},
				},
			},
			expected: int32(0),
		},
		{
			name:       "maxUnavailable less than replicas with percents",
			deployment: deployment(10, intstr.FromString("50%")),
			expected:   int32(5),
		},
		{
			name:       "maxUnavailable equal replicas with percents",
			deployment: deployment(10, intstr.FromString("100%")),
			expected:   int32(10),
		},
		{
			name:       "maxUnavailable greater than replicas with percents",
			deployment: deployment(5, intstr.FromString("100%")),
			expected:   int32(5),
		},
	}

	for _, test := range tests {
		t.Log(test.name)
		t.Run(test.name, func(t *testing.T) {
			maxUnavailable := MaxUnavailable(test.deployment)
			if test.expected != maxUnavailable {
				t.Fatalf("expected:%v, got:%v", test.expected, maxUnavailable)
			}
		})
	}
}

//Set of simple tests for annotation related util functions
func TestAnnotationUtils(t *testing.T) {

	//Setup
	tDeployment := generateDeployment("nginx")
	tRS := generateRS(tDeployment)
	tDeployment.Annotations[RevisionAnnotation] = "1"

	//Test Case 1: Check if anotations are copied properly from deployment to RS
	t.Run("SetNewReplicaSetAnnotations", func(t *testing.T) {
		//Try to set the increment revision from 1 through 20
		for i := 0; i < 20; i++ {

			nextRevision := fmt.Sprintf("%d", i+1)
			SetNewReplicaSetAnnotations(&tDeployment, &tRS, nextRevision, true)
			//Now the ReplicaSets Revision Annotation should be i+1

			if tRS.Annotations[RevisionAnnotation] != nextRevision {
				t.Errorf("Revision Expected=%s Obtained=%s", nextRevision, tRS.Annotations[RevisionAnnotation])
			}
		}
	})

	//Test Case 2:  Check if annotations are set properly
	t.Run("SetReplicasAnnotations", func(t *testing.T) {
		updated := SetReplicasAnnotations(&tRS, 10, 11)
		if !updated {
			t.Errorf("SetReplicasAnnotations() failed")
		}
		value, ok := tRS.Annotations[DesiredReplicasAnnotation]
		if !ok {
			t.Errorf("SetReplicasAnnotations did not set DesiredReplicasAnnotation")
		}
		if value != "10" {
			t.Errorf("SetReplicasAnnotations did not set DesiredReplicasAnnotation correctly value=%s", value)
		}
		if value, ok = tRS.Annotations[MaxReplicasAnnotation]; !ok {
			t.Errorf("SetReplicasAnnotations did not set DesiredReplicasAnnotation")
		}
		if value != "11" {
			t.Errorf("SetReplicasAnnotations did not set MaxReplicasAnnotation correctly value=%s", value)
		}
	})

	//Test Case 3:  Check if annotations reflect deployments state
	tRS.Annotations[DesiredReplicasAnnotation] = "1"
	tRS.Status.AvailableReplicas = 1
	tRS.Spec.Replicas = new(int32)
	*tRS.Spec.Replicas = 1

	t.Run("IsSaturated", func(t *testing.T) {
		saturated := IsSaturated(&tDeployment, &tRS)
		if !saturated {
			t.Errorf("SetReplicasAnnotations Expected=true Obtained=false")
		}
	})
	//Tear Down
}

func TestReplicasAnnotationsNeedUpdate(t *testing.T) {

	desiredReplicas := fmt.Sprintf("%d", int32(10))
	maxReplicas := fmt.Sprintf("%d", int32(20))

	tests := []struct {
		name       string
		replicaSet *apps.ReplicaSet
		expected   bool
	}{
		{
			name: "test Annotations nil",
			replicaSet: &apps.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{Name: "hello", Namespace: "test"},
				Spec: apps.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
			},
			expected: true,
		},
		{
			name: "test desiredReplicas update",
			replicaSet: &apps.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "hello",
					Namespace:   "test",
					Annotations: map[string]string{DesiredReplicasAnnotation: "8", MaxReplicasAnnotation: maxReplicas},
				},
				Spec: apps.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
			},
			expected: true,
		},
		{
			name: "test maxReplicas update",
			replicaSet: &apps.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "hello",
					Namespace:   "test",
					Annotations: map[string]string{DesiredReplicasAnnotation: desiredReplicas, MaxReplicasAnnotation: "16"},
				},
				Spec: apps.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
			},
			expected: true,
		},
		{
			name: "test needn't update",
			replicaSet: &apps.ReplicaSet{
				ObjectMeta: metav1.ObjectMeta{
					Name:        "hello",
					Namespace:   "test",
					Annotations: map[string]string{DesiredReplicasAnnotation: desiredReplicas, MaxReplicasAnnotation: maxReplicas},
				},
				Spec: apps.ReplicaSetSpec{
					Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"foo": "bar"}},
				},
			},
			expected: false,
		},
	}

	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := ReplicasAnnotationsNeedUpdate(test.replicaSet, 10, 20)
			if result != test.expected {
				t.Errorf("case[%d]:%s Expected %v, Got: %v", i, test.name, test.expected, result)
			}
		})
	}
}
