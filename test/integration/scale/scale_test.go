/*
Copyright 2017 The Kubernetes Authors.

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

package scale

import (
	"encoding/json"
	"path"
	"strings"
	"testing"

	_ "github.com/coreos/etcd/etcdserver/api/v3rpc" // Force package logger init.
	"github.com/coreos/pkg/capnslog"

	appsv1beta2 "k8s.io/api/apps/v1beta2"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes"
	apitesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

type subresourceTest struct {
	resource schema.GroupVersionResource
	kind     schema.GroupVersionKind
}

func makeGVR(group, version, resource string) schema.GroupVersionResource {
	return schema.GroupVersionResource{Group: group, Version: version, Resource: resource}
}
func makeGVK(group, version, kind string) schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: group, Version: version, Kind: kind}
}

func TestMain(m *testing.M) {
	framework.EtcdMain(m.Run)
}

func TestScaleSubresources(t *testing.T) {
	clientSet, tearDown := setup(t)
	defer tearDown()

	resourceLists, err := clientSet.Discovery().ServerResources()
	if err != nil {
		t.Fatal(err)
	}

	expectedScaleSubresources := map[schema.GroupVersionResource]schema.GroupVersionKind{
		makeGVR("", "v1", "replicationcontrollers/scale"): makeGVK("autoscaling", "v1", "Scale"),

		makeGVR("extensions", "v1beta1", "deployments/scale"):            makeGVK("extensions", "v1beta1", "Scale"),
		makeGVR("extensions", "v1beta1", "replicationcontrollers/scale"): makeGVK("extensions", "v1beta1", "Scale"),
		makeGVR("extensions", "v1beta1", "replicasets/scale"):            makeGVK("extensions", "v1beta1", "Scale"),

		makeGVR("apps", "v1beta1", "deployments/scale"):  makeGVK("apps", "v1beta1", "Scale"),
		makeGVR("apps", "v1beta1", "statefulsets/scale"): makeGVK("apps", "v1beta1", "Scale"),

		makeGVR("apps", "v1beta2", "deployments/scale"):  makeGVK("apps", "v1beta2", "Scale"),
		makeGVR("apps", "v1beta2", "replicasets/scale"):  makeGVK("apps", "v1beta2", "Scale"),
		makeGVR("apps", "v1beta2", "statefulsets/scale"): makeGVK("apps", "v1beta2", "Scale"),

		makeGVR("apps", "v1", "deployments/scale"):  makeGVK("autoscaling", "v1", "Scale"),
		makeGVR("apps", "v1", "replicasets/scale"):  makeGVK("autoscaling", "v1", "Scale"),
		makeGVR("apps", "v1", "statefulsets/scale"): makeGVK("autoscaling", "v1", "Scale"),
	}

	autoscalingGVK := schema.GroupVersionKind{Group: "autoscaling", Version: "v1", Kind: "Scale"}

	discoveredScaleSubresources := map[schema.GroupVersionResource]schema.GroupVersionKind{}
	for _, resourceList := range resourceLists {
		containingGV, err := schema.ParseGroupVersion(resourceList.GroupVersion)
		if err != nil {
			t.Fatalf("error getting group version for %#v: %v", resourceList, err)
		}

		for _, resource := range resourceList.APIResources {
			if !strings.HasSuffix(resource.Name, "/scale") {
				continue
			}

			gvr := containingGV.WithResource(resource.Name)
			if _, exists := discoveredScaleSubresources[gvr]; exists {
				t.Errorf("scale subresource %#v listed multiple times in discovery", gvr)
				continue
			}

			gvk := containingGV.WithKind(resource.Kind)
			if resource.Group != "" {
				gvk.Group = resource.Group
			}
			if resource.Version != "" {
				gvk.Version = resource.Version
			}
			discoveredScaleSubresources[gvr] = gvk
		}
	}

	// Ensure nothing is missing
	for gvr, gvk := range expectedScaleSubresources {
		if _, ok := discoveredScaleSubresources[gvr]; !ok {
			t.Errorf("expected scale subresource %#v of kind %#v was missing from discovery", gvr, gvk)
		}
	}

	// Ensure discovery lists expected types
	for gvr, gvk := range discoveredScaleSubresources {
		if expectedGVK, expected := expectedScaleSubresources[gvr]; !expected {
			if gvk == autoscalingGVK {
				t.Errorf("unexpected scale subresource %#v of kind %#v. new scale subresource should be added to expectedScaleSubresources", gvr, gvk)
			} else {
				t.Errorf("unexpected scale subresource %#v of kind %#v. new scale resources are expected to use Scale from the autoscaling/v1 API group", gvr, gvk)
			}
			continue
		} else if expectedGVK != gvk {
			t.Errorf("scale subresource %#v should be of kind %#v, but %#v was listed in discovery", gvr, expectedGVK, gvk)
			continue
		}
	}

	// Create objects required to exercise scale subresources
	if _, err := clientSet.CoreV1().ReplicationControllers("default").Create(rcStub); err != nil {
		t.Fatal(err)
	}
	if _, err := clientSet.AppsV1beta2().ReplicaSets("default").Create(rsStub); err != nil {
		t.Fatal(err)
	}
	if _, err := clientSet.AppsV1beta2().Deployments("default").Create(deploymentStub); err != nil {
		t.Fatal(err)
	}
	if _, err := clientSet.AppsV1beta2().StatefulSets("default").Create(ssStub); err != nil {
		t.Fatal(err)
	}

	// Ensure scale subresources return and accept expected kinds
	for gvr, gvk := range discoveredScaleSubresources {
		prefix := "/apis"
		if gvr.Group == corev1.GroupName {
			prefix = "/api"
		}

		resourceParts := strings.SplitN(gvr.Resource, "/", 2)

		urlPath := path.Join(prefix, gvr.Group, gvr.Version, "namespaces", "default", resourceParts[0], "test", resourceParts[1])
		obj := &unstructured.Unstructured{}

		getData, err := clientSet.CoreV1().RESTClient().Get().AbsPath(urlPath).DoRaw()
		if err != nil {
			t.Errorf("error fetching %s: %v", urlPath, err)
			continue
		}
		if err := json.Unmarshal(getData, obj); err != nil {
			t.Errorf("error decoding %s: %v", urlPath, err)
			t.Log(string(getData))
			continue
		}

		if obj.GetObjectKind().GroupVersionKind() != gvk {
			t.Errorf("expected %#v, got %#v from %s", gvk, obj.GetObjectKind().GroupVersionKind(), urlPath)
			t.Log(string(getData))
			continue
		}

		updateData, err := clientSet.CoreV1().RESTClient().Put().AbsPath(urlPath).Body(getData).DoRaw()
		if err != nil {
			t.Errorf("error putting to %s: %v", urlPath, err)
			t.Log(string(getData))
			t.Log(string(updateData))
			continue
		}
	}
}

var (
	replicas = int32(1)

	podStub = corev1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"foo": "bar"}},
		Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "test", Image: "busybox"}}},
	}

	rcStub = &corev1.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{Name: "test"},
		Spec:       corev1.ReplicationControllerSpec{Selector: podStub.Labels, Replicas: &replicas, Template: &podStub},
	}

	rsStub = &appsv1beta2.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{Name: "test"},
		Spec:       appsv1beta2.ReplicaSetSpec{Selector: &metav1.LabelSelector{MatchLabels: podStub.Labels}, Replicas: &replicas, Template: podStub},
	}

	deploymentStub = &appsv1beta2.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test"},
		Spec:       appsv1beta2.DeploymentSpec{Selector: &metav1.LabelSelector{MatchLabels: podStub.Labels}, Replicas: &replicas, Template: podStub},
	}

	ssStub = &appsv1beta2.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{Name: "test"},
		Spec:       appsv1beta2.StatefulSetSpec{Selector: &metav1.LabelSelector{MatchLabels: podStub.Labels}, Replicas: &replicas, Template: podStub},
	}
)

func setup(t *testing.T) (client kubernetes.Interface, tearDown func()) {
	result := apitesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())

	// TODO: Disable logging here until we resolve teardown issues which result in
	// massive log spam. Another path forward would be to refactor
	// StartTestServerOrDie to work with the etcd instance already started by the
	// integration test scripts.
	// See https://github.com/kubernetes/kubernetes/issues/49489.
	repo, err := capnslog.GetRepoLogger("github.com/coreos/etcd")
	if err != nil {
		t.Fatalf("couldn't configure logging: %v", err)
	}
	repo.SetLogLevel(map[string]capnslog.LogLevel{
		"etcdserver/api/v3rpc": capnslog.CRITICAL,
	})

	result.ClientConfig.AcceptContentTypes = ""
	result.ClientConfig.ContentType = ""
	result.ClientConfig.NegotiatedSerializer = nil
	clientSet, err := kubernetes.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("error creating clientset: %v", err)
	}

	return clientSet, result.TearDownFn
}
