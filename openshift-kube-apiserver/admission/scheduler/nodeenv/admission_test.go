package nodeenv

import (
	"context"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	kapi "k8s.io/kubernetes/pkg/apis/core"

	projectv1 "github.com/openshift/api/project/v1"
	"github.com/openshift/apiserver-library-go/pkg/labelselector"
)

// TestPodAdmission verifies various scenarios involving pod/project/global node label selectors
func TestPodAdmission(t *testing.T) {
	namespace := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "testProject",
			Namespace: "",
		},
	}

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "worker-1",
			Namespace: "",
			Labels: map[string]string{
				"worker": "true",
			},
		},
	}

	handler := &podNodeEnvironment{}
	pod := &kapi.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "testPod"},
	}

	tests := []struct {
		defaultNodeSelector       string
		projectNodeSelector       string
		podNodeSelector           map[string]string
		podNodeName               string
		mergedNodeSelector        map[string]string
		ignoreProjectNodeSelector bool
		admit                     bool
		testName                  string
	}{
		{
			defaultNodeSelector:       "",
			podNodeSelector:           map[string]string{},
			mergedNodeSelector:        map[string]string{},
			ignoreProjectNodeSelector: true,
			admit:                     true,
			testName:                  "No node selectors",
		},
		{
			defaultNodeSelector:       "infra = false",
			podNodeSelector:           map[string]string{},
			mergedNodeSelector:        map[string]string{"infra": "false"},
			ignoreProjectNodeSelector: true,
			admit:                     true,
			testName:                  "Default node selector and no conflicts",
		},
		{
			defaultNodeSelector: "",
			projectNodeSelector: "infra = false",
			podNodeSelector:     map[string]string{},
			mergedNodeSelector:  map[string]string{"infra": "false"},
			admit:               true,
			testName:            "Project node selector and no conflicts",
		},
		{
			defaultNodeSelector: "infra = false",
			projectNodeSelector: "",
			podNodeSelector:     map[string]string{},
			mergedNodeSelector:  map[string]string{},
			admit:               true,
			testName:            "Empty project node selector and no conflicts",
		},
		{
			defaultNodeSelector: "infra = false",
			projectNodeSelector: "infra=true",
			podNodeSelector:     map[string]string{},
			mergedNodeSelector:  map[string]string{"infra": "true"},
			admit:               true,
			testName:            "Default and project node selector, no conflicts",
		},
		{
			defaultNodeSelector: "infra = false",
			projectNodeSelector: "infra=true",
			podNodeSelector:     map[string]string{"env": "test"},
			mergedNodeSelector:  map[string]string{"infra": "true", "env": "test"},
			admit:               true,
			testName:            "Project and pod node selector, no conflicts",
		},
		{
			defaultNodeSelector: "env = test",
			projectNodeSelector: "infra=true",
			podNodeSelector:     map[string]string{"infra": "false"},
			mergedNodeSelector:  map[string]string{"infra": "false"},
			admit:               false,
			testName:            "Conflicting pod and project node selector, one label",
		},
		{
			defaultNodeSelector: "env=dev",
			projectNodeSelector: "infra=false, env = test",
			podNodeSelector:     map[string]string{"env": "dev", "color": "blue"},
			mergedNodeSelector:  map[string]string{"env": "dev", "color": "blue"},
			admit:               false,
			testName:            "Conflicting pod and project node selector, multiple labels",
		},
		{
			defaultNodeSelector: "",
			projectNodeSelector: "worker=true",
			podNodeName:         "worker-1",
			podNodeSelector:     nil,
			mergedNodeSelector:  map[string]string{"worker": "true"},
			admit:               true,
			testName:            "node referenced in pod.nodeName does not conflict with project node selector",
		},
		{
			defaultNodeSelector: "",
			projectNodeSelector: "",
			podNodeName:         "worker-1",
			podNodeSelector:     map[string]string{"worker": "false"},
			mergedNodeSelector:  map[string]string{"worker": "false"},
			admit:               true,
			// default to kube behavior: let this fail by kubelet
			testName: "node referenced in pod spec.nodeName can conflict with its own node selector when no project node selector is specified",
		},
		{
			defaultNodeSelector: "worker = true",
			projectNodeSelector: "worker=false",
			podNodeName:         "worker-1",
			podNodeSelector:     nil,
			mergedNodeSelector:  nil,
			admit:               false,
			testName:            "node referenced in pod spec.nodeName conflicts with project node selector",
		},
		{
			defaultNodeSelector: "",
			projectNodeSelector: "worker=true",
			podNodeName:         "worker-2",
			podNodeSelector:     nil,
			mergedNodeSelector:  nil,
			admit:               false,
			testName:            "missing node referenced in pod spec.nodeName does not admit",
		},
	}
	for _, test := range tests {
		indexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
		indexer.Add(namespace)
		indexer.Add(node)
		handler.nsLister = corev1listers.NewNamespaceLister(indexer)
		handler.nsListerSynced = func() bool { return true }
		handler.nodeLister = corev1listers.NewNodeLister(indexer)
		handler.nodeListerSynced = func() bool { return true }
		handler.defaultNodeSelector = test.defaultNodeSelector

		if !test.ignoreProjectNodeSelector {
			namespace.ObjectMeta.Annotations = map[string]string{projectv1.ProjectNodeSelector: test.projectNodeSelector}
		}
		pod.Spec = kapi.PodSpec{NodeSelector: test.podNodeSelector, NodeName: test.podNodeName}

		attrs := admission.NewAttributesRecord(pod, nil, kapi.Kind("Pod").WithVersion("version"), "testProject", namespace.ObjectMeta.Name, kapi.Resource("pods").WithVersion("version"), "", admission.Create, nil, false, nil)
		err := handler.Admit(context.TODO(), attrs, nil)
		if test.admit && err != nil {
			t.Errorf("Test: %s, expected no error but got: %s", test.testName, err)
		} else if !test.admit && err == nil {
			t.Errorf("Test: %s, expected an error", test.testName)
		} else if err == nil {
			if err := handler.Validate(context.TODO(), attrs, nil); err != nil {
				t.Errorf("Test: %s, unexpected Validate error after Admit succeeded: %v", test.testName, err)
			}
		}

		if !labelselector.Equals(test.mergedNodeSelector, pod.Spec.NodeSelector) {
			t.Errorf("Test: %s, expected: %s but got: %s", test.testName, test.mergedNodeSelector, pod.Spec.NodeSelector)
		} else if len(test.projectNodeSelector) > 0 {
			firstProjectKey := strings.TrimSpace(strings.Split(test.projectNodeSelector, "=")[0])
			delete(pod.Spec.NodeSelector, firstProjectKey)
			if err := handler.Validate(context.TODO(), attrs, nil); err == nil {
				t.Errorf("Test: %s, expected Validate error after removing project key %q", test.testName, firstProjectKey)
			}
		}
	}
}

func TestHandles(t *testing.T) {
	for op, shouldHandle := range map[admission.Operation]bool{
		admission.Create:  true,
		admission.Update:  false,
		admission.Connect: false,
		admission.Delete:  false,
	} {
		nodeEnvionment, err := NewPodNodeEnvironment()
		if err != nil {
			t.Errorf("%v: error getting node environment: %v", op, err)
			continue
		}

		if e, a := shouldHandle, nodeEnvionment.Handles(op); e != a {
			t.Errorf("%v: shouldHandle=%t, handles=%t", op, e, a)
		}
	}
}
