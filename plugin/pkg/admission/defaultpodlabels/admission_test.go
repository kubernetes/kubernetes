package defaultpodlabels

import (
	"fmt"
	"testing"
	"time"

	"encoding/json"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/admission"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
	"reflect"
	"strings"
)

func newHandlerForTest(c clientset.Interface, config string) (admission.Interface, informers.SharedInformerFactory, error) {
	f := informers.NewSharedInformerFactory(c, 5*time.Minute)
	handler, err := NewDefaultPodLabels(strings.NewReader(config))
	if err != nil {
		return nil, nil, err
	}
	pluginInitializer := kubeapiserveradmission.NewPluginInitializer(c, nil, f, nil, nil, nil, nil)
	pluginInitializer.Initialize(handler)
	err = admission.Validate(handler)
	return handler, f, err
}

func newMockClientForTest(namespaceToLabels map[string]map[string]string) *fake.Clientset {
	mockClient := &fake.Clientset{}
	mockClient.AddReactor("list", "namespaces", func(action core.Action) (bool, runtime.Object, error) {
		namespaceList := &api.NamespaceList{
			ListMeta: metav1.ListMeta{
				ResourceVersion: fmt.Sprintf("%d", len(namespaceToLabels)),
			},
		}

		for name, labels := range namespaceToLabels {
			namespaceList.Items = append(namespaceList.Items, api.Namespace{
				ObjectMeta: metav1.ObjectMeta{
					Name:            name,
					Labels:          labels,
					ResourceVersion: "0",
				},
			})
		}

		return true, namespaceList, nil
	})
	return mockClient
}

func newPod(namespace string, labels map[string]string) api.Pod {
	return api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "testpod",
			Namespace: namespace,
			Labels:    labels,
		},
		Spec: api.PodSpec{
			Volumes:    []api.Volume{{Name: "vol"}},
			Containers: []api.Container{{Name: "ctr", Image: "image"}},
		},
	}
}

func testPod(handler admission.Interface, namespace string, labels map[string]string, expectedLabels map[string]string) error {
	pod := newPod(namespace, labels)

	err := handler.Admit(admission.NewAttributesRecord(&pod, nil, api.Kind("Pod").WithVersion("version"), pod.Namespace, pod.Name, api.Resource("pods").WithVersion("version"), "", admission.Create, nil))
	if err != nil {
		return err
	}

	if !reflect.DeepEqual(pod.Labels, expectedLabels) {
		return fmt.Errorf("Pod has labels %s, expected %s", pod.Labels, expectedLabels)
	}

	return nil
}

// TestAdmissionNamespaceExists verifies pod is admitted only if namespace exists.
func TestEmptyConfig(t *testing.T) {
	handler, informerFactory, err := newHandlerForTest(newMockClientForTest(map[string]map[string]string{}), "{\"labels\": []}")
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)

	podLabels := map[string]string{"dont-change-me": "foo"}

	if err := testPod(handler, "test", podLabels, podLabels); err != nil {
		t.Error(err)
	}
}

func TestItemDefault(t *testing.T) {
	config := DefaultPodLabelsConfig{
		Labels: []LabelConfigItem{
			{
				Name:    "label1",
				Default: "config-default1",
			},
			{
				Name:    "label2",
				Default: "config-default2",
			},
			{
				Name:          "label3",
				SkipNamespace: true,
				Default:       "config-default3",
			},
		},
	}

	configJSON, err := json.Marshal(config)
	if err != nil {
		panic(err)
	}

	mockClient := newMockClientForTest(map[string]map[string]string{"test": {"label1": "namespace-default1", "label2": "namespace-default2"}, "test2": {}})
	handler, informerFactory, err := newHandlerForTest(mockClient, string(configJSON))
	if err != nil {
		t.Errorf("unexpected error initializing handler: %v", err)
	}
	informerFactory.Start(wait.NeverStop)

	// test with a namespace default
	expectedLabels := map[string]string{
		"label1": "pod-default",
		"label2": "namespace-default2",
		"label3": "config-default3",
	}
	if err := testPod(handler, "test", map[string]string{"label1": "pod-default"}, expectedLabels); err != nil {
		t.Errorf("Error testing with namespace default: %s", err)
		return
	}

	// test without a namespace default
	expectedLabels = map[string]string{
		"label1": "pod-default",
		"label2": "config-default2",
		"label3": "config-default3",
	}
	if err := testPod(handler, "test2", map[string]string{"label1": "pod-default"}, expectedLabels); err != nil {
		t.Errorf("Error testing without namespace default: %s", err)
		return
	}

}
