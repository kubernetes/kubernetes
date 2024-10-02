/*
Copyright 2021 The Kubernetes Authors.

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

package completion

import (
	"net/http"
	"sort"
	"testing"

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubectl/pkg/cmd/get"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	"k8s.io/kubectl/pkg/scheme"
)

func TestClusterCompletionFunc(t *testing.T) {
	setMockFactory(api.Config{
		Clusters: map[string]*api.Cluster{
			"bar": {},
			"baz": {},
			"foo": {},
		},
	})

	comps, directive := ClusterCompletionFunc(nil, []string{}, "")
	checkCompletion(t, comps, []string{"bar", "baz", "foo"}, directive, cobra.ShellCompDirectiveNoFileComp)

	comps, directive = ClusterCompletionFunc(nil, []string{}, "b")
	checkCompletion(t, comps, []string{"bar", "baz"}, directive, cobra.ShellCompDirectiveNoFileComp)

	comps, directive = ClusterCompletionFunc(nil, []string{}, "ba")
	checkCompletion(t, comps, []string{"bar", "baz"}, directive, cobra.ShellCompDirectiveNoFileComp)

	comps, directive = ClusterCompletionFunc(nil, []string{}, "bar")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)

	comps, directive = ClusterCompletionFunc(nil, []string{}, "bart")
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestContextCompletionFunc(t *testing.T) {
	setMockFactory(api.Config{
		Contexts: map[string]*api.Context{
			"bar": {},
			"baz": {},
			"foo": {},
		},
	})

	comps, directive := ContextCompletionFunc(nil, []string{}, "")
	checkCompletion(t, comps, []string{"bar", "baz", "foo"}, directive, cobra.ShellCompDirectiveNoFileComp)

	comps, directive = ContextCompletionFunc(nil, []string{}, "b")
	checkCompletion(t, comps, []string{"bar", "baz"}, directive, cobra.ShellCompDirectiveNoFileComp)

	comps, directive = ContextCompletionFunc(nil, []string{}, "ba")
	checkCompletion(t, comps, []string{"bar", "baz"}, directive, cobra.ShellCompDirectiveNoFileComp)

	comps, directive = ContextCompletionFunc(nil, []string{}, "bar")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)

	comps, directive = ContextCompletionFunc(nil, []string{}, "bart")
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestUserCompletionFunc(t *testing.T) {
	setMockFactory(api.Config{
		AuthInfos: map[string]*api.AuthInfo{
			"bar": {},
			"baz": {},
			"foo": {},
		},
	})

	comps, directive := UserCompletionFunc(nil, []string{}, "")
	checkCompletion(t, comps, []string{"bar", "baz", "foo"}, directive, cobra.ShellCompDirectiveNoFileComp)

	comps, directive = UserCompletionFunc(nil, []string{}, "b")
	checkCompletion(t, comps, []string{"bar", "baz"}, directive, cobra.ShellCompDirectiveNoFileComp)

	comps, directive = UserCompletionFunc(nil, []string{}, "ba")
	checkCompletion(t, comps, []string{"bar", "baz"}, directive, cobra.ShellCompDirectiveNoFileComp)

	comps, directive = UserCompletionFunc(nil, []string{}, "bar")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)

	comps, directive = UserCompletionFunc(nil, []string{}, "bart")
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceTypeAndNameCompletionFuncOneArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := ResourceTypeAndNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod"}, "b")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceTypeAndNameCompletionFuncRepeating(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := ResourceTypeAndNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod", "bar"}, "")
	// The other pods should be completed, but not the already specified ones
	checkCompletion(t, comps, []string{"foo"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceTypeAndNameCompletionFuncJointForm(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := ResourceTypeAndNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{}, "pod/b")
	checkCompletion(t, comps, []string{"pod/bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceTypeAndNameCompletionFuncJointFormRepeating(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := ResourceTypeAndNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod/bar"}, "pod/")
	// The other pods should be completed, but not the already specified ones
	checkCompletion(t, comps, []string{"pod/foo"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionFuncNoArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := SpecifiedResourceTypeAndNameCompletionFunc(tf, []string{"pod", "service", "statefulset"})
	comps, directive := compFunc(cmd, []string{}, "s")
	checkCompletion(t, comps, []string{"service", "statefulset"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionFuncOneArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := SpecifiedResourceTypeAndNameCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{"pod"}, "b")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionFuncRepeating(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := SpecifiedResourceTypeAndNameCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{"pod", "bar"}, "")
	// The other pods should be completed, but not the already specified ones
	checkCompletion(t, comps, []string{"foo"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionFuncJointFormOneArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := SpecifiedResourceTypeAndNameCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{}, "pod/b")
	checkCompletion(t, comps, []string{"pod/bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionFuncJointFormRepeating(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := SpecifiedResourceTypeAndNameCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{"pod/bar"}, "pod/")
	// The other pods should be completed, but not the already specified ones
	checkCompletion(t, comps, []string{"pod/foo"}, directive, cobra.ShellCompDirectiveNoFileComp)
}
func TestSpecifiedResourceTypeAndNameCompletionNoRepeatFuncOneArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := SpecifiedResourceTypeAndNameNoRepeatCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{"pod"}, "b")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionNoRepeatFuncMultiArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := SpecifiedResourceTypeAndNameNoRepeatCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{"pod", "bar"}, "")
	// There should not be any more pods shown as this function should not repeat the completion
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionNoRepeatFuncJointFormOneArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := SpecifiedResourceTypeAndNameNoRepeatCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{}, "pod/b")
	checkCompletion(t, comps, []string{"pod/bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionNoRepeatFuncJointFormMultiArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := SpecifiedResourceTypeAndNameNoRepeatCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{"pod/bar"}, "pod/")
	// There should not be any more pods shown as this function should not repeat the completion
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceNameCompletionFuncNoArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := ResourceNameCompletionFunc(tf, "pod")
	comps, directive := compFunc(cmd, []string{}, "b")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceNameCompletionFuncTooManyArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := ResourceNameCompletionFunc(tf, "pod")
	comps, directive := compFunc(cmd, []string{"pod-name"}, "")
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceNameCompletionFuncJointFormNoArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := ResourceNameCompletionFunc(tf, "pod")
	comps, directive := compFunc(cmd, []string{}, "pod/b")
	// The <type>/<name> should NOT be supported by this function
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameCompletionFuncNoArgsPodName(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := PodResourceNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{}, "b")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameCompletionFuncNoArgsResources(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := PodResourceNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{}, "d")
	checkCompletion(
		t, comps, []string{"daemonsets/", "deployments/"},
		directive, cobra.ShellCompDirectiveNoFileComp|cobra.ShellCompDirectiveNoSpace)
}

func TestPodResourceNameCompletionFuncTooManyArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := PodResourceNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod-name"}, "")
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameCompletionFuncJointFormNoArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := PodResourceNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{}, "pod/b")
	// The <type>/<name> SHOULD be supported by this function
	checkCompletion(t, comps, []string{"pod/bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameCompletionFuncJointFormTooManyArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	pods, _, _ := cmdtesting.TestData()
	addResourceToFactory(tf, pods)

	compFunc := PodResourceNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod/name"}, "pod/b")
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceAndContainerNameCompletionFunc(t *testing.T) {
	barPod := getTestPod()

	testCases := []struct {
		name              string
		args              []string
		toComplete        string
		expectedComps     []string
		expectedDirective cobra.ShellCompDirective
	}{
		{
			name:              "no args pod name",
			args:              []string{},
			toComplete:        "b",
			expectedComps:     []string{"bar"},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:              "no args resources",
			args:              []string{},
			toComplete:        "s",
			expectedComps:     []string{"services/", "statefulsets/"},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp | cobra.ShellCompDirectiveNoSpace,
		},
		{
			name:              "joint form no args",
			args:              []string{},
			toComplete:        "pod/b",
			expectedComps:     []string{"pod/bar"},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:              "joint form too many args",
			args:              []string{"pod/pod-name", "container-name"},
			toComplete:        "",
			expectedComps:     []string{},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:              "complete all containers' names",
			args:              []string{"bar"},
			toComplete:        "",
			expectedComps:     []string{"bar", "foo"},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:              "complete specific container name",
			args:              []string{"bar"},
			toComplete:        "b",
			expectedComps:     []string{"bar"},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tf, cmd := prepareCompletionTest()
			addResourceToFactory(tf, barPod)
			compFunc := PodResourceNameAndContainerCompletionFunc(tf)
			comps, directive := compFunc(cmd, tc.args, tc.toComplete)
			checkCompletion(t, comps, tc.expectedComps, directive, tc.expectedDirective)
		})
	}
}

func TestResourceAndPortCompletionFunc(t *testing.T) {
	barPod := getTestPod()
	bazService := getTestService()

	testCases := []struct {
		name              string
		obj               runtime.Object
		args              []string
		toComplete        string
		expectedComps     []string
		expectedDirective cobra.ShellCompDirective
	}{
		{
			name:              "no args pod name",
			obj:               barPod,
			args:              []string{},
			toComplete:        "b",
			expectedComps:     []string{"bar"},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:              "no args resources",
			obj:               barPod,
			args:              []string{},
			toComplete:        "s",
			expectedComps:     []string{"services/", "statefulsets/"},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp | cobra.ShellCompDirectiveNoSpace,
		},
		{
			name:              "too many args",
			obj:               barPod,
			args:              []string{"pod-name", "port-number"},
			toComplete:        "",
			expectedComps:     []string{},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:              "joint from no args",
			obj:               barPod,
			args:              []string{},
			toComplete:        "pod/b",
			expectedComps:     []string{"pod/bar"},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:              "joint from too many args",
			obj:               barPod,
			args:              []string{"pod/pod-name", "port-number"},
			toComplete:        "",
			expectedComps:     []string{},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:              "complete container port with default exposed port",
			obj:               barPod,
			args:              []string{"bar"},
			toComplete:        "",
			expectedComps:     []string{"80:80", "81:81"},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:              "complete container port with custom exposed port",
			obj:               barPod,
			args:              []string{"bar"},
			toComplete:        "90",
			expectedComps:     []string{"90:80", "90:81"},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:              "complete service port with default exposed port",
			obj:               bazService,
			args:              []string{"service/baz"},
			toComplete:        "",
			expectedComps:     []string{"8080:8080", "8081:8081"},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:              "complete container port with custom exposed port",
			obj:               bazService,
			args:              []string{"service/baz"},
			toComplete:        "9090",
			expectedComps:     []string{"9090:8080", "9090:8081"},
			expectedDirective: cobra.ShellCompDirectiveNoFileComp,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tf, cmd := prepareCompletionTest()
			addResourceToFactory(tf, tc.obj)
			compFunc := ResourceAndPortCompletionFunc(tf)
			comps, directive := compFunc(cmd, tc.args, tc.toComplete)
			checkCompletion(t, comps, tc.expectedComps, directive, tc.expectedDirective)
		})
	}
}

func setMockFactory(config api.Config) {
	clientConfig := clientcmd.NewDefaultClientConfig(config, nil)
	testFactory := cmdtesting.NewTestFactory().WithClientConfig(clientConfig)
	SetFactoryForCompletion(testFactory)
}

func prepareCompletionTest() (*cmdtesting.TestFactory, *cobra.Command) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	streams, _, _, _ := genericiooptions.NewTestIOStreams()
	cmd := get.NewCmdGet("kubectl", tf, streams)
	return tf, cmd
}

func addResourceToFactory(tf *cmdtesting.TestFactory, obj runtime.Object) {
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, obj)},
	}
}

func checkCompletion(t *testing.T, comps, expectedComps []string, directive, expectedDirective cobra.ShellCompDirective) {
	if e, d := expectedDirective, directive; e != d {
		t.Errorf("expected directive\n%v\nbut got\n%v", e, d)
	}

	sort.Strings(comps)
	sort.Strings(expectedComps)

	if len(expectedComps) != len(comps) {
		t.Fatalf("expected completions\n%v\nbut got\n%v", expectedComps, comps)
	}

	for i := range comps {
		if expectedComps[i] != comps[i] {
			t.Errorf("expected completions\n%v\nbut got\n%v", expectedComps, comps)
			break
		}
	}
}

func getTestPod() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "test", ResourceVersion: "11"},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name: "bar",
					Ports: []corev1.ContainerPort{
						{
							ContainerPort: 80,
						},
						{
							ContainerPort: 81,
						},
					},
				},
				{
					Name: "foo",
				},
			},
		},
	}
}

func getTestService() *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "baz", Namespace: "test", ResourceVersion: "12"},
		Spec: corev1.ServiceSpec{
			Ports: []corev1.ServicePort{
				{
					Port: 8080,
				},
				{
					Port: 8081,
				},
			},
		},
	}
}
