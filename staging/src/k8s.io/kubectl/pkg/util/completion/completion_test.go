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
	addPodsToFactory(tf)

	compFunc := ResourceTypeAndNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod"}, "b")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceTypeAndNameCompletionFuncRepeating(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := ResourceTypeAndNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod", "bar"}, "")
	// The other pods should be completed, but not the already specified ones
	checkCompletion(t, comps, []string{"foo"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceTypeAndNameCompletionFuncJointForm(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := ResourceTypeAndNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{}, "pod/b")
	checkCompletion(t, comps, []string{"pod/bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceTypeAndNameCompletionFuncJointFormRepeating(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := ResourceTypeAndNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod/bar"}, "pod/")
	// The other pods should be completed, but not the already specified ones
	checkCompletion(t, comps, []string{"pod/foo"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionFuncNoArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := SpecifiedResourceTypeAndNameCompletionFunc(tf, []string{"pod", "service", "statefulset"})
	comps, directive := compFunc(cmd, []string{}, "s")
	checkCompletion(t, comps, []string{"service", "statefulset"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionFuncOneArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := SpecifiedResourceTypeAndNameCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{"pod"}, "b")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionFuncRepeating(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := SpecifiedResourceTypeAndNameCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{"pod", "bar"}, "")
	// The other pods should be completed, but not the already specified ones
	checkCompletion(t, comps, []string{"foo"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionFuncJointFormOneArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := SpecifiedResourceTypeAndNameCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{}, "pod/b")
	checkCompletion(t, comps, []string{"pod/bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionFuncJointFormRepeating(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := SpecifiedResourceTypeAndNameCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{"pod/bar"}, "pod/")
	// The other pods should be completed, but not the already specified ones
	checkCompletion(t, comps, []string{"pod/foo"}, directive, cobra.ShellCompDirectiveNoFileComp)
}
func TestSpecifiedResourceTypeAndNameCompletionNoRepeatFuncOneArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := SpecifiedResourceTypeAndNameNoRepeatCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{"pod"}, "b")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionNoRepeatFuncMultiArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := SpecifiedResourceTypeAndNameNoRepeatCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{"pod", "bar"}, "")
	// There should not be any more pods shown as this function should not repeat the completion
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionNoRepeatFuncJointFormOneArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := SpecifiedResourceTypeAndNameNoRepeatCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{}, "pod/b")
	checkCompletion(t, comps, []string{"pod/bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestSpecifiedResourceTypeAndNameCompletionNoRepeatFuncJointFormMultiArg(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := SpecifiedResourceTypeAndNameNoRepeatCompletionFunc(tf, []string{"pod"})
	comps, directive := compFunc(cmd, []string{"pod/bar"}, "pod/")
	// There should not be any more pods shown as this function should not repeat the completion
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceNameCompletionFuncNoArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := ResourceNameCompletionFunc(tf, "pod")
	comps, directive := compFunc(cmd, []string{}, "b")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceNameCompletionFuncTooManyArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := ResourceNameCompletionFunc(tf, "pod")
	comps, directive := compFunc(cmd, []string{"pod-name"}, "")
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestResourceNameCompletionFuncJointFormNoArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := ResourceNameCompletionFunc(tf, "pod")
	comps, directive := compFunc(cmd, []string{}, "pod/b")
	// The <type>/<name> should NOT be supported by this function
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameCompletionFuncNoArgsPodName(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := PodResourceNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{}, "b")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameCompletionFuncNoArgsResources(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := PodResourceNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{}, "d")
	checkCompletion(
		t, comps, []string{"daemonsets/", "deployments/"},
		directive, cobra.ShellCompDirectiveNoFileComp|cobra.ShellCompDirectiveNoSpace)
}

func TestPodResourceNameCompletionFuncTooManyArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := PodResourceNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod-name"}, "")
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameCompletionFuncJointFormNoArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := PodResourceNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{}, "pod/b")
	// The <type>/<name> SHOULD be supported by this function
	checkCompletion(t, comps, []string{"pod/bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameCompletionFuncJointFormTooManyArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := PodResourceNameCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod/name"}, "pod/b")
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameAndContainerCompletionFuncNoArgsPodName(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := PodResourceNameAndContainerCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{}, "b")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameAndContainerCompletionFuncNoArgsResources(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := PodResourceNameAndContainerCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{}, "s")
	checkCompletion(
		t, comps, []string{"services/", "statefulsets/"},
		directive, cobra.ShellCompDirectiveNoFileComp|cobra.ShellCompDirectiveNoSpace)

}

func TestPodResourceNameAndContainerCompletionFuncTooManyArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := PodResourceNameAndContainerCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod-name", "container-name"}, "")
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameAndContainerCompletionFuncJointFormNoArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := PodResourceNameAndContainerCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{}, "pod/b")
	checkCompletion(t, comps, []string{"pod/bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameAndContainerCompletionFuncJointFormTooManyArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := PodResourceNameAndContainerCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod/pod-name", "container-name"}, "")
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
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

func addPodsToFactory(tf *cmdtesting.TestFactory) {
	pods, _, _ := cmdtesting.TestData()
	codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
	tf.UnstructuredClient = &fake.RESTClient{
		NegotiatedSerializer: resource.UnstructuredPlusDefaultContentConfig().NegotiatedSerializer,
		Resp:                 &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: cmdtesting.ObjBody(codec, pods)},
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
