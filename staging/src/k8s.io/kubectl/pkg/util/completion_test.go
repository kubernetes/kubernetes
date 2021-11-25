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

package util

import (
	"net/http"
	"sort"
	"testing"

	"github.com/spf13/cobra"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/rest/fake"
	"k8s.io/kubectl/pkg/cmd/get"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"

	// cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/scheme"
)

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

func TestPodResourceNameAndContainerCompletionFuncNoArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := PodResourceNameAndContainerCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{}, "b")
	checkCompletion(t, comps, []string{"bar"}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func TestPodResourceNameAndContainerCompletionFuncTooManyArgs(t *testing.T) {
	tf, cmd := prepareCompletionTest()
	addPodsToFactory(tf)

	compFunc := PodResourceNameAndContainerCompletionFunc(tf)
	comps, directive := compFunc(cmd, []string{"pod-name", "container-name"}, "")
	checkCompletion(t, comps, []string{}, directive, cobra.ShellCompDirectiveNoFileComp)
}

func prepareCompletionTest() (*cmdtesting.TestFactory, *cobra.Command) {
	tf := cmdtesting.NewTestFactory().WithNamespace("test")
	defer tf.Cleanup()

	streams, _, _, _ := genericclioptions.NewTestIOStreams()
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
