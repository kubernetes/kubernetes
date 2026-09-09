/*
Copyright The Kubernetes Authors.

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

package explain

import (
	"path/filepath"
	"slices"
	"testing"

	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/discovery"
	discoveryfake "k8s.io/client-go/discovery/fake"
	openapiclient "k8s.io/client-go/openapi"
	"k8s.io/client-go/rest"
	clientgotesting "k8s.io/client-go/testing"
	clienttestutil "k8s.io/client-go/util/testing"
)

// completionDiscoveryClient serves the API resources the RESTMapper is built
// from, and the OpenAPI V3 documents field completion reads. Anything else the
// interface requires is inherited from the fake discovery client.
type completionDiscoveryClient struct {
	discovery.DiscoveryInterface
	openAPIV3 openapiclient.Client
}

func (c *completionDiscoveryClient) OpenAPIV3() openapiclient.Client { return c.openAPIV3 }
func (c *completionDiscoveryClient) Fresh() bool                     { return true }
func (c *completionDiscoveryClient) Invalidate()                     {}

var _ discovery.CachedDiscoveryInterface = &completionDiscoveryClient{}

// completionAPIResources are the resources the completion tests resolve. The
// group versions match the OpenAPI V3 documents under testdata, except for the
// autoscaling group, which is served in two versions on purpose: v2 is the
// preferred one, so pinning v1 through --api-version has to be visible in the
// completions.
var completionAPIResources = []*metav1.APIResourceList{
	{
		GroupVersion: "v1",
		APIResources: []metav1.APIResource{
			{Name: "pods", Namespaced: true, Kind: "Pod"},
		},
	},
	{
		GroupVersion: "apps/v1",
		APIResources: []metav1.APIResource{
			{Name: "deployments", Namespaced: true, Kind: "Deployment"},
		},
	},
	{
		GroupVersion: "autoscaling/v2",
		APIResources: []metav1.APIResource{
			{Name: "horizontalpodautoscalers", Namespaced: true, Kind: "HorizontalPodAutoscaler"},
		},
	},
	{
		GroupVersion: "autoscaling/v1",
		APIResources: []metav1.APIResource{
			{Name: "horizontalpodautoscalers", Namespaced: true, Kind: "HorizontalPodAutoscaler"},
		},
	},
}

// newCompletionRESTClientGetter returns a RESTClientGetter backed by the fake
// API resources above and by the OpenAPI V3 documents under testdata.
func newCompletionRESTClientGetter(t *testing.T) genericclioptions.RESTClientGetter {
	t.Helper()

	// explain_test.go lives in the external test package, so its testdata path
	// is spelled out again here.
	fakeServer, err := clienttestutil.NewFakeOpenAPIV3Server(filepath.Join("..", "..", "..", "testdata", "openapi", "v3"))
	if err != nil {
		t.Fatalf("error starting fake openapi server: %v", err)
	}
	t.Cleanup(fakeServer.HttpServer.Close)

	openAPIV3 := discovery.NewDiscoveryClientForConfigOrDie(&rest.Config{Host: fakeServer.HttpServer.URL}).OpenAPIV3()
	discoveryClient := &completionDiscoveryClient{
		DiscoveryInterface: &discoveryfake.FakeDiscovery{
			Fake: &clientgotesting.Fake{Resources: completionAPIResources},
		},
		openAPIV3: openAPIV3,
	}

	return genericclioptions.NewTestConfigFlags().WithDiscoveryClient(discoveryClient)
}

func TestResourceFieldCompletion(t *testing.T) {
	restClientGetter := newCompletionRESTClientGetter(t)
	completeFn := resourceFieldCompletionFunc(restClientGetter, func() string { return "" })

	noSpace := cobra.ShellCompDirectiveNoFileComp | cobra.ShellCompDirectiveNoSpace

	cases := []struct {
		toComplete    string
		mustContain   string   // one completion that must be present
		exactResults  []string // if set, the full result must equal this
		wantDirective cobra.ShellCompDirective
	}{
		{
			toComplete:    "pods.",
			mustContain:   "pods.spec.",
			wantDirective: noSpace,
		},
		{
			toComplete:    "pods.spec.",
			mustContain:   "pods.spec.containers.",
			wantDirective: noSpace,
		},
		{
			toComplete:    "pods.spec.con",
			exactResults:  []string{"pods.spec.containers."},
			wantDirective: noSpace,
		},
		{
			// Array field: containers is []Container, so its sub-fields should be offered.
			toComplete:    "pods.spec.containers.",
			mustContain:   "pods.spec.containers.image",
			wantDirective: noSpace,
		},
		{
			toComplete:    "pods.sp",
			mustContain:   "pods.spec.",
			wantDirective: noSpace,
		},
		{
			// Leaf (string) field must appear without a trailing dot, and the shell
			// must insert a space after selection (no NoSpace directive for leaf-only results).
			toComplete:    "pods.metadata.na",
			mustContain:   "pods.metadata.name",
			wantDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			// After selecting a leaf field the completion must return nothing.
			toComplete:    "pods.metadata.name.",
			exactResults:  []string{},
			wantDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			// Map field: nodeSelector is map[string]string, which cannot be drilled into.
			toComplete:    "pods.spec.nodeSelector.",
			exactResults:  []string{},
			wantDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			toComplete:    "deployments.apps.",
			mustContain:   "deployments.apps.spec.",
			wantDirective: noSpace,
		},
		{
			// A group-qualified resource name with no field matching the prefix.
			toComplete:    "deployments.app",
			exactResults:  []string{"deployments.apps."},
			wantDirective: noSpace,
		},
		{
			// Mid-word group suffix: the group-qualified name is offered exactly
			// once even though the resource is served in multiple versions, and
			// matching field names are offered alongside it.
			toComplete:    "horizontalpodautoscalers.a",
			exactResults:  []string{"horizontalpodautoscalers.apiVersion", "horizontalpodautoscalers.autoscaling."},
			wantDirective: noSpace,
		},
		{
			// Fields come from the group's preferred version (autoscaling/v2),
			// which is the version explain describes without --api-version:
			// metrics only exists there.
			toComplete:    "horizontalpodautoscalers.spec.m",
			exactResults:  []string{"horizontalpodautoscalers.spec.metrics.", "horizontalpodautoscalers.spec.maxReplicas", "horizontalpodautoscalers.spec.minReplicas"},
			wantDirective: noSpace,
		},
		{
			toComplete:    "horizontalpodautoscalers.spec.metrics.",
			mustContain:   "horizontalpodautoscalers.spec.metrics.resource.",
			wantDirective: noSpace,
		},
	}

	// Second argument should always return nothing.
	comps, directive := completeFn(newCompletionCommand(), []string{"pods"}, "pods.sp")
	if len(comps) != 0 || directive != cobra.ShellCompDirectiveNoFileComp {
		t.Errorf("expected no completions for second arg, got %v (%v)", comps, directive)
	}

	for _, tc := range cases {
		t.Run(tc.toComplete, func(t *testing.T) {
			comps, directive := completeFn(newCompletionCommand(), []string{}, tc.toComplete)
			if directive != tc.wantDirective {
				t.Errorf("directive: got %v, want %v", directive, tc.wantDirective)
			}
			if tc.exactResults != nil {
				if len(comps) != len(tc.exactResults) {
					t.Fatalf("completions: got %v, want %v", comps, tc.exactResults)
				}
				for i, want := range tc.exactResults {
					if comps[i] != want {
						t.Errorf("completion[%d]: got %q, want %q", i, comps[i], want)
					}
				}
			}
			if tc.mustContain != "" {
				if !slices.Contains(comps, tc.mustContain) {
					t.Errorf("expected %q in completions, got %v", tc.mustContain, comps)
				}
			}
		})
	}
}

func TestResourceFieldCompletionWithAPIVersion(t *testing.T) {
	restClientGetter := newCompletionRESTClientGetter(t)

	cases := []struct {
		name          string
		apiVersion    string
		toComplete    string
		exactResults  []string
		wantDirective cobra.ShellCompDirective
	}{
		{
			// autoscaling/v1 has targetCPUUtilizationPercentage, the preferred
			// version the mapper resolves to (autoscaling/v2) does not, so the
			// completions have to follow the pinned version.
			name:          "fields come from the pinned version",
			apiVersion:    "autoscaling/v1",
			toComplete:    "horizontalpodautoscalers.spec.t",
			exactResults:  []string{"horizontalpodautoscalers.spec.targetCPUUtilizationPercentage"},
			wantDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			// The fields of the preferred version must not leak into a pinned one.
			name:          "fields absent from the pinned version are not offered",
			apiVersion:    "autoscaling/v1",
			toComplete:    "horizontalpodautoscalers.spec.metric",
			exactResults:  []string{},
			wantDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			// With --api-version set, explain does not accept group-qualified
			// resource names, so they must not be offered.
			name:          "group-qualified resource names are not offered",
			apiVersion:    "apps/v1",
			toComplete:    "deployments.app",
			exactResults:  []string{},
			wantDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			// A version the server does not serve has no fields to offer.
			name:          "unknown version yields no completions",
			apiVersion:    "autoscaling/v2beta1",
			toComplete:    "horizontalpodautoscalers.spec.m",
			exactResults:  []string{},
			wantDirective: cobra.ShellCompDirectiveNoFileComp,
		},
		{
			name:          "malformed api version yields no completions",
			apiVersion:    "a/b/c",
			toComplete:    "pods.spec.",
			exactResults:  []string{},
			wantDirective: cobra.ShellCompDirectiveNoFileComp,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			completeFn := resourceFieldCompletionFunc(restClientGetter, func() string { return tc.apiVersion })
			comps, directive := completeFn(newCompletionCommand(), []string{}, tc.toComplete)
			if directive != tc.wantDirective {
				t.Errorf("directive: got %v, want %v", directive, tc.wantDirective)
			}
			if len(comps) != len(tc.exactResults) {
				t.Fatalf("completions: got %v, want %v", comps, tc.exactResults)
			}
			for i, want := range tc.exactResults {
				if comps[i] != want {
					t.Errorf("completion[%d]: got %q, want %q", i, comps[i], want)
				}
			}
		})
	}
}

// newCompletionCommand returns the command the completion function is invoked
// with. It reads no flags off it, so a bare command is enough.
func newCompletionCommand() *cobra.Command {
	return &cobra.Command{}
}
