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

	openapi_v2 "github.com/google/gnostic-models/openapiv2"
	"github.com/spf13/cobra"

	sptest "k8s.io/apimachinery/pkg/util/strategicpatch/testing"
	"k8s.io/client-go/discovery"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
)

// openAPIV2DiscoveryClient serves the fake OpenAPI V2 document used by explain's
// field completion. It embeds cmdtesting.FakeCachedDiscoveryClient so it satisfies
// discovery.CachedDiscoveryInterface while only providing the OpenAPI schema.
type openAPIV2DiscoveryClient struct {
	*cmdtesting.FakeCachedDiscoveryClient
	schema *sptest.Fake
}

func (c *openAPIV2DiscoveryClient) OpenAPISchema() (*openapi_v2.Document, error) {
	return c.schema.OpenAPISchema()
}

func newCompletionTestFactory(t *testing.T) *cmdtesting.TestFactory {
	t.Helper()
	tf := cmdtesting.NewTestFactory()
	t.Cleanup(tf.Cleanup)

	fakeSchema := &sptest.Fake{Path: filepath.Join("..", "..", "..", "testdata", "openapi", "swagger.json")}
	dc := &openAPIV2DiscoveryClient{
		FakeCachedDiscoveryClient: cmdtesting.NewFakeCachedDiscoveryClient(),
		schema:                    fakeSchema,
	}
	// The RESTMapper set by NewTestFactory takes precedence, so wiring a discovery
	// client only changes where the OpenAPI schema comes from.
	tf.WithDiscoveryClient(dc)
	return tf
}

// newCompletionCommand returns a command carrying only the flags that
// resourceFieldCompletionFunc reads, so the tests exercise the completion
// function itself rather than the full explain command wiring.
func newCompletionCommand() *cobra.Command {
	cmd := &cobra.Command{}
	cmd.Flags().String("api-version", "", "")
	return cmd
}

var _ discovery.CachedDiscoveryInterface = &openAPIV2DiscoveryClient{}

func TestResourceFieldCompletion(t *testing.T) {
	tf := newCompletionTestFactory(t)
	cmd := newCompletionCommand()
	completeFn := resourceFieldCompletionFunc(tf)

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
			// Group-qualified resource: fields come from a version present in the
			// OpenAPI doc (apps/v1) even if the mapper lists apps/v1beta1 first.
			toComplete:    "deployments.apps.",
			mustContain:   "deployments.apps.spec.",
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
			toComplete:    "cronjobs.b",
			exactResults:  []string{"cronjobs.batch."},
			wantDirective: noSpace,
		},
	}

	// Second argument should always return nothing.
	comps, directive := completeFn(cmd, []string{"pods"}, "pods.sp")
	if len(comps) != 0 || directive != cobra.ShellCompDirectiveNoFileComp {
		t.Errorf("expected no completions for second arg, got %v (%v)", comps, directive)
	}

	for _, tc := range cases {
		t.Run(tc.toComplete, func(t *testing.T) {
			comps, directive := completeFn(cmd, []string{}, tc.toComplete)
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
	tf := newCompletionTestFactory(t)
	cmd := newCompletionCommand()
	completeFn := resourceFieldCompletionFunc(tf)

	// With --api-version set, explain does not accept group-qualified resource
	// names, so they must not be offered.
	if err := cmd.Flags().Set("api-version", "batch/v1beta1"); err != nil {
		t.Fatal(err)
	}
	comps, directive := completeFn(cmd, []string{}, "cronjobs.b")
	if len(comps) != 0 || directive != cobra.ShellCompDirectiveNoFileComp {
		t.Errorf("expected no completions with --api-version set, got %v (%v)", comps, directive)
	}
}
