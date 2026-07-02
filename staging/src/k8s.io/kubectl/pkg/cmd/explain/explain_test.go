/*
Copyright 2022 The Kubernetes Authors.

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

package explain_test

import (
	"errors"
	"path/filepath"
	"regexp"
	"slices"
	"testing"

	"github.com/spf13/cobra"
	"k8s.io/apimachinery/pkg/api/meta"
	sptest "k8s.io/apimachinery/pkg/util/strategicpatch/testing"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/discovery"
	openapiclient "k8s.io/client-go/openapi"
	"k8s.io/client-go/rest"
	clienttestutil "k8s.io/client-go/util/testing"
	"k8s.io/kubectl/pkg/cmd/explain"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/openapi"
)

var (
	testDataPath      = filepath.Join("..", "..", "..", "testdata")
	fakeSchema        = sptest.Fake{Path: filepath.Join(testDataPath, "openapi", "swagger.json")}
	FakeOpenAPISchema = testOpenAPISchema{
		OpenAPISchemaFn: func() (openapi.Resources, error) {
			s, err := fakeSchema.OpenAPISchema()
			if err != nil {
				return nil, err
			}
			return openapi.NewOpenAPIData(s)
		},
	}
)

type testOpenAPISchema struct {
	OpenAPISchemaFn func() (openapi.Resources, error)
}

func TestExplainInvalidArgs(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	flags := explain.NewExplainFlags(genericiooptions.NewTestIOStreamsDiscard())

	opts, err := flags.ToOptions(tf, "kubectl", []string{})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	err = opts.Validate()
	if err.Error() != "You must specify the type of resource to explain. Use \"kubectl api-resources\" for a complete list of supported resources.\n" {
		t.Error("unexpected non-error")
	}

	opts, err = flags.ToOptions(tf, "kubectl", []string{"resource1", "resource2"})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	err = opts.Validate()
	if err.Error() != "We accept only this format: explain RESOURCE\n" {
		t.Error("unexpected non-error")
	}
}

func TestExplainCommandRegistersCompletion(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	ioStreams := genericiooptions.NewTestIOStreamsDiscard()
	cmd := explain.NewCmdExplain("kubectl", tf, ioStreams)
	if cmd.ValidArgsFunction == nil {
		t.Fatal("expected explain command to have ValidArgsFunction set")
	}
}

func TestExplainNotExistResource(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	flags := explain.NewExplainFlags(genericiooptions.NewTestIOStreamsDiscard())

	opts, err := flags.ToOptions(tf, "kubectl", []string{"foo"})
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	err = opts.Validate()
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	err = opts.Run()
	if _, ok := err.(*meta.NoResourceMatchError); !ok {
		t.Fatalf("unexpected error %v", err)
	}
}

type explainTestCase struct {
	Name               string
	Args               []string
	Flags              map[string]string
	ExpectPattern      []string
	ExpectErrorPattern string

	// Custom OpenAPI V3 client to use for the test. If nil, a default one will
	// be provided
	OpenAPIV3SchemaFn func() (openapiclient.Client, error)
}

var explainV2Cases = []explainTestCase{
	{
		Name:          "Basic",
		Args:          []string{"pods"},
		ExpectPattern: []string{`\s*KIND:[\t ]*Pod\s*`},
	},
	{
		Name:          "Recursive",
		Args:          []string{"pods"},
		Flags:         map[string]string{"recursive": "true"},
		ExpectPattern: []string{`\s*KIND:[\t ]*Pod\s*`},
	},
	{
		Name:          "DefaultAPIVersion",
		Args:          []string{"horizontalpodautoscalers"},
		Flags:         map[string]string{"api-version": "autoscaling/v1"},
		ExpectPattern: []string{`\s*VERSION:[\t ]*(v1|autoscaling/v1)\s*`},
	},
	{
		Name:               "NonExistingAPIVersion",
		Args:               []string{"pods"},
		Flags:              map[string]string{"api-version": "v99"},
		ExpectErrorPattern: `couldn't find resource for \"/v99, (Kind=Pod|Resource=pods)\"`,
	},
	{
		Name:               "NonExistingResource",
		Args:               []string{"foo"},
		ExpectErrorPattern: `the server doesn't have a resource type "foo"`,
	},
}

func TestExplainOpenAPIV2(t *testing.T) {
	runExplainTestCases(t, explainV2Cases)
}

func TestExplainOpenAPIV3(t *testing.T) {

	fallbackV3SchemaFn := func() (openapiclient.Client, error) {
		fakeDiscoveryClient := discovery.NewDiscoveryClientForConfigOrDie(&rest.Config{Host: "https://not.a.real.site:65543/"})
		return fakeDiscoveryClient.OpenAPIV3(), nil
	}
	// Returns a client that causes fallback to v2 implementation
	cases := []explainTestCase{
		{
			// No --output, but OpenAPIV3 enabled should fall back to v2 if
			// v2 is not available. Shows this by making openapiv3 client
			// point to a bad URL. So the fact the proper data renders is
			// indication v2 was used instead.
			Name:              "Fallback",
			Args:              []string{"pods"},
			ExpectPattern:     []string{`\s*KIND:[\t ]*Pod\s*`},
			OpenAPIV3SchemaFn: fallbackV3SchemaFn,
		},
		{
			Name:          "NonDefaultAPIVersion",
			Args:          []string{"horizontalpodautoscalers"},
			Flags:         map[string]string{"api-version": "autoscaling/v2"},
			ExpectPattern: []string{`\s*VERSION:[\t ]*(v2|autoscaling/v2)\s*`},
		},
		{
			// Show that explicitly specifying --output plaintext-openapiv2 causes
			// old implementation to be used even though OpenAPIV3 is enabled
			Name:              "OutputPlaintextV2",
			Args:              []string{"pods"},
			Flags:             map[string]string{"output": "plaintext-openapiv2"},
			ExpectPattern:     []string{`\s*KIND:[\t ]*Pod\s*`},
			OpenAPIV3SchemaFn: fallbackV3SchemaFn,
		},
	}
	cases = append(cases, explainV2Cases...)

	runExplainTestCases(t, cases)
}

func runExplainTestCases(t *testing.T, cases []explainTestCase) {
	fakeServer, err := clienttestutil.NewFakeOpenAPIV3Server(filepath.Join(testDataPath, "openapi", "v3"))
	if err != nil {
		t.Fatalf("error starting fake openapi server: %v", err.Error())
	}
	defer fakeServer.HttpServer.Close()

	openapiV3SchemaFn := func() (openapiclient.Client, error) {
		fakeDiscoveryClient := discovery.NewDiscoveryClientForConfigOrDie(&rest.Config{Host: fakeServer.HttpServer.URL})
		return fakeDiscoveryClient.OpenAPIV3(), nil
	}

	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()

	tf.OpenAPISchemaFunc = FakeOpenAPISchema.OpenAPISchemaFn
	tf.ClientConfigVal = cmdtesting.DefaultClientConfig()

	ioStreams, _, buf, _ := genericiooptions.NewTestIOStreams()

	type catchFatal error

	for _, tcase := range cases {

		t.Run(tcase.Name, func(t *testing.T) {

			// Catch os.Exit calls for tests which expect them
			// and replace them with panics that we catch in each test
			// to check if it is expected.
			cmdutil.BehaviorOnFatal(func(str string, code int) {
				panic(catchFatal(errors.New(str)))
			})
			defer cmdutil.DefaultBehaviorOnFatal()

			var err error

			func() {
				defer func() {
					// Catch panic and check at end of test if it is
					// expected.
					if panicErr := recover(); panicErr != nil {
						if e := panicErr.(catchFatal); e != nil {
							err = e
						} else {
							panic(panicErr)
						}
					}
				}()

				if tcase.OpenAPIV3SchemaFn != nil {
					tf.OpenAPIV3ClientFunc = tcase.OpenAPIV3SchemaFn
				} else {
					tf.OpenAPIV3ClientFunc = openapiV3SchemaFn
				}

				cmd := explain.NewCmdExplain("kubectl", tf, ioStreams)
				for k, v := range tcase.Flags {
					if err := cmd.Flags().Set(k, v); err != nil {
						t.Fatal(err)
					}
				}
				cmd.Run(cmd, tcase.Args)
			}()

			for _, rexp := range tcase.ExpectPattern {
				if matched, err := regexp.MatchString(rexp, buf.String()); err != nil || !matched {
					if err != nil {
						t.Error(err)
					} else {
						t.Errorf("expected output to match regex:\n\t%s\ninstead got:\n\t%s", rexp, buf.String())
					}
				}
			}

			if err != nil {
				if matched, regexErr := regexp.MatchString(tcase.ExpectErrorPattern, err.Error()); len(tcase.ExpectErrorPattern) == 0 || regexErr != nil || !matched {
					t.Fatalf("unexpected error: %s did not match regex %s (%v)", err.Error(),
						tcase.ExpectErrorPattern, regexErr)
				}
			} else if len(tcase.ExpectErrorPattern) > 0 {
				t.Fatalf("did not trigger expected error: %s in output:\n%s", tcase.ExpectErrorPattern, buf.String())
			}
		})

		buf.Reset()
	}
}

func TestExplainFieldCompletion(t *testing.T) {
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()
	tf.OpenAPISchemaFunc = FakeOpenAPISchema.OpenAPISchemaFn

	cmd := explain.NewCmdExplain("kubectl", tf, genericiooptions.NewTestIOStreamsDiscard())

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
	comps, directive := cmd.ValidArgsFunction(cmd, []string{"pods"}, "pods.sp")
	if len(comps) != 0 || directive != cobra.ShellCompDirectiveNoFileComp {
		t.Errorf("expected no completions for second arg, got %v (%v)", comps, directive)
	}

	for _, tc := range cases {
		t.Run(tc.toComplete, func(t *testing.T) {
			comps, directive := cmd.ValidArgsFunction(cmd, []string{}, tc.toComplete)
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
				found := slices.Contains(comps, tc.mustContain)
				if !found {
					t.Errorf("expected %q in completions, got %v", tc.mustContain, comps)
				}
			}
		})
	}

	// With --api-version set, explain does not accept group-qualified resource
	// names, so they must not be offered.
	if err := cmd.Flags().Set("api-version", "batch/v1beta1"); err != nil {
		t.Fatal(err)
	}
	comps, directive = cmd.ValidArgsFunction(cmd, []string{}, "cronjobs.b")
	if len(comps) != 0 || directive != cobra.ShellCompDirectiveNoFileComp {
		t.Errorf("expected no completions with --api-version set, got %v (%v)", comps, directive)
	}
}

// OpenAPI V2 specifications retrieval -- should never be called.
func panicOpenAPISchemaFn() (openapi.Resources, error) {
	panic("should never be called")
}

// OpenAPI V3 specifications retrieval does *not* retrieve V2 specifications.
func TestExplainOpenAPIV3DoesNotLoadOpenAPIV2Specs(t *testing.T) {
	// Set up OpenAPI V3 specifications endpoint for explain.
	fakeServer, err := clienttestutil.NewFakeOpenAPIV3Server(filepath.Join(testDataPath, "openapi", "v3"))
	if err != nil {
		t.Fatalf("error starting fake openapi server: %v", err.Error())
	}
	defer fakeServer.HttpServer.Close()
	tf := cmdtesting.NewTestFactory()
	defer tf.Cleanup()
	tf.OpenAPIV3ClientFunc = func() (openapiclient.Client, error) {
		fakeDiscoveryClient := discovery.NewDiscoveryClientForConfigOrDie(&rest.Config{Host: fakeServer.HttpServer.URL})
		return fakeDiscoveryClient.OpenAPIV3(), nil
	}
	// OpenAPI V2 specifications retrieval will panic if called.
	tf.OpenAPISchemaFunc = panicOpenAPISchemaFn

	// Explain the following resources, validating the command does not panic.
	cmd := explain.NewCmdExplain("kubectl", tf, genericiooptions.NewTestIOStreamsDiscard())
	resources := []string{"pods", "services", "endpoints", "configmaps"}
	for _, resource := range resources {
		cmd.Run(cmd, []string{resource})
	}
	// Verify retrieving OpenAPI V2 specifications will panic.
	defer func() {
		if panicErr := recover(); panicErr == nil {
			t.Fatal("expecting panic for openapi v2 retrieval")
		}
	}()
	// Set OpenAPI V2 output flag for explain.
	if err := cmd.Flags().Set("output", "plaintext-openapiv2"); err != nil {
		t.Fatal(err)
	}
	cmd.Run(cmd, []string{"pods"})
}
