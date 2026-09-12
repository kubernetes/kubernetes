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
	"slices"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/openapi3"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kubectl/pkg/explain"
	"k8s.io/kubectl/pkg/util/completion"
)

// schemaRefPrefix is the prefix of the references OpenAPI V3 documents use to
// point at the reusable schemas in their components section.
const schemaRefPrefix = "#/components/schemas/"

// resourceFieldCompletionFunc returns a completion function for kubectl explain that completes:
// - resource types when no dot is present (e.g., "pods", "deploy")
// - field paths when a dot is present (e.g., "pods.spec", "pods.spec.containers")
// apiVersion reports the group/version the user pinned, if any; it is read at
// completion time, after the flags have been parsed.
func resourceFieldCompletionFunc(restClientGetter genericclioptions.RESTClientGetter, apiVersion func() string) completion.CompletionFunc {
	return func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		if len(args) > 0 || strings.Contains(toComplete, "..") {
			return nil, cobra.ShellCompDirectiveNoFileComp
		}

		if !strings.Contains(toComplete, ".") {
			// Complete resource type names, appending "." so the next tab press moves
			// into field completion. No verb filter is passed because explain works for
			// any resource, not just those that support GET.
			var comps []string
			for _, res := range completion.CompGetResourceList(restClientGetter, cmd, toComplete) {
				comps = append(comps, res+".")
			}
			return comps, cobra.ShellCompDirectiveNoFileComp | cobra.ShellCompDirectiveNoSpace
		}

		mapper, err := restClientGetter.ToRESTMapper()
		if err != nil {
			return nil, cobra.ShellCompDirectiveNoFileComp
		}

		// Everything up to the last dot is a complete "resource[.group][.fields...]"
		// path; the remainder is the prefix to filter completions by. Parse the path
		// with the same functions explain's Run uses, so completions are offered
		// exactly for the arguments explain accepts: group-qualified resource names
		// are allowed without --api-version, but parsed as field paths with it.
		lastDot := strings.LastIndex(toComplete, ".")
		path, prefix := toComplete[:lastDot], toComplete[lastDot+1:]

		var gvr schema.GroupVersionResource
		var fieldsPath []string
		requestedAPIVersion := apiVersion()
		if requestedAPIVersion == "" {
			gvr, fieldsPath, err = explain.SplitAndParseResourceRequestWithMatchingPrefix(path, mapper)
		} else {
			gvr, fieldsPath, err = explain.SplitAndParseResourceRequest(path, mapper)
		}
		if err != nil {
			return nil, cobra.ShellCompDirectiveNoFileComp
		}

		// The parsers resolve the version through the RESTMapper, which does not
		// know about --api-version. Explain's Run applies the flag on top of the
		// parsed result, so completion has to do the same to describe the same
		// fields the command would print.
		if requestedAPIVersion != "" {
			gv, err := schema.ParseGroupVersion(requestedAPIVersion)
			if err != nil {
				return nil, cobra.ShellCompDirectiveNoFileComp
			}
			gvr.Group = gv.Group
			gvr.Version = gv.Version
		}

		var comps []string
		hasExpandable := false
		expandable, leaves := fieldNamesForGVR(restClientGetter, mapper, gvr, fieldsPath)
		for _, name := range expandable {
			if strings.HasPrefix(name, prefix) {
				comps = append(comps, toComplete[:lastDot+1]+name+".")
				hasExpandable = true
			}
		}
		for _, name := range leaves {
			if strings.HasPrefix(name, prefix) {
				comps = append(comps, toComplete[:lastDot+1]+name)
			}
		}

		if requestedAPIVersion == "" {
			// The user may also still be typing a group-qualified resource name
			// (e.g. "deployments.ap" → "deployments.apps."). Offer the names that
			// extend toComplete and that the parser resolves back to themselves,
			// i.e. that are not shadowed by a shorter resource name.
			resource, _, _ := strings.Cut(toComplete, ".")
			gvrs, _ := mapper.ResourcesFor(schema.GroupVersionResource{Resource: resource})
			seen := map[string]bool{}
			for _, g := range gvrs {
				gr := g.GroupResource().String()
				if seen[gr] || !strings.HasPrefix(gr, toComplete) {
					continue
				}
				seen[gr] = true
				if selected, fields, err := explain.SplitAndParseResourceRequestWithMatchingPrefix(gr, mapper); err == nil && len(fields) == 0 && selected.GroupResource() == g.GroupResource() {
					comps = append(comps, gr+".")
					hasExpandable = true
				}
			}
		}

		// Only suppress the trailing space when there are expandable (dot-ending)
		// completions. For leaf-only results the shell should insert a space after
		// the completed field name.
		directive := cobra.ShellCompDirectiveNoFileComp
		if hasExpandable {
			directive |= cobra.ShellCompDirectiveNoSpace
		}
		return comps, directive
	}
}

// fieldNamesForGVR returns the expandable and leaf field names at fieldsPath within the
// OpenAPI v3 schema for gvr.
func fieldNamesForGVR(restClientGetter genericclioptions.RESTClientGetter, mapper meta.RESTMapper, gvr schema.GroupVersionResource, fieldsPath []string) (expandable, leaves []string) {
	gvk, err := mapper.KindFor(gvr)
	if err != nil || gvk.Empty() {
		// The version may be one the RESTMapper does not know about, for instance
		// when it comes from --api-version. Resolve the kind from the group
		// resource then, but keep looking the schema up in the requested version.
		preferred, err := mapper.KindFor(gvr.GroupResource().WithVersion(""))
		if err != nil || preferred.Empty() {
			return nil, nil
		}
		gvk = gvr.GroupVersion().WithKind(preferred.Kind)
	}

	discoveryClient, err := restClientGetter.ToDiscoveryClient()
	if err != nil {
		return nil, nil
	}
	gvSpec, err := openapi3.NewRoot(discoveryClient.OpenAPIV3()).GVSpec(gvk.GroupVersion())
	if err != nil || gvSpec.Components == nil {
		return nil, nil
	}
	schemas := gvSpec.Components.Schemas

	s := schemaForGVK(schemas, gvk)
	for _, field := range fieldsPath {
		object := resolveToObject(s, schemas, map[string]bool{})
		if object == nil {
			return nil, nil
		}
		next, ok := object.Properties[field]
		if !ok {
			return nil, nil
		}
		s = &next
	}

	object := resolveToObject(s, schemas, map[string]bool{})
	if object == nil {
		return nil, nil
	}
	for name, field := range object.Properties {
		if resolveToObject(&field, schemas, map[string]bool{}) != nil {
			expandable = append(expandable, name)
		} else {
			leaves = append(leaves, name)
		}
	}
	slices.Sort(expandable)
	slices.Sort(leaves)
	return expandable, leaves
}

// schemaForGVK returns the schema the OpenAPI V3 document defines for gvk, or nil
// when the document does not describe that kind.
func schemaForGVK(schemas map[string]*spec.Schema, gvk schema.GroupVersionKind) *spec.Schema {
	for _, s := range schemas {
		values, ok := s.Extensions["x-kubernetes-group-version-kind"].([]interface{})
		if !ok {
			continue
		}
		for _, value := range values {
			candidate, ok := value.(map[string]interface{})
			if !ok {
				continue
			}
			if candidate["group"] == gvk.Group && candidate["version"] == gvk.Version && candidate["kind"] == gvk.Kind {
				return s
			}
		}
	}
	return nil
}

// resolveToObject unwraps references and arrays until it reaches the schema
// holding named sub-fields, or nil for schemas that cannot be drilled into
// (primitives, maps, ...). visited guards against reference cycles in the schema.
func resolveToObject(s *spec.Schema, schemas map[string]*spec.Schema, visited map[string]bool) *spec.Schema {
	if s == nil {
		return nil
	}
	if ref, ok := referencedSchemaName(s); ok {
		if visited[ref] {
			return nil
		}
		visited[ref] = true
		return resolveToObject(schemas[ref], schemas, visited)
	}
	if s.Items != nil && s.Items.Schema != nil {
		return resolveToObject(s.Items.Schema, schemas, visited)
	}
	if len(s.Properties) > 0 {
		return s
	}
	return nil
}

// referencedSchemaName returns the name of the components schema s points at,
// either through $ref directly or through the single-element allOf Kubernetes
// generates when a description accompanies the reference.
func referencedSchemaName(s *spec.Schema) (string, bool) {
	ref := s.Ref.String()
	if ref == "" && len(s.AllOf) == 1 {
		ref = s.AllOf[0].Ref.String()
	}
	return strings.CutPrefix(ref, schemaRefPrefix)
}
