/*
Copyright 2026 The Kubernetes Authors.

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
	"bytes"
	"io"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/kube-openapi/pkg/util/proto"
	"k8s.io/kubectl/pkg/cmd/apiresources"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/explain"
)

// resourceFieldCompletionFunc returns a completion function for kubectl explain that completes:
// - resource types when no dot is present (e.g., "pods", "deploy")
// - field paths when a dot is present (e.g., "pods.spec", "pods.spec.containers")
func resourceFieldCompletionFunc(f cmdutil.Factory) func(*cobra.Command, []string, string) ([]string, cobra.ShellCompDirective) {
	return func(cmd *cobra.Command, args []string, toComplete string) ([]string, cobra.ShellCompDirective) {
		if len(args) > 0 || strings.Contains(toComplete, "..") {
			return nil, cobra.ShellCompDirectiveNoFileComp
		}

		if !strings.Contains(toComplete, ".") {
			// Complete resource type names, appending "." so the next tab press moves
			// into field completion. Listed directly (not via the shared
			// util/completion helper) because that helper hardcodes a "get" verb
			// filter, and explain works for any resource, not just those that
			// support GET.
			buf := new(bytes.Buffer)
			o := apiresources.NewAPIResourceOptions(genericiooptions.IOStreams{In: os.Stdin, Out: buf, ErrOut: io.Discard})
			o.PrintFlags.OutputFormat = new("name")
			o.Cached = true
			if err := o.Complete(f, cmd, nil); err != nil {
				return nil, cobra.ShellCompDirectiveNoFileComp
			}
			_ = o.RunAPIResources()
			var comps []string
			for res := range strings.SplitSeq(buf.String(), "\n") {
				if res != "" && strings.HasPrefix(res, toComplete) {
					comps = append(comps, res+".")
				}
			}
			return comps, cobra.ShellCompDirectiveNoFileComp | cobra.ShellCompDirectiveNoSpace
		}

		mapper, err := f.ToRESTMapper()
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
		apiVersion, _ := cmd.Flags().GetString("api-version")
		if apiVersion == "" {
			gvr, fieldsPath, err = explain.SplitAndParseResourceRequestWithMatchingPrefix(path, mapper)
		} else {
			gvr, fieldsPath, err = explain.SplitAndParseResourceRequest(path, mapper)
		}
		if err != nil {
			return nil, cobra.ShellCompDirectiveNoFileComp
		}

		var comps []string
		hasExpandable := false
		expandable, leaves := fieldNamesForGVR(f, mapper, gvr, fieldsPath)
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

		if apiVersion == "" {
			// The user may also still be typing a group-qualified resource name
			// (e.g. "deployments.ap" → "deployments.apps."). Offer the names that
			// extend toComplete and that the parser resolves back to themselves,
			// i.e. that are not shadowed by a shorter resource name.
			resource := toComplete[:strings.Index(toComplete, ".")]
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
// OpenAPI v2 schema for gvr.
// TODO: use the OpenAPI v3 schema so that CRD fields are always complete.
func fieldNamesForGVR(f cmdutil.Factory, mapper meta.RESTMapper, gvr schema.GroupVersionResource, fieldsPath []string) (expandable, leaves []string) {
	openAPIResources, err := f.OpenAPISchema()
	if err != nil {
		return nil, nil
	}
	// The version picked by the parser, or the group's preferred version, may be
	// absent from the OpenAPI document; try both.
	var s proto.Schema
	for _, v := range []schema.GroupVersionResource{gvr, gvr.GroupResource().WithVersion("")} {
		if gvk, err := mapper.KindFor(v); err == nil && !gvk.Empty() {
			if s = openAPIResources.LookupResource(gvk); s != nil {
				break
			}
		}
	}
	if s == nil {
		return nil, nil
	}
	s, err = explain.LookupSchemaForField(s, fieldsPath)
	if err != nil {
		return nil, nil
	}
	kind := resolveToKind(s, map[string]bool{})
	if kind == nil {
		return nil, nil
	}
	for _, name := range kind.Keys() {
		if resolveToKind(kind.Fields[name], map[string]bool{}) != nil {
			expandable = append(expandable, name)
		} else {
			leaves = append(leaves, name)
		}
	}
	return expandable, leaves
}

// resolveToKind unwraps references and arrays until it reaches the object schema
// holding named sub-fields, or nil for schemas that cannot be drilled into
// (primitives, maps, ...). visited guards against reference cycles in the schema.
func resolveToKind(s proto.Schema, visited map[string]bool) *proto.Kind {
	switch t := s.(type) {
	case *proto.Kind:
		return t
	case *proto.Array:
		if t.SubType != nil {
			return resolveToKind(t.SubType, visited)
		}
	case proto.Reference:
		if sub := t.SubSchema(); sub != nil && !visited[t.Reference()] {
			visited[t.Reference()] = true
			return resolveToKind(sub, visited)
		}
	}
	return nil
}
