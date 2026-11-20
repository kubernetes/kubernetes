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

package coverage

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
)

// Shape is one request shape of a resource.
type Shape struct {
	Operation   admission.Operation
	Subresource string // "" for the main resource
}

// Requirement declares that every admission hook that intercepts any of
// Sources on Resource must also intercept Target on Resource; otherwise the
// hook is in violation for Target requests.
//
// Version is intentionally absent: coverage is version-agnostic, and target and
// sources share Resource so a source can be evaluated as the same request with
// a different operation and subresource.
//
// Hook selectors are evaluated only against the target request's objects, so
// the target's admission object should carry the same metadata as the
// sources' objects (true for subresources backed by the parent object, such
// as pods/resize). Avoid requirements on namespaces unless that has been
// thought through: the namespace selector reads a namespace's labels from the
// request object only when there is no subresource.
type Requirement struct {
	Resource schema.GroupResource
	Target   Shape
	Sources  []Shape
}

// Violation describes a coverage violation for one hook and one request.
type Violation struct {
	Resource schema.GroupResource
	Target   Shape
	// Source is the first source, in declaration order, that the hook
	// intercepts. Callers leave Source empty when coverage could not be
	// determined (for example, because selector evaluation failed).
	Source Shape
}

// TargetString returns the target as used in messages, e.g. "UPDATE pods/dynamic".
func (v Violation) TargetString() string { return shapeString(v.Resource, v.Target) }

// SourceString returns the source as used in messages, e.g. "CREATE pods", or
// "" if Source is empty, so that it can't render as a bare resource.
func (v Violation) SourceString() string {
	if v.Source.Operation == "" {
		return ""
	}
	return shapeString(v.Resource, v.Source)
}

// RuleHint describes the rule a hook needs in order to intercept the target,
// for use in "add <RuleHint> to the hook's rules" messages, e.g.
// `a rule for UPDATE on "pods/dynamic" (or "pods/*")`, or
// `a rule for UPDATE on "pods" (or "pods/*")` for a main-resource target, since
// "pods/*" matches the main resource too. The operation is included because
// adding the resource to a rule that lacks the target's operation does not
// help. Rules carry the group separately, so it is omitted.
func (v Violation) RuleHint() string {
	entry := v.Resource.Resource
	if v.Target.Subresource != "" {
		entry += "/" + v.Target.Subresource
	}
	return fmt.Sprintf("a rule for %s on %q (or %q)", v.Target.Operation, entry, v.Resource.Resource+"/*")
}

func shapeString(gr schema.GroupResource, s Shape) string {
	r := gr.String()
	if s.Subresource != "" {
		r += "/" + s.Subresource
	}
	return string(s.Operation) + " " + r
}

// HookKind identifies the kind of admission hook, and so the plugin that
// dispatches it. The values are stable: they name the plugin's coverage audit
// annotation (see AuditAnnotationKey).
type HookKind string

// Hook kinds, one per admission plugin that dispatches configurable hooks.
const (
	WebhookValidating HookKind = "validating-webhook"
	WebhookMutating   HookKind = "mutating-webhook"
	PolicyValidating  HookKind = "validating-admission-policy"
	PolicyMutating    HookKind = "mutating-admission-policy"
)
