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

// Package equivalents enforces admission-equivalent coverage uniformly for all dynamic admission
// plugins: a hook that applies to an admission equivalent of a request (see admission.Equivalent)
// must also apply to the request itself.
//
// Plugins call Targets once per request, Check for each hook, and Reject with the violations.
package equivalents

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	admissionmetrics "k8s.io/apiserver/pkg/admission/metrics"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	"k8s.io/klog/v2"
)

const (
	// AuditAnnotationKey records the violations that caused a request to be rejected for
	// admission-equivalent coverage.
	AuditAnnotationKey = "equivalents.admission.k8s.io/violations"

	// maxViolationsInMessage bounds the error message. The audit annotation and metrics include
	// all violations.
	maxViolationsInMessage = 5
)

// Hook identifies a dynamic admission hook in coverage errors, audit annotations and metrics.
type Hook struct {
	// Plugin is the admission plugin name, e.g. "ValidatingAdmissionWebhook".
	Plugin string
	// Name is the webhook name or the policy name.
	Name string
	// Configuration is the webhook configuration name. Empty for policies.
	Configuration string
	// Binding is the policy binding name. Empty for webhooks.
	Binding string
}

// Targets returns the admission equivalents of the request attr admitted with o, resolved to
// resource operations. It returns nil if the request's resource or subresource declares no
// admission equivalents, in which case there is nothing to check.
func Targets(attr admission.Attributes, o admission.ObjectInterfaces) []ResourceOperation {
	g, ok := o.(admission.EquivalentsGetter)
	if !ok {
		return nil
	}
	return expand(resourceOperationOf(attr), g.GetAdmissionEquivalents())
}

// Check checks the given hook against the request attr and its admission equivalents targets,
// and returns a violation if the hook applies to a target but not to the request.
//
// matches reports whether the hook's static match criteria (rules and selectors, but not
// matchConditions) select the given attributes. Its bool is ignored when it returns an error.
// If the hook's rules don't cover the attributes, matches must return (false, nil), even if a
// selector would fail to evaluate. Errors that don't depend on the resource, subresource or
// operation are fine.
//
// The hook covers the request only if matches returns (true, nil) for it. Otherwise:
//   - If the request was evaluated cleanly, a target that matches or fails to evaluate is a
//     violation, so that evaluation failures fail closed.
//   - If the request failed to evaluate, only a target that matches cleanly is a violation.
func Check(attr admission.Attributes, targets []ResourceOperation, hook Hook, matches func(admission.Attributes) (bool, error)) (Violation, bool) {
	ok, sourceErr := matches(attr)
	if sourceErr == nil && ok {
		return Violation{}, false
	}
	for _, t := range targets {
		ok, err := matches(&attrWithResourceOperation{Attributes: attr, target: t})
		switch {
		case sourceErr != nil:
			// Dispatch hits the same error on the request and handles it as it would on the
			// target (webhooks reject; policies apply failurePolicy), so a target error isn't a
			// violation; a clean target match is. Because matches cleanly rejects attributes its
			// rules don't cover, a source error can't hide a rules gap.
			if err == nil && ok {
				return Violation{hook: hook, target: t, sourceErr: sourceErr}, true
			}
		case err != nil || ok:
			return Violation{hook: hook, target: t, targetErr: err}, true
		}
	}
	return Violation{}, false
}

// Reject returns a Forbidden error describing violations, in order, and records the audit
// annotation and metrics for the rejection. It returns nil if there are no violations.
func Reject(ctx context.Context, attr admission.Attributes, violations []Violation) error {
	if len(violations) == 0 {
		return nil
	}
	source := resourceOperationOf(attr)
	messages := make([]string, len(violations))
	hint := false
	for i, v := range violations {
		messages[i] = v.message(source)
		hint = hint || v.sourceErr == nil
		admissionmetrics.Metrics.ObserveEquivalentCoverageRejection(ctx, v.hook.Plugin, source.resource.GroupResource().String(), source.subresource, v.hook.Name)
	}
	if err := attr.AddAnnotation(AuditAnnotationKey, strings.Join(messages, "; ")); err != nil {
		klog.ErrorS(err, "Failed to record admission-equivalent coverage audit annotation")
	}

	shown := messages
	if len(shown) > maxViolationsInMessage {
		shown = shown[:maxViolationsInMessage]
	}
	var b strings.Builder
	fmt.Fprintf(&b, "request to %s is not covered by admission hooks that apply to its admission equivalents: %s", source.path(), strings.Join(shown, "; "))
	if more := len(messages) - len(shown); more > 0 {
		fmt.Fprintf(&b, "; and %d more", more)
	}
	if hint {
		fmt.Fprintf(&b, ". Add %q to the rules of these hooks, or remove their rules for the listed resources.", source.path())
	} else {
		b.WriteString(".")
	}
	return admission.NewForbidden(attr, errors.New(b.String()))
}

// Violation is an admission failure caused by a hook that covers an admission equivalent of the
// request but not the request itself.
type Violation struct {
	hook   Hook
	target ResourceOperation
	// targetErr is set if the hook's match criteria could not be evaluated for target.
	targetErr error
	// sourceErr is set if the hook's match criteria could not be evaluated for the request, but
	// matched target.
	sourceErr error
}

// message builds a user-facing summary of the violation.
func (v Violation) message(source ResourceOperation) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s %q", v.hook.Plugin, v.hook.Name)
	switch {
	case v.hook.Binding != "":
		fmt.Fprintf(&b, " (binding %q)", v.hook.Binding)
	case v.hook.Configuration != "":
		fmt.Fprintf(&b, " (configuration %q)", v.hook.Configuration)
	}
	if v.sourceErr != nil {
		fmt.Fprintf(&b, " applies to %s but could not be evaluated for %s (%v)", v.target, source, v.sourceErr)
		return b.String()
	}
	fmt.Fprintf(&b, " applies to %s but not %s", v.target, source)
	if v.targetErr != nil {
		fmt.Fprintf(&b, " (match criteria could not be evaluated: %v)", v.targetErr)
	}
	return b.String()
}

// ResourceOperation is a resource or subresource and the operation requested on it: the part of
// a request that differs between a request and its admission equivalents.
type ResourceOperation struct {
	resource    schema.GroupVersionResource
	subresource string
	operation   admission.Operation
}

func resourceOperationOf(attr admission.Attributes) ResourceOperation {
	return ResourceOperation{resource: attr.GetResource(), subresource: attr.GetSubresource(), operation: attr.GetOperation()}
}

// path returns "resource" or "resource/subresource".
func (r ResourceOperation) path() string {
	if r.subresource == "" {
		return r.resource.Resource
	}
	return r.resource.Resource + "/" + r.subresource
}

// String returns e.g. "widgets/resize (UPDATE)".
func (r ResourceOperation) String() string {
	return fmt.Sprintf("%s (%s)", r.path(), r.operation)
}

// expand resolves eqs against source into targets, in declaration order, then operation order.
func expand(source ResourceOperation, eqs []admission.Equivalent) []ResourceOperation {
	var targets []ResourceOperation
	for _, eq := range eqs {
		ops := eq.Operations
		if len(ops) == 0 {
			ops = []admission.Operation{source.operation}
		}
		for _, op := range ops {
			targets = append(targets, ResourceOperation{resource: source.resource, subresource: eq.Subresource, operation: op})
		}
	}
	return targets
}

// attrWithResourceOperation describes a hypothetical request to an admission equivalent, for
// matchers only. The kind is unchanged: matchers use it to describe a match, not to decide one.
// Annotations are rejected: the request is never made.
type attrWithResourceOperation struct {
	admission.Attributes
	target ResourceOperation
}

var errReadOnly = errors.New("attributes of a hypothetical admission-equivalent request are read-only")

func (a *attrWithResourceOperation) GetResource() schema.GroupVersionResource {
	return a.target.resource
}
func (a *attrWithResourceOperation) GetSubresource() string             { return a.target.subresource }
func (a *attrWithResourceOperation) GetOperation() admission.Operation  { return a.target.operation }
func (a *attrWithResourceOperation) AddAnnotation(string, string) error { return errReadOnly }
func (a *attrWithResourceOperation) AddAnnotationWithLevel(string, string, auditinternal.Level) error {
	return errReadOnly
}
