/*
Copyright 2019 The Kubernetes Authors.

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

package filterconfig

import (
	"bytes"
	"fmt"

	fcv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
)

var _ fmt.GoStringer = RequestDigest{}

// GoString produces a golang source expression of the value.
func (rd RequestDigest) GoString() string {
	return fmt.Sprintf("RequestDigest{RequestInfo: %#+v, User: %#+v}", rd.RequestInfo, rd.User)
}

// FmtFlowSchema produces a golang source expression of the value.
func FmtFlowSchema(fs *fcv1a1.FlowSchema) string {
	if fs == nil {
		return "nil"
	}
	var buf bytes.Buffer
	buf.WriteString(fmt.Sprintf("&v1alpha1.FlowSchema{ObjectMeta: %#+v, Spec: v1alpha1.FlowSchemaSpec{PriorityLevelConfiguration: %#+v, MatchingPrecedence: %d, DistinguisherMethod: ",
		fs.ObjectMeta, fs.Spec.PriorityLevelConfiguration,
		fs.Spec.MatchingPrecedence))
	if fs.Spec.DistinguisherMethod == nil {
		buf.WriteString("nil")
	} else {
		buf.WriteString(fmt.Sprintf("&%#+v", *fs.Spec.DistinguisherMethod))
	}
	buf.WriteString(", Rules: []v1alpha1.PolicyRulesWithSubjects{")
	for idx, rule := range fs.Spec.Rules {
		if idx > 0 {
			buf.WriteString(", ")
		}
		buf.WriteString(FmtPolicyRuleSlim(rule))
	}
	buf.WriteString(fmt.Sprintf("}}, Status: %#+v}", fs.Status))
	return buf.String()
}

// FmtPolicyRule produces a golang source expression of the value.
func FmtPolicyRule(rule fcv1a1.PolicyRulesWithSubjects) string {
	return "v1alpha1.PolicyRulesWithSubjects" + FmtPolicyRuleSlim(rule)
}

// FmtPolicyRuleSlim produces a golang source expression of the value
// but without the leading type name.  See above for an example
// context where this is useful.
func FmtPolicyRuleSlim(rule fcv1a1.PolicyRulesWithSubjects) string {
	var buf bytes.Buffer
	buf.WriteString("{Subjects: []v1alpha1.Subject{")
	for jdx, subj := range rule.Subjects {
		if jdx > 0 {
			buf.WriteString(", ")
		}
		buf.WriteString(fmt.Sprintf("{Kind: %q", subj.Kind))
		if subj.User != nil {
			buf.WriteString(fmt.Sprintf(", User: &%#+v", *subj.User))
		}
		if subj.Group != nil {
			buf.WriteString(fmt.Sprintf(", Group: &%#+v", *subj.Group))
		}
		if subj.ServiceAccount != nil {
			buf.WriteString(fmt.Sprintf(", ServiceAcount: &%#+v", *subj.ServiceAccount))
		}
		buf.WriteString("}")
	}
	buf.WriteString(fmt.Sprintf("}, ResourceRules: %#+v, NonResourceRules: %#+v}", rule.ResourceRules, rule.NonResourceRules))
	return buf.String()
}

// FmtUsers produces a golang source expression of the value.
func FmtUsers(list []user.Info) string {
	var buf bytes.Buffer
	buf.WriteString("[]user.Info{")
	for idx, member := range list {
		if idx > 0 {
			buf.WriteString(", ")
		}
		buf.WriteString(fmt.Sprintf("%#+v", member))
	}
	buf.WriteString("}")
	return buf.String()
}

// FmtRequests produces a golang source expression of the value.
func FmtRequests(list []*request.RequestInfo) string {
	var buf bytes.Buffer
	buf.WriteString("[]*request.RequestInfo{")
	for idx, member := range list {
		if idx > 0 {
			buf.WriteString(", ")
		}
		buf.WriteString(fmt.Sprintf("%#+v", member))
	}
	buf.WriteString("}")
	return buf.String()
}
