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

package format

import (
	"bytes"
	"encoding/json"
	"fmt"

	fcv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// This file provides an easy way to mark a value for formatting to
// `%s` in full detail IF it is printed but without costing a lot of
// CPU or memory if the value is NOT printed.  The API Priority and
// Fairness API objects are formatted into JSON.  The other types of
// objects here are formatted into golang source.

// Stringer marks the given value for custom formatting by this package.
type Stringer struct{ val interface{} }

// Fmt marks the given value for custom formatting by this package.
func Fmt(val interface{}) Stringer {
	return Stringer{val}
}

// String formats to a string in full detail
func (sr Stringer) String() string {
	if sr.val == nil {
		return "nil"
	}
	switch typed := sr.val.(type) {
	case *fcv1a1.FlowSchema,
		fcv1a1.FlowSchema,
		fcv1a1.FlowSchemaSpec,
		fcv1a1.FlowDistinguisherMethod,
		*fcv1a1.FlowDistinguisherMethod,
		*fcv1a1.PolicyRulesWithSubjects,
		fcv1a1.PolicyRulesWithSubjects,
		fcv1a1.Subject,
		fcv1a1.ResourcePolicyRule,
		fcv1a1.NonResourcePolicyRule,
		fcv1a1.FlowSchemaCondition,
		*fcv1a1.PriorityLevelConfiguration,
		fcv1a1.PriorityLevelConfiguration,
		fcv1a1.PriorityLevelConfigurationSpec,
		*fcv1a1.LimitedPriorityLevelConfiguration,
		fcv1a1.LimitedPriorityLevelConfiguration,
		fcv1a1.LimitResponse,
		*fcv1a1.QueuingConfiguration,
		fcv1a1.QueuingConfiguration:
		return ToJSON(sr.val)
	case []user.Info:
		return FmtUsers(typed)
	case []*request.RequestInfo:
		return FmtRequests(typed)
	default:
		return fmt.Sprintf("%#+v", sr.val)
	}
}

// ToJSON converts using encoding/json and handles errors by
// formatting them
func ToJSON(val interface{}) string {
	bs, err := json.Marshal(val)
	str := string(bs)
	if err != nil {
		str = str + "<" + err.Error() + ">"
	}
	return str
}

// FmtPriorityLevelConfiguration returns a golang source expression
// equivalent to the given value
func FmtPriorityLevelConfiguration(pl *fcv1a1.PriorityLevelConfiguration) string {
	if pl == nil {
		return "nil"
	}
	var buf bytes.Buffer
	buf.WriteString(fmt.Sprintf("&v1alpha1.PriorityLevelConfiguration{ObjectMeta: %#+v, Spec: ",
		pl.ObjectMeta))
	BufferPriorityLevelConfigurationSpec(&buf, &pl.Spec)
	buf.WriteString(fmt.Sprintf(", Status: %#+v}", pl.Status))
	return buf.String()
}

// FmtPriorityLevelConfigurationSpec returns a golang source
// expression equivalent to the given value
func FmtPriorityLevelConfigurationSpec(plSpec *fcv1a1.PriorityLevelConfigurationSpec) string {
	var buf bytes.Buffer
	BufferPriorityLevelConfigurationSpec(&buf, plSpec)
	return buf.String()
}

// BufferPriorityLevelConfigurationSpec writes a golang source
// expression for the given value to the given buffer
func BufferPriorityLevelConfigurationSpec(buf *bytes.Buffer, plSpec *fcv1a1.PriorityLevelConfigurationSpec) {
	buf.WriteString(fmt.Sprintf("v1alpha1.PriorityLevelConfigurationSpec{Type: %#v", plSpec.Type))
	if plSpec.Limited != nil {
		buf.WriteString(fmt.Sprintf(", Limited: &v1alpha1.LimitedPriorityLevelConfiguration{AssuredConcurrencyShares:%d, LimitResponse:v1alpha1.LimitResponse{Type:%#v", plSpec.Limited.AssuredConcurrencyShares, plSpec.Limited.LimitResponse.Type))
		if plSpec.Limited.LimitResponse.Queuing != nil {
			buf.WriteString(fmt.Sprintf(", Queuing:&%#+v", *plSpec.Limited.LimitResponse.Queuing))
		}
		buf.WriteString(" } }")
	}
	buf.WriteString("}")
}

// FmtFlowSchema produces a golang source expression of the value.
func FmtFlowSchema(fs *fcv1a1.FlowSchema) string {
	if fs == nil {
		return "nil"
	}
	var buf bytes.Buffer
	buf.WriteString(fmt.Sprintf("&v1alpha1.FlowSchema{ObjectMeta: %#+v, Spec: ",
		fs.ObjectMeta))
	BufferFlowSchemaSpec(&buf, &fs.Spec)
	buf.WriteString(fmt.Sprintf(", Status: %#+v}", fs.Status))
	return buf.String()
}

// FmtFlowSchemaSpec produces a golang source expression equivalent to
// the given spec
func FmtFlowSchemaSpec(fsSpec *fcv1a1.FlowSchemaSpec) string {
	var buf bytes.Buffer
	BufferFlowSchemaSpec(&buf, fsSpec)
	return buf.String()
}

// BufferFlowSchemaSpec writes a golang source expression for the
// given value to the given buffer
func BufferFlowSchemaSpec(buf *bytes.Buffer, fsSpec *fcv1a1.FlowSchemaSpec) {
	buf.WriteString(fmt.Sprintf("v1alpha1.FlowSchemaSpec{PriorityLevelConfiguration: %#+v, MatchingPrecedence: %d, DistinguisherMethod: ",
		fsSpec.PriorityLevelConfiguration,
		fsSpec.MatchingPrecedence))
	if fsSpec.DistinguisherMethod == nil {
		buf.WriteString("nil")
	} else {
		buf.WriteString(fmt.Sprintf("&%#+v", *fsSpec.DistinguisherMethod))
	}
	buf.WriteString(", Rules: []v1alpha1.PolicyRulesWithSubjects{")
	for idx, rule := range fsSpec.Rules {
		if idx > 0 {
			buf.WriteString(", ")
		}
		BufferFmtPolicyRulesWithSubjectsSlim(buf, rule)
	}
	buf.WriteString("}}")
}

// FmtPolicyRulesWithSubjects produces a golang source expression of the value.
func FmtPolicyRulesWithSubjects(rule fcv1a1.PolicyRulesWithSubjects) string {
	return "v1alpha1.PolicyRulesWithSubjects" + FmtPolicyRulesWithSubjectsSlim(rule)
}

// FmtPolicyRulesWithSubjectsSlim produces a golang source expression
// of the value but without the leading type name.  See above for an
// example context where this is useful.
func FmtPolicyRulesWithSubjectsSlim(rule fcv1a1.PolicyRulesWithSubjects) string {
	var buf bytes.Buffer
	BufferFmtPolicyRulesWithSubjectsSlim(&buf, rule)
	return buf.String()
}

// BufferFmtPolicyRulesWithSubjectsSlim writes a golang source
// expression for the given value to the given buffer but excludes the
// leading type name
func BufferFmtPolicyRulesWithSubjectsSlim(buf *bytes.Buffer, rule fcv1a1.PolicyRulesWithSubjects) {
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
