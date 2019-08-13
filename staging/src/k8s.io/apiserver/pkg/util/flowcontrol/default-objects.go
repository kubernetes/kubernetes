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

package flowcontrol

import (
	rmv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
)

// DefaultPriorityLevelConfigurationObjects returns the instances of
// the ones defined in the KEP, with system-top in position 0 and
// workload-low in position 1
func DefaultPriorityLevelConfigurationObjects() []*rmv1a1.PriorityLevelConfiguration {
	return []*rmv1a1.PriorityLevelConfiguration{
		pl("system-top",
			rmv1a1.PriorityLevelConfigurationSpec{Exempt: true},
		),
		pl("workload-low", rmv1a1.PriorityLevelConfigurationSpec{
			GlobalDefault:            true,
			AssuredConcurrencyShares: 100,
			Queues:                   128,
			HandSize:                 6,
			QueueLengthLimit:         100}),
		pl("system-high", rmv1a1.PriorityLevelConfigurationSpec{
			AssuredConcurrencyShares: 100,
			Queues:                   128,
			HandSize:                 6,
			QueueLengthLimit:         10}),
		pl("system-low", rmv1a1.PriorityLevelConfigurationSpec{
			AssuredConcurrencyShares: 30,
			Queues:                   1,
			QueueLengthLimit:         1000}),
		pl("workload-high", rmv1a1.PriorityLevelConfigurationSpec{
			AssuredConcurrencyShares: 30,
			Queues:                   128,
			HandSize:                 6,
			QueueLengthLimit:         100}),
	}
}

// DefaultFlowSchemaObjects returns instances of the FlowSchema
// objects shown in the Example Configuration section of the KEP
func DefaultFlowSchemaObjects() []*rmv1a1.FlowSchema {
	verbAll := []string{rmv1a1.VerbAll}
	apiGroupAll := []string{rmv1a1.APIGroupAll}
	apiGroupCore := []string{""}
	resourceAll := []string{rmv1a1.ResourceAll}
	ruleRscAll := rule(verbAll, apiGroupAll, resourceAll)
	ruleNonRscAll := rmv1a1.PolicyRule{
		Verbs:           verbAll,
		NonResourceURLs: []string{rmv1a1.NonResourceAll}}
	subjectsAll := groups(user.AllAuthenticated, user.AllUnauthenticated)
	return []*rmv1a1.FlowSchema{
		fs("system-top", 1500, "",
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: groups(user.SystemPrivilegedGroup),
				Rule:     ruleRscAll},
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: groups(user.SystemPrivilegedGroup),
				Rule:     ruleNonRscAll},
		),
		fs("system-high", 2500, rmv1a1.FlowDistinguisherMethodByUserType,
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: groups(user.NodesGroup),
				Rule:     rule(verbAll, apiGroupCore, []string{"nodes"})},
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: groups(user.NodesGroup),
				// Namespace: "kube-system",
				Rule: ruleRscAll},
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: svcAccount("*"),
				// Namespace: "kube-system",
				Rule: rule(verbAll, apiGroupCore, []string{"endpoints", "configmaps", "leases"})},
		),
		fs("system-low", 3500, rmv1a1.FlowDistinguisherMethodByUserType,
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: svcAccount("generic-garbage-collector"),
				Rule:     ruleRscAll}),
		// The KEP calls for workload-high to do a negated match on
		// subject, but the models do not support negated matches.  So
		// instead we carve out an exception using a positive match on
		// Subject at a logically higher matching precedence.
		fs("workload-low", 5500,
			rmv1a1.FlowDistinguisherMethodByNamespaceType,
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: svcAccount("*"),
				Rule:     ruleRscAll}),
		newFSAllObj("workload-high", "workload-high", 7500, false, subjectsAll),
	}
}

func pl(name string, spec rmv1a1.PriorityLevelConfigurationSpec) *rmv1a1.PriorityLevelConfiguration {
	return &rmv1a1.PriorityLevelConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       spec}
}

func fs(name string, matchingPrecedence int32, dmType rmv1a1.FlowDistinguisherMethodType, rules ...rmv1a1.PolicyRuleWithSubjects) *rmv1a1.FlowSchema {
	return fs2(name, name, matchingPrecedence, dmType, rules...)
}

func fs2(name, plName string, matchingPrecedence int32, dmType rmv1a1.FlowDistinguisherMethodType, rules ...rmv1a1.PolicyRuleWithSubjects) *rmv1a1.FlowSchema {
	var dm *rmv1a1.FlowDistinguisherMethod
	if dmType != "" {
		dm = &rmv1a1.FlowDistinguisherMethod{Type: dmType}
	}
	return &rmv1a1.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: rmv1a1.FlowSchemaSpec{
			PriorityLevelConfiguration: rmv1a1.PriorityLevelConfigurationReference{plName},
			MatchingPrecedence:         matchingPrecedence,
			DistinguisherMethod:        dm,
			Rules:                      rules},
	}

}

func newFSAllObj(fsName, plName string, matchingPrecedence int32, exempt bool, subjects []rmv1a1.Subject) *rmv1a1.FlowSchema {
	verbAll := []string{rmv1a1.VerbAll}
	apiGroupAll := []string{rmv1a1.APIGroupAll}
	resourceAll := []string{rmv1a1.ResourceAll}
	dmType := rmv1a1.FlowDistinguisherMethodByNamespaceType
	if exempt {
		dmType = rmv1a1.FlowDistinguisherMethodByUserType
	}
	return fs2(fsName, plName, matchingPrecedence, dmType,
		rmv1a1.PolicyRuleWithSubjects{
			Subjects: subjects,
			Rule:     rule(verbAll, apiGroupAll, resourceAll)},
		rmv1a1.PolicyRuleWithSubjects{
			Subjects: subjects,
			Rule: rmv1a1.PolicyRule{
				Verbs:           verbAll,
				NonResourceURLs: []string{rmv1a1.NonResourceAll}}},
	)
}

func groups(names ...string) []rmv1a1.Subject {
	ans := make([]rmv1a1.Subject, len(names))
	for idx, name := range names {
		ans[idx] = rmv1a1.Subject{Kind: "Group", Name: name}
	}
	return ans
}

func svcAccount(name string) []rmv1a1.Subject {
	return []rmv1a1.Subject{{Kind: "ServiceAccount", Name: name, Namespace: "kube-system"}}
}

func rule(verbs []string, groups []string, resources []string) rmv1a1.PolicyRule {
	return rmv1a1.PolicyRule{Verbs: verbs, APIGroups: groups, Resources: resources}
}
