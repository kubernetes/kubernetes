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

package bootstrap

import (
	rmv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
)

// all the preserving objects
var (
	PreservingPriorityLevelConfigurations = []*rmv1a1.PriorityLevelConfiguration{
		DefaultPriorityLevelConfigurationExempt,
		DefaultPriorityLevelConfigurationDefault,
	}
	PreservingFlowSchemas = []*rmv1a1.FlowSchema{
		DefaultFlowSchemaSystemTop,
		DefaultFlowSchemaCatchAll,
	}
)

// preserving default priority-levels
var (
	DefaultPriorityLevelConfigurationExempt = pl(
		rmv1a1.PriorityLevelConfigurationNameExempt,
		rmv1a1.PriorityLevelConfigurationSpec{Exempt: true},
	)
	DefaultPriorityLevelConfigurationDefault = pl(
		"default",
		rmv1a1.PriorityLevelConfigurationSpec{
			AssuredConcurrencyShares: 100,
			Queues:                   128,
			HandSize:                 6,
			QueueLengthLimit:         100,
		})
)

// preserving default flow-schemas
var (
	DefaultFlowSchemaSystemTop = fs(
		"system-top",
		rmv1a1.PriorityLevelConfigurationNameExempt,
		1500, "",
		rmv1a1.PolicyRuleWithSubjects{
			Subjects: groups(user.SystemPrivilegedGroup),
			Rule: resourceRule(
				[]string{rmv1a1.VerbAll},
				[]string{rmv1a1.APIGroupAll},
				[]string{rmv1a1.ResourceAll},
			)},
		rmv1a1.PolicyRuleWithSubjects{
			Subjects: groups(user.SystemPrivilegedGroup),
			Rule: nonResourceRule(
				[]string{rmv1a1.VerbAll},
				[]string{rmv1a1.NonResourceAll},
			)},
	)
	DefaultFlowSchemaCatchAll = fs(
		"catch-all",
		"default",
		1500, rmv1a1.FlowDistinguisherMethodByUserType,
		rmv1a1.PolicyRuleWithSubjects{
			Subjects: groups(user.AllUnauthenticated, user.AllAuthenticated),
			Rule: resourceRule(
				[]string{rmv1a1.VerbAll},
				[]string{rmv1a1.APIGroupAll},
				[]string{rmv1a1.ResourceAll},
			)},
		rmv1a1.PolicyRuleWithSubjects{
			Subjects: groups(user.AllUnauthenticated, user.AllAuthenticated),
			Rule: nonResourceRule(
				[]string{rmv1a1.VerbAll},
				[]string{rmv1a1.NonResourceAll},
			)},
	)
)

// DefaultPriorityLevelConfigurations returns the instances of
// the ones defined in the KEP, with system-top in position 0 and
// workload-low in position 1
func DefaultPriorityLevelConfigurations() []*rmv1a1.PriorityLevelConfiguration {
	return []*rmv1a1.PriorityLevelConfiguration{
		DefaultPriorityLevelConfigurationExempt,
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
		pl("workload-low", rmv1a1.PriorityLevelConfigurationSpec{
			AssuredConcurrencyShares: 30,
			Queues:                   128,
			HandSize:                 6,
			QueueLengthLimit:         100}),
		DefaultPriorityLevelConfigurationDefault,
	}
}

// DefaultFlowSchemas returns instances of the FlowSchema
// objects shown in the Example Configuration section of the KEP
func DefaultFlowSchemas() []*rmv1a1.FlowSchema {
	verbAll := []string{rmv1a1.VerbAll}
	apiGroupAll := []string{rmv1a1.APIGroupAll}
	apiGroupCore := ""
	resourceAll := []string{rmv1a1.ResourceAll}
	ruleRscAll := resourceRule(verbAll, apiGroupAll, resourceAll)
	ruleNonRscAll := rmv1a1.PolicyRule{
		Verbs:           verbAll,
		NonResourceURLs: []string{rmv1a1.NonResourceAll}}
	subjectsAll := groups(user.AllAuthenticated, user.AllUnauthenticated)
	return []*rmv1a1.FlowSchema{
		DefaultFlowSchemaSystemTop,
		fs("system-high", "system-high", 2500,
			rmv1a1.FlowDistinguisherMethodByUserType,
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: groups(user.NodesGroup),
				Rule:     resourceRule(verbAll, []string{apiGroupCore}, []string{"nodes"})},
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: groups(user.NodesGroup),
				Rule:     ruleRscAll},
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: users(user.KubeControllerManager, user.KubeScheduler),
				Rule:     resourceRule(verbAll, []string{apiGroupCore}, []string{"endpoints", "configmaps", "leases"})},
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: users(user.KubeControllerManager, user.KubeScheduler),
				Rule:     ruleNonRscAll},
		),
		fs("system-low", "system-low", 3500,
			rmv1a1.FlowDistinguisherMethodByUserType,
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: kubeSystemServiceAccount("generic-garbage-collector"),
				Rule:     ruleRscAll,
			},
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: kubeSystemServiceAccount("generic-garbage-collector"),
				Rule:     nonResourceRule([]string{rmv1a1.VerbAll}, []string{rmv1a1.NonResourceAll}),
			},
		),
		fs("workload-high", "workload-high", 7500,
			rmv1a1.FlowDistinguisherMethodByNamespaceType,
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: subjectsAll,
				Rule:     ruleRscAll},
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: subjectsAll,
				Rule:     ruleNonRscAll}),
		DefaultFlowSchemaCatchAll,
	}
}

func pl(name string, spec rmv1a1.PriorityLevelConfigurationSpec) *rmv1a1.PriorityLevelConfiguration {
	return &rmv1a1.PriorityLevelConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       spec}
}

func fs(name, plName string, matchingPrecedence int32, dmType rmv1a1.FlowDistinguisherMethodType, rules ...rmv1a1.PolicyRuleWithSubjects) *rmv1a1.FlowSchema {
	var dm *rmv1a1.FlowDistinguisherMethod
	if dmType != "" {
		dm = &rmv1a1.FlowDistinguisherMethod{Type: dmType}
	}
	return &rmv1a1.FlowSchema{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: rmv1a1.FlowSchemaSpec{
			PriorityLevelConfiguration: rmv1a1.PriorityLevelConfigurationReference{
				Name: plName,
			},
			MatchingPrecedence:  matchingPrecedence,
			DistinguisherMethod: dm,
			Rules:               rules},
	}

}

func groups(names ...string) []rmv1a1.Subject {
	ans := make([]rmv1a1.Subject, len(names))
	for idx, name := range names {
		ans[idx] = rmv1a1.Subject{Kind: rmv1a1.GroupKind, Name: name}
	}
	return ans
}

func users(names ...string) []rmv1a1.Subject {
	ans := make([]rmv1a1.Subject, len(names))
	for idx, name := range names {
		ans[idx] = rmv1a1.Subject{Kind: rmv1a1.UserKind, Name: name}
	}
	return ans
}

func kubeSystemServiceAccount(name string) []rmv1a1.Subject {
	return []rmv1a1.Subject{{Kind: rmv1a1.ServiceAccountKind, Name: name, Namespace: metav1.NamespaceSystem}}
}

func resourceRule(verbs []string, groups []string, resources []string) rmv1a1.PolicyRule {
	return rmv1a1.PolicyRule{Verbs: verbs, APIGroups: groups, Resources: resources}
}

func nonResourceRule(verbs []string, nonResourceURLs []string) rmv1a1.PolicyRule {
	return rmv1a1.PolicyRule{Verbs: verbs, NonResourceURLs: nonResourceURLs}
}
