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
	"math"

	rmv1a1 "k8s.io/api/flowcontrol/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/authentication/user"
)

// The objects that define an apiserver's initial behavior
var (
	// InitialPriorityLevelConfigurations lists exempt first, default second
	InitialPriorityLevelConfigurations = []*rmv1a1.PriorityLevelConfiguration{
		DefaultPriorityLevelConfigurationExempt,
		DefaultPriorityLevelConfigurationDefault,
	}
	InitialFlowSchemas = []*rmv1a1.FlowSchema{
		DefaultFlowSchemaExempt,
		DefaultFlowSchemaDefault,
	}
)

// Initial PriorityLevelConfiguration objects
var (
	DefaultPriorityLevelConfigurationExempt = pl(
		rmv1a1.PriorityLevelConfigurationNameExempt,
		rmv1a1.PriorityLevelConfigurationSpec{Exempt: true},
	)
	DefaultPriorityLevelConfigurationDefault = setDefault(pl(
		"default",
		rmv1a1.PriorityLevelConfigurationSpec{
			AssuredConcurrencyShares: 100,
			Queues:                   128,
			HandSize:                 6,
			QueueLengthLimit:         100,
		}))
)

// Initial FlowSchema objects
var (
	DefaultFlowSchemaExempt = NewFSAllGroups(
		"exempt",
		rmv1a1.PriorityLevelConfigurationNameExempt,
		1500, "",
		user.SystemPrivilegedGroup,
	)
	DefaultFlowSchemaDefault = NewFSAllGroups(
		"default",
		"default",
		math.MaxInt32, rmv1a1.FlowDistinguisherMethodByUserType,
		user.AllUnauthenticated, user.AllAuthenticated,
	)
)

// PredefinedPriorityLevelConfigurations returns the instances of
// the ones defined in the KEP
// with the exempt one appearing first
func PredefinedPriorityLevelConfigurations() []*rmv1a1.PriorityLevelConfiguration {
	return []*rmv1a1.PriorityLevelConfiguration{
		DefaultPriorityLevelConfigurationExempt,
		pl("system", rmv1a1.PriorityLevelConfigurationSpec{
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

var (
	verbAll       = []string{rmv1a1.VerbAll}
	apiGroupAll   = []string{rmv1a1.APIGroupAll}
	apiGroupCore  = ""
	resourceAll   = []string{rmv1a1.ResourceAll}
	ruleRscAll    = resourceRule(verbAll, apiGroupAll, resourceAll)
	ruleNonRscAll = rmv1a1.PolicyRule{
		Verbs:           verbAll,
		NonResourceURLs: []string{rmv1a1.NonResourceAll}}
)

// PredefinedFlowSchemas returns instances of the FlowSchema
// objects shown in the Example Configuration section of the KEP
func PredefinedFlowSchemas() []*rmv1a1.FlowSchema {
	allServiceAccountKubeControllerManager := []string{
		"attachdetach-controller",
		"certificate-controller",
		"clusterrole-aggregation-controller",
		"cronjob-controller",
		"daemon-set-controller",
		"deployment-controller",
		"disruption-controller",
		"endpoint-controller",
		"expand-controller",
		"generic-garbage-collector",
		"horizontal-pod-autoscaler",
		"job-controller",
		"namespace-controller",
		"node-controller",
		"persistent-volume-binder",
		"pod-garbage-collector",
		"pv-protection-controller",
		"pvc-protection-controller",
		"replicaset-controller",
		"replication-controller",
		"resourcequota-controller",
		"route-controller",
		"service-account-controller",
		"service-controller",
		"statefulset-controller",
		"ttl-controller",
	}
	return []*rmv1a1.FlowSchema{
		DefaultFlowSchemaExempt,
		fs("system-nodes", "system", 2500,
			rmv1a1.FlowDistinguisherMethodByUserType,
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: groups(user.NodesGroup),
				Rule:     ruleRscAll,
			},
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: kubeSystemServiceAccount(rmv1a1.NameAll),
				Rule:     resourceRule(verbAll, []string{apiGroupCore}, []string{"endpoints", "configmaps", "leases"}),
			},
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: kubeSystemServiceAccount(rmv1a1.NameAll),
				Rule:     ruleNonRscAll,
			},
		),
		fs("system-leader-election", "system", 2500,
			rmv1a1.FlowDistinguisherMethodByUserType,
			rmv1a1.PolicyRuleWithSubjects{
				Subjects: users(user.KubeControllerManager, user.KubeScheduler),
				Rule:     resourceRule(verbAll, []string{"coordination.k8s.io"}, []string{"leases"}),
			},
		),
		NewFSAll("kube-controller-manager", "workload-high", 3500, rmv1a1.FlowDistinguisherMethodByNamespaceType, append(users(user.KubeControllerManager), kubeSystemServiceAccount(allServiceAccountKubeControllerManager...)...)...),
		NewFSAllUsers("kube-scheduler", "workload-high", 3500, rmv1a1.FlowDistinguisherMethodByNamespaceType, user.KubeScheduler),
		NewFSAllGroups("service-accounts", "workload-low", 7500, rmv1a1.FlowDistinguisherMethodByUserType, "system:serviceaccounts"),
		DefaultFlowSchemaDefault,
	}
}

func NewFSAllUsers(name, plName string, matchingPrecedence int32, dmType rmv1a1.FlowDistinguisherMethodType, userNames ...string) *rmv1a1.FlowSchema {
	return NewFSAll(name, plName, matchingPrecedence, dmType, users(userNames...)...)
}

func NewFSAllGroups(name, plName string, matchingPrecedence int32, dmType rmv1a1.FlowDistinguisherMethodType, groupNames ...string) *rmv1a1.FlowSchema {
	return NewFSAll(name, plName, matchingPrecedence, dmType, groups(groupNames...)...)
}

func NewFSAll(name, plName string, matchingPrecedence int32, dmType rmv1a1.FlowDistinguisherMethodType, subjects ...rmv1a1.Subject) *rmv1a1.FlowSchema {
	return fs(name, plName, matchingPrecedence, dmType,
		rmv1a1.PolicyRuleWithSubjects{
			Subjects: subjects,
			Rule:     ruleRscAll},
		rmv1a1.PolicyRuleWithSubjects{
			Subjects: subjects,
			Rule:     ruleNonRscAll},
	)
}

func pl(name string, spec rmv1a1.PriorityLevelConfigurationSpec) *rmv1a1.PriorityLevelConfiguration {
	return &rmv1a1.PriorityLevelConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       spec}
}

func setDefault(pl *rmv1a1.PriorityLevelConfiguration) *rmv1a1.PriorityLevelConfiguration {
	pl.Spec.GlobalDefault = true
	return pl
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

func kubeSystemServiceAccount(names ...string) []rmv1a1.Subject {
	subjects := []rmv1a1.Subject{}
	for _, name := range names {
		subjects = append(subjects, rmv1a1.Subject{
			Kind:      rmv1a1.ServiceAccountKind,
			Name:      name,
			Namespace: metav1.NamespaceSystem,
		})
	}
	return subjects
}

func resourceRule(verbs []string, groups []string, resources []string) rmv1a1.PolicyRule {
	return rmv1a1.PolicyRule{Verbs: verbs, APIGroups: groups, Resources: resources}
}

func nonResourceRule(verbs []string, nonResourceURLs []string) rmv1a1.PolicyRule {
	return rmv1a1.PolicyRule{Verbs: verbs, NonResourceURLs: nonResourceURLs}
}
