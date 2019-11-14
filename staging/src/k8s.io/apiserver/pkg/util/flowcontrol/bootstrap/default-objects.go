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
		nil)

	DefaultPriorityLevelConfigurationDefault = plq(
		rmv1a1.PriorityLevelConfigurationNameCatchAll, 100,
		&rmv1a1.QueuingConfiguration{
			Queues:           128,
			HandSize:         6,
			QueueLengthLimit: 100,
		})
)

// Initial FlowSchema objects
var (
	DefaultFlowSchemaExempt = NewFSAllGroups(
		rmv1a1.FlowSchemaNameExempt,
		rmv1a1.PriorityLevelConfigurationNameExempt,
		1500, "",
		user.SystemPrivilegedGroup,
	)
	DefaultFlowSchemaDefault = NewFSAllGroups(
		rmv1a1.FlowSchemaNameCatchAll,
		rmv1a1.PriorityLevelConfigurationNameCatchAll,
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
		spl(plq("system", 30, &rmv1a1.QueuingConfiguration{
			Queues:           1,
			QueueLengthLimit: 1000})),
		spl(plq("workload-high", 30, &rmv1a1.QueuingConfiguration{
			Queues:           128,
			HandSize:         6,
			QueueLengthLimit: 100})),
		spl(plq("workload-low", 30, &rmv1a1.QueuingConfiguration{
			Queues:           128,
			HandSize:         6,
			QueueLengthLimit: 100})),
		DefaultPriorityLevelConfigurationDefault,
	}
}

var (
	verbAll       = []string{rmv1a1.VerbAll}
	apiGroupAll   = []string{rmv1a1.APIGroupAll}
	apiGroupCore  = ""
	nsAll         = []string{"c s", "*"}
	nsSystem      = []string{"kube-system"}
	resourceAll   = rmv1a1.ResourceAll
	ruleRscAll    = []rmv1a1.ResourcePolicyRule{resourceRule(verbAll, apiGroupAll, nsAll, resourceAll)}
	ruleNonRscAll = []rmv1a1.NonResourcePolicyRule{{
		Verbs:           verbAll,
		NonResourceURLs: []string{rmv1a1.NonResourceAll}}}
)

func prws(subjects []rmv1a1.Subject, nrr []rmv1a1.NonResourcePolicyRule, rr ...rmv1a1.ResourcePolicyRule) rmv1a1.PolicyRulesWithSubjects {
	return rmv1a1.PolicyRulesWithSubjects{
		Subjects:         subjects,
		ResourceRules:    rr,
		NonResourceRules: nrr,
	}
}

// All the particular usernames available from the controller manager
var _ = []string{
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

// PredefinedFlowSchemas returns instances of the FlowSchema
// objects shown in the Example Configuration section of the KEP
func PredefinedFlowSchemas() []*rmv1a1.FlowSchema {
	return []*rmv1a1.FlowSchema{
		DefaultFlowSchemaExempt,
		sfs(fs("system-nodes", "system", 2500,
			rmv1a1.FlowDistinguisherMethodByUserType,
			prws(
				groups(user.NodesGroup), ruleNonRscAll,
				resourceRule(verbAll, []string{apiGroupCore}, nsAll, "nodes"),
				resourceRule(verbAll, apiGroupAll, nsSystem, resourceAll),
			),
		)),
		sfs(fs("system-leader-election", "system", 2500,
			rmv1a1.FlowDistinguisherMethodByUserType,
			prws(
				kubeSystemServiceAccount(rmv1a1.NameAll), nil,
				resourceRule(verbAll, []string{"coordination.k8s.io"}, nsAll, "leases"),
			),
		)),
		NewFSAll("system-low", "system-low", 3500, rmv1a1.FlowDistinguisherMethodByNamespaceType, kubeSystemServiceAccount("generic-garbage-collector")...),
		NewFSAll("workload-low", "workload-low", 4500, rmv1a1.FlowDistinguisherMethodByNamespaceType, kubeSystemServiceAccount(rmv1a1.NameAll)...),
		NewFSAllGroups("workload-high", "workload-high", 5500, rmv1a1.FlowDistinguisherMethodByNamespaceType, user.AllUnauthenticated, user.AllAuthenticated),
		DefaultFlowSchemaDefault,
	}
}

// NewFSAllUsers constructs a FlowSchema that matches the given subjects regardless of verb and object.
// The subjects are usernames.
func NewFSAllUsers(name, plName string, matchingPrecedence int32, dmType rmv1a1.FlowDistinguisherMethodType, userNames ...string) *rmv1a1.FlowSchema {
	return NewFSAll(name, plName, matchingPrecedence, dmType, users(userNames...)...)
}

// NewFSAllGroups constructs a FlowSchema that matches the given subjects regardless of verb and object.
// The subjects are user group names.
func NewFSAllGroups(name, plName string, matchingPrecedence int32, dmType rmv1a1.FlowDistinguisherMethodType, groupNames ...string) *rmv1a1.FlowSchema {
	return NewFSAll(name, plName, matchingPrecedence, dmType, groups(groupNames...)...)
}

// NewFSAll constructs a FlowSchema that matches the given subjects regardless of verb and target object
func NewFSAll(name, plName string, matchingPrecedence int32, dmType rmv1a1.FlowDistinguisherMethodType, subjects ...rmv1a1.Subject) *rmv1a1.FlowSchema {
	return fs(name, plName, matchingPrecedence, dmType,
		rmv1a1.PolicyRulesWithSubjects{
			Subjects:         subjects,
			ResourceRules:    ruleRscAll,
			NonResourceRules: ruleNonRscAll},
	)
}

func pl(name string, lc *rmv1a1.LimitedPriorityLevelConfiguration) *rmv1a1.PriorityLevelConfiguration {
	spec := rmv1a1.PriorityLevelConfigurationSpec{Type: rmv1a1.PriorityLevelEnablementExempt}
	if lc != nil {
		spec.Type = rmv1a1.PriorityLevelEnablementLimited
		spec.Limited = lc
	}
	return &rmv1a1.PriorityLevelConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec:       spec}
}

func plq(name string, acs int32, qc *rmv1a1.QueuingConfiguration) *rmv1a1.PriorityLevelConfiguration {
	lr := rmv1a1.LimitResponse{Type: rmv1a1.LimitResponseTypeReject}
	if qc != nil {
		lr.Type = rmv1a1.LimitResponseTypeQueue
		lr.Queuing = qc
	}
	return pl(name, &rmv1a1.LimitedPriorityLevelConfiguration{
		AssuredConcurrencyShares: acs,
		LimitResponse:            lr})
}

func spl(obj *rmv1a1.PriorityLevelConfiguration) *rmv1a1.PriorityLevelConfiguration {
	obj.Labels = map[string]string{"suggested": "true"}
	return obj
}

func fs(name, plName string, matchingPrecedence int32, dmType rmv1a1.FlowDistinguisherMethodType, rules ...rmv1a1.PolicyRulesWithSubjects) *rmv1a1.FlowSchema {
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

func sfs(obj *rmv1a1.FlowSchema) *rmv1a1.FlowSchema {
	obj.Labels = map[string]string{"suggested": "true"}
	return obj
}

func groups(names ...string) []rmv1a1.Subject {
	ans := make([]rmv1a1.Subject, len(names))
	for idx, name := range names {
		ans[idx] = rmv1a1.Subject{Kind: rmv1a1.SubjectKindGroup,
			Group: &rmv1a1.GroupSubject{Name: name}}
	}
	return ans
}

func users(names ...string) []rmv1a1.Subject {
	ans := make([]rmv1a1.Subject, len(names))
	for idx, name := range names {
		ans[idx] = rmv1a1.Subject{Kind: rmv1a1.SubjectKindUser,
			User: &rmv1a1.UserSubject{Name: name}}
	}
	return ans
}

func kubeSystemServiceAccount(names ...string) []rmv1a1.Subject {
	subjects := []rmv1a1.Subject{}
	for _, name := range names {
		subjects = append(subjects, rmv1a1.Subject{
			Kind: rmv1a1.SubjectKindServiceAccount,
			ServiceAccount: &rmv1a1.ServiceAccountSubject{
				Name:      name,
				Namespace: metav1.NamespaceSystem,
			}})
	}
	return subjects
}

func resourceRule(verbs, groups, namespaces []string, resources ...string) rmv1a1.ResourcePolicyRule {
	clusterScope := namespaces[0] == "c s"
	if clusterScope {
		namespaces = namespaces[1:]
	}
	return rmv1a1.ResourcePolicyRule{Verbs: verbs,
		APIGroups:    groups,
		Resources:    resources,
		ClusterScope: clusterScope,
		Namespaces:   namespaces}
}
