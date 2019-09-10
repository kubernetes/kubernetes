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

package deletion

import (
	"fmt"
	"sort"
	"strings"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/discovery"
)

// NamespaceConditionUpdater interface that translates namespace deleter errors
// into namespace status conditions.
type NamespaceConditionUpdater interface {
	ProcessDiscoverResourcesErr(e error)
	ProcessGroupVersionErr(e error)
	ProcessDeleteContentErr(e error)
	Update(*v1.Namespace) bool
}

type namespaceConditionUpdater struct {
	newConditions       []v1.NamespaceCondition
	deleteContentErrors []error
}

var _ NamespaceConditionUpdater = &namespaceConditionUpdater{}

var (
	// conditionTypes Namespace condition types that are maintained by namespace_deleter controller.
	conditionTypes = []v1.NamespaceConditionType{
		v1.NamespaceDeletionDiscoveryFailure,
		v1.NamespaceDeletionGVParsingFailure,
		v1.NamespaceDeletionContentFailure,
	}
	okMessages = map[v1.NamespaceConditionType]string{
		v1.NamespaceDeletionDiscoveryFailure: "All resources successfully discovered",
		v1.NamespaceDeletionGVParsingFailure: "All legacy kube types successfully parsed",
		v1.NamespaceDeletionContentFailure:   "All content successfully deleted",
	}
	okReasons = map[v1.NamespaceConditionType]string{
		v1.NamespaceDeletionDiscoveryFailure: "ResourcesDiscovered",
		v1.NamespaceDeletionGVParsingFailure: "ParsedGroupVersions",
		v1.NamespaceDeletionContentFailure:   "ContentDeleted",
	}
)

// ProcessGroupVersionErr creates error condition if parsing GroupVersion of resources fails.
func (u *namespaceConditionUpdater) ProcessGroupVersionErr(err error) {
	d := v1.NamespaceCondition{
		Type:               v1.NamespaceDeletionGVParsingFailure,
		Status:             v1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Reason:             "GroupVersionParsingFailed",
		Message:            err.Error(),
	}
	u.newConditions = append(u.newConditions, d)
}

// ProcessDiscoverResourcesErr creates error condition from ErrGroupDiscoveryFailed.
func (u *namespaceConditionUpdater) ProcessDiscoverResourcesErr(err error) {
	var msg string
	if derr, ok := err.(*discovery.ErrGroupDiscoveryFailed); ok {
		msg = fmt.Sprintf("Discovery failed for some groups, %d failing: %v", len(derr.Groups), err)
	} else {
		msg = err.Error()
	}
	d := v1.NamespaceCondition{
		Type:               v1.NamespaceDeletionDiscoveryFailure,
		Status:             v1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Reason:             "DiscoveryFailed",
		Message:            msg,
	}
	u.newConditions = append(u.newConditions, d)

}

// ProcessDeleteContentErr creates error condition from multiple delete content errors.
func (u *namespaceConditionUpdater) ProcessDeleteContentErr(err error) {
	u.deleteContentErrors = append(u.deleteContentErrors, err)
}

// Update compiles processed errors from namespace deletion into status conditions.
func (u *namespaceConditionUpdater) Update(ns *v1.Namespace) bool {
	if c := getCondition(u.newConditions, v1.NamespaceDeletionContentFailure); c == nil {
		if c := makeDeleteContentCondition(u.deleteContentErrors); c != nil {
			u.newConditions = append(u.newConditions, *c)
		}
	}
	return updateConditions(&ns.Status, u.newConditions)
}

func makeDeleteContentCondition(err []error) *v1.NamespaceCondition {
	if len(err) == 0 {
		return nil
	}
	msgs := make([]string, 0, len(err))
	for _, e := range err {
		msgs = append(msgs, e.Error())
	}
	sort.Strings(msgs)
	return &v1.NamespaceCondition{
		Type:               v1.NamespaceDeletionContentFailure,
		Status:             v1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Reason:             "ContentDeletionFailed",
		Message:            fmt.Sprintf("Failed to delete all resource types, %d remaining: %v", len(err), strings.Join(msgs, ", ")),
	}
}

func updateConditions(status *v1.NamespaceStatus, newConditions []v1.NamespaceCondition) (hasChanged bool) {
	for _, conditionType := range conditionTypes {
		newCondition := getCondition(newConditions, conditionType)
		// if we weren't failing, then this returned nil.  We should set the "ok" variant of the condition
		if newCondition == nil {
			newCondition = newSuccessfulCondition(conditionType)
		}
		oldCondition := getCondition(status.Conditions, conditionType)

		// only new condition of this type exists, add to the list
		if oldCondition == nil {
			status.Conditions = append(status.Conditions, *newCondition)
			hasChanged = true

		} else if oldCondition.Status != newCondition.Status || oldCondition.Message != newCondition.Message || oldCondition.Reason != newCondition.Reason {
			// old condition needs to be updated
			if oldCondition.Status != newCondition.Status {
				oldCondition.LastTransitionTime = metav1.Now()
			}
			oldCondition.Type = newCondition.Type
			oldCondition.Status = newCondition.Status
			oldCondition.Reason = newCondition.Reason
			oldCondition.Message = newCondition.Message
			hasChanged = true
		}
	}
	return
}

func newSuccessfulCondition(conditionType v1.NamespaceConditionType) *v1.NamespaceCondition {
	return &v1.NamespaceCondition{
		Type:               conditionType,
		Status:             v1.ConditionFalse,
		LastTransitionTime: metav1.Now(),
		Reason:             okReasons[conditionType],
		Message:            okMessages[conditionType],
	}
}

func getCondition(conditions []v1.NamespaceCondition, conditionType v1.NamespaceConditionType) *v1.NamespaceCondition {
	for i := range conditions {
		if conditions[i].Type == conditionType {
			return &(conditions[i])
		}
	}
	return nil
}
