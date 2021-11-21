package wait

import (
	"fmt"
	"io"
	"strings"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/resource"
)

// ConditionalWaiter hold information to check an API status condition
type ConditionalWaiter struct {
	conditionName   string
	conditionStatus string
	// errOut is written to if an error occurs
	errOut io.Writer
}

func NewConditionalWaiter(name, value string, errOut io.Writer) Waiter {
	return ConditionalWaiter{
		conditionName:   name,
		conditionStatus: value,
		errOut:          errOut,
	}
}

func (w ConditionalWaiter) VisitResource(info *resource.Info, o *WaitOptions) (runtime.Object, bool, error) {
	return getObjAndCheckCondition(info, o, w.isConditionMet, w.checkCondition)
}

func (w ConditionalWaiter) OnWaitLoopCompletion(visitedCount int, err error) error {
	if visitedCount == 0 {
		return errNoMatchingResources
	} else {
		return err
	}
}

func (w ConditionalWaiter) checkCondition(obj *unstructured.Unstructured) (bool, error) {
	conditions, found, err := unstructured.NestedSlice(obj.Object, "status", "conditions")
	if err != nil {
		return false, err
	}
	if !found {
		return false, nil
	}
	for _, conditionUncast := range conditions {
		condition := conditionUncast.(map[string]interface{})
		name, found, err := unstructured.NestedString(condition, "type")
		if !found || err != nil || !strings.EqualFold(name, w.conditionName) {
			continue
		}
		status, found, err := unstructured.NestedString(condition, "status")
		if !found || err != nil {
			continue
		}
		generation, found, _ := unstructured.NestedInt64(obj.Object, "metadata", "generation")
		if found {
			observedGeneration, found := getObservedGeneration(obj, condition)
			if found && observedGeneration < generation {
				return false, nil
			}
		}
		return strings.EqualFold(status, w.conditionStatus), nil
	}

	return false, nil
}

func (w ConditionalWaiter) isConditionMet(event watch.Event) (bool, error) {
	if event.Type == watch.Error {
		// keep waiting in the event we see an error - we expect the watch to be closed by
		// the server
		err := errors.FromObject(event.Object)
		fmt.Fprintf(w.errOut, "error: An error occurred while waiting for the condition to be satisfied: %v", err)
		return false, nil
	}
	if event.Type == watch.Deleted {
		// this will chain back out, result in another get and an return false back up the chain
		return false, nil
	}
	obj := event.Object.(*unstructured.Unstructured)
	return w.checkCondition(obj)
}

func getObservedGeneration(obj *unstructured.Unstructured, condition map[string]interface{}) (int64, bool) {
	conditionObservedGeneration, found, _ := unstructured.NestedInt64(condition, "observedGeneration")
	if found {
		return conditionObservedGeneration, true
	}
	statusObservedGeneration, found, _ := unstructured.NestedInt64(obj.Object, "status", "observedGeneration")
	return statusObservedGeneration, found
}
