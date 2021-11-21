package wait

import (
	"errors"
	"fmt"
	"io"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/util/jsonpath"
	"k8s.io/kubectl/pkg/cmd/get"
)

// A Waiter defines the behavior of waiting for the desired state, including interpreting any errors that the
// ResourceFinder encounters in the OnWaitLoopCompletion method.
type Waiter interface {
	// VisitResource is called once for each resource found during the wait loop.
	VisitResource(info *resource.Info, o *WaitOptions) (runtime.Object, bool, error)

	// OnWaitLoopCompletion is called at the end of each wait loop cycle and may be used to supress or raise errors
	// based on the number of resources visited.
	OnWaitLoopCompletion(visitedCount int, err error) error
}

func waiterFor(condition string, errOut io.Writer) (Waiter, error) {
	if strings.ToLower(condition) == "delete" {
		return NewDeletionWaiter(errOut), nil
	}
	if strings.HasPrefix(condition, "condition=") {
		conditionName := condition[len("condition="):]
		conditionValue := "true"
		if equalsIndex := strings.Index(conditionName, "="); equalsIndex != -1 {
			conditionValue = conditionName[equalsIndex+1:]
			conditionName = conditionName[0:equalsIndex]
		}

		return NewConditionalWaiter(conditionName, conditionValue, errOut), nil
	}
	if strings.HasPrefix(condition, "jsonpath=") {
		splitStr := strings.Split(condition, "=")
		if len(splitStr) != 3 {
			return nil, fmt.Errorf("jsonpath wait format must be --for=jsonpath='{.status.readyReplicas}'=3")
		}
		jsonPathExp, jsonPathCond, err := processJSONPathInput(splitStr[1], splitStr[2])
		if err != nil {
			return nil, err
		}
		j, err := newJSONPathParser(jsonPathExp)
		if err != nil {
			return nil, err
		}
		return NewJSONPathWaiter(jsonPathCond, j, errOut), nil
	}

	return nil, fmt.Errorf("unrecognized condition: %q", condition)
}

// newJSONPathParser will create a new JSONPath parser based on the jsonPathExpression
func newJSONPathParser(jsonPathExpression string) (*jsonpath.JSONPath, error) {
	j := jsonpath.New("wait")
	if jsonPathExpression == "" {
		return nil, errors.New("jsonpath expression cannot be empty")
	}
	if err := j.Parse(jsonPathExpression); err != nil {
		return nil, err
	}
	return j, nil
}

// processJSONPathInput will parses the user's JSONPath input and process the string
func processJSONPathInput(jsonPathExpression, jsonPathCond string) (string, string, error) {
	relaxedJSONPathExp, err := get.RelaxedJSONPathExpression(jsonPathExpression)
	if err != nil {
		return "", "", err
	}
	if jsonPathCond == "" {
		return "", "", errors.New("jsonpath wait condition cannot be empty")
	}
	jsonPathCond = strings.Trim(jsonPathCond, `'"`)

	return relaxedJSONPathExp, jsonPathCond, nil
}
