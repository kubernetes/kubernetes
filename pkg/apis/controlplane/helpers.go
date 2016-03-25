package controlplane

import (
	"fmt"

	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/sets"
)
func ClusterSelectorRequirementsAsSelector(nsm []ClusterSelectorRequirement) (labels.Selector, error) {
	if len(nsm) == 0 {
		return labels.Nothing(), nil
	}
	selector := labels.NewSelector()
	for _, expr := range nsm {
		var op labels.Operator
		switch expr.Operator {
		case ClusterSelectorOpIn:
			op = labels.InOperator
		case ClusterSelectorOpNotIn:
			op = labels.NotInOperator
		default:
			return nil, fmt.Errorf("%q is not a valid node selector operator", expr.Operator)
		}
		r, err := labels.NewRequirement(expr.Key, op, sets.NewString(expr.Values...))
		if err != nil {
			return nil, err
		}
		selector = selector.Add(*r)
	}
	return selector, nil
}



