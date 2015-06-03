package autoscaler

import (
	"fmt"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

type testunit struct {
	Name        string
	Actions     []AutoScaleAction
	Expectation AutoScaleAction
}

func getTestUnits() []testunit {
	return []testunit{
		testunit{
			Name:    "simple-noop-test",
			Actions: []AutoScaleAction{
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeNone},
			},
			Expectation: AutoScaleAction{
				ScaleType: api.AutoScaleActionTypeNone,
				ScaleBy:   0,
			},
		},
		testunit{
			Name:    "simple-scale-up-test",
			Actions: []AutoScaleAction{
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 2},
			},
			Expectation: AutoScaleAction{
				ScaleType: api.AutoScaleActionTypeScaleUp,
				ScaleBy:   2,
			},
		},
		testunit{
			Name:    "simple-scale-down-test",
			Actions: []AutoScaleAction{
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 4},
			},
			Expectation: AutoScaleAction{
				ScaleType: api.AutoScaleActionTypeScaleDown,
				ScaleBy:   4,
			},
		},
		testunit{
			Name:    "multi-noop-test",
			Actions: []AutoScaleAction{
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeNone},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeNone, ScaleBy: -1},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeNone, ScaleBy: 1},
			},
			Expectation: AutoScaleAction{
				ScaleType: api.AutoScaleActionTypeNone,
				ScaleBy:   0,
			},
		},
		testunit{
			Name:    "multi-scale-up-test",
			Actions: []AutoScaleAction{
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 2},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 3},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 4},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 3},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 2},
			},
			Expectation: AutoScaleAction{
				ScaleType: api.AutoScaleActionTypeScaleUp,
				ScaleBy:   4,
			},
		},
		testunit{
			Name:    "multi-scale-up-zero-negative-test",
			Actions: []AutoScaleAction{
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 1},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 2},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: -1},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 2},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 0},
			},
			Expectation: AutoScaleAction{
				ScaleType: api.AutoScaleActionTypeScaleUp,
				ScaleBy:   2,
			},
		},
		testunit{
			Name:    "multi-scale-down-test",
			Actions: []AutoScaleAction{
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 4},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 6},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 6},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 6},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 4},
			},
			Expectation: AutoScaleAction{
				ScaleType: api.AutoScaleActionTypeScaleDown,
				ScaleBy:   4,
			},
		},
		testunit{
			Name:    "multi-scale-down-negative-test",
			Actions: []AutoScaleAction{
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 1},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 1},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 2},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 3},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 5},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 8},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: -1},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: -1},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: -2},
			},
			Expectation: AutoScaleAction{
				ScaleType: api.AutoScaleActionTypeScaleDown,
				ScaleBy:   1,
			},
		},
		testunit{
			Name:    "combo-scale-test",
			Actions: []AutoScaleAction{
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeNone, ScaleBy: -1},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 22},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 11},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 5},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 4},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 3},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 2},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 1},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 42},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 21},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleUp, ScaleBy: 10},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 10},
			},
			Expectation: AutoScaleAction{
				ScaleType: api.AutoScaleActionTypeScaleUp,
				ScaleBy:   42,
			},
		},
		testunit{
			Name:    "combo-test-2",
			Actions: []AutoScaleAction{
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeNone, ScaleBy: -1},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 4},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 2},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 8},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeScaleDown, ScaleBy: 6},
				AutoScaleAction{ScaleType: api.AutoScaleActionTypeNone},
			},
			Expectation: AutoScaleAction{
				ScaleType: api.AutoScaleActionTypeScaleDown,
				ScaleBy:   2,
			},
		},
	}
}

func matchExpectations(result, expected AutoScaleAction, triggerCheck bool) bool {
	if result.ScaleType != expected.ScaleType {
		return false
	}

	if result.ScaleBy != expected.ScaleBy {
		return false
	}

	if triggerCheck && result.Trigger != expected.Trigger {
		return false
	}

	return true
}

func checkExpectations(result, expected AutoScaleAction, triggerCheck bool) error {
	if result.ScaleType != expected.ScaleType {
		return fmt.Errorf("got scale type %q, expected %q", result.ScaleType, expected.ScaleType)
	}

	if result.ScaleBy != expected.ScaleBy {
		return fmt.Errorf("got scale by %v, expected %v", result.ScaleBy, expected.ScaleBy)
	}

	if triggerCheck && result.Trigger != expected.Trigger {
		return fmt.Errorf("triggers does not match expectation")
	}

	return nil
}

func TestReconcileActions(t *testing.T) {
	for _, unit := range getTestUnits() {
		result := ReconcileActions(unit.Actions)
		if err := checkExpectations(result, unit.Expectation, false); err != nil {
			t.Errorf("Test case %s %s", unit.Name, err)
		}
	}
}
