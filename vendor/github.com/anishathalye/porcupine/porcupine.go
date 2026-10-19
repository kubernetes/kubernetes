package porcupine

import "time"

// CheckOperations checks whether a history is linearizable.
func CheckOperations(model Model, history []Operation) bool {
	res, _ := checkOperations(model, history, false, 0)
	return res == Ok
}

// CheckOperationsTimeout checks whether a history is linearizable, with a
// timeout.
//
// A timeout of 0 is interpreted as an unlimited timeout.
func CheckOperationsTimeout(model Model, history []Operation, timeout time.Duration) CheckResult {
	res, _ := checkOperations(model, history, false, timeout)
	return res
}

// CheckOperationsVerbose checks whether a history is linearizable while
// computing data that can be used to visualize the history and linearization.
//
// The returned LinearizationInfo can be used with [Visualize].
func CheckOperationsVerbose(model Model, history []Operation, timeout time.Duration) (CheckResult, LinearizationInfo) {
	return checkOperations(model, history, true, timeout)
}

// CheckEvents checks whether a history is linearizable.
func CheckEvents(model Model, history []Event) bool {
	res, _ := checkEvents(model, history, false, 0)
	return res == Ok
}

// CheckEventsTimeout checks whether a history is linearizable, with a timeout.
//
// A timeout of 0 is interpreted as an unlimited timeout.
func CheckEventsTimeout(model Model, history []Event, timeout time.Duration) CheckResult {
	res, _ := checkEvents(model, history, false, timeout)
	return res
}

// CheckEventsVerbose checks whether a history is linearizable while computing
// data that can be used to visualize the history and linearization.
//
// The returned LinearizationInfo can be used with [Visualize].
func CheckEventsVerbose(model Model, history []Event, timeout time.Duration) (CheckResult, LinearizationInfo) {
	return checkEvents(model, history, true, timeout)
}
