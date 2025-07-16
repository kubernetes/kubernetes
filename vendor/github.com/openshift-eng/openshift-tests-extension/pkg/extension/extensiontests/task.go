package extensiontests

import "sync/atomic"

type SpecTask struct {
	fn func(spec ExtensionTestSpec)
}

func (t *SpecTask) Run(spec ExtensionTestSpec) {
	t.fn(spec)
}

type TestResultTask struct {
	fn func(result *ExtensionTestResult)
}

func (t *TestResultTask) Run(result *ExtensionTestResult) {
	t.fn(result)
}

type OneTimeTask struct {
	fn       func()
	executed int32 // Atomic boolean to indicate whether the function has been run
}

func (t *OneTimeTask) Run() {
	// Ensure one-time tasks are only run once
	if atomic.CompareAndSwapInt32(&t.executed, 0, 1) {
		t.fn()
	}
}
