package testrunner

type RunResult struct {
	Passed               bool
	HasProgrammaticFocus bool
}

func PassingRunResult() RunResult {
	return RunResult{
		Passed:               true,
		HasProgrammaticFocus: false,
	}
}

func FailingRunResult() RunResult {
	return RunResult{
		Passed:               false,
		HasProgrammaticFocus: false,
	}
}

func (r RunResult) Merge(o RunResult) RunResult {
	return RunResult{
		Passed:               r.Passed && o.Passed,
		HasProgrammaticFocus: r.HasProgrammaticFocus || o.HasProgrammaticFocus,
	}
}
