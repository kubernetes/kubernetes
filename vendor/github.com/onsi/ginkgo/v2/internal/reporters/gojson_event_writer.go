package reporters

type GoJSONEventWriter struct {
	enc encoder
	specSystemErrFn specSystemExtractFn
	specSystemOutFn specSystemExtractFn
}

func NewGoJSONEventWriter(enc encoder, errFn specSystemExtractFn, outFn specSystemExtractFn) *GoJSONEventWriter {
	return &GoJSONEventWriter{
		enc: enc,
		specSystemErrFn: errFn,
		specSystemOutFn: outFn,
	}
}

func (r *GoJSONEventWriter) writeEvent(e *gojsonEvent) error {
	return r.enc.Encode(e)
}

func (r *GoJSONEventWriter) WriteSuiteStart(report *gojsonReport) error {
	e := &gojsonEvent{
		Time:        &report.o.StartTime,
		Action:      GoJSONStart,
		Package:     report.goPkg,
		Output:      nil,
		FailedBuild: "",
	}
	return r.writeEvent(e)
}

func (r *GoJSONEventWriter) WriteSuiteResult(report *gojsonReport) error {
	var action GoJSONAction
	switch {
	case report.o.PreRunStats.SpecsThatWillRun == 0:
		action = GoJSONSkip
	case report.o.SuiteSucceeded:
		action = GoJSONPass
	default:
		action = GoJSONFail
	}
	e := &gojsonEvent{
		Time:        &report.o.EndTime,
		Action:      action,
		Package:     report.goPkg,
		Output:      nil,
		FailedBuild: "",
		Elapsed:     ptr(report.elapsed),
	}
	return r.writeEvent(e)
}

func (r *GoJSONEventWriter) WriteSpecStart(report *gojsonReport, specReport *gojsonSpecReport) error {
	e := &gojsonEvent{
		Time:        &specReport.o.StartTime,
		Action:      GoJSONRun,
		Test:        specReport.testName,
		Package:     report.goPkg,
		Output:      nil,
		FailedBuild: "",
	}
	return r.writeEvent(e)
}

func (r *GoJSONEventWriter) WriteSpecOut(report *gojsonReport, specReport *gojsonSpecReport) error {
	events := []*gojsonEvent{}

	stdErr := r.specSystemErrFn(specReport.o)
	if stdErr != "" {
		events = append(events, &gojsonEvent{
			Time:        &specReport.o.EndTime,
			Action:      GoJSONOutput,
			Test:        specReport.testName,
			Package:     report.goPkg,
			Output:      ptr(stdErr),
			FailedBuild: "",
		})
	}
	stdOut := r.specSystemOutFn(specReport.o)
	if stdOut != "" {
		events = append(events, &gojsonEvent{
			Time:        &specReport.o.EndTime,
			Action:      GoJSONOutput,
			Test:        specReport.testName,
			Package:     report.goPkg,
			Output:      ptr(stdOut),
			FailedBuild: "",
		})
	}

	for _, ev := range events {
		err := r.writeEvent(ev)
		if err != nil {
			return err
		}
	}
	return nil
}

func (r *GoJSONEventWriter) WriteSpecResult(report *gojsonReport, specReport *gojsonSpecReport) error {
	e := &gojsonEvent{
		Time:        &specReport.o.EndTime,
		Action:      specReport.action,
		Test:        specReport.testName,
		Package:     report.goPkg,
		Elapsed:     ptr(specReport.elapsed),
		Output:      nil,
		FailedBuild: "",
	}
	return r.writeEvent(e)
}
