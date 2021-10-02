package stenographer

import (
	"sync"

	"github.com/onsi/ginkgo/types"
)

func NewFakeStenographerCall(method string, args ...interface{}) FakeStenographerCall {
	return FakeStenographerCall{
		Method: method,
		Args:   args,
	}
}

type FakeStenographer struct {
	calls []FakeStenographerCall
	lock  *sync.Mutex
}

type FakeStenographerCall struct {
	Method string
	Args   []interface{}
}

func NewFakeStenographer() *FakeStenographer {
	stenographer := &FakeStenographer{
		lock: &sync.Mutex{},
	}
	stenographer.Reset()
	return stenographer
}

func (stenographer *FakeStenographer) Calls() []FakeStenographerCall {
	stenographer.lock.Lock()
	defer stenographer.lock.Unlock()

	return stenographer.calls
}

func (stenographer *FakeStenographer) Reset() {
	stenographer.lock.Lock()
	defer stenographer.lock.Unlock()

	stenographer.calls = make([]FakeStenographerCall, 0)
}

func (stenographer *FakeStenographer) CallsTo(method string) []FakeStenographerCall {
	stenographer.lock.Lock()
	defer stenographer.lock.Unlock()

	results := make([]FakeStenographerCall, 0)
	for _, call := range stenographer.calls {
		if call.Method == method {
			results = append(results, call)
		}
	}

	return results
}

func (stenographer *FakeStenographer) registerCall(method string, args ...interface{}) {
	stenographer.lock.Lock()
	defer stenographer.lock.Unlock()

	stenographer.calls = append(stenographer.calls, NewFakeStenographerCall(method, args...))
}

func (stenographer *FakeStenographer) AnnounceSuite(description string, randomSeed int64, randomizingAll bool, succinct bool) {
	stenographer.registerCall("AnnounceSuite", description, randomSeed, randomizingAll, succinct)
}

func (stenographer *FakeStenographer) AnnounceAggregatedParallelRun(nodes int, succinct bool) {
	stenographer.registerCall("AnnounceAggregatedParallelRun", nodes, succinct)
}

func (stenographer *FakeStenographer) AnnounceParallelRun(node int, nodes int, succinct bool) {
	stenographer.registerCall("AnnounceParallelRun", node, nodes, succinct)
}

func (stenographer *FakeStenographer) AnnounceNumberOfSpecs(specsToRun int, total int, succinct bool) {
	stenographer.registerCall("AnnounceNumberOfSpecs", specsToRun, total, succinct)
}

func (stenographer *FakeStenographer) AnnounceTotalNumberOfSpecs(total int, succinct bool) {
	stenographer.registerCall("AnnounceTotalNumberOfSpecs", total, succinct)
}

func (stenographer *FakeStenographer) AnnounceSpecRunCompletion(summary *types.SuiteSummary, succinct bool) {
	stenographer.registerCall("AnnounceSpecRunCompletion", summary, succinct)
}

func (stenographer *FakeStenographer) AnnounceSpecWillRun(spec *types.SpecSummary) {
	stenographer.registerCall("AnnounceSpecWillRun", spec)
}

func (stenographer *FakeStenographer) AnnounceBeforeSuiteFailure(summary *types.SetupSummary, succinct bool, fullTrace bool) {
	stenographer.registerCall("AnnounceBeforeSuiteFailure", summary, succinct, fullTrace)
}

func (stenographer *FakeStenographer) AnnounceAfterSuiteFailure(summary *types.SetupSummary, succinct bool, fullTrace bool) {
	stenographer.registerCall("AnnounceAfterSuiteFailure", summary, succinct, fullTrace)
}
func (stenographer *FakeStenographer) AnnounceCapturedOutput(output string) {
	stenographer.registerCall("AnnounceCapturedOutput", output)
}

func (stenographer *FakeStenographer) AnnounceSuccessfulSpec(spec *types.SpecSummary) {
	stenographer.registerCall("AnnounceSuccessfulSpec", spec)
}

func (stenographer *FakeStenographer) AnnounceSuccessfulSlowSpec(spec *types.SpecSummary, succinct bool) {
	stenographer.registerCall("AnnounceSuccessfulSlowSpec", spec, succinct)
}

func (stenographer *FakeStenographer) AnnounceSuccessfulMeasurement(spec *types.SpecSummary, succinct bool) {
	stenographer.registerCall("AnnounceSuccessfulMeasurement", spec, succinct)
}

func (stenographer *FakeStenographer) AnnouncePendingSpec(spec *types.SpecSummary, noisy bool) {
	stenographer.registerCall("AnnouncePendingSpec", spec, noisy)
}

func (stenographer *FakeStenographer) AnnounceSkippedSpec(spec *types.SpecSummary, succinct bool, fullTrace bool) {
	stenographer.registerCall("AnnounceSkippedSpec", spec, succinct, fullTrace)
}

func (stenographer *FakeStenographer) AnnounceSpecTimedOut(spec *types.SpecSummary, succinct bool, fullTrace bool) {
	stenographer.registerCall("AnnounceSpecTimedOut", spec, succinct, fullTrace)
}

func (stenographer *FakeStenographer) AnnounceSpecPanicked(spec *types.SpecSummary, succinct bool, fullTrace bool) {
	stenographer.registerCall("AnnounceSpecPanicked", spec, succinct, fullTrace)
}

func (stenographer *FakeStenographer) AnnounceSpecFailed(spec *types.SpecSummary, succinct bool, fullTrace bool) {
	stenographer.registerCall("AnnounceSpecFailed", spec, succinct, fullTrace)
}

func (stenographer *FakeStenographer) SummarizeFailures(summaries []*types.SpecSummary) {
	stenographer.registerCall("SummarizeFailures", summaries)
}
