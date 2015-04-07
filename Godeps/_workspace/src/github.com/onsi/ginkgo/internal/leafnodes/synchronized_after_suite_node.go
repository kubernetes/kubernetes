package leafnodes

import (
	"encoding/json"
	"github.com/onsi/ginkgo/internal/failer"
	"github.com/onsi/ginkgo/types"
	"io/ioutil"
	"net/http"
	"time"
)

type synchronizedAfterSuiteNode struct {
	runnerA *runner
	runnerB *runner

	outcome types.SpecState
	failure types.SpecFailure
	runTime time.Duration
}

func NewSynchronizedAfterSuiteNode(bodyA interface{}, bodyB interface{}, codeLocation types.CodeLocation, timeout time.Duration, failer *failer.Failer) SuiteNode {
	return &synchronizedAfterSuiteNode{
		runnerA: newRunner(bodyA, codeLocation, timeout, failer, types.SpecComponentTypeAfterSuite, 0),
		runnerB: newRunner(bodyB, codeLocation, timeout, failer, types.SpecComponentTypeAfterSuite, 0),
	}
}

func (node *synchronizedAfterSuiteNode) Run(parallelNode int, parallelTotal int, syncHost string) bool {
	node.outcome, node.failure = node.runnerA.run()

	if parallelNode == 1 {
		if parallelTotal > 1 {
			node.waitUntilOtherNodesAreDone(syncHost)
		}

		outcome, failure := node.runnerB.run()

		if node.outcome == types.SpecStatePassed {
			node.outcome, node.failure = outcome, failure
		}
	}

	return node.outcome == types.SpecStatePassed
}

func (node *synchronizedAfterSuiteNode) Passed() bool {
	return node.outcome == types.SpecStatePassed
}

func (node *synchronizedAfterSuiteNode) Summary() *types.SetupSummary {
	return &types.SetupSummary{
		ComponentType: node.runnerA.nodeType,
		CodeLocation:  node.runnerA.codeLocation,
		State:         node.outcome,
		RunTime:       node.runTime,
		Failure:       node.failure,
	}
}

func (node *synchronizedAfterSuiteNode) waitUntilOtherNodesAreDone(syncHost string) {
	for {
		if node.canRun(syncHost) {
			return
		}

		time.Sleep(50 * time.Millisecond)
	}
}

func (node *synchronizedAfterSuiteNode) canRun(syncHost string) bool {
	resp, err := http.Get(syncHost + "/RemoteAfterSuiteData")
	if err != nil || resp.StatusCode != http.StatusOK {
		return false
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return false
	}
	resp.Body.Close()

	afterSuiteData := types.RemoteAfterSuiteData{}
	err = json.Unmarshal(body, &afterSuiteData)
	if err != nil {
		return false
	}

	return afterSuiteData.CanRun
}
