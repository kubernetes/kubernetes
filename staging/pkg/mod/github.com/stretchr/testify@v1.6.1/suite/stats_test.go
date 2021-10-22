package suite

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestPassedReturnsTrueWhenAllTestsPass(t *testing.T) {
	sinfo := newSuiteInformation()
	sinfo.TestStats = map[string]*TestInformation{
		"Test1": {TestName: "Test1", Passed: true},
		"Test2": {TestName: "Test2", Passed: true},
		"Test3": {TestName: "Test3", Passed: true},
	}

	assert.True(t, sinfo.Passed())
}

func TestPassedReturnsFalseWhenSomeTestFails(t *testing.T) {
	sinfo := newSuiteInformation()
	sinfo.TestStats = map[string]*TestInformation{
		"Test1": {TestName: "Test1", Passed: true},
		"Test2": {TestName: "Test2", Passed: false},
		"Test3": {TestName: "Test3", Passed: true},
	}

	assert.False(t, sinfo.Passed())
}
