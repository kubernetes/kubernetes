package steps

import (
	"github.com/DATA-DOG/godog"
)

func existingTest(arg1 string) error {
	return godog.ErrPending
}

func iRunTheTest() error {
	return godog.ErrPending
}

func theExistingTestWillPass() error {
	return godog.ErrPending
}

func thisIsFine() error {
	return godog.ErrPending
}

func FirstStepsFeatureContext(s *godog.Suite) {
	s.Step(`^existing test "([^"]*)"$`, existingTest)
	s.Step(`^I run the test$`, iRunTheTest)
	s.Step(`^the existing test will pass$`, theExistingTestWillPass)
	s.Step(`^this is fine$`, thisIsFine)
}
