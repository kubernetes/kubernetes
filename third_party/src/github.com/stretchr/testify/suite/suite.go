package suite

import (
	"flag"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"testing"
)

var matchMethod = flag.String("m", "", "regular expression to select tests of the suite to run")

// Suite is a basic testing suite with methods for storing and
// retrieving the current *testing.T context.
type Suite struct {
	t *testing.T
}

// T retrieves the current *testing.T context.
func (suite *Suite) T() *testing.T {
	return suite.t
}

// SetT sets the current *testing.T context.
func (suite *Suite) SetT(t *testing.T) {
	suite.t = t
}

// Run takes a testing suite and runs all of the tests attached
// to it.
func Run(t *testing.T, suite TestingSuite) {
	suite.SetT(t)

	if setupAllSuite, ok := suite.(SetupAllSuite); ok {
		setupAllSuite.SetupSuite()
	}

	methodFinder := reflect.TypeOf(suite)
	tests := []testing.InternalTest{}
	for index := 0; index < methodFinder.NumMethod(); index++ {
		method := methodFinder.Method(index)
		ok, err := methodFilter(method.Name)
		if err != nil {
			fmt.Fprintf(os.Stderr, "testify: invalid regexp for -m: %s\n", err)
			os.Exit(1)
		}
		if ok {
			test := testing.InternalTest{
				Name: method.Name,
				F: func(t *testing.T) {
					parentT := suite.T()
					suite.SetT(t)
					if setupTestSuite, ok := suite.(SetupTestSuite); ok {
						setupTestSuite.SetupTest()
					}
					method.Func.Call([]reflect.Value{reflect.ValueOf(suite)})
					if tearDownTestSuite, ok := suite.(TearDownTestSuite); ok {
						tearDownTestSuite.TearDownTest()
					}
					suite.SetT(parentT)
				},
			}
			tests = append(tests, test)
		}
	}

	if !testing.RunTests(func(_, _ string) (bool, error) { return true, nil },
		tests) {
		t.Fail()
	}

	if tearDownAllSuite, ok := suite.(TearDownAllSuite); ok {
		tearDownAllSuite.TearDownSuite()
	}
}

// Filtering method according to set regular expression
// specified command-line argument -m
func methodFilter(name string) (bool, error) {
	if ok, _ := regexp.MatchString("^Test", name); !ok {
		return false, nil
	}
	return regexp.MatchString(*matchMethod, name)
}
