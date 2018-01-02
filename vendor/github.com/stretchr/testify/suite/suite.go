package suite

import (
	"flag"
	"fmt"
	"os"
	"reflect"
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var allTestsFilter = func(_, _ string) (bool, error) { return true, nil }
var matchMethod = flag.String("testify.m", "", "regular expression to select tests of the testify suite to run")

// Suite is a basic testing suite with methods for storing and
// retrieving the current *testing.T context.
type Suite struct {
	*assert.Assertions
	require *require.Assertions
	t       *testing.T
}

// T retrieves the current *testing.T context.
func (suite *Suite) T() *testing.T {
	return suite.t
}

// SetT sets the current *testing.T context.
func (suite *Suite) SetT(t *testing.T) {
	suite.t = t
	suite.Assertions = assert.New(t)
	suite.require = require.New(t)
}

// Require returns a require context for suite.
func (suite *Suite) Require() *require.Assertions {
	if suite.require == nil {
		suite.require = require.New(suite.T())
	}
	return suite.require
}

// Assert returns an assert context for suite.  Normally, you can call
// `suite.NoError(expected, actual)`, but for situations where the embedded
// methods are overridden (for example, you might want to override
// assert.Assertions with require.Assertions), this method is provided so you
// can call `suite.Assert().NoError()`.
func (suite *Suite) Assert() *assert.Assertions {
	if suite.Assertions == nil {
		suite.Assertions = assert.New(suite.T())
	}
	return suite.Assertions
}

// Run takes a testing suite and runs all of the tests attached
// to it.
func Run(t *testing.T, suite TestingSuite) {
	suite.SetT(t)

	if setupAllSuite, ok := suite.(SetupAllSuite); ok {
		setupAllSuite.SetupSuite()
	}
	defer func() {
		if tearDownAllSuite, ok := suite.(TearDownAllSuite); ok {
			tearDownAllSuite.TearDownSuite()
		}
	}()

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
					if beforeTestSuite, ok := suite.(BeforeTest); ok {
						beforeTestSuite.BeforeTest(methodFinder.Elem().Name(), method.Name)
					}
					defer func() {
						if afterTestSuite, ok := suite.(AfterTest); ok {
							afterTestSuite.AfterTest(methodFinder.Elem().Name(), method.Name)
						}
						if tearDownTestSuite, ok := suite.(TearDownTestSuite); ok {
							tearDownTestSuite.TearDownTest()
						}
						suite.SetT(parentT)
					}()
					method.Func.Call([]reflect.Value{reflect.ValueOf(suite)})
				},
			}
			tests = append(tests, test)
		}
	}
	runTests(t, tests)
}

func runTests(t testing.TB, tests []testing.InternalTest) {
	r, ok := t.(runner)
	if !ok { // backwards compatibility with Go 1.6 and below
		if !testing.RunTests(allTestsFilter, tests) {
			t.Fail()
		}
		return
	}

	for _, test := range tests {
		r.Run(test.Name, test.F)
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

type runner interface {
	Run(name string, f func(t *testing.T)) bool
}
