package internal

import (
	"errors"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"

	"github.com/onsi/ginkgo/v2/types"
)

const TIMEOUT_ELAPSED_FAILURE_REASON = "Suite did not run because the timeout elapsed"
const PRIOR_FAILURES_FAILURE_REASON = "Suite did not run because prior suites failed and --keep-going is not set"
const EMPTY_SKIP_FAILURE_REASON = "Suite did not run go test reported that no test files were found"

type TestSuiteState uint

const (
	TestSuiteStateInvalid TestSuiteState = iota

	TestSuiteStateUncompiled
	TestSuiteStateCompiled

	TestSuiteStatePassed

	TestSuiteStateSkippedDueToEmptyCompilation
	TestSuiteStateSkippedByFilter
	TestSuiteStateSkippedDueToPriorFailures

	TestSuiteStateFailed
	TestSuiteStateFailedDueToTimeout
	TestSuiteStateFailedToCompile
)

var TestSuiteStateFailureStates = []TestSuiteState{TestSuiteStateFailed, TestSuiteStateFailedDueToTimeout, TestSuiteStateFailedToCompile}

func (state TestSuiteState) Is(states ...TestSuiteState) bool {
	for _, suiteState := range states {
		if suiteState == state {
			return true
		}
	}

	return false
}

type TestSuite struct {
	Path        string
	PackageName string
	IsGinkgo    bool

	Precompiled        bool
	PathToCompiledTest string
	CompilationError   error

	HasProgrammaticFocus bool
	State                TestSuiteState
}

func (ts TestSuite) AbsPath() string {
	path, _ := filepath.Abs(ts.Path)
	return path
}

func (ts TestSuite) NamespacedName() string {
	name := relPath(ts.Path)
	name = strings.TrimLeft(name, "."+string(filepath.Separator))
	name = strings.ReplaceAll(name, string(filepath.Separator), "_")
	name = strings.ReplaceAll(name, " ", "_")
	if name == "" {
		return ts.PackageName
	}
	return name
}

type TestSuites []TestSuite

func (ts TestSuites) AnyHaveProgrammaticFocus() bool {
	for _, suite := range ts {
		if suite.HasProgrammaticFocus {
			return true
		}
	}

	return false
}

func (ts TestSuites) ThatAreGinkgoSuites() TestSuites {
	out := TestSuites{}
	for _, suite := range ts {
		if suite.IsGinkgo {
			out = append(out, suite)
		}
	}
	return out
}

func (ts TestSuites) CountWithState(states ...TestSuiteState) int {
	n := 0
	for _, suite := range ts {
		if suite.State.Is(states...) {
			n += 1
		}
	}

	return n
}

func (ts TestSuites) WithState(states ...TestSuiteState) TestSuites {
	out := TestSuites{}
	for _, suite := range ts {
		if suite.State.Is(states...) {
			out = append(out, suite)
		}
	}

	return out
}

func (ts TestSuites) WithoutState(states ...TestSuiteState) TestSuites {
	out := TestSuites{}
	for _, suite := range ts {
		if !suite.State.Is(states...) {
			out = append(out, suite)
		}
	}

	return out
}

func (ts TestSuites) ShuffledCopy(seed int64) TestSuites {
	out := make(TestSuites, len(ts))
	permutation := rand.New(rand.NewSource(seed)).Perm(len(ts))
	for i, j := range permutation {
		out[i] = ts[j]
	}
	return out
}

func FindSuites(args []string, cliConfig types.CLIConfig, allowPrecompiled bool) TestSuites {
	suites := TestSuites{}

	if len(args) > 0 {
		for _, arg := range args {
			if allowPrecompiled {
				suite, err := precompiledTestSuite(arg)
				if err == nil {
					suites = append(suites, suite)
					continue
				}
			}
			recurseForSuite := cliConfig.Recurse
			if strings.HasSuffix(arg, "/...") && arg != "/..." {
				arg = arg[:len(arg)-4]
				recurseForSuite = true
			}
			suites = append(suites, suitesInDir(arg, recurseForSuite)...)
		}
	} else {
		suites = suitesInDir(".", cliConfig.Recurse)
	}

	if cliConfig.SkipPackage != "" {
		skipFilters := strings.Split(cliConfig.SkipPackage, ",")
		for idx := range suites {
			for _, skipFilter := range skipFilters {
				if strings.Contains(suites[idx].Path, skipFilter) {
					suites[idx].State = TestSuiteStateSkippedByFilter
					break
				}
			}
		}
	}

	return suites
}

func precompiledTestSuite(path string) (TestSuite, error) {
	info, err := os.Stat(path)
	if err != nil {
		return TestSuite{}, err
	}

	if info.IsDir() {
		return TestSuite{}, errors.New("this is a directory, not a file")
	}

	if filepath.Ext(path) != ".test" && filepath.Ext(path) != ".exe" {
		return TestSuite{}, errors.New("this is not a .test binary")
	}

	if filepath.Ext(path) == ".test" && runtime.GOOS != "windows" && info.Mode()&0111 == 0 {
		return TestSuite{}, errors.New("this is not executable")
	}

	dir := relPath(filepath.Dir(path))
	packageName := strings.TrimSuffix(filepath.Base(path), ".exe")
	packageName = strings.TrimSuffix(packageName, ".test")

	path, err = filepath.Abs(path)
	if err != nil {
		return TestSuite{}, err
	}

	return TestSuite{
		Path:               dir,
		PackageName:        packageName,
		IsGinkgo:           true,
		Precompiled:        true,
		PathToCompiledTest: path,
		State:              TestSuiteStateCompiled,
	}, nil
}

func suitesInDir(dir string, recurse bool) TestSuites {
	suites := TestSuites{}

	if path.Base(dir) == "vendor" {
		return suites
	}

	files, _ := os.ReadDir(dir)
	re := regexp.MustCompile(`^[^._].*_test\.go$`)
	for _, file := range files {
		if !file.IsDir() && re.MatchString(file.Name()) {
			suite := TestSuite{
				Path:        relPath(dir),
				PackageName: packageNameForSuite(dir),
				IsGinkgo:    filesHaveGinkgoSuite(dir, files),
				State:       TestSuiteStateUncompiled,
			}
			suites = append(suites, suite)
			break
		}
	}

	if recurse {
		re = regexp.MustCompile(`^[._]`)
		for _, file := range files {
			if file.IsDir() && !re.MatchString(file.Name()) {
				suites = append(suites, suitesInDir(dir+"/"+file.Name(), recurse)...)
			}
		}
	}

	return suites
}

func relPath(dir string) string {
	dir, _ = filepath.Abs(dir)
	cwd, _ := os.Getwd()
	dir, _ = filepath.Rel(cwd, filepath.Clean(dir))

	if string(dir[0]) != "." {
		dir = "." + string(filepath.Separator) + dir
	}

	return dir
}

func packageNameForSuite(dir string) string {
	path, _ := filepath.Abs(dir)
	return filepath.Base(path)
}

func filesHaveGinkgoSuite(dir string, files []os.DirEntry) bool {
	reTestFile := regexp.MustCompile(`_test\.go$`)
	reGinkgo := regexp.MustCompile(`package ginkgo|\/ginkgo"|\/ginkgo\/v2"|\/ginkgo\/v2/dsl/`)

	for _, file := range files {
		if !file.IsDir() && reTestFile.MatchString(file.Name()) {
			contents, _ := os.ReadFile(dir + "/" + file.Name())
			if reGinkgo.Match(contents) {
				return true
			}
		}
	}

	return false
}
