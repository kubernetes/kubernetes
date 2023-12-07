package internal

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"

	"github.com/onsi/ginkgo/v2/types"
)

func CompileSuite(suite TestSuite, goFlagsConfig types.GoFlagsConfig) TestSuite {
	if suite.PathToCompiledTest != "" {
		return suite
	}

	suite.CompilationError = nil

	path, err := filepath.Abs(filepath.Join(suite.Path, suite.PackageName+".test"))
	if err != nil {
		suite.State = TestSuiteStateFailedToCompile
		suite.CompilationError = fmt.Errorf("Failed to compute compilation target path:\n%s", err.Error())
		return suite
	}

	ginkgoInvocationPath, _ := os.Getwd()
	ginkgoInvocationPath, _ = filepath.Abs(ginkgoInvocationPath)
	packagePath := suite.AbsPath()
	pathToInvocationPath, err := filepath.Rel(packagePath, ginkgoInvocationPath)
	if err != nil {
		suite.State = TestSuiteStateFailedToCompile
		suite.CompilationError = fmt.Errorf("Failed to get relative path from package to the current working directory:\n%s", err.Error())
		return suite
	}
	args, err := types.GenerateGoTestCompileArgs(goFlagsConfig, path, "./", pathToInvocationPath)
	if err != nil {
		suite.State = TestSuiteStateFailedToCompile
		suite.CompilationError = fmt.Errorf("Failed to generate go test compile flags:\n%s", err.Error())
		return suite
	}

	cmd := exec.Command("go", args...)
	cmd.Dir = suite.Path
	output, err := cmd.CombinedOutput()
	if err != nil {
		if len(output) > 0 {
			suite.State = TestSuiteStateFailedToCompile
			suite.CompilationError = fmt.Errorf("Failed to compile %s:\n\n%s", suite.PackageName, output)
		} else {
			suite.State = TestSuiteStateFailedToCompile
			suite.CompilationError = fmt.Errorf("Failed to compile %s\n%s", suite.PackageName, err.Error())
		}
		return suite
	}

	if strings.Contains(string(output), "[no test files]") {
		suite.State = TestSuiteStateSkippedDueToEmptyCompilation
		return suite
	}

	if len(output) > 0 {
		fmt.Println(string(output))
	}

	if !FileExists(path) {
		suite.State = TestSuiteStateFailedToCompile
		suite.CompilationError = fmt.Errorf("Failed to compile %s:\nOutput file %s could not be found", suite.PackageName, path)
		return suite
	}

	suite.State = TestSuiteStateCompiled
	suite.PathToCompiledTest = path
	return suite
}

func Cleanup(goFlagsConfig types.GoFlagsConfig, suites ...TestSuite) {
	if goFlagsConfig.BinaryMustBePreserved() {
		return
	}
	for _, suite := range suites {
		if !suite.Precompiled {
			os.Remove(suite.PathToCompiledTest)
		}
	}
}

type parallelSuiteBundle struct {
	suite    TestSuite
	compiled chan TestSuite
}

type OrderedParallelCompiler struct {
	mutex        *sync.Mutex
	stopped      bool
	numCompilers int

	idx                int
	numSuites          int
	completionChannels []chan TestSuite
}

func NewOrderedParallelCompiler(numCompilers int) *OrderedParallelCompiler {
	return &OrderedParallelCompiler{
		mutex:        &sync.Mutex{},
		numCompilers: numCompilers,
	}
}

func (opc *OrderedParallelCompiler) StartCompiling(suites TestSuites, goFlagsConfig types.GoFlagsConfig) {
	opc.stopped = false
	opc.idx = 0
	opc.numSuites = len(suites)
	opc.completionChannels = make([]chan TestSuite, opc.numSuites)

	toCompile := make(chan parallelSuiteBundle, opc.numCompilers)
	for compiler := 0; compiler < opc.numCompilers; compiler++ {
		go func() {
			for bundle := range toCompile {
				c, suite := bundle.compiled, bundle.suite
				opc.mutex.Lock()
				stopped := opc.stopped
				opc.mutex.Unlock()
				if !stopped {
					suite = CompileSuite(suite, goFlagsConfig)
				}
				c <- suite
			}
		}()
	}

	for idx, suite := range suites {
		opc.completionChannels[idx] = make(chan TestSuite, 1)
		toCompile <- parallelSuiteBundle{suite, opc.completionChannels[idx]}
		if idx == 0 { //compile first suite serially
			suite = <-opc.completionChannels[0]
			opc.completionChannels[0] <- suite
		}
	}

	close(toCompile)
}

func (opc *OrderedParallelCompiler) Next() (int, TestSuite) {
	if opc.idx >= opc.numSuites {
		return opc.numSuites, TestSuite{}
	}

	idx := opc.idx
	suite := <-opc.completionChannels[idx]
	opc.idx = opc.idx + 1

	return idx, suite
}

func (opc *OrderedParallelCompiler) StopAndDrain() {
	opc.mutex.Lock()
	opc.stopped = true
	opc.mutex.Unlock()
}
