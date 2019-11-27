package testrunner

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/onsi/ginkgo/config"
	"github.com/onsi/ginkgo/ginkgo/testsuite"
	"github.com/onsi/ginkgo/internal/remote"
	"github.com/onsi/ginkgo/reporters/stenographer"
	colorable "github.com/onsi/ginkgo/reporters/stenographer/support/go-colorable"
	"github.com/onsi/ginkgo/types"
)

type TestRunner struct {
	Suite testsuite.TestSuite

	compiled              bool
	compilationTargetPath string

	numCPU         int
	parallelStream bool
	timeout        time.Duration
	goOpts         map[string]interface{}
	additionalArgs []string
	stderr         *bytes.Buffer

	CoverageFile string
}

func New(suite testsuite.TestSuite, numCPU int, parallelStream bool, timeout time.Duration, goOpts map[string]interface{}, additionalArgs []string) *TestRunner {
	runner := &TestRunner{
		Suite:          suite,
		numCPU:         numCPU,
		parallelStream: parallelStream,
		goOpts:         goOpts,
		additionalArgs: additionalArgs,
		timeout:        timeout,
		stderr:         new(bytes.Buffer),
	}

	if !suite.Precompiled {
		dir, err := ioutil.TempDir("", "ginkgo")
		if err != nil {
			panic(fmt.Sprintf("couldn't create temporary directory... might be time to rm -rf:\n%s", err.Error()))
		}
		runner.compilationTargetPath = filepath.Join(dir, suite.PackageName+".test")
	}

	return runner
}

func (t *TestRunner) Compile() error {
	return t.CompileTo(t.compilationTargetPath)
}

func (t *TestRunner) BuildArgs(path string) []string {
	args := make([]string, len(buildArgs), len(buildArgs)+3)
	copy(args, buildArgs)
	args = append(args, "-o", path, t.Suite.Path)

	if t.getCoverMode() != "" {
		args = append(args, "-cover", fmt.Sprintf("-covermode=%s", t.getCoverMode()))
	} else {
		if t.shouldCover() || t.getCoverPackage() != "" {
			args = append(args, "-cover", "-covermode=atomic")
		}
	}

	boolOpts := []string{
		"a",
		"n",
		"msan",
		"race",
		"x",
		"work",
		"linkshared",
	}

	for _, opt := range boolOpts {
		if s, found := t.goOpts[opt].(*bool); found && *s {
			args = append(args, fmt.Sprintf("-%s", opt))
		}
	}

	intOpts := []string{
		"memprofilerate",
		"blockprofilerate",
	}

	for _, opt := range intOpts {
		if s, found := t.goOpts[opt].(*int); found {
			args = append(args, fmt.Sprintf("-%s=%d", opt, *s))
		}
	}

	stringOpts := []string{
		"asmflags",
		"buildmode",
		"compiler",
		"gccgoflags",
		"installsuffix",
		"ldflags",
		"pkgdir",
		"toolexec",
		"coverprofile",
		"cpuprofile",
		"memprofile",
		"outputdir",
		"coverpkg",
		"tags",
		"gcflags",
		"vet",
		"mod",
	}

	for _, opt := range stringOpts {
		if s, found := t.goOpts[opt].(*string); found && *s != "" {
			args = append(args, fmt.Sprintf("-%s=%s", opt, *s))
		}
	}
	return args
}

func (t *TestRunner) CompileTo(path string) error {
	if t.compiled {
		return nil
	}

	if t.Suite.Precompiled {
		return nil
	}

	args := t.BuildArgs(path)
	cmd := exec.Command("go", args...)

	output, err := cmd.CombinedOutput()

	if err != nil {
		if len(output) > 0 {
			return fmt.Errorf("Failed to compile %s:\n\n%s", t.Suite.PackageName, output)
		}
		return fmt.Errorf("Failed to compile %s", t.Suite.PackageName)
	}

	if len(output) > 0 {
		fmt.Println(string(output))
	}

	if !fileExists(path) {
		compiledFile := t.Suite.PackageName + ".test"
		if fileExists(compiledFile) {
			// seems like we are on an old go version that does not support the -o flag on go test
			// move the compiled test file to the desired location by hand
			err = os.Rename(compiledFile, path)
			if err != nil {
				// We cannot move the file, perhaps because the source and destination
				// are on different partitions. We can copy the file, however.
				err = copyFile(compiledFile, path)
				if err != nil {
					return fmt.Errorf("Failed to copy compiled file: %s", err)
				}
			}
		} else {
			return fmt.Errorf("Failed to compile %s: output file %q could not be found", t.Suite.PackageName, path)
		}
	}

	t.compiled = true

	return nil
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil || !os.IsNotExist(err)
}

// copyFile copies the contents of the file named src to the file named
// by dst. The file will be created if it does not already exist. If the
// destination file exists, all it's contents will be replaced by the contents
// of the source file.
func copyFile(src, dst string) error {
	srcInfo, err := os.Stat(src)
	if err != nil {
		return err
	}
	mode := srcInfo.Mode()

	in, err := os.Open(src)
	if err != nil {
		return err
	}

	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return err
	}

	defer func() {
		closeErr := out.Close()
		if err == nil {
			err = closeErr
		}
	}()

	_, err = io.Copy(out, in)
	if err != nil {
		return err
	}

	err = out.Sync()
	if err != nil {
		return err
	}

	return out.Chmod(mode)
}

func (t *TestRunner) Run() RunResult {
	if t.Suite.IsGinkgo {
		if t.numCPU > 1 {
			if t.parallelStream {
				return t.runAndStreamParallelGinkgoSuite()
			} else {
				return t.runParallelGinkgoSuite()
			}
		} else {
			return t.runSerialGinkgoSuite()
		}
	} else {
		return t.runGoTestSuite()
	}
}

func (t *TestRunner) CleanUp() {
	if t.Suite.Precompiled {
		return
	}
	os.RemoveAll(filepath.Dir(t.compilationTargetPath))
}

func (t *TestRunner) runSerialGinkgoSuite() RunResult {
	ginkgoArgs := config.BuildFlagArgs("ginkgo", config.GinkgoConfig, config.DefaultReporterConfig)
	return t.run(t.cmd(ginkgoArgs, os.Stdout, 1), nil)
}

func (t *TestRunner) runGoTestSuite() RunResult {
	return t.run(t.cmd([]string{"-test.v"}, os.Stdout, 1), nil)
}

func (t *TestRunner) runAndStreamParallelGinkgoSuite() RunResult {
	completions := make(chan RunResult)
	writers := make([]*logWriter, t.numCPU)

	server, err := remote.NewServer(t.numCPU)
	if err != nil {
		panic("Failed to start parallel spec server")
	}

	server.Start()
	defer server.Close()

	for cpu := 0; cpu < t.numCPU; cpu++ {
		config.GinkgoConfig.ParallelNode = cpu + 1
		config.GinkgoConfig.ParallelTotal = t.numCPU
		config.GinkgoConfig.SyncHost = server.Address()

		ginkgoArgs := config.BuildFlagArgs("ginkgo", config.GinkgoConfig, config.DefaultReporterConfig)

		writers[cpu] = newLogWriter(os.Stdout, cpu+1)

		cmd := t.cmd(ginkgoArgs, writers[cpu], cpu+1)

		server.RegisterAlive(cpu+1, func() bool {
			if cmd.ProcessState == nil {
				return true
			}
			return !cmd.ProcessState.Exited()
		})

		go t.run(cmd, completions)
	}

	res := PassingRunResult()

	for cpu := 0; cpu < t.numCPU; cpu++ {
		res = res.Merge(<-completions)
	}

	for _, writer := range writers {
		writer.Close()
	}

	os.Stdout.Sync()

	if t.shouldCombineCoverprofiles() {
		t.combineCoverprofiles()
	}

	return res
}

func (t *TestRunner) runParallelGinkgoSuite() RunResult {
	result := make(chan bool)
	completions := make(chan RunResult)
	writers := make([]*logWriter, t.numCPU)
	reports := make([]*bytes.Buffer, t.numCPU)

	stenographer := stenographer.New(!config.DefaultReporterConfig.NoColor, config.GinkgoConfig.FlakeAttempts > 1, colorable.NewColorableStdout())
	aggregator := remote.NewAggregator(t.numCPU, result, config.DefaultReporterConfig, stenographer)

	server, err := remote.NewServer(t.numCPU)
	if err != nil {
		panic("Failed to start parallel spec server")
	}
	server.RegisterReporters(aggregator)
	server.Start()
	defer server.Close()

	for cpu := 0; cpu < t.numCPU; cpu++ {
		config.GinkgoConfig.ParallelNode = cpu + 1
		config.GinkgoConfig.ParallelTotal = t.numCPU
		config.GinkgoConfig.SyncHost = server.Address()
		config.GinkgoConfig.StreamHost = server.Address()

		ginkgoArgs := config.BuildFlagArgs("ginkgo", config.GinkgoConfig, config.DefaultReporterConfig)

		reports[cpu] = &bytes.Buffer{}
		writers[cpu] = newLogWriter(reports[cpu], cpu+1)

		cmd := t.cmd(ginkgoArgs, writers[cpu], cpu+1)

		server.RegisterAlive(cpu+1, func() bool {
			if cmd.ProcessState == nil {
				return true
			}
			return !cmd.ProcessState.Exited()
		})

		go t.run(cmd, completions)
	}

	res := PassingRunResult()

	for cpu := 0; cpu < t.numCPU; cpu++ {
		res = res.Merge(<-completions)
	}

	//all test processes are done, at this point
	//we should be able to wait for the aggregator to tell us that it's done

	select {
	case <-result:
		fmt.Println("")
	case <-time.After(time.Second):
		//the aggregator never got back to us!  something must have gone wrong
		fmt.Println(`
	 -------------------------------------------------------------------
	|                                                                   |
	|  Ginkgo timed out waiting for all parallel nodes to report back!  |
	|                                                                   |
	 -------------------------------------------------------------------`)
		fmt.Println("\n", t.Suite.PackageName, "timed out. path:", t.Suite.Path)
		os.Stdout.Sync()

		for _, writer := range writers {
			writer.Close()
		}

		for _, report := range reports {
			fmt.Print(report.String())
		}

		os.Stdout.Sync()
	}

	if t.shouldCombineCoverprofiles() {
		t.combineCoverprofiles()
	}

	return res
}

const CoverProfileSuffix = ".coverprofile"

func (t *TestRunner) cmd(ginkgoArgs []string, stream io.Writer, node int) *exec.Cmd {
	args := []string{"--test.timeout=" + t.timeout.String()}

	coverProfile := t.getCoverProfile()

	if t.shouldCombineCoverprofiles() {

		testCoverProfile := "--test.coverprofile="

		coverageFile := ""
		// Set default name for coverage results
		if coverProfile == "" {
			coverageFile = t.Suite.PackageName + CoverProfileSuffix
		} else {
			coverageFile = coverProfile
		}

		testCoverProfile += coverageFile

		t.CoverageFile = filepath.Join(t.Suite.Path, coverageFile)

		if t.numCPU > 1 {
			testCoverProfile = fmt.Sprintf("%s.%d", testCoverProfile, node)
		}
		args = append(args, testCoverProfile)
	}

	args = append(args, ginkgoArgs...)
	args = append(args, t.additionalArgs...)

	path := t.compilationTargetPath
	if t.Suite.Precompiled {
		path, _ = filepath.Abs(filepath.Join(t.Suite.Path, fmt.Sprintf("%s.test", t.Suite.PackageName)))
	}

	cmd := exec.Command(path, args...)

	cmd.Dir = t.Suite.Path
	cmd.Stderr = io.MultiWriter(stream, t.stderr)
	cmd.Stdout = stream

	return cmd
}

func (t *TestRunner) shouldCover() bool {
	return *t.goOpts["cover"].(*bool)
}

func (t *TestRunner) shouldRequireSuite() bool {
	return *t.goOpts["requireSuite"].(*bool)
}

func (t *TestRunner) getCoverProfile() string {
	return *t.goOpts["coverprofile"].(*string)
}

func (t *TestRunner) getCoverPackage() string {
	return *t.goOpts["coverpkg"].(*string)
}

func (t *TestRunner) getCoverMode() string {
	return *t.goOpts["covermode"].(*string)
}

func (t *TestRunner) shouldCombineCoverprofiles() bool {
	return t.shouldCover() || t.getCoverPackage() != "" || t.getCoverMode() != ""
}

func (t *TestRunner) run(cmd *exec.Cmd, completions chan RunResult) RunResult {
	var res RunResult

	defer func() {
		if completions != nil {
			completions <- res
		}
	}()

	err := cmd.Start()
	if err != nil {
		fmt.Printf("Failed to run test suite!\n\t%s", err.Error())
		return res
	}

	cmd.Wait()

	exitStatus := cmd.ProcessState.Sys().(syscall.WaitStatus).ExitStatus()
	res.Passed = (exitStatus == 0) || (exitStatus == types.GINKGO_FOCUS_EXIT_CODE)
	res.HasProgrammaticFocus = (exitStatus == types.GINKGO_FOCUS_EXIT_CODE)

	if strings.Contains(t.stderr.String(), "warning: no tests to run") {
		if t.shouldRequireSuite() {
			res.Passed = false
		}
		fmt.Fprintf(os.Stderr, `Found no test suites, did you forget to run "ginkgo bootstrap"?`)
	}

	return res
}

func (t *TestRunner) combineCoverprofiles() {
	profiles := []string{}

	coverProfile := t.getCoverProfile()

	for cpu := 1; cpu <= t.numCPU; cpu++ {
		var coverFile string
		if coverProfile == "" {
			coverFile = fmt.Sprintf("%s%s.%d", t.Suite.PackageName, CoverProfileSuffix, cpu)
		} else {
			coverFile = fmt.Sprintf("%s.%d", coverProfile, cpu)
		}

		coverFile = filepath.Join(t.Suite.Path, coverFile)
		coverProfile, err := ioutil.ReadFile(coverFile)
		os.Remove(coverFile)

		if err == nil {
			profiles = append(profiles, string(coverProfile))
		}
	}

	if len(profiles) != t.numCPU {
		return
	}

	lines := map[string]int{}
	lineOrder := []string{}
	for i, coverProfile := range profiles {
		for _, line := range strings.Split(coverProfile, "\n")[1:] {
			if len(line) == 0 {
				continue
			}
			components := strings.Split(line, " ")
			count, _ := strconv.Atoi(components[len(components)-1])
			prefix := strings.Join(components[0:len(components)-1], " ")
			lines[prefix] += count
			if i == 0 {
				lineOrder = append(lineOrder, prefix)
			}
		}
	}

	output := []string{"mode: atomic"}
	for _, line := range lineOrder {
		output = append(output, fmt.Sprintf("%s %d", line, lines[line]))
	}
	finalOutput := strings.Join(output, "\n")

	finalFilename := ""

	if coverProfile != "" {
		finalFilename = coverProfile
	} else {
		finalFilename = fmt.Sprintf("%s%s", t.Suite.PackageName, CoverProfileSuffix)
	}

	coverageFilepath := filepath.Join(t.Suite.Path, finalFilename)
	ioutil.WriteFile(coverageFilepath, []byte(finalOutput), 0666)

	t.CoverageFile = coverageFilepath
}
