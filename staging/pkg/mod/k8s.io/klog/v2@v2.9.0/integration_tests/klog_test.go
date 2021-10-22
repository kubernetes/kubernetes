package integration_tests_test

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"testing"
)

const (
	infoLog    = "this is a info log line"
	warningLog = "this is a warning log line"
	errorLog   = "this is a error log line"
	fatalLog   = "this is a fatal log line"
)

// res is a type alias to a slice of pointers to regular expressions.
type res = []*regexp.Regexp

var (
	infoLogRE    = regexp.MustCompile(regexp.QuoteMeta(infoLog))
	warningLogRE = regexp.MustCompile(regexp.QuoteMeta(warningLog))
	errorLogRE   = regexp.MustCompile(regexp.QuoteMeta(errorLog))
	fatalLogRE   = regexp.MustCompile(regexp.QuoteMeta(fatalLog))

	stackTraceRE = regexp.MustCompile(`\ngoroutine \d+ \[[^]]+\]:\n`)

	allLogREs = res{infoLogRE, warningLogRE, errorLogRE, fatalLogRE, stackTraceRE}

	defaultExpectedInDirREs = map[int]res{
		0: {stackTraceRE, fatalLogRE, errorLogRE, warningLogRE, infoLogRE},
		1: {stackTraceRE, fatalLogRE, errorLogRE, warningLogRE},
		2: {stackTraceRE, fatalLogRE, errorLogRE},
		3: {stackTraceRE, fatalLogRE},
	}
	expectedOneOutputInDirREs = map[int]res{
		0: {infoLogRE},
		1: {warningLogRE},
		2: {errorLogRE},
		3: {fatalLogRE},
	}

	defaultNotExpectedInDirREs = map[int]res{
		0: {},
		1: {infoLogRE},
		2: {infoLogRE, warningLogRE},
		3: {infoLogRE, warningLogRE, errorLogRE},
	}
)

func TestDestinationsWithDifferentFlags(t *testing.T) {
	tests := map[string]struct {
		// logfile states if the flag -log_file should be set
		logfile bool
		// logdir states if the flag -log_dir should be set
		logdir bool
		// flags is for additional flags to pass to the klog'ed executable
		flags []string

		// expectedLogFile states if we generally expect the log file to exist.
		// If this is not set, we expect the file not to exist and will error if it
		// does.
		expectedLogFile bool
		// expectedLogDir states if we generally expect the log files in the log
		// dir to exist.
		// If this is not set, we expect the log files in the log dir not to exist and
		// will error if they do.
		expectedLogDir bool

		// expectedOnStderr is a list of REs we expect to find on stderr
		expectedOnStderr res
		// notExpectedOnStderr is a list of REs that we must not find on stderr
		notExpectedOnStderr res
		// expectedInFile is a list of REs we expect to find in the log file
		expectedInFile res
		// notExpectedInFile is a list of REs we must not find in the log file
		notExpectedInFile res

		// expectedInDir is a list of REs we expect to find in the log files in the
		// log dir, specified by log severity (0 = warning, 1 = info, ...)
		expectedInDir map[int]res
		// notExpectedInDir is a list of REs we must not find in the log files in
		// the log dir, specified by log severity (0 = warning, 1 = info, ...)
		notExpectedInDir map[int]res
	}{
		"default flags": {
			// Everything, including the trace on fatal, goes to stderr

			expectedOnStderr: allLogREs,
		},
		"everything disabled": {
			// Nothing, including the trace on fatal, is showing anywhere

			flags: []string{"-logtostderr=false", "-alsologtostderr=false", "-stderrthreshold=1000"},

			notExpectedOnStderr: allLogREs,
		},
		"everything disabled but low stderrthreshold": {
			// Everything above -stderrthreshold, including the trace on fatal, will
			// be logged to stderr, even if we set -logtostderr to false.

			flags: []string{"-logtostderr=false", "-alsologtostderr=false", "-stderrthreshold=1"},

			expectedOnStderr:    res{warningLogRE, errorLogRE, stackTraceRE},
			notExpectedOnStderr: res{infoLogRE},
		},
		"with logtostderr only": {
			// Everything, including the trace on fatal, goes to stderr

			flags: []string{"-logtostderr=true", "-alsologtostderr=false", "-stderrthreshold=1000"},

			expectedOnStderr: allLogREs,
		},
		"with log file only": {
			// Everything, including the trace on fatal, goes to the single log file

			logfile: true,
			flags:   []string{"-logtostderr=false", "-alsologtostderr=false", "-stderrthreshold=1000"},

			expectedLogFile: true,

			notExpectedOnStderr: allLogREs,
			expectedInFile:      allLogREs,
		},
		"with log dir only": {
			// Everything, including the trace on fatal, goes to the log files in the log dir

			logdir: true,
			flags:  []string{"-logtostderr=false", "-alsologtostderr=false", "-stderrthreshold=1000"},

			expectedLogDir: true,

			notExpectedOnStderr: allLogREs,
			expectedInDir:       defaultExpectedInDirREs,
			notExpectedInDir:    defaultNotExpectedInDirREs,
		},
		"with log dir only and one_output": {
			// Everything, including the trace on fatal, goes to the log files in the log dir

			logdir: true,
			flags:  []string{"-logtostderr=false", "-alsologtostderr=false", "-stderrthreshold=1000", "-one_output=true"},

			expectedLogDir: true,

			notExpectedOnStderr: allLogREs,
			expectedInDir:       expectedOneOutputInDirREs,
			notExpectedInDir:    defaultNotExpectedInDirREs,
		},
		"with log dir and logtostderr": {
			// Everything, including the trace on fatal, goes to stderr. The -log_dir is
			// ignored, nothing goes to the log files in the log dir.

			logdir: true,
			flags:  []string{"-logtostderr=true", "-alsologtostderr=false", "-stderrthreshold=1000"},

			expectedOnStderr: allLogREs,
		},
		"with log file and log dir": {
			// Everything, including the trace on fatal, goes to the single log file.
			// The -log_dir is ignored, nothing goes to the log file in the log dir.

			logdir:  true,
			logfile: true,
			flags:   []string{"-logtostderr=false", "-alsologtostderr=false", "-stderrthreshold=1000"},

			expectedLogFile: true,

			notExpectedOnStderr: allLogREs,
			expectedInFile:      allLogREs,
		},
		"with log file and alsologtostderr": {
			// Everything, including the trace on fatal, goes to the single log file
			// AND to stderr.

			flags:   []string{"-alsologtostderr=true", "-logtostderr=false", "-stderrthreshold=1000"},
			logfile: true,

			expectedLogFile: true,

			expectedOnStderr: allLogREs,
			expectedInFile:   allLogREs,
		},
		"with log dir and alsologtostderr": {
			// Everything, including the trace on fatal, goes to the log file in the
			// log dir AND to stderr.

			logdir: true,
			flags:  []string{"-alsologtostderr=true", "-logtostderr=false", "-stderrthreshold=1000"},

			expectedLogDir: true,

			expectedOnStderr: allLogREs,
			expectedInDir:    defaultExpectedInDirREs,
			notExpectedInDir: defaultNotExpectedInDirREs,
		},
		"with log dir, alsologtostderr and one_output": {
			// Everything, including the trace on fatal, goes to the log file in the
			// log dir AND to stderr.

			logdir: true,
			flags:  []string{"-alsologtostderr=true", "-logtostderr=false", "-stderrthreshold=1000", "-one_output=true"},

			expectedLogDir: true,

			expectedOnStderr: allLogREs,
			expectedInDir:    expectedOneOutputInDirREs,
			notExpectedInDir: defaultNotExpectedInDirREs,
		},
	}

	binaryFileExtention := ""
	if runtime.GOOS == "windows" {
		binaryFileExtention = ".exe"
	}

	for tcName, tc := range tests {
		tc := tc
		t.Run(tcName, func(t *testing.T) {
			t.Parallel()
			withTmpDir(t, func(logdir string) {
				// :: Setup
				flags := tc.flags
				stderr := &bytes.Buffer{}
				logfile := filepath.Join(logdir, "the_single_log_file") // /some/tmp/dir/the_single_log_file

				if tc.logfile {
					flags = append(flags, "-log_file="+logfile)
				}
				if tc.logdir {
					flags = append(flags, "-log_dir="+logdir)
				}

				// :: Execute
				klogRun(t, flags, stderr)

				// :: Assert
				// check stderr
				checkForLogs(t, tc.expectedOnStderr, tc.notExpectedOnStderr, stderr.String(), "stderr")

				// check log_file
				if tc.expectedLogFile {
					content := getFileContent(t, logfile)
					checkForLogs(t, tc.expectedInFile, tc.notExpectedInFile, content, "logfile")
				} else {
					assertFileIsAbsent(t, logfile)
				}

				// check files in log_dir
				for level, levelName := range logFileLevels {
					binaryName := "main" + binaryFileExtention
					logfile, err := getLogFilePath(logdir, binaryName, levelName)
					if tc.expectedLogDir {
						if err != nil {
							t.Errorf("Unable to find log file: %v", err)
						}
						content := getFileContent(t, logfile)
						checkForLogs(t, tc.expectedInDir[level], tc.notExpectedInDir[level], content, "logfile["+logfile+"]")
					} else {
						if err == nil {
							t.Errorf("Unexpectedly found log file %s", logfile)
						}
					}
				}
			})
		})
	}
}

const klogExampleGoFile = "./internal/main.go"

// klogRun spawns a simple executable that uses klog, to later inspect its
// stderr and potentially created log files
func klogRun(t *testing.T, flags []string, stderr io.Writer) {
	callFlags := []string{"run", klogExampleGoFile}
	callFlags = append(callFlags, flags...)

	cmd := exec.Command("go", callFlags...)
	cmd.Stderr = stderr
	cmd.Env = append(os.Environ(),
		"KLOG_INFO_LOG="+infoLog,
		"KLOG_WARNING_LOG="+warningLog,
		"KLOG_ERROR_LOG="+errorLog,
		"KLOG_FATAL_LOG="+fatalLog,
	)

	err := cmd.Run()

	if _, ok := err.(*exec.ExitError); !ok {
		t.Fatalf("Run failed: %v", err)
	}
}

var logFileLevels = map[int]string{
	0: "INFO",
	1: "WARNING",
	2: "ERROR",
	3: "FATAL",
}

func getFileContent(t *testing.T, filePath string) string {
	content, err := ioutil.ReadFile(filePath)
	if err != nil {
		t.Errorf("Could not read file '%s': %v", filePath, err)
	}
	return string(content)
}

func assertFileIsAbsent(t *testing.T, filePath string) {
	if _, err := os.Stat(filePath); !os.IsNotExist(err) {
		t.Errorf("Expected file '%s' not to exist", filePath)
	}
}

func checkForLogs(t *testing.T, expected, disallowed res, content, name string) {
	for _, re := range expected {
		checkExpected(t, true, name, content, re)
	}
	for _, re := range disallowed {
		checkExpected(t, false, name, content, re)
	}
}

func checkExpected(t *testing.T, expected bool, where string, haystack string, needle *regexp.Regexp) {
	found := needle.MatchString(haystack)

	if expected && !found {
		t.Errorf("Expected to find '%s' in %s", needle, where)
	}
	if !expected && found {
		t.Errorf("Expected not to find '%s' in %s", needle, where)
	}
}

func withTmpDir(t *testing.T, f func(string)) {
	tmpDir, err := ioutil.TempDir("", "klog_e2e_")
	if err != nil {
		t.Fatalf("Could not create temp directory: %v", err)
	}
	defer func() {
		if err := os.RemoveAll(tmpDir); err != nil {
			t.Fatalf("Could not remove temp directory '%s': %v", tmpDir, err)
		}
	}()

	f(tmpDir)
}

// getLogFileFromDir returns the path of either the symbolic link to the logfile, or the the logfile itself. This must
// be done as the creation of a symlink is not guaranteed on any platform. On Windows, only users with administration
// privileges can create a symlink.
func getLogFilePath(dir, binaryName, levelName string) (string, error) {
	symlink := filepath.Join(dir, binaryName+"."+levelName)
	if _, err := os.Stat(symlink); err == nil {
		return symlink, nil
	}
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		return "", fmt.Errorf("could not read directory %s: %v", dir, err)
	}
	var foundFile string
	for _, file := range files {
		if strings.HasPrefix(file.Name(), binaryName) && strings.Contains(file.Name(), levelName) {
			if foundFile != "" {
				return "", fmt.Errorf("found multiple matching files")
			}
			foundFile = file.Name()
		}
	}
	if foundFile != "" {
		return filepath.Join(dir, foundFile), nil
	}
	return "", fmt.Errorf("file missing from directory")
}
