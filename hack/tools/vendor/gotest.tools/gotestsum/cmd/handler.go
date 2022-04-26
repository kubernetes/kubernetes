package cmd

import (
	"fmt"
	"io"
	"os"
	"os/exec"

	"github.com/pkg/errors"
	"gotest.tools/gotestsum/internal/junitxml"
	"gotest.tools/gotestsum/log"
	"gotest.tools/gotestsum/testjson"
)

type eventHandler struct {
	formatter testjson.EventFormatter
	err       io.Writer
	jsonFile  io.WriteCloser
	maxFails  int
}

func (h *eventHandler) Err(text string) error {
	_, _ = h.err.Write([]byte(text + "\n"))
	// always return nil, no need to stop scanning if the stderr write fails
	return nil
}

func (h *eventHandler) Event(event testjson.TestEvent, execution *testjson.Execution) error {
	// ignore artificial events with no raw Bytes()
	if h.jsonFile != nil && len(event.Bytes()) > 0 {
		_, err := h.jsonFile.Write(append(event.Bytes(), '\n'))
		if err != nil {
			return errors.Wrap(err, "failed to write JSON file")
		}
	}

	err := h.formatter.Format(event, execution)
	if err != nil {
		return errors.Wrap(err, "failed to format event")
	}

	if h.maxFails > 0 && len(execution.Failed()) >= h.maxFails {
		return fmt.Errorf("ending test run because max failures was reached")
	}
	return nil
}

func (h *eventHandler) Close() error {
	if h.jsonFile != nil {
		if err := h.jsonFile.Close(); err != nil {
			log.Errorf("Failed to close JSON file: %v", err)
		}
	}
	return nil
}

var _ testjson.EventHandler = &eventHandler{}

func newEventHandler(opts *options) (*eventHandler, error) {
	formatter := testjson.NewEventFormatter(opts.stdout, opts.format)
	if formatter == nil {
		return nil, errors.Errorf("unknown format %s", opts.format)
	}
	handler := &eventHandler{
		formatter: formatter,
		err:       opts.stderr,
		maxFails:  opts.maxFails,
	}
	var err error
	if opts.jsonFile != "" {
		handler.jsonFile, err = os.Create(opts.jsonFile)
		if err != nil {
			return handler, errors.Wrap(err, "failed to open JSON file")
		}
	}
	return handler, nil
}

func writeJUnitFile(opts *options, execution *testjson.Execution) error {
	if opts.junitFile == "" {
		return nil
	}
	junitFile, err := os.Create(opts.junitFile)
	if err != nil {
		return fmt.Errorf("failed to open JUnit file: %v", err)
	}
	defer func() {
		if err := junitFile.Close(); err != nil {
			log.Errorf("Failed to close JUnit file: %v", err)
		}
	}()

	return junitxml.Write(junitFile, execution, junitxml.Config{
		FormatTestSuiteName:     opts.junitTestSuiteNameFormat.Value(),
		FormatTestCaseClassname: opts.junitTestCaseClassnameFormat.Value(),
	})
}

func postRunHook(opts *options, execution *testjson.Execution) error {
	command := opts.postRunHookCmd.Value()
	if len(command) == 0 {
		return nil
	}

	cmd := exec.Command(command[0], command[1:]...)
	cmd.Stdout = opts.stdout
	cmd.Stderr = opts.stderr
	cmd.Env = append(
		os.Environ(),
		"GOTESTSUM_JSONFILE="+opts.jsonFile,
		"GOTESTSUM_JUNITFILE="+opts.junitFile,
		fmt.Sprintf("TESTS_TOTAL=%d", execution.Total()),
		fmt.Sprintf("TESTS_FAILED=%d", len(execution.Failed())),
		fmt.Sprintf("TESTS_SKIPPED=%d", len(execution.Skipped())),
		fmt.Sprintf("TESTS_ERRORS=%d", len(execution.Errors())),
	)
	// TODO: send a more detailed report to stdin?
	return cmd.Run()
}
