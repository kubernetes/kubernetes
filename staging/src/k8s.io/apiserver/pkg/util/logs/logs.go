/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package logs

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/wait"
)

const (
	logFlushFreqFlagName = "log-flush-frequency"
	logFileFlagName      = "log-file"
	teeLogsFlagName      = "tee-logs"
)

var (
	logFlushFreq = pflag.Duration(logFlushFreqFlagName, 5*time.Second, "Maximum number of seconds between log flushes")
	logFile      = pflag.String(logFileFlagName, "", "The file to send stdout & stderr output to. Defaults to outputting to not redirecting.")
	teeLogs      = pflag.Bool(teeLogsFlagName, false, fmt.Sprintf("Whether to still output to stdout & stderr when --%s is set. If --log-file is unset, this value is ignored.", logFileFlagName))
)

// TODO(thockin): This is temporary until we agree on log dirs and put those into each cmd.
func init() {
	flag.Set("logtostderr", "true")
}

// AddFlags registers this package's flags on arbitrary FlagSets, such that they point to the
// same value as the global flags.
func AddFlags(fs *pflag.FlagSet) {
	fs.AddFlag(pflag.Lookup(logFlushFreqFlagName))
	fs.AddFlag(pflag.Lookup(logFileFlagName))
	fs.AddFlag(pflag.Lookup(teeLogsFlagName))
}

// GlogWriter serves as a bridge between the standard log package and the glog package.
type GlogWriter struct{}

// Write implements the io.Writer interface.
func (writer GlogWriter) Write(data []byte) (n int, err error) {
	glog.InfoDepth(1, string(data))
	return len(data), nil
}

// InitLogs initializes logs the way we want for kubernetes.
func InitLogs() {
	initLogFile()

	log.SetOutput(GlogWriter{})
	log.SetFlags(0)
	// The default glog flush interval is 5 seconds.
	go wait.Forever(glog.Flush, *logFlushFreq)
}

// NewLogger creates a new log.Logger which sends logs to glog.Info.
func NewLogger(prefix string) *log.Logger {
	return log.New(GlogWriter{}, prefix, 0)
}

// GlogSetter is a setter to set glog level.
func GlogSetter(val string) (string, error) {
	var level glog.Level
	if err := level.Set(val); err != nil {
		return "", fmt.Errorf("failed set glog.logging.verbosity %s: %v", val, err)
	}
	return fmt.Sprintf("successfully set glog.logging.verbosity to %s", val), nil
}

var (
	outputFile *os.File
	logTeeWG   sync.WaitGroup
)

// InitLogFile sets up the log file, and redirects stdout & stderr to the file. This must be called
// before any logging, otherwise those log lines will not be captured.
func initLogFile() {
	if !flag.Parsed() {
		os.Stderr.WriteString("ERROR: InitLogFile before flag.Parse")
		return
	}

	if *logFile == "" {
		// If log-file is unset, nothing to do.
		return
	}

	// If log-file is set, glog's logtostderr MUST be set to capture logs.
	if f := flag.Lookup("logtostderr"); f != nil {
		if f.Value.String() != "true" {
			defer glog.Warningf("Forcing --logtostderr to true") // Defer to capture logs to file.
			if err := f.Value.Set("true"); err != nil {
				glog.Errorf("Failed to set --logstderr=true: %v", err)
			}
		}
	}

	// Setup the log file.
	file, err := os.OpenFile(*logFile, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	if err != nil {
		glog.Errorf("Failed to open %s for logging: %v", *logFile, err)
		return
	}
	outputFile = file

	if *teeLogs {
		stdout := os.Stdout
		pipeR, pipeW, err := os.Pipe()
		if err != nil {
			glog.Errorf("failed to create stdout pipe: %v", err)
		}

		// Since tee causes log writing to be asynchronous, we combine stdout & stderr into a single
		// stream (tee'd to stdout) to maintain ordering.
		os.Stdout = pipeW
		os.Stderr = pipeW

		logTeeWG.Add(1)
		go func() {
			io.Copy(io.MultiWriter(stdout, file), pipeR)
			logTeeWG.Done()
		}()
	} else {
		os.Stdout = file
		os.Stderr = file
	}
}

func ShutDownLogs() {
	glog.Flush()
	if *logFile == "" {
		return
	}
	if *teeLogs {
		// Close the stdout pipe, and wait for the tee routine to flush its buffer and shutdown.
		os.Stdout.Close()
		logTeeWG.Wait()
	}
	outputFile.Close()
}
