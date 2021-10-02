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

// Package logs contains support for logging options, flags and setup.
// Commands must explicitly enable command line flags. They no longer
// get added automatically when importing this package.
package logs

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/spf13/pflag"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
)

const logFlushFreqFlagName = "log-flush-frequency"

var (
	packageFlags = flag.NewFlagSet("logging", flag.ContinueOnError)
	logFlushFreq time.Duration
)

func init() {
	klog.InitFlags(packageFlags)
	packageFlags.DurationVar(&logFlushFreq, logFlushFreqFlagName, 5*time.Second, "Maximum number of seconds between log flushes")
}

// AddFlags registers this package's flags on arbitrary FlagSets. This includes
// the klog flags, with the original underscore as separator between. If
// commands want hyphens as separators, they can set
// k8s.io/component-base/cli/flag/WordSepNormalizeFunc as normalization
// function on the flag set before calling AddFlags.
//
// May be called more than once.
func AddFlags(fs *pflag.FlagSet) {
	// Determine whether the flags are already present by looking up one
	// which always should exist.
	if f := fs.Lookup("v"); f != nil {
		return
	}
	fs.AddGoFlagSet(packageFlags)
}

// AddGoFlags is a variant of AddFlags for traditional Go flag.FlagSet.
// Commands should use pflag whenever possible for the sake of consistency.
// Cases where this function is needed include tests (they have to set up flags
// in flag.CommandLine) and commands that for historic reasons use Go
// flag.Parse and cannot change to pflag because it would break their command
// line interface.
func AddGoFlags(fs *flag.FlagSet) {
	packageFlags.VisitAll(func(f *flag.Flag) {
		fs.Var(f.Value, f.Name, f.Usage)
	})
}

// KlogWriter serves as a bridge between the standard log package and the glog package.
type KlogWriter struct{}

// Write implements the io.Writer interface.
func (writer KlogWriter) Write(data []byte) (n int, err error) {
	klog.InfoDepth(1, string(data))
	return len(data), nil
}

// InitLogs initializes logs the way we want for Kubernetes.
// It should be called after parsing flags. If called before that,
// it will use the default log settings.
func InitLogs() {
	log.SetOutput(KlogWriter{})
	log.SetFlags(0)
	// The default klog flush interval is 5 seconds.
	go wait.Forever(klog.Flush, logFlushFreq)
}

// FlushLogs flushes logs immediately. This should be called at the end of
// the main function via defer to ensure that all pending log messages
// are printed before exiting the program.
func FlushLogs() {
	klog.Flush()
}

// NewLogger creates a new log.Logger which sends logs to klog.Info.
func NewLogger(prefix string) *log.Logger {
	return log.New(KlogWriter{}, prefix, 0)
}

// GlogSetter is a setter to set glog level.
func GlogSetter(val string) (string, error) {
	var level klog.Level
	if err := level.Set(val); err != nil {
		return "", fmt.Errorf("failed set klog.logging.verbosity %s: %v", val, err)
	}
	return fmt.Sprintf("successfully set klog.logging.verbosity to %s", val), nil
}
