/*
Copyright 2015 The Kubernetes Authors.

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

// Package framework contains provider-independent helper code for
// building and running E2E tests with Ginkgo. The actual Ginkgo test
// suites gets assembled by combining this framework, the optional
// provider support code and specific tests via a separate .go file
// like Kubernetes' test/e2e.go.
package framework

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	ginkgotypes "github.com/onsi/ginkgo/v2/types"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/textlogger"

	_ "k8s.io/component-base/logs/testinit" // Ensure command line flags are registered.
)

var (
	logConfig = textlogger.NewConfig(
		textlogger.Output(ginkgo.GinkgoWriter),
		textlogger.Backtrace(unwind),
	)
	ginkgoLogger = textlogger.NewLogger(logConfig)
	TimeNow      = time.Now    // Can be stubbed out for testing.
	Pid          = os.Getpid() // Can be stubbed out for testing.
)

func init() {
	// ktesting and testinit already registered the -v and -vmodule
	// command line flags. To configure the textlogger and klog
	// consistently, we need to intercept the Set call. This
	// can be done by swapping out the flag.Value for the -v and
	// -vmodule flags with a wrapper which calls both.
	var fs flag.FlagSet
	logConfig.AddFlags(&fs)
	fs.VisitAll(func(loggerFlag *flag.Flag) {
		klogFlag := flag.CommandLine.Lookup(loggerFlag.Name)
		if klogFlag != nil {
			klogFlag.Value = &valueChain{Value: loggerFlag.Value, parentValue: klogFlag.Value}
		}
	})

	// Now install the textlogger as the klog default logger.
	// Calls like klog.Info then will write to ginkgo.GingoWriter
	// through the textlogger.
	//
	// However, stack unwinding is then still being done by klog and thus
	// ignores ginkgo.GingkoHelper. Tests should use framework.Logf or
	// structured, contextual logging.
	writer, _ := ginkgoLogger.GetSink().(textlogger.KlogBufferWriter)
	opts := []klog.LoggerOption{
		klog.ContextualLogger(true),
		klog.WriteKlogBuffer(writer.WriteKlogBuffer),
	}
	klog.SetLoggerWithOptions(ginkgoLogger, opts...)
}

type valueChain struct {
	flag.Value
	parentValue flag.Value
}

func (v *valueChain) Set(value string) error {
	if err := v.Value.Set(value); err != nil {
		return err
	}
	if err := v.parentValue.Set(value); err != nil {
		return err
	}
	return nil
}

func unwind(skip int) (string, int) {
	location := ginkgotypes.NewCodeLocation(skip + 1)
	return location.FileName, location.LineNumber
}

// log re-implements klog.Info: same header, but stack unwinding
// with support for ginkgo.GinkgoWriter and skipping stack levels.
func log(offset int, msg string) {
	now := TimeNow()
	file, line := unwind(offset + 1)
	if file == "" {
		file = "???"
		line = 1
	} else if slash := strings.LastIndex(file, "/"); slash >= 0 {
		file = file[slash+1:]
	}
	_, month, day := now.Date()
	hour, minute, second := now.Clock()
	header := fmt.Sprintf("I%02d%02d %02d:%02d:%02d.%06d %d %s:%d]",
		month, day, hour, minute, second, now.Nanosecond()/1000, Pid, file, line)

	fmt.Fprintln(ginkgo.GinkgoWriter, header, msg)
}
