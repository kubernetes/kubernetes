/*

This file is intended to be used as a standin for a klog'ed executable.

It is called by the integration test via `go run` and with different klog
flags to assert on klog behaviour, especially where klog logs its output
when different combinations of the klog flags are at play.

This file is not intended to be used outside of the integration tests and
is not supposed to be a (good) example on how to use klog.

*/

package main

import (
	"flag"
	"fmt"
	"os"

	"k8s.io/klog/v2"
)

func main() {
	infoLogLine := getEnvOrDie("KLOG_INFO_LOG")
	warningLogLine := getEnvOrDie("KLOG_WARNING_LOG")
	errorLogLine := getEnvOrDie("KLOG_ERROR_LOG")
	fatalLogLine := getEnvOrDie("KLOG_FATAL_LOG")

	klog.InitFlags(nil)
	flag.Parse()
	klog.Info(infoLogLine)
	klog.Warning(warningLogLine)
	klog.Error(errorLogLine)
	klog.Flush()
	klog.Fatal(fatalLogLine)
}

func getEnvOrDie(name string) string {
	val, ok := os.LookupEnv(name)
	if !ok {
		fmt.Fprintf(os.Stderr, name+" could not be found in environment")
		os.Exit(1)
	}
	return val
}
