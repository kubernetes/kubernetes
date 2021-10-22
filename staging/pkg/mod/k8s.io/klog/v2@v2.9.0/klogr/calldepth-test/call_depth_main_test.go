// Package calldepth does black-box testing.
//
// Another intentional effect is that "go test" compiles
// this into a separate binary which we need because
// we have to configure klog differently that TestOutput.

package calldepth

import (
	"bytes"
	"flag"
	"strings"
	"testing"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/klogr"
)

func TestCallDepth(t *testing.T) {
	klog.InitFlags(nil)
	flag.CommandLine.Set("v", "10")
	flag.CommandLine.Set("skip_headers", "false")
	flag.CommandLine.Set("logtostderr", "false")
	flag.CommandLine.Set("alsologtostderr", "false")
	flag.CommandLine.Set("stderrthreshold", "10")
	flag.Parse()

	t.Run("call-depth", func(t *testing.T) {
		logr := klogr.New()

		// hijack the klog output
		tmpWriteBuffer := bytes.NewBuffer(nil)
		klog.SetOutput(tmpWriteBuffer)

		validate := func(t *testing.T) {
			output := tmpWriteBuffer.String()
			if !strings.Contains(output, "call_depth_main_test.go:") {
				t.Fatalf("output should have contained call_depth_main_test.go, got instead: %s", output)
			}
		}

		t.Run("direct", func(t *testing.T) {
			logr.Info("hello world")
			validate(t)
		})

		t.Run("indirect", func(t *testing.T) {
			myInfo(logr, "hello world")
			validate(t)
		})

		t.Run("nested", func(t *testing.T) {
			myInfo2(logr, "hello world")
			validate(t)
		})
	})
}
