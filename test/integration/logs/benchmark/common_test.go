/*
Copyright 2021 The Kubernetes Authors.

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

package benchmark

import (
	"flag"
	"os"
	"testing"

	"k8s.io/klog/v2"
)

func init() {
	// hack/make-rules/test-integration.sh expects that all unit tests
	// support -v and -vmodule.
	klog.InitFlags(nil)

	// Write all output into a single file.
	flag.Set("alsologtostderr", "false")
	flag.Set("logtostderr", "false")
	flag.Set("one_output", "true")
	flag.Set("stderrthreshold", "FATAL")
}

func newBytesWritten(tb testing.TB, filename string) *bytesWritten {
	out, err := os.Create(filename)
	if err != nil {
		tb.Fatalf("open fake output: %v", err)
	}
	tb.Cleanup(func() { _ = out.Close() })
	return &bytesWritten{
		out: out,
	}
}

type bytesWritten struct {
	out          *os.File
	bytesWritten int64
}

func (b *bytesWritten) Write(data []byte) (int, error) {
	b.bytesWritten += int64(len(data))
	return b.out.Write(data)
}

func printf(item logMessage) {
	if item.isError {
		klog.Errorf("%s: %v %s", item.msg, item.err, item.kvs)
	} else {
		klog.Infof("%s: %v", item.msg, item.kvs)
	}
}

func prints(logger klog.Logger, item logMessage) {
	if item.isError {
		logger.Error(item.err, item.msg, item.kvs...) // nolint: logcheck
	} else {
		logger.Info(item.msg, item.kvs...) // nolint: logcheck
	}
}

func printLogger(item logMessage) {
	prints(klog.Background(), item)
}
