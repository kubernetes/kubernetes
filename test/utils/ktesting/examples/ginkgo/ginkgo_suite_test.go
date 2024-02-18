//go:build example
// +build example

/*
Copyright 2024 The Kubernetes Authors.

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

package ginkgo_test

import (
	"testing"

	"github.com/onsi/ginkgo/v2"
	ginkgotypes "github.com/onsi/ginkgo/v2/types"
	"github.com/onsi/gomega"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/textlogger"
)

// Initialize klog to log through Ginkgo. The textlogger supports
// delegating stack unwinding to Ginkgo, therefore it gets used here.
func init() {
	config := textlogger.NewConfig(
		textlogger.Output(ginkgo.GinkgoWriter),
		textlogger.Backtrace(unwind),
	)
	logger := textlogger.NewLogger(config)
	writer, _ := logger.GetSink().(textlogger.KlogBufferWriter)
	opts := []klog.LoggerOption{
		klog.ContextualLogger(true),
		klog.WriteKlogBuffer(writer.WriteKlogBuffer),
	}
	klog.SetLoggerWithOptions(logger, opts...)

	// No command line flags, ktesting already added them
	// for klog. This can be solved, but isn't needed
	// for this simple example.
}

func unwind(skip int) (string, int) {
	location := ginkgotypes.NewCodeLocation(skip + 1)
	return location.FileName, location.LineNumber
}

func TestGinkgo(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t, "Ginkgo Suite")
}
