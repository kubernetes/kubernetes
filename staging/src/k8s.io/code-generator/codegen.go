/*
Copyright 2023 The Kubernetes Authors.

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

package main

import (
	"flag"
	"k8s.io/code-generator/internal/codegen"
	"k8s.io/code-generator/internal/codegen/execution"
	"k8s.io/klog/v2"
)

var options []execution.Option

func main() {
	klog.InitFlags(nil)
	_ = flag.Set("logtostderr", "true")
	codegen.Run(options...)
}

// RunMain is a wrapper for main() to allow for testing.
func RunMain(opts ...execution.Option) {
	old := options
	defer func() { options = old }()
	options = append(make([]execution.Option, 0, len(opts)), opts...)
	main()
}
