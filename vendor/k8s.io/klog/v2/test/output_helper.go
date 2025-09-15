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

package test

import (
	"github.com/go-logr/logr"

	"k8s.io/klog/v2"
)

func loggerHelper(logger logr.Logger, msg string, kv []interface{}) {
	logger = logger.WithCallDepth(1)
	logger.Info(msg, kv...)
}

func klogHelper(level klog.Level, msg string, kv []interface{}) {
	klog.V(level).InfoSDepth(1, msg, kv...)
}
