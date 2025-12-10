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

package internal

import "k8s.io/klog/v2"

func Log(logger *klog.Logger, level int, msg string, keyAndValues ...any) {
	if logger == nil {
		return
	}
	logger.V(level).Info(msg, keyAndValues...)
}

func LogErr(logger *klog.Logger, err error, msg string, keyAndValues ...any) {
	if logger == nil {
		return
	}
	logger.Error(err, msg, keyAndValues...)
}
