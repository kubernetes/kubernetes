/*
Copyright 2018 The Kubernetes Authors.

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

package etcd3

import (
	"fmt"

	"google.golang.org/grpc/grpclog"
	"k8s.io/klog/v2"
)

func init() {
	grpclog.SetLoggerV2(klogWrapper{})
}

type klogWrapper struct{}

const klogWrapperDepth = 4

func (klogWrapper) Info(args ...interface{}) {
	if klogV := klog.V(5); klogV.Enabled() {
		klogV.InfoSDepth(klogWrapperDepth, fmt.Sprint(args...))
	}
}

func (klogWrapper) Infoln(args ...interface{}) {
	if klogV := klog.V(5); klogV.Enabled() {
		klogV.InfoSDepth(klogWrapperDepth, fmt.Sprintln(args...))
	}
}

func (klogWrapper) Infof(format string, args ...interface{}) {
	if klogV := klog.V(5); klogV.Enabled() {
		klog.V(5).InfoSDepth(klogWrapperDepth, fmt.Sprintf(format, args...))
	}
}

func (klogWrapper) Warning(args ...interface{}) {
	klog.WarningDepth(klogWrapperDepth, args...)
}

func (klogWrapper) Warningln(args ...interface{}) {
	klog.WarningDepth(klogWrapperDepth, fmt.Sprintln(args...))
}

func (klogWrapper) Warningf(format string, args ...interface{}) {
	klog.WarningDepth(klogWrapperDepth, fmt.Sprintf(format, args...))
}

func (klogWrapper) Error(args ...interface{}) {
	klog.ErrorDepth(klogWrapperDepth, args...)
}

func (klogWrapper) Errorln(args ...interface{}) {
	klog.ErrorDepth(klogWrapperDepth, fmt.Sprintln(args...))
}

func (klogWrapper) Errorf(format string, args ...interface{}) {
	klog.ErrorDepth(klogWrapperDepth, fmt.Sprintf(format, args...))
}

func (klogWrapper) Fatal(args ...interface{}) {
	klog.FatalDepth(klogWrapperDepth, args...)
}

func (klogWrapper) Fatalln(args ...interface{}) {
	klog.FatalDepth(klogWrapperDepth, fmt.Sprintln(args...))
}

func (klogWrapper) Fatalf(format string, args ...interface{}) {
	klog.FatalDepth(klogWrapperDepth, fmt.Sprintf(format, args...))
}

func (klogWrapper) V(l int) bool {
	return bool(klog.V(klog.Level(l)).Enabled())
}
