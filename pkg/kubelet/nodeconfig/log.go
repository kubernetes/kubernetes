/*
Copyright 2017 The Kubernetes Authors.

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

package nodeconfig

import (
	"fmt"
	"github.com/golang/glog"
)

// This file contains shims for inserting nodeconfigLogFmt, but still reporting the call site of the message.

const nodeconfigLogFmt = "nodeconfig controller: %s"

func fatalf(format string, args ...interface{}) {
	var s string
	if len(args) > 0 {
		s = fmt.Sprintf(format, args...)
	} else {
		s = format
	}
	msg := fmt.Sprintf(nodeconfigLogFmt, s)
	glog.ErrorDepth(1, msg)
	panic(fmt.Errorf(msg))
}

func errorf(format string, args ...interface{}) {
	var s string
	if len(args) > 0 {
		s = fmt.Sprintf(format, args...)
	} else {
		s = format
	}
	glog.ErrorDepth(1, fmt.Sprintf(nodeconfigLogFmt, s))
}

func infof(format string, args ...interface{}) {
	var s string
	if len(args) > 0 {
		s = fmt.Sprintf(format, args...)
	} else {
		s = format
	}
	glog.InfoDepth(1, fmt.Sprintf(nodeconfigLogFmt, s))
}
