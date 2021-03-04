/*
Copyright 2020 The Kubernetes Authors.

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

package sanitization

import (
	"fmt"

	"k8s.io/component-base/logs/datapol"
)

const (
	datapolMsgFmt = "Log message has been redacted. Log argument #%d contains: %v"
	datapolMsg    = "Log message has been redacted."
)

// SanitizingFilter implements the LogFilter interface from klog with a set of functions that inspects the arguments with the datapol library
type SanitizingFilter struct{}

// Filter is the filter function for the non-formatting logging functions of klog.
func (sf *SanitizingFilter) Filter(args []interface{}) []interface{} {
	for i, v := range args {
		types := datapol.Verify(v)
		if len(types) > 0 {
			return []interface{}{fmt.Sprintf(datapolMsgFmt, i, types)}
		}
	}
	return args
}

// FilterF is the filter function for the formatting logging functions of klog
func (sf *SanitizingFilter) FilterF(fmt string, args []interface{}) (string, []interface{}) {
	for i, v := range args {
		types := datapol.Verify(v)
		if len(types) > 0 {
			return datapolMsgFmt, []interface{}{i, types}
		}
	}
	return fmt, args

}

// FilterS is the filter for the structured logging functions of klog.
func (sf *SanitizingFilter) FilterS(msg string, keysAndValues []interface{}) (string, []interface{}) {
	for i, v := range keysAndValues {
		types := datapol.Verify(v)
		if len(types) > 0 {
			if i%2 == 0 {
				return datapolMsg, []interface{}{"key_index", i, "types", types}
			}
			// since we scanned linearly we can safely log the key.
			return datapolMsg, []interface{}{"key", keysAndValues[i-1], "types", types}
		}
	}
	return msg, keysAndValues
}
