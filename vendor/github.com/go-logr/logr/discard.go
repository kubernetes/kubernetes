/*
Copyright 2020 The logr Authors.

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

package logr

// Discard returns a valid Logger that discards all messages logged to it.
// It can be used whenever the caller is not interested in the logs.
func Discard() Logger {
	return DiscardLogger{}
}

// DiscardLogger is a Logger that discards all messages.
type DiscardLogger struct{}

func (l DiscardLogger) Enabled() bool {
	return false
}

func (l DiscardLogger) Info(msg string, keysAndValues ...interface{}) {
}

func (l DiscardLogger) Error(err error, msg string, keysAndValues ...interface{}) {
}

func (l DiscardLogger) V(level int) Logger {
	return l
}

func (l DiscardLogger) WithValues(keysAndValues ...interface{}) Logger {
	return l
}

func (l DiscardLogger) WithName(name string) Logger {
	return l
}

// Verify that it actually implements the interface
var _ Logger = DiscardLogger{}
