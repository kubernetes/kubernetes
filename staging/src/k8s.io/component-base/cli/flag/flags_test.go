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

package flag

import (
	"bytes"
	"testing"

	"github.com/go-logr/logr"
	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
	"k8s.io/klog/v2"
)

type FakeLogger struct {
	logr.Logger
	infoBuffer  bytes.Buffer
	errorBuffer bytes.Buffer
}

func (logger *FakeLogger) Init(info logr.RuntimeInfo) {}
func (logger *FakeLogger) Enabled(lvl int) bool       { return true }
func (logger *FakeLogger) Info(lvl int, msg string, keysAndValues ...interface{}) {
	logger.infoBuffer.WriteString(msg)
}
func (logger *FakeLogger) Error(err error, msg string, keysAndValues ...interface{}) {
	logger.errorBuffer.WriteString(msg)
}
func (logger *FakeLogger) WithValues(keysAndValues ...interface{}) logr.LogSink { return logger }
func (logger *FakeLogger) WithName(name string) logr.LogSink                    { return logger }

var _ logr.LogSink = &FakeLogger{}

func TestWordSepNormalizeFunc(t *testing.T) {
	cases := []struct {
		flagName         string
		expectedFlagName string
	}{
		{
			flagName:         "foo",
			expectedFlagName: "foo",
		},
		{
			flagName:         "foo-bar",
			expectedFlagName: "foo-bar",
		},
		{
			flagName:         "foo_bar",
			expectedFlagName: "foo-bar",
		},
	}
	for _, tc := range cases {
		t.Run(tc.flagName, func(t *testing.T) {
			fakeLogger := &FakeLogger{}
			klog.SetLogger(logr.New(fakeLogger))
			result := WordSepNormalizeFunc(nil, tc.flagName)
			assert.Equal(t, pflag.NormalizedName(tc.expectedFlagName), result)
			assert.Equal(t, "", fakeLogger.infoBuffer.String())
			assert.Equal(t, "", fakeLogger.errorBuffer.String())
		})
	}
}

func TestWarnWordSepNormalizeFunc(t *testing.T) {
	cases := []struct {
		flagName         string
		expectedFlagName string
		expectedWarning  string
	}{
		{
			flagName:         "foo",
			expectedFlagName: "foo",
			expectedWarning:  "",
		},
		{
			flagName:         "foo-bar",
			expectedFlagName: "foo-bar",
			expectedWarning:  "",
		},
		{
			flagName:         "foo_bar",
			expectedFlagName: "foo-bar",
			expectedWarning:  "using an underscore in a flag name is not supported. foo_bar has been converted to foo-bar.",
		},
	}
	for _, tc := range cases {
		t.Run(tc.flagName, func(t *testing.T) {
			fakeLogger := &FakeLogger{}
			klog.SetLogger(logr.New(fakeLogger))
			result := WarnWordSepNormalizeFunc(nil, tc.flagName)
			assert.Equal(t, pflag.NormalizedName(tc.expectedFlagName), result)
			assert.Equal(t, tc.expectedWarning, fakeLogger.infoBuffer.String())
			assert.Equal(t, "", fakeLogger.errorBuffer.String())
		})
	}
}
