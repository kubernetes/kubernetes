/*
Copyright 2022 The Kubernetes Authors.

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

package tracelog

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestTraceLog(t *testing.T) {
	trace := NewTraceLog("test", 30*time.Second)
	trace.LogWithFrequency("aaa-111", "test 111")
	trace.LogWithFrequency("aaa-222", "test 222")
	require.Equal(t, len(trace.traceLogModules), 2)
	require.NotNil(t, trace.getLastLogTime("aaa-111"))
	trace.CleanLogModule("aaa-111")
	trace.CleanLogModule("aaa-222")
	require.Equal(t, len(trace.traceLogModules), 0)
}
