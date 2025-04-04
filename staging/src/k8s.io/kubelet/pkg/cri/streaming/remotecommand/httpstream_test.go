/*
Copyright 2025 The Kubernetes Authors.

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

package remotecommand

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
	"go.uber.org/goleak"

	"k8s.io/client-go/tools/remotecommand"
)

func TestHandleResizeEvents(t *testing.T) {
	var testTerminalSize remotecommand.TerminalSize
	rawTerminalSize, err := json.Marshal(&testTerminalSize)
	require.NoError(t, err)

	testCases := []struct {
		name             string
		resizeStreamData []byte
		cancelContext    bool
		readFromChannel  bool
	}{
		{
			name:             "data attempted to be sent on the channel; channel not read; context canceled",
			resizeStreamData: rawTerminalSize,
			cancelContext:    true,
		},
		{
			name:             "data attempted to be sent on the channel; channel read; context not canceled",
			resizeStreamData: rawTerminalSize,
			readFromChannel:  true,
		},
		{
			name:          "no data attempted to be sent on the channel; context canceled",
			cancelContext: true,
		},
		{
			name: "no data attempted to be sent on the channel; context not canceled",
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.TODO())
			connCtx := connectionContext{
				resizeStream: io.NopCloser(bytes.NewReader(testCase.resizeStreamData)),
				resizeChan:   make(chan remotecommand.TerminalSize),
			}

			go handleResizeEvents(ctx, connCtx.resizeStream, connCtx.resizeChan)
			if testCase.readFromChannel {
				gotTerminalSize := <-connCtx.resizeChan
				require.Equal(t, gotTerminalSize, testTerminalSize)
			}
			if testCase.cancelContext {
				cancel()
			}

			goleak.VerifyNone(t)
			cancel()
		})
	}
}
