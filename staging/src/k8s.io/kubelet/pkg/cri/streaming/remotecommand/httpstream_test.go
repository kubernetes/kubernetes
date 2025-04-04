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
