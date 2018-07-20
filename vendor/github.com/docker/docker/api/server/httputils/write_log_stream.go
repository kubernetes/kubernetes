package httputils

import (
	"fmt"
	"io"
	"net/url"
	"sort"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/docker/docker/pkg/stdcopy"
)

// WriteLogStream writes an encoded byte stream of log messages from the
// messages channel, multiplexing them with a stdcopy.Writer if mux is true
func WriteLogStream(_ context.Context, w io.Writer, msgs <-chan *backend.LogMessage, config *types.ContainerLogsOptions, mux bool) {
	wf := ioutils.NewWriteFlusher(w)
	defer wf.Close()

	wf.Flush()

	outStream := io.Writer(wf)
	errStream := outStream
	sysErrStream := errStream
	if mux {
		sysErrStream = stdcopy.NewStdWriter(outStream, stdcopy.Systemerr)
		errStream = stdcopy.NewStdWriter(outStream, stdcopy.Stderr)
		outStream = stdcopy.NewStdWriter(outStream, stdcopy.Stdout)
	}

	for {
		msg, ok := <-msgs
		if !ok {
			return
		}
		// check if the message contains an error. if so, write that error
		// and exit
		if msg.Err != nil {
			fmt.Fprintf(sysErrStream, "Error grabbing logs: %v\n", msg.Err)
			continue
		}
		logLine := msg.Line
		if config.Details {
			logLine = append(attrsByteSlice(msg.Attrs), ' ')
			logLine = append(logLine, msg.Line...)
		}
		if config.Timestamps {
			logLine = append([]byte(msg.Timestamp.Format(jsonmessage.RFC3339NanoFixed)+" "), logLine...)
		}
		if msg.Source == "stdout" && config.ShowStdout {
			outStream.Write(logLine)
		}
		if msg.Source == "stderr" && config.ShowStderr {
			errStream.Write(logLine)
		}
	}
}

type byKey []backend.LogAttr

func (b byKey) Len() int           { return len(b) }
func (b byKey) Less(i, j int) bool { return b[i].Key < b[j].Key }
func (b byKey) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }

func attrsByteSlice(a []backend.LogAttr) []byte {
	// Note this sorts "a" in-place. That is fine here - nothing else is
	// going to use Attrs or care about the order.
	sort.Sort(byKey(a))

	var ret []byte
	for i, pair := range a {
		k, v := url.QueryEscape(pair.Key), url.QueryEscape(pair.Value)
		ret = append(ret, []byte(k)...)
		ret = append(ret, '=')
		ret = append(ret, []byte(v)...)
		if i != len(a)-1 {
			ret = append(ret, ',')
		}
	}
	return ret
}
