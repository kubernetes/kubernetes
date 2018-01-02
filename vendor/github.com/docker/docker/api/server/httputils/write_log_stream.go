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
	"github.com/docker/docker/pkg/jsonlog"
	"github.com/docker/docker/pkg/stdcopy"
)

// WriteLogStream writes an encoded byte stream of log messages from the
// messages channel, multiplexing them with a stdcopy.Writer if mux is true
func WriteLogStream(ctx context.Context, w io.Writer, msgs <-chan *backend.LogMessage, config *types.ContainerLogsOptions, mux bool) {
	wf := ioutils.NewWriteFlusher(w)
	defer wf.Close()

	wf.Flush()

	// this might seem like doing below is clear:
	//   var outStream io.Writer = wf
	// however, this GREATLY DISPLEASES golint, and if you do that, it will
	// fail CI. we need outstream to be type writer because if we mux streams,
	// we will need to reassign all of the streams to be stdwriters, which only
	// conforms to the io.Writer interface.
	var outStream io.Writer
	outStream = wf
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
			// TODO(dperny) the format is defined in
			// daemon/logger/logger.go as logger.TimeFormat. importing
			// logger is verboten (not part of backend) so idk if just
			// importing the same thing from jsonlog is good enough
			logLine = append([]byte(msg.Timestamp.Format(jsonlog.RFC3339NanoFixed)+" "), logLine...)
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
