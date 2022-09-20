/*

The remote package provides the pieces to allow Ginkgo test suites to report to remote listeners.
This is used, primarily, to enable streaming parallel test output but has, in principal, broader applications (e.g. streaming test output to a browser).

*/

package parallel_support

import (
	"encoding/json"
	"io"
	"net"
	"net/http"

	"github.com/onsi/ginkgo/v2/reporters"
	"github.com/onsi/ginkgo/v2/types"
)

/*
httpServer spins up on an automatically selected port and listens for communication from the forwarding reporter.
It then forwards that communication to attached reporters.
*/
type httpServer struct {
	listener net.Listener
	handler  *ServerHandler
}

//Create a new server, automatically selecting a port
func newHttpServer(parallelTotal int, reporter reporters.Reporter) (*httpServer, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	return &httpServer{
		listener: listener,
		handler:  newServerHandler(parallelTotal, reporter),
	}, nil
}

//Start the server.  You don't need to `go s.Start()`, just `s.Start()`
func (server *httpServer) Start() {
	httpServer := &http.Server{}
	mux := http.NewServeMux()
	httpServer.Handler = mux

	//streaming endpoints
	mux.HandleFunc("/suite-will-begin", server.specSuiteWillBegin)
	mux.HandleFunc("/did-run", server.didRun)
	mux.HandleFunc("/suite-did-end", server.specSuiteDidEnd)
	mux.HandleFunc("/emit-output", server.emitOutput)
	mux.HandleFunc("/progress-report", server.emitProgressReport)

	//synchronization endpoints
	mux.HandleFunc("/before-suite-completed", server.handleBeforeSuiteCompleted)
	mux.HandleFunc("/before-suite-state", server.handleBeforeSuiteState)
	mux.HandleFunc("/have-nonprimary-procs-finished", server.handleHaveNonprimaryProcsFinished)
	mux.HandleFunc("/aggregated-nonprimary-procs-report", server.handleAggregatedNonprimaryProcsReport)
	mux.HandleFunc("/counter", server.handleCounter)
	mux.HandleFunc("/up", server.handleUp)
	mux.HandleFunc("/abort", server.handleAbort)

	go httpServer.Serve(server.listener)
}

//Stop the server
func (server *httpServer) Close() {
	server.listener.Close()
}

//The address the server can be reached it.  Pass this into the `ForwardingReporter`.
func (server *httpServer) Address() string {
	return "http://" + server.listener.Addr().String()
}

func (server *httpServer) GetSuiteDone() chan interface{} {
	return server.handler.done
}

func (server *httpServer) GetOutputDestination() io.Writer {
	return server.handler.outputDestination
}

func (server *httpServer) SetOutputDestination(w io.Writer) {
	server.handler.outputDestination = w
}

func (server *httpServer) RegisterAlive(node int, alive func() bool) {
	server.handler.registerAlive(node, alive)
}

//
// Streaming Endpoints
//

//The server will forward all received messages to Ginkgo reporters registered with `RegisterReporters`
func (server *httpServer) decode(writer http.ResponseWriter, request *http.Request, object interface{}) bool {
	defer request.Body.Close()
	if json.NewDecoder(request.Body).Decode(object) != nil {
		writer.WriteHeader(http.StatusBadRequest)
		return false
	}
	return true
}

func (server *httpServer) handleError(err error, writer http.ResponseWriter) bool {
	if err == nil {
		return false
	}
	switch err {
	case ErrorEarly:
		writer.WriteHeader(http.StatusTooEarly)
	case ErrorGone:
		writer.WriteHeader(http.StatusGone)
	case ErrorFailed:
		writer.WriteHeader(http.StatusFailedDependency)
	default:
		writer.WriteHeader(http.StatusInternalServerError)
	}
	return true
}

func (server *httpServer) specSuiteWillBegin(writer http.ResponseWriter, request *http.Request) {
	var report types.Report
	if !server.decode(writer, request, &report) {
		return
	}

	server.handleError(server.handler.SpecSuiteWillBegin(report, voidReceiver), writer)
}

func (server *httpServer) didRun(writer http.ResponseWriter, request *http.Request) {
	var report types.SpecReport
	if !server.decode(writer, request, &report) {
		return
	}

	server.handleError(server.handler.DidRun(report, voidReceiver), writer)
}

func (server *httpServer) specSuiteDidEnd(writer http.ResponseWriter, request *http.Request) {
	var report types.Report
	if !server.decode(writer, request, &report) {
		return
	}
	server.handleError(server.handler.SpecSuiteDidEnd(report, voidReceiver), writer)
}

func (server *httpServer) emitOutput(writer http.ResponseWriter, request *http.Request) {
	output, err := io.ReadAll(request.Body)
	if err != nil {
		writer.WriteHeader(http.StatusInternalServerError)
		return
	}
	var n int
	server.handleError(server.handler.EmitOutput(output, &n), writer)
}

func (server *httpServer) emitProgressReport(writer http.ResponseWriter, request *http.Request) {
	var report types.ProgressReport
	if !server.decode(writer, request, &report) {
		return
	}
	server.handleError(server.handler.EmitProgressReport(report, voidReceiver), writer)
}

func (server *httpServer) handleBeforeSuiteCompleted(writer http.ResponseWriter, request *http.Request) {
	var beforeSuiteState BeforeSuiteState
	if !server.decode(writer, request, &beforeSuiteState) {
		return
	}

	server.handleError(server.handler.BeforeSuiteCompleted(beforeSuiteState, voidReceiver), writer)
}

func (server *httpServer) handleBeforeSuiteState(writer http.ResponseWriter, request *http.Request) {
	var beforeSuiteState BeforeSuiteState
	if server.handleError(server.handler.BeforeSuiteState(voidSender, &beforeSuiteState), writer) {
		return
	}
	json.NewEncoder(writer).Encode(beforeSuiteState)
}

func (server *httpServer) handleHaveNonprimaryProcsFinished(writer http.ResponseWriter, request *http.Request) {
	if server.handleError(server.handler.HaveNonprimaryProcsFinished(voidSender, voidReceiver), writer) {
		return
	}
	writer.WriteHeader(http.StatusOK)
}

func (server *httpServer) handleAggregatedNonprimaryProcsReport(writer http.ResponseWriter, request *http.Request) {
	var aggregatedReport types.Report
	if server.handleError(server.handler.AggregatedNonprimaryProcsReport(voidSender, &aggregatedReport), writer) {
		return
	}
	json.NewEncoder(writer).Encode(aggregatedReport)
}

func (server *httpServer) handleCounter(writer http.ResponseWriter, request *http.Request) {
	var n int
	if server.handleError(server.handler.Counter(voidSender, &n), writer) {
		return
	}
	json.NewEncoder(writer).Encode(ParallelIndexCounter{Index: n})
}

func (server *httpServer) handleUp(writer http.ResponseWriter, request *http.Request) {
	writer.WriteHeader(http.StatusOK)
}

func (server *httpServer) handleAbort(writer http.ResponseWriter, request *http.Request) {
	if request.Method == "GET" {
		var shouldAbort bool
		server.handler.ShouldAbort(voidSender, &shouldAbort)
		if shouldAbort {
			writer.WriteHeader(http.StatusGone)
		} else {
			writer.WriteHeader(http.StatusOK)
		}
	} else {
		server.handler.Abort(voidSender, voidReceiver)
	}
}
