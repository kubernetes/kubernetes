/*

The remote package provides the pieces to allow Ginkgo test suites to report to remote listeners.
This is used, primarily, to enable streaming parallel test output but has, in principal, broader applications (e.g. streaming test output to a browser).

*/

package parallel_support

import (
	"io"
	"net"
	"net/http"
	"net/rpc"

	"github.com/onsi/ginkgo/v2/reporters"
)

/*
RPCServer spins up on an automatically selected port and listens for communication from the forwarding reporter.
It then forwards that communication to attached reporters.
*/
type RPCServer struct {
	listener net.Listener
	handler  *ServerHandler
}

// Create a new server, automatically selecting a port
func newRPCServer(parallelTotal int, reporter reporters.Reporter) (*RPCServer, error) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return nil, err
	}
	return &RPCServer{
		listener: listener,
		handler:  newServerHandler(parallelTotal, reporter),
	}, nil
}

// Start the server.  You don't need to `go s.Start()`, just `s.Start()`
func (server *RPCServer) Start() {
	rpcServer := rpc.NewServer()
	rpcServer.RegisterName("Server", server.handler) //register the handler's methods as the server

	httpServer := &http.Server{}
	httpServer.Handler = rpcServer

	go httpServer.Serve(server.listener)
}

// Stop the server
func (server *RPCServer) Close() {
	server.listener.Close()
}

// The address the server can be reached it.  Pass this into the `ForwardingReporter`.
func (server *RPCServer) Address() string {
	return server.listener.Addr().String()
}

func (server *RPCServer) GetSuiteDone() chan any {
	return server.handler.done
}

func (server *RPCServer) GetOutputDestination() io.Writer {
	return server.handler.outputDestination
}

func (server *RPCServer) SetOutputDestination(w io.Writer) {
	server.handler.outputDestination = w
}

func (server *RPCServer) RegisterAlive(node int, alive func() bool) {
	server.handler.registerAlive(node, alive)
}
