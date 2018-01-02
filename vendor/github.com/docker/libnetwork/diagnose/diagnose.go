package diagnose

import (
	"fmt"
	"net"
	"net/http"
	"sync"

	"github.com/sirupsen/logrus"
)

// HTTPHandlerFunc TODO
type HTTPHandlerFunc func(interface{}, http.ResponseWriter, *http.Request)

type httpHandlerCustom struct {
	ctx interface{}
	F   func(interface{}, http.ResponseWriter, *http.Request)
}

// ServeHTTP TODO
func (h httpHandlerCustom) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.F(h.ctx, w, r)
}

var diagPaths2Func = map[string]HTTPHandlerFunc{
	"/":      notImplemented,
	"/help":  help,
	"/ready": ready,
}

// Server when the debug is enabled exposes a
// This data structure is protected by the Agent mutex so does not require and additional mutex here
type Server struct {
	sk                net.Listener
	port              int
	mux               *http.ServeMux
	registeredHanders []string
	sync.Mutex
}

// Init TODO
func (n *Server) Init() {
	n.mux = http.NewServeMux()

	// Register local handlers
	n.RegisterHandler(n, diagPaths2Func)
}

// RegisterHandler TODO
func (n *Server) RegisterHandler(ctx interface{}, hdlrs map[string]HTTPHandlerFunc) {
	n.Lock()
	defer n.Unlock()
	for path, fun := range hdlrs {
		n.mux.Handle(path, httpHandlerCustom{ctx, fun})
		n.registeredHanders = append(n.registeredHanders, path)
	}
}

// EnableDebug opens a TCP socket to debug the passed network DB
func (n *Server) EnableDebug(ip string, port int) {
	n.Lock()
	defer n.Unlock()

	n.port = port
	logrus.SetLevel(logrus.DebugLevel)

	if n.sk != nil {
		logrus.Infof("The server is already up and running")
		return
	}

	logrus.Infof("Starting the server listening on %d for commands", port)

	// // Create the socket
	// var err error
	// n.sk, err = net.Listen("tcp", listeningAddr)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	//
	// go func() {
	// 	http.Serve(n.sk, n.mux)
	// }()
	http.ListenAndServe(":8000", n.mux)
}

// DisableDebug stop the dubug and closes the tcp socket
func (n *Server) DisableDebug() {
	n.Lock()
	defer n.Unlock()
	n.sk.Close()
	n.sk = nil
}

// IsDebugEnable returns true when the debug is enabled
func (n *Server) IsDebugEnable() bool {
	n.Lock()
	defer n.Unlock()
	return n.sk != nil
}

func notImplemented(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "URL path: %s no method implemented check /help\n", r.URL.Path)
}

func help(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	n, ok := ctx.(*Server)
	if ok {
		for _, path := range n.registeredHanders {
			fmt.Fprintf(w, "%s\n", path)
		}
	}
}

func ready(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	fmt.Fprintf(w, "OK\n")
}

// DebugHTTPForm TODO
func DebugHTTPForm(r *http.Request) {
	r.ParseForm()
	for k, v := range r.Form {
		logrus.Debugf("Form[%q] = %q\n", k, v)
	}
}

// HTTPReplyError TODO
func HTTPReplyError(w http.ResponseWriter, message, usage string) {
	fmt.Fprintf(w, "%s\n", message)
	if usage != "" {
		fmt.Fprintf(w, "Usage: %s\n", usage)
	}
}
