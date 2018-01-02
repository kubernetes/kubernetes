package server

import (
	"fmt"
	"net"
	"net/http"
	"os"
	"path"
	
	"go.pedge.io/dlog"

	"github.com/gorilla/mux"
)

// Route is a specification and  handler for a REST endpoint.
type Route struct {
	verb string
	path string
	fn   func(http.ResponseWriter, *http.Request)
}

func (r *Route) GetVerb() string {
	return r.verb
}

func (r *Route) GetPath() string {
	return r.path
}

func (r *Route) GetFn() func(http.ResponseWriter, *http.Request) {
	return r.fn
}

// StartGraphAPI starts a REST server to receive GraphDriver commands from
// the Linux container engine.
func StartGraphAPI(name string, restBase string) error {
	graphPlugin := newGraphPlugin(name)
	if err := startServer(name, restBase, 0, graphPlugin.Routes()); err != nil {
		return err
	}

	return nil
}

// StartPluginAPI starts a REST server to receive volume API commands from the
// Linux container engine and volume management commands from the CLI/UX.
func StartPluginAPI(
	name string,
	mgmtBase string,
	pluginBase string,
	mgmtPort uint16,
	pluginPort uint16,
) error {
	if err := StartVolumeMgmtAPI(
		name,
		mgmtBase,
		mgmtPort,
	); err != nil {
		return err
	}
	if err := StartVolumePluginAPI(
		name,
		pluginBase,
		pluginPort,
	); err != nil {
		return err
	}
	return nil
}

// StartVolumeMgmtAPI starts a REST server to receive volume management API commands
func StartVolumeMgmtAPI(
	name string,
	mgmtBase string,
	mgmtPort uint16,
) error {
	volMgmtApi := newVolumeAPI(name)
	if err := startServer(
		name,
		mgmtBase,
		mgmtPort,
		volMgmtApi.Routes(),
	); err != nil {
		return err
	}
	return nil
}

func GetVolumeAPIRoutes(name string) []*Route {
	volMgmtApi := newVolumeAPI(name)
	return volMgmtApi.Routes()
}

// StartVolumePluginAPI starts a REST server to receive volume API commands
// from the linux container  engine
func StartVolumePluginAPI(
	name string,
	pluginBase string,
	pluginPort uint16,
) error {

	volPluginApi := newVolumePlugin(name)
	if err := startServer(
		name,
		pluginBase,
		pluginPort,
		volPluginApi.Routes(),
	); err != nil {
		return err
	}
	return nil
}

// StartClusterAPI starts a REST server to receive driver configuration commands
// from the CLI/UX to control the OSD cluster.
func StartClusterAPI(clusterApiBase string, clusterPort uint16) error {
	clusterApi := newClusterAPI()
	if err := startServer("osd", clusterApiBase, clusterPort, clusterApi.Routes()); err != nil {
		return err
	}

	return nil
}

func GetClusterAPIRoutes() []*Route {
	clusterApi := newClusterAPI()
	return clusterApi.Routes()
}

func startServer(name string, sockBase string, port uint16, routes []*Route) error {
	var (
		listener net.Listener
		err      error
	)
	router := mux.NewRouter()
	router.NotFoundHandler = http.HandlerFunc(notFound)
	for _, v := range routes {
		router.Methods(v.verb).Path(v.path).HandlerFunc(v.fn)
	}
	socket := path.Join(sockBase, name+".sock")
	os.Remove(socket)
	os.MkdirAll(path.Dir(socket), 0755)

	dlog.Printf("Starting REST service on socket : %+v", socket)
	listener, err = net.Listen("unix", socket)
	if err != nil {
		dlog.Warnln("Cannot listen on UNIX socket: ", err)
		return err
	}
	go http.Serve(listener, router)
	if port != 0 {
		dlog.Printf("Starting REST service on port : %v", port)
		go http.ListenAndServe(fmt.Sprintf(":%d", port), router)
	}
	return nil
}

type restServer interface {
	Routes() []*Route
	String() string
	logRequest(request string, id string) dlog.Logger
	sendError(request string, id string, w http.ResponseWriter, msg string, code int)
}

type restBase struct {
	restServer
	version string
	name    string
}

func (rest *restBase) logRequest(request string, id string) dlog.Logger {
	return dlog.WithFields(map[string]interface{}{
		"Driver":  rest.name,
		"Request": request,
		"ID":      id,
	})
}
func (rest *restBase) sendError(request string, id string, w http.ResponseWriter, msg string, code int) {
	rest.logRequest(request, id).Warnln(code, " ", msg)
	http.Error(w, msg, code)
}

func notFound(w http.ResponseWriter, r *http.Request) {
	dlog.Warnf("Not found: %+v ", r.URL)
	http.NotFound(w, r)
}
