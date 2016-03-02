package main

import (
	"github.com/emicklei/go-restful"
	"io"
	"log"
	"os"
	"runtime/pprof"
)

// ProfilingService is a WebService that can start/stop a CPU profile and write results to a file
// 	GET /{rootPath}/start will activate CPU profiling
//	GET /{rootPath}/stop will stop profiling
//
// NewProfileService("/profiler", "ace.prof").AddWebServiceTo(restful.DefaultContainer)
//
type ProfilingService struct {
	rootPath   string   // the base (root) of the service, e.g. /profiler
	cpuprofile string   // the output filename to write profile results, e.g. myservice.prof
	cpufile    *os.File // if not nil, then profiling is active
}

func NewProfileService(rootPath string, outputFilename string) *ProfilingService {
	ps := new(ProfilingService)
	ps.rootPath = rootPath
	ps.cpuprofile = outputFilename
	return ps
}

// Add this ProfileService to a restful Container
func (p ProfilingService) AddWebServiceTo(container *restful.Container) {
	ws := new(restful.WebService)
	ws.Path(p.rootPath).Consumes("*/*").Produces(restful.MIME_JSON)
	ws.Route(ws.GET("/start").To(p.startProfiler))
	ws.Route(ws.GET("/stop").To(p.stopProfiler))
	container.Add(ws)
}

func (p *ProfilingService) startProfiler(req *restful.Request, resp *restful.Response) {
	if p.cpufile != nil {
		io.WriteString(resp.ResponseWriter, "[restful] CPU profiling already running")
		return // error?
	}
	cpufile, err := os.Create(p.cpuprofile)
	if err != nil {
		log.Fatal(err)
	}
	// remember for close
	p.cpufile = cpufile
	pprof.StartCPUProfile(cpufile)
	io.WriteString(resp.ResponseWriter, "[restful] CPU profiling started, writing on:"+p.cpuprofile)
}

func (p *ProfilingService) stopProfiler(req *restful.Request, resp *restful.Response) {
	if p.cpufile == nil {
		io.WriteString(resp.ResponseWriter, "[restful] CPU profiling not active")
		return // error?
	}
	pprof.StopCPUProfile()
	p.cpufile.Close()
	p.cpufile = nil
	io.WriteString(resp.ResponseWriter, "[restful] CPU profiling stopped, closing:"+p.cpuprofile)
}

func main() {} // exists for example compilation only
