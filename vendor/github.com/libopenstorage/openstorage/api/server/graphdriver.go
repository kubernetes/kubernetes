package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/pkg/archive"
	"github.com/libopenstorage/openstorage/config"
	"github.com/libopenstorage/openstorage/graph"
)

const (
	// GraphDriver is the string returned in the handshake protocol.
	GraphDriver = "GraphDriver"
)

// Implementation of the Docker GraphgraphDriver plugin specification.
type graphDriver struct {
	restBase
	gd graphdriver.Driver
}

type graphRequest struct {
	ID         string `json:",omitempty"`
	Parent     string `json:",omitempty"`
	MountLabel string `json:",omitempty"`
}

type graphResponse struct {
	Err      error             `json:",omitempty"`
	Dir      string            `json:",omitempty"`
	Exists   bool              `json:",omitempty"`
	Status   [][2]string       `json:",omitempty"`
	Metadata map[string]string `json:",omitempty"`
	Changes  []archive.Change  `json:",omitempty"`
	Size     int64             `json:",omitempty"`
}

func newGraphPlugin(name string) restServer {
	return &graphDriver{restBase{name: name, version: "0.3"}, nil}
}

func (d *graphDriver) String() string {
	return d.name
}

func graphDriverPath(method string) string {
	return fmt.Sprintf("/%s.%s", GraphDriver, method)
}

func (d *graphDriver) Routes() []*Route {
	return []*Route{
		{verb: "POST", path: graphDriverPath("Init"), fn: d.init},
		{verb: "POST", path: graphDriverPath("Create"), fn: d.create},
		{verb: "POST", path: graphDriverPath("Remove"), fn: d.remove},
		{verb: "POST", path: graphDriverPath("Get"), fn: d.get},
		{verb: "POST", path: graphDriverPath("Put"), fn: d.put},
		{verb: "POST", path: graphDriverPath("Exists"), fn: d.exists},
		{verb: "POST", path: graphDriverPath("Status"), fn: d.graphStatus},
		{verb: "POST", path: graphDriverPath("GetMetadata"), fn: d.getMetadata},
		{verb: "POST", path: graphDriverPath("Cleanup"), fn: d.cleanup},
		{verb: "POST", path: graphDriverPath("Diff"), fn: d.diff},
		{verb: "POST", path: graphDriverPath("Changes"), fn: d.changes},
		{verb: "POST", path: graphDriverPath("ApplyDiff"), fn: d.applyDiff},
		{verb: "POST", path: graphDriverPath("DiffSize"), fn: d.diffSize},
		{verb: "POST", path: "/Plugin.Activate", fn: d.handshake},
	}
}

func (d *graphDriver) emptyResponse(w http.ResponseWriter) {
	json.NewEncoder(w).Encode(&graphResponse{})
}

func (d *graphDriver) errResponse(method string, w http.ResponseWriter, err error) {
	d.logRequest(method, "").Warnf("%v", err)
	fmt.Fprintln(w, fmt.Sprintf(`{"Err": %q}`, err.Error()))
}

func (d *graphDriver) decodeError(method string, w http.ResponseWriter, err error) {
	e := fmt.Errorf("Unable to decode JSON payload")
	d.sendError(method, "", w, e.Error()+":"+err.Error(), http.StatusBadRequest)
	return
}

func (d *graphDriver) decode(method string, w http.ResponseWriter, r *http.Request) (*graphRequest, error) {
	var request graphRequest
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		d.decodeError(method, w, err)
		return nil, err
	}
	if len(request.Parent) != 0 {
		d.logRequest(method, request.ID).Debugln("Parent: ", request.Parent)
	} else {
		d.logRequest(method, request.ID).Debugln("")
	}
	return &request, nil
}

func (d *graphDriver) handshake(w http.ResponseWriter, r *http.Request) {
	h := struct {
		Implements []string
	}{Implements: []string{GraphDriver}}

	err := json.NewEncoder(w).Encode(&h)
	if err != nil {
		d.sendError("handshake", "", w, "encode error", http.StatusInternalServerError)
		return
	}
	d.logRequest("handshake", "").Debugln("Handshake completed")
}

func (d *graphDriver) init(w http.ResponseWriter, r *http.Request) {
	method := "init"
	var request struct {
		Home string
		Opts []string
	}
	d.logRequest(method, request.Home).Infoln("")
	if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
		d.decodeError(method, w, err)
		return
	}
	gd, err := graph.Get(d.name)
	if err != nil {
		gd, err = graph.New(d.name, config.GraphDriverAPIBase, request.Opts)
		if err != nil {
			d.errResponse(method, w, err)
			return
		}
	}
	d.gd = gd
	d.emptyResponse(w)
}

func (d *graphDriver) create(w http.ResponseWriter, r *http.Request) {
	method := "create"
	if d.gd == nil {
		d.errResponse(method, w, errors.New("Graph driver not yet initialized."))
		return
	}

	request, err := d.decode(method, w, r)
	if err != nil {
		return
	}
	if err := d.gd.Create(request.ID, request.Parent, "", nil); err != nil {
		d.errResponse(method, w, err)
		return
	}
	d.emptyResponse(w)
}

func (d *graphDriver) remove(w http.ResponseWriter, r *http.Request) {
	method := "remove"
	if d.gd == nil {
		d.errResponse(method, w, errors.New("Graph driver not yet initialized."))
		return
	}

	request, err := d.decode(method, w, r)
	if err != nil {
		return
	}
	if err := d.gd.Remove(request.ID); err != nil {
		d.errResponse(method, w, err)
		return
	}
	d.emptyResponse(w)
}

func (d *graphDriver) get(w http.ResponseWriter, r *http.Request) {
	var response graphResponse
	method := "get"
	if d.gd == nil {
		d.errResponse(method, w, errors.New("Graph driver not yet initialized."))
		return
	}

	request, err := d.decode(method, w, r)
	if err != nil {
		return
	}
	response.Dir, response.Err = d.gd.Get(request.ID, request.MountLabel)
	if response.Err != nil {
		d.errResponse(method, w, response.Err)
		return
	}
	json.NewEncoder(w).Encode(&response)
}

func (d *graphDriver) put(w http.ResponseWriter, r *http.Request) {
	method := "put"
	request, err := d.decode(method, w, r)
	if d.gd == nil {
		d.errResponse(method, w, errors.New("Graph driver not yet initialized."))
		return
	}

	if err != nil {
		return
	}
	err = d.gd.Put(request.ID)
	if err != nil {
		d.errResponse(method, w, err)
		return
	}
	d.emptyResponse(w)
}

func (d *graphDriver) exists(w http.ResponseWriter, r *http.Request) {
	var response graphResponse
	method := "put"
	if d.gd == nil {
		d.errResponse(method, w, errors.New("Graph driver not yet initialized."))
		return
	}

	request, err := d.decode(method, w, r)
	if err != nil {
		return
	}
	response.Exists = d.gd.Exists(request.ID)
	json.NewEncoder(w).Encode(&response)
}

func (d *graphDriver) graphStatus(w http.ResponseWriter, r *http.Request) {
	var response graphResponse
	response.Status = d.gd.Status()
	json.NewEncoder(w).Encode(&response)
}

func (d *graphDriver) getMetadata(w http.ResponseWriter, r *http.Request) {
	var response graphResponse
	method := "getMetadata"
	if d.gd == nil {
		d.errResponse(method, w, errors.New("Graph driver not yet initialized."))
		return
	}

	request, err := d.decode(method, w, r)
	if err != nil {
		return
	}
	response.Metadata, response.Err = d.gd.GetMetadata(request.ID)
	if response.Err != nil {
		d.errResponse(method, w, response.Err)
		return
	}
	json.NewEncoder(w).Encode(&response)
}

func (d *graphDriver) cleanup(w http.ResponseWriter, r *http.Request) {
	method := "cleanup"
	if d.gd == nil {
		d.errResponse(method, w, errors.New("Graph driver not yet initialized."))
		return
	}

	err := d.gd.Cleanup()
	if err != nil {
		d.errResponse(method, w, err)
		return
	}
	d.emptyResponse(w)
}

func (d *graphDriver) diff(w http.ResponseWriter, r *http.Request) {
	method := "diff"
	if d.gd == nil {
		d.errResponse(method, w, errors.New("Graph driver not yet initialized."))
		return
	}

	request, err := d.decode(method, w, r)
	if err != nil {
		return
	}
	archive, err := d.gd.Diff(request.ID, request.Parent)
	if err != nil {
		d.errResponse(method, w, err)
		return
	}
	io.Copy(w, archive)
}

func (d *graphDriver) changes(w http.ResponseWriter, r *http.Request) {
	method := "changes"
	if d.gd == nil {
		d.errResponse(method, w, errors.New("Graph driver not yet initialized."))
		return
	}

	request, err := d.decode(method, w, r)
	if err != nil {
		return
	}
	changes, err := d.gd.Changes(request.ID, request.Parent)
	if err != nil {
		d.errResponse(method, w, err)
		return
	}
	json.NewEncoder(w).Encode(&graphResponse{Changes: changes})
}

func (d *graphDriver) applyDiff(w http.ResponseWriter, r *http.Request) {
	method := "applyDiff"
	if d.gd == nil {
		d.errResponse(method, w, errors.New("Graph driver not yet initialized."))
		return
	}

	id := r.URL.Query().Get("id")
	parent := r.URL.Query().Get("parent")
	d.logRequest(method, id).Debugf("Parent %v", parent)
	size, err := d.gd.ApplyDiff(id, parent, r.Body)
	if err != nil {
		d.errResponse(method, w, err)
		return
	}
	json.NewEncoder(w).Encode(&graphResponse{Size: size})
}

func (d *graphDriver) diffSize(w http.ResponseWriter, r *http.Request) {
	method := "diffSize"
	if d.gd == nil {
		d.errResponse(method, w, errors.New("Graph driver not yet initialized."))
		return
	}

	request, err := d.decode(method, w, r)
	if err != nil {
		return
	}
	size, err := d.gd.DiffSize(request.ID, request.Parent)
	if err != nil {
		d.errResponse(method, w, err)
		return
	}
	json.NewEncoder(w).Encode(&graphResponse{Size: size})
}
