package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/gorilla/mux"

	"github.com/Sirupsen/logrus"
	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/api/errors"
	"github.com/libopenstorage/openstorage/volume"
	"github.com/libopenstorage/openstorage/volume/drivers"
)

const schedDriverPostFix = "-sched"

type volAPI struct {
	restBase
}

func responseStatus(err error) string {
	if err == nil {
		return ""
	}
	return err.Error()
}

func newVolumeAPI(name string) restServer {
	return &volAPI{restBase{version: volume.APIVersion, name: name}}
}

func (vd *volAPI) String() string {
	return vd.name
}

func (vd *volAPI) getVolDriver(r *http.Request) (volume.VolumeDriver, error) {
	// Check if the driver has registered by it's user agent name
	userAgent := r.Header.Get("User-Agent")
	if len(userAgent) > 0 {
		clientName := strings.Split(userAgent, "/")
		if len(clientName) > 0 {
			d, err := volumedrivers.Get(clientName[0])
			if err == nil {
				return d, nil
			}
		}
	}

	// Check if the driver has registered a scheduler-based driver
	d, err := volumedrivers.Get(vd.name + schedDriverPostFix)
	if err == nil {
		return d, nil
	}

	// default
	return volumedrivers.Get(vd.name)
}

func (vd *volAPI) parseID(r *http.Request) (string, error) {
	vars := mux.Vars(r)
	if id, ok := vars["id"]; ok {
		return string(id), nil
	}
	return "", fmt.Errorf("could not parse snap ID")
}

func (vd *volAPI) create(w http.ResponseWriter, r *http.Request) {
	var dcRes api.VolumeCreateResponse
	var dcReq api.VolumeCreateRequest
	method := "create"

	if err := json.NewDecoder(r.Body).Decode(&dcReq); err != nil {
		vd.sendError(vd.name, method, w, err.Error(), http.StatusBadRequest)
		return
	}

	d, err := vd.getVolDriver(r)
	if err != nil {
		notFound(w, r)
		return
	}
	id, err := d.Create(dcReq.Locator, dcReq.Source, dcReq.Spec)
	dcRes.VolumeResponse = &api.VolumeResponse{Error: responseStatus(err)}
	dcRes.Id = id

	vd.logRequest(method, id).Infoln("")

	json.NewEncoder(w).Encode(&dcRes)
}

func processErrorForVolSetResponse(action *api.VolumeStateAction, err error, resp *api.VolumeSetResponse) {
	if err == nil || resp == nil {
		return
	}

	if action != nil && (action.Mount == api.VolumeActionParam_VOLUME_ACTION_PARAM_OFF ||
		action.Attach == api.VolumeActionParam_VOLUME_ACTION_PARAM_OFF) {
		switch err.(type) {
		case *errors.ErrNotFound:
			logrus.Warnf("Ignoring volumeSet on non-existent volume. Err: %v. Actions are Mount=%v Attach=%v",
				err, action.Mount, action.Attach)
			resp.VolumeResponse = &api.VolumeResponse{}
			resp.Volume = &api.Volume{}
		default:
			resp.VolumeResponse = &api.VolumeResponse{
				Error: err.Error(),
			}
		}
	} else if err != nil {
		resp.VolumeResponse = &api.VolumeResponse{
			Error: err.Error(),
		}
	}
}

func (vd *volAPI) volumeSet(w http.ResponseWriter, r *http.Request) {
	var (
		volumeID string
		err      error
		req      api.VolumeSetRequest
		resp     api.VolumeSetResponse
	)
	method := "volumeSet"

	err = json.NewDecoder(r.Body).Decode(&req)
	if err != nil {
		vd.sendError(vd.name, method, w, err.Error(), http.StatusBadRequest)
		return
	}

	if volumeID, err = vd.parseID(r); err != nil {
		vd.sendError(vd.name, method, w, err.Error(), http.StatusBadRequest)
		return
	}

	setActions := ""
	if req.Action != nil {
		setActions = fmt.Sprintf("Mount=%v Attach=%v", req.Action.Mount, req.Action.Attach)
	}

	vd.logRequest(method, string(volumeID)).Infoln(setActions)

	d, err := vd.getVolDriver(r)
	if err != nil {
		notFound(w, r)
		return
	}

	if req.Locator != nil || req.Spec != nil {
		err = d.Set(volumeID, req.Locator, req.Spec)
	}

	for err == nil && req.Action != nil {
		if req.Action.Attach != api.VolumeActionParam_VOLUME_ACTION_PARAM_NONE {
			if req.Action.Attach == api.VolumeActionParam_VOLUME_ACTION_PARAM_ON {
				_, err = d.Attach(volumeID, req.Options)
			} else {
				err = d.Detach(volumeID, req.Action.UnmountBeforeDetach)
			}
			if err != nil {
				break
			}
		}

		if req.Action.Mount != api.VolumeActionParam_VOLUME_ACTION_PARAM_NONE {
			if req.Action.Mount == api.VolumeActionParam_VOLUME_ACTION_PARAM_ON {
				if req.Action.MountPath == "" {
					err = fmt.Errorf("Invalid mount path")
					break
				}
				err = d.Mount(volumeID, req.Action.MountPath)
			} else {
				err = d.Unmount(volumeID, req.Action.MountPath)
			}
			if err != nil {
				break
			}
		}
		break
	}

	if err != nil {
		processErrorForVolSetResponse(req.Action, err, &resp)
	} else {
		v, err := d.Inspect([]string{volumeID})
		if err != nil {
			processErrorForVolSetResponse(req.Action, err, &resp)
		} else if v == nil || len(v) != 1 {
			processErrorForVolSetResponse(req.Action, &errors.ErrNotFound{Type: "Volume", ID: volumeID}, &resp)
		} else {
			v0 := v[0]
			resp.Volume = v0
		}
	}

	json.NewEncoder(w).Encode(resp)

}

func (vd *volAPI) inspect(w http.ResponseWriter, r *http.Request) {
	var err error
	var volumeID string

	method := "inspect"
	d, err := vd.getVolDriver(r)
	if err != nil {
		notFound(w, r)
		return
	}

	if volumeID, err = vd.parseID(r); err != nil {
		e := fmt.Errorf("Failed to parse parse volumeID: %s", err.Error())
		vd.sendError(vd.name, method, w, e.Error(), http.StatusBadRequest)
		return
	}

	vd.logRequest(method, string(volumeID)).Infoln("")

	dk, err := d.Inspect([]string{volumeID})
	if err != nil {
		vd.sendError(vd.name, method, w, err.Error(), http.StatusNotFound)
		return
	}

	json.NewEncoder(w).Encode(dk)
}

func (vd *volAPI) delete(w http.ResponseWriter, r *http.Request) {
	var volumeID string
	var err error

	method := "delete"
	if volumeID, err = vd.parseID(r); err != nil {
		e := fmt.Errorf("Failed to parse parse volumeID: %s", err.Error())
		vd.sendError(vd.name, method, w, e.Error(), http.StatusBadRequest)
		return
	}

	vd.logRequest(method, volumeID).Infoln("")

	d, err := vd.getVolDriver(r)
	if err != nil {
		notFound(w, r)
		return
	}

	volumes, err := d.Inspect([]string{volumeID})

	if len(volumes) < 1 {
		e := fmt.Errorf("Volume %s does not exist", volumeID)
		vd.sendError(vd.name, method, w, e.Error(), http.StatusBadRequest)
		return
	}
	vol := volumes[0]

	volumeResponse := &api.VolumeResponse{}

	if vol.Spec.Sticky {
		volumeResponse.Error = "Cannot delete a sticky volume"
		json.NewEncoder(w).Encode(volumeResponse)
		return
	}

	if err := d.Delete(volumeID); err != nil {
		volumeResponse.Error = err.Error()
	}
	json.NewEncoder(w).Encode(volumeResponse)
}

func (vd *volAPI) enumerate(w http.ResponseWriter, r *http.Request) {
	var locator api.VolumeLocator
	var configLabels map[string]string
	var err error
	var vols []*api.Volume

	method := "enumerate"

	d, err := vd.getVolDriver(r)
	if err != nil {
		notFound(w, r)
		return
	}
	params := r.URL.Query()
	v := params[string(api.OptName)]
	if v != nil {
		locator.Name = v[0]
	}
	v = params[string(api.OptLabel)]
	if v != nil {
		if err = json.Unmarshal([]byte(v[0]), &locator.VolumeLabels); err != nil {
			e := fmt.Errorf("Failed to parse parse VolumeLabels: %s", err.Error())
			vd.sendError(vd.name, method, w, e.Error(), http.StatusBadRequest)
		}
	}
	v = params[string(api.OptConfigLabel)]
	if v != nil {
		if err = json.Unmarshal([]byte(v[0]), &configLabels); err != nil {
			e := fmt.Errorf("Failed to parse parse configLabels: %s", err.Error())
			vd.sendError(vd.name, method, w, e.Error(), http.StatusBadRequest)
		}
	}
	v = params[string(api.OptVolumeID)]
	if v != nil {
		ids := make([]string, len(v))
		for i, s := range v {
			ids[i] = string(s)
		}
		vols, err = d.Inspect(ids)
		if err != nil {
			e := fmt.Errorf("Failed to inspect volumeID: %s", err.Error())
			vd.sendError(vd.name, method, w, e.Error(), http.StatusBadRequest)
			return
		}
	} else {
		vols, err = d.Enumerate(&locator, configLabels)
		if err != nil {
			vd.sendError(vd.name, method, w, err.Error(), http.StatusInternalServerError)
			return
		}
	}
	json.NewEncoder(w).Encode(vols)
}

func (vd *volAPI) snap(w http.ResponseWriter, r *http.Request) {
	var snapReq api.SnapCreateRequest
	var snapRes api.SnapCreateResponse
	method := "snap"

	if err := json.NewDecoder(r.Body).Decode(&snapReq); err != nil {
		vd.sendError(vd.name, method, w, err.Error(), http.StatusBadRequest)
		return
	}
	d, err := vd.getVolDriver(r)
	if err != nil {
		notFound(w, r)
		return
	}

	vd.logRequest(method, string(snapReq.Id)).Infoln("")

	id, err := d.Snapshot(snapReq.Id, snapReq.Readonly, snapReq.Locator)
	snapRes.VolumeCreateResponse = &api.VolumeCreateResponse{
		Id: id,
		VolumeResponse: &api.VolumeResponse{
			Error: responseStatus(err),
		},
	}
	json.NewEncoder(w).Encode(&snapRes)
}

func (vd *volAPI) restore(w http.ResponseWriter, r *http.Request) {
	var volumeID, snapID string
	var err error
	method := "restore"

	if volumeID, err = vd.parseID(r); err != nil {
		e := fmt.Errorf("Failed to parse parse volumeID: %s", err.Error())
		vd.sendError(vd.name, method, w, e.Error(), http.StatusBadRequest)
		return
	}

	d, err := vd.getVolDriver(r)
	if err != nil {
		notFound(w, r)
		return
	}

	params := r.URL.Query()
	v := params[api.OptSnapID]
	if v != nil {
		snapID = v[0]
	} else {
		vd.sendError(vd.name, method, w, "Missing "+api.OptSnapID+" param",
			http.StatusBadRequest)
		return
	}

	volumeResponse := &api.VolumeResponse{}
	if err := d.Restore(volumeID, snapID); err != nil {
		volumeResponse.Error = responseStatus(err)
	}
	json.NewEncoder(w).Encode(volumeResponse)
}

func (vd *volAPI) snapEnumerate(w http.ResponseWriter, r *http.Request) {
	var err error
	var labels map[string]string
	var ids []string

	method := "snapEnumerate"
	d, err := vd.getVolDriver(r)
	if err != nil {
		notFound(w, r)
		return
	}
	params := r.URL.Query()
	v := params[string(api.OptLabel)]
	if v != nil {
		if err = json.Unmarshal([]byte(v[0]), &labels); err != nil {
			e := fmt.Errorf("Failed to parse parse VolumeLabels: %s", err.Error())
			vd.sendError(vd.name, method, w, e.Error(), http.StatusBadRequest)
		}
	}

	v, ok := params[string(api.OptVolumeID)]
	if v != nil && ok {
		ids = make([]string, len(params))
		for i, s := range v {
			ids[i] = string(s)
		}
	}

	snaps, err := d.SnapEnumerate(ids, labels)
	if err != nil {
		e := fmt.Errorf("Failed to enumerate snaps: %s", err.Error())
		vd.sendError(vd.name, method, w, e.Error(), http.StatusBadRequest)
		return
	}

	json.NewEncoder(w).Encode(snaps)
}

func (vd *volAPI) stats(w http.ResponseWriter, r *http.Request) {
	var volumeID string
	var err error

	if volumeID, err = vd.parseID(r); err != nil {
		e := fmt.Errorf("Failed to parse volumeID: %s", err.Error())
		http.Error(w, e.Error(), http.StatusBadRequest)
		return
	}

	params := r.URL.Query()
	// By default always report /proc/diskstats style stats.
	cumulative := true
	if opt, ok := params[string(api.OptCumulative)]; ok {
		if boolValue, err := strconv.ParseBool(strings.Join(opt[:], "")); !ok {
			e := fmt.Errorf("Failed to parse %s option: %s",
				api.OptCumulative, err.Error())
			http.Error(w, e.Error(), http.StatusBadRequest)
			return
		} else {
			cumulative = boolValue
		}
	}

	d, err := vd.getVolDriver(r)
	if err != nil {
		notFound(w, r)
		return
	}

	stats, err := d.Stats(volumeID, cumulative)
	if err != nil {
		e := fmt.Errorf("Failed to get stats: %s", err.Error())
		http.Error(w, e.Error(), http.StatusBadRequest)
		return
	}
	json.NewEncoder(w).Encode(stats)
}

func (vd *volAPI) usedsize(w http.ResponseWriter, r *http.Request) {
	var volumeID string
	var err error

	if volumeID, err = vd.parseID(r); err != nil {
		e := fmt.Errorf("Failed to parse volumeID: %s", err.Error())
		http.Error(w, e.Error(), http.StatusBadRequest)
		return
	}

	d, err := vd.getVolDriver(r)
	if err != nil {
		notFound(w, r)
		return
	}

	used, err := d.UsedSize(volumeID)
	if err != nil {
		e := fmt.Errorf("Failed to get used size: %s", err.Error())
		http.Error(w, e.Error(), http.StatusBadRequest)
		return
	}
	json.NewEncoder(w).Encode(used)
}

func (vd *volAPI) requests(w http.ResponseWriter, r *http.Request) {
	var err error

	method := "requests"

	d, err := vd.getVolDriver(r)
	if err != nil {
		notFound(w, r)
		return
	}

	requests, err := d.GetActiveRequests()
	if err != nil {
		e := fmt.Errorf("Failed to get active requests: %s", err.Error())
		vd.sendError(vd.name, method, w, e.Error(), http.StatusBadRequest)
		return
	}
	json.NewEncoder(w).Encode(requests)
}

func (vd *volAPI) versions(w http.ResponseWriter, r *http.Request) {
	versions := []string{
		volume.APIVersion,
		// Update supported versions by adding them here
	}
	json.NewEncoder(w).Encode(versions)
}

func volVersion(route, version string) string {
	if version == "" {
		return "/" + route
	} else {
		return "/" + version + "/" + route
	}
}

func volPath(route, version string) string {
	return volVersion(api.OsdVolumePath+route, version)
}

func snapPath(route, version string) string {
	return volVersion(api.OsdSnapshotPath+route, version)
}

func (vd *volAPI) Routes() []*Route {
	return []*Route{
		{verb: "GET", path: "/" + api.OsdVolumePath + "/versions", fn: vd.versions},
		{verb: "POST", path: volPath("", volume.APIVersion), fn: vd.create},
		{verb: "PUT", path: volPath("/{id}", volume.APIVersion), fn: vd.volumeSet},
		{verb: "GET", path: volPath("", volume.APIVersion), fn: vd.enumerate},
		{verb: "GET", path: volPath("/{id}", volume.APIVersion), fn: vd.inspect},
		{verb: "DELETE", path: volPath("/{id}", volume.APIVersion), fn: vd.delete},
		{verb: "GET", path: volPath("/stats", volume.APIVersion), fn: vd.stats},
		{verb: "GET", path: volPath("/stats/{id}", volume.APIVersion), fn: vd.stats},
		{verb: "GET", path: volPath("/usedsize", volume.APIVersion), fn: vd.usedsize},
		{verb: "GET", path: volPath("/usedsize/{id}", volume.APIVersion), fn: vd.usedsize},
		{verb: "GET", path: volPath("/requests", volume.APIVersion), fn: vd.requests},
		{verb: "GET", path: volPath("/requests/{id}", volume.APIVersion), fn: vd.requests},
		{verb: "POST", path: snapPath("", volume.APIVersion), fn: vd.snap},
		{verb: "GET", path: snapPath("", volume.APIVersion), fn: vd.snapEnumerate},
		{verb: "POST", path: snapPath("/restore/{id}", volume.APIVersion), fn: vd.restore},
	}
}
