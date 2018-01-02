package api

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"

	"github.com/docker/libnetwork"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/netutils"
	"github.com/docker/libnetwork/types"
	"github.com/gorilla/mux"
)

var (
	successResponse  = responseStatus{Status: "Success", StatusCode: http.StatusOK}
	createdResponse  = responseStatus{Status: "Created", StatusCode: http.StatusCreated}
	mismatchResponse = responseStatus{Status: "Body/URI parameter mismatch", StatusCode: http.StatusBadRequest}
	badQueryResponse = responseStatus{Status: "Unsupported query", StatusCode: http.StatusBadRequest}
)

const (
	// Resource name regex
	// Gorilla mux encloses the passed pattern with '^' and '$'. So we need to do some tricks
	// to have mux eventually build a query regex which matches empty or word string (`^$|[\w]+`)
	regex = "[a-zA-Z_0-9-]+"
	qregx = "$|" + regex
	// Router URL variable definition
	nwName   = "{" + urlNwName + ":" + regex + "}"
	nwNameQr = "{" + urlNwName + ":" + qregx + "}"
	nwID     = "{" + urlNwID + ":" + regex + "}"
	nwPIDQr  = "{" + urlNwPID + ":" + qregx + "}"
	epName   = "{" + urlEpName + ":" + regex + "}"
	epNameQr = "{" + urlEpName + ":" + qregx + "}"
	epID     = "{" + urlEpID + ":" + regex + "}"
	epPIDQr  = "{" + urlEpPID + ":" + qregx + "}"
	sbID     = "{" + urlSbID + ":" + regex + "}"
	sbPIDQr  = "{" + urlSbPID + ":" + qregx + "}"
	cnIDQr   = "{" + urlCnID + ":" + qregx + "}"
	cnPIDQr  = "{" + urlCnPID + ":" + qregx + "}"

	// Internal URL variable name.They can be anything as
	// long as they do not collide with query fields.
	urlNwName = "network-name"
	urlNwID   = "network-id"
	urlNwPID  = "network-partial-id"
	urlEpName = "endpoint-name"
	urlEpID   = "endpoint-id"
	urlEpPID  = "endpoint-partial-id"
	urlSbID   = "sandbox-id"
	urlSbPID  = "sandbox-partial-id"
	urlCnID   = "container-id"
	urlCnPID  = "container-partial-id"
)

// NewHTTPHandler creates and initialize the HTTP handler to serve the requests for libnetwork
func NewHTTPHandler(c libnetwork.NetworkController) func(w http.ResponseWriter, req *http.Request) {
	h := &httpHandler{c: c}
	h.initRouter()
	return h.handleRequest
}

type responseStatus struct {
	Status     string
	StatusCode int
}

func (r *responseStatus) isOK() bool {
	return r.StatusCode == http.StatusOK || r.StatusCode == http.StatusCreated
}

type processor func(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus)

type httpHandler struct {
	c libnetwork.NetworkController
	r *mux.Router
}

func (h *httpHandler) handleRequest(w http.ResponseWriter, req *http.Request) {
	// Make sure the service is there
	if h.c == nil {
		http.Error(w, "NetworkController is not available", http.StatusServiceUnavailable)
		return
	}

	// Get handler from router and execute it
	h.r.ServeHTTP(w, req)
}

func (h *httpHandler) initRouter() {
	m := map[string][]struct {
		url string
		qrs []string
		fct processor
	}{
		"GET": {
			// Order matters
			{"/networks", []string{"name", nwNameQr}, procGetNetworks},
			{"/networks", []string{"partial-id", nwPIDQr}, procGetNetworks},
			{"/networks", nil, procGetNetworks},
			{"/networks/" + nwID, nil, procGetNetwork},
			{"/networks/" + nwID + "/endpoints", []string{"name", epNameQr}, procGetEndpoints},
			{"/networks/" + nwID + "/endpoints", []string{"partial-id", epPIDQr}, procGetEndpoints},
			{"/networks/" + nwID + "/endpoints", nil, procGetEndpoints},
			{"/networks/" + nwID + "/endpoints/" + epID, nil, procGetEndpoint},
			{"/services", []string{"network", nwNameQr}, procGetServices},
			{"/services", []string{"name", epNameQr}, procGetServices},
			{"/services", []string{"partial-id", epPIDQr}, procGetServices},
			{"/services", nil, procGetServices},
			{"/services/" + epID, nil, procGetService},
			{"/services/" + epID + "/backend", nil, procGetSandbox},
			{"/sandboxes", []string{"partial-container-id", cnPIDQr}, procGetSandboxes},
			{"/sandboxes", []string{"container-id", cnIDQr}, procGetSandboxes},
			{"/sandboxes", []string{"partial-id", sbPIDQr}, procGetSandboxes},
			{"/sandboxes", nil, procGetSandboxes},
			{"/sandboxes/" + sbID, nil, procGetSandbox},
		},
		"POST": {
			{"/networks", nil, procCreateNetwork},
			{"/networks/" + nwID + "/endpoints", nil, procCreateEndpoint},
			{"/networks/" + nwID + "/endpoints/" + epID + "/sandboxes", nil, procJoinEndpoint},
			{"/services", nil, procPublishService},
			{"/services/" + epID + "/backend", nil, procAttachBackend},
			{"/sandboxes", nil, procCreateSandbox},
		},
		"DELETE": {
			{"/networks/" + nwID, nil, procDeleteNetwork},
			{"/networks/" + nwID + "/endpoints/" + epID, nil, procDeleteEndpoint},
			{"/networks/" + nwID + "/endpoints/" + epID + "/sandboxes/" + sbID, nil, procLeaveEndpoint},
			{"/services/" + epID, nil, procUnpublishService},
			{"/services/" + epID + "/backend/" + sbID, nil, procDetachBackend},
			{"/sandboxes/" + sbID, nil, procDeleteSandbox},
		},
	}

	h.r = mux.NewRouter()
	for method, routes := range m {
		for _, route := range routes {
			r := h.r.Path("/{.*}" + route.url).Methods(method).HandlerFunc(makeHandler(h.c, route.fct))
			if route.qrs != nil {
				r.Queries(route.qrs...)
			}

			r = h.r.Path(route.url).Methods(method).HandlerFunc(makeHandler(h.c, route.fct))
			if route.qrs != nil {
				r.Queries(route.qrs...)
			}
		}
	}
}

func makeHandler(ctrl libnetwork.NetworkController, fct processor) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		var (
			body []byte
			err  error
		)
		if req.Body != nil {
			body, err = ioutil.ReadAll(req.Body)
			if err != nil {
				http.Error(w, "Invalid body: "+err.Error(), http.StatusBadRequest)
				return
			}
		}

		res, rsp := fct(ctrl, mux.Vars(req), body)
		if !rsp.isOK() {
			http.Error(w, rsp.Status, rsp.StatusCode)
			return
		}
		if res != nil {
			writeJSON(w, rsp.StatusCode, res)
		}
	}
}

/*****************
 Resource Builders
******************/

func buildNetworkResource(nw libnetwork.Network) *networkResource {
	r := &networkResource{}
	if nw != nil {
		r.Name = nw.Name()
		r.ID = nw.ID()
		r.Type = nw.Type()
		epl := nw.Endpoints()
		r.Endpoints = make([]*endpointResource, 0, len(epl))
		for _, e := range epl {
			epr := buildEndpointResource(e)
			r.Endpoints = append(r.Endpoints, epr)
		}
	}
	return r
}

func buildEndpointResource(ep libnetwork.Endpoint) *endpointResource {
	r := &endpointResource{}
	if ep != nil {
		r.Name = ep.Name()
		r.ID = ep.ID()
		r.Network = ep.Network()
	}
	return r
}

func buildSandboxResource(sb libnetwork.Sandbox) *sandboxResource {
	r := &sandboxResource{}
	if sb != nil {
		r.ID = sb.ID()
		r.Key = sb.Key()
		r.ContainerID = sb.ContainerID()
	}
	return r
}

/****************
 Options Parsers
*****************/

func (sc *sandboxCreate) parseOptions() []libnetwork.SandboxOption {
	var setFctList []libnetwork.SandboxOption
	if sc.HostName != "" {
		setFctList = append(setFctList, libnetwork.OptionHostname(sc.HostName))
	}
	if sc.DomainName != "" {
		setFctList = append(setFctList, libnetwork.OptionDomainname(sc.DomainName))
	}
	if sc.HostsPath != "" {
		setFctList = append(setFctList, libnetwork.OptionHostsPath(sc.HostsPath))
	}
	if sc.ResolvConfPath != "" {
		setFctList = append(setFctList, libnetwork.OptionResolvConfPath(sc.ResolvConfPath))
	}
	if sc.UseDefaultSandbox {
		setFctList = append(setFctList, libnetwork.OptionUseDefaultSandbox())
	}
	if sc.UseExternalKey {
		setFctList = append(setFctList, libnetwork.OptionUseExternalKey())
	}
	if sc.DNS != nil {
		for _, d := range sc.DNS {
			setFctList = append(setFctList, libnetwork.OptionDNS(d))
		}
	}
	if sc.ExtraHosts != nil {
		for _, e := range sc.ExtraHosts {
			setFctList = append(setFctList, libnetwork.OptionExtraHost(e.Name, e.Address))
		}
	}
	if sc.ExposedPorts != nil {
		setFctList = append(setFctList, libnetwork.OptionExposedPorts(sc.ExposedPorts))
	}
	if sc.PortMapping != nil {
		setFctList = append(setFctList, libnetwork.OptionPortMapping(sc.PortMapping))
	}
	return setFctList
}

func (ej *endpointJoin) parseOptions() []libnetwork.EndpointOption {
	// priority will go here
	return []libnetwork.EndpointOption{}
}

/******************
 Process functions
*******************/

func processCreateDefaults(c libnetwork.NetworkController, nc *networkCreate) {
	if nc.NetworkType == "" {
		nc.NetworkType = c.Config().Daemon.DefaultDriver
	}
}

/***************************
 NetworkController interface
****************************/
func procCreateNetwork(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	var create networkCreate

	err := json.Unmarshal(body, &create)
	if err != nil {
		return nil, &responseStatus{Status: "Invalid body: " + err.Error(), StatusCode: http.StatusBadRequest}
	}
	processCreateDefaults(c, &create)

	options := []libnetwork.NetworkOption{}
	if val, ok := create.NetworkOpts[netlabel.Internal]; ok {
		internal, err := strconv.ParseBool(val)
		if err != nil {
			return nil, &responseStatus{Status: err.Error(), StatusCode: http.StatusBadRequest}
		}
		if internal {
			options = append(options, libnetwork.NetworkOptionInternalNetwork())
		}
	}
	if val, ok := create.NetworkOpts[netlabel.EnableIPv6]; ok {
		enableIPv6, err := strconv.ParseBool(val)
		if err != nil {
			return nil, &responseStatus{Status: err.Error(), StatusCode: http.StatusBadRequest}
		}
		options = append(options, libnetwork.NetworkOptionEnableIPv6(enableIPv6))
	}
	if len(create.DriverOpts) > 0 {
		options = append(options, libnetwork.NetworkOptionDriverOpts(create.DriverOpts))
	}

	if len(create.IPv4Conf) > 0 {
		ipamV4Conf := &libnetwork.IpamConf{
			PreferredPool: create.IPv4Conf[0].PreferredPool,
			SubPool:       create.IPv4Conf[0].SubPool,
		}

		options = append(options, libnetwork.NetworkOptionIpam("default", "", []*libnetwork.IpamConf{ipamV4Conf}, nil, nil))
	}

	nw, err := c.NewNetwork(create.NetworkType, create.Name, create.ID, options...)
	if err != nil {
		return nil, convertNetworkError(err)
	}

	return nw.ID(), &createdResponse
}

func procGetNetwork(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	t, by := detectNetworkTarget(vars)
	nw, errRsp := findNetwork(c, t, by)
	if !errRsp.isOK() {
		return nil, errRsp
	}
	return buildNetworkResource(nw), &successResponse
}

func procGetNetworks(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	var list []*networkResource

	// Look for query filters and validate
	name, queryByName := vars[urlNwName]
	shortID, queryByPid := vars[urlNwPID]
	if queryByName && queryByPid {
		return nil, &badQueryResponse
	}

	if queryByName {
		if nw, errRsp := findNetwork(c, name, byName); errRsp.isOK() {
			list = append(list, buildNetworkResource(nw))
		}
	} else if queryByPid {
		// Return all the prefix-matching networks
		l := func(nw libnetwork.Network) bool {
			if strings.HasPrefix(nw.ID(), shortID) {
				list = append(list, buildNetworkResource(nw))
			}
			return false
		}
		c.WalkNetworks(l)
	} else {
		for _, nw := range c.Networks() {
			list = append(list, buildNetworkResource(nw))
		}
	}

	return list, &successResponse
}

func procCreateSandbox(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	var create sandboxCreate

	err := json.Unmarshal(body, &create)
	if err != nil {
		return "", &responseStatus{Status: "Invalid body: " + err.Error(), StatusCode: http.StatusBadRequest}
	}

	sb, err := c.NewSandbox(create.ContainerID, create.parseOptions()...)
	if err != nil {
		return "", convertNetworkError(err)
	}

	return sb.ID(), &createdResponse
}

/******************
 Network interface
*******************/
func procCreateEndpoint(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	var ec endpointCreate

	err := json.Unmarshal(body, &ec)
	if err != nil {
		return "", &responseStatus{Status: "Invalid body: " + err.Error(), StatusCode: http.StatusBadRequest}
	}

	nwT, nwBy := detectNetworkTarget(vars)
	n, errRsp := findNetwork(c, nwT, nwBy)
	if !errRsp.isOK() {
		return "", errRsp
	}

	var setFctList []libnetwork.EndpointOption
	for _, str := range ec.MyAliases {
		setFctList = append(setFctList, libnetwork.CreateOptionMyAlias(str))
	}

	ep, err := n.CreateEndpoint(ec.Name, setFctList...)
	if err != nil {
		return "", convertNetworkError(err)
	}

	return ep.ID(), &createdResponse
}

func procGetEndpoint(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	nwT, nwBy := detectNetworkTarget(vars)
	epT, epBy := detectEndpointTarget(vars)

	ep, errRsp := findEndpoint(c, nwT, epT, nwBy, epBy)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	return buildEndpointResource(ep), &successResponse
}

func procGetEndpoints(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	// Look for query filters and validate
	name, queryByName := vars[urlEpName]
	shortID, queryByPid := vars[urlEpPID]
	if queryByName && queryByPid {
		return nil, &badQueryResponse
	}

	nwT, nwBy := detectNetworkTarget(vars)
	nw, errRsp := findNetwork(c, nwT, nwBy)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	var list []*endpointResource

	// If query parameter is specified, return a filtered collection
	if queryByName {
		if ep, errRsp := findEndpoint(c, nwT, name, nwBy, byName); errRsp.isOK() {
			list = append(list, buildEndpointResource(ep))
		}
	} else if queryByPid {
		// Return all the prefix-matching endpoints
		l := func(ep libnetwork.Endpoint) bool {
			if strings.HasPrefix(ep.ID(), shortID) {
				list = append(list, buildEndpointResource(ep))
			}
			return false
		}
		nw.WalkEndpoints(l)
	} else {
		for _, ep := range nw.Endpoints() {
			epr := buildEndpointResource(ep)
			list = append(list, epr)
		}
	}

	return list, &successResponse
}

func procDeleteNetwork(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	target, by := detectNetworkTarget(vars)

	nw, errRsp := findNetwork(c, target, by)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	err := nw.Delete()
	if err != nil {
		return nil, convertNetworkError(err)
	}

	return nil, &successResponse
}

/******************
 Endpoint interface
*******************/
func procJoinEndpoint(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	var ej endpointJoin
	var setFctList []libnetwork.EndpointOption
	err := json.Unmarshal(body, &ej)
	if err != nil {
		return nil, &responseStatus{Status: "Invalid body: " + err.Error(), StatusCode: http.StatusBadRequest}
	}

	nwT, nwBy := detectNetworkTarget(vars)
	epT, epBy := detectEndpointTarget(vars)

	ep, errRsp := findEndpoint(c, nwT, epT, nwBy, epBy)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	sb, errRsp := findSandbox(c, ej.SandboxID, byID)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	for _, str := range ej.Aliases {
		name, alias, err := netutils.ParseAlias(str)
		if err != nil {
			return "", convertNetworkError(err)
		}
		setFctList = append(setFctList, libnetwork.CreateOptionAlias(name, alias))
	}

	err = ep.Join(sb, setFctList...)
	if err != nil {
		return nil, convertNetworkError(err)
	}
	return sb.Key(), &successResponse
}

func procLeaveEndpoint(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	nwT, nwBy := detectNetworkTarget(vars)
	epT, epBy := detectEndpointTarget(vars)

	ep, errRsp := findEndpoint(c, nwT, epT, nwBy, epBy)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	sb, errRsp := findSandbox(c, vars[urlSbID], byID)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	err := ep.Leave(sb)
	if err != nil {
		return nil, convertNetworkError(err)
	}

	return nil, &successResponse
}

func procDeleteEndpoint(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	nwT, nwBy := detectNetworkTarget(vars)
	epT, epBy := detectEndpointTarget(vars)

	ep, errRsp := findEndpoint(c, nwT, epT, nwBy, epBy)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	err := ep.Delete(false)
	if err != nil {
		return nil, convertNetworkError(err)
	}

	return nil, &successResponse
}

/******************
 Service interface
*******************/
func procGetServices(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	// Look for query filters and validate
	nwName, filterByNwName := vars[urlNwName]
	svName, queryBySvName := vars[urlEpName]
	shortID, queryBySvPID := vars[urlEpPID]

	if filterByNwName && queryBySvName || filterByNwName && queryBySvPID || queryBySvName && queryBySvPID {
		return nil, &badQueryResponse
	}

	var list []*endpointResource

	switch {
	case filterByNwName:
		// return all service present on the specified network
		nw, errRsp := findNetwork(c, nwName, byName)
		if !errRsp.isOK() {
			return list, &successResponse
		}
		for _, ep := range nw.Endpoints() {
			epr := buildEndpointResource(ep)
			list = append(list, epr)
		}
	case queryBySvName:
		// Look in each network for the service with the specified name
		l := func(ep libnetwork.Endpoint) bool {
			if ep.Name() == svName {
				list = append(list, buildEndpointResource(ep))
				return true
			}
			return false
		}
		for _, nw := range c.Networks() {
			nw.WalkEndpoints(l)
		}
	case queryBySvPID:
		// Return all the prefix-matching services
		l := func(ep libnetwork.Endpoint) bool {
			if strings.HasPrefix(ep.ID(), shortID) {
				list = append(list, buildEndpointResource(ep))
			}
			return false
		}
		for _, nw := range c.Networks() {
			nw.WalkEndpoints(l)
		}
	default:
		for _, nw := range c.Networks() {
			for _, ep := range nw.Endpoints() {
				epr := buildEndpointResource(ep)
				list = append(list, epr)
			}
		}
	}

	return list, &successResponse
}

func procGetService(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	epT, epBy := detectEndpointTarget(vars)
	sv, errRsp := findService(c, epT, epBy)
	if !errRsp.isOK() {
		return nil, endpointToService(errRsp)
	}
	return buildEndpointResource(sv), &successResponse
}

func procPublishService(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	var sp servicePublish

	err := json.Unmarshal(body, &sp)
	if err != nil {
		return "", &responseStatus{Status: "Invalid body: " + err.Error(), StatusCode: http.StatusBadRequest}
	}

	n, errRsp := findNetwork(c, sp.Network, byName)
	if !errRsp.isOK() {
		return "", errRsp
	}

	var setFctList []libnetwork.EndpointOption
	for _, str := range sp.MyAliases {
		setFctList = append(setFctList, libnetwork.CreateOptionMyAlias(str))
	}

	ep, err := n.CreateEndpoint(sp.Name, setFctList...)
	if err != nil {
		return "", endpointToService(convertNetworkError(err))
	}

	return ep.ID(), &createdResponse
}

func procUnpublishService(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	var sd serviceDelete

	if body != nil {
		err := json.Unmarshal(body, &sd)
		if err != nil {
			return "", &responseStatus{Status: "Invalid body: " + err.Error(), StatusCode: http.StatusBadRequest}
		}
	}

	epT, epBy := detectEndpointTarget(vars)
	sv, errRsp := findService(c, epT, epBy)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	if err := sv.Delete(sd.Force); err != nil {
		return nil, endpointToService(convertNetworkError(err))
	}
	return nil, &successResponse
}

func procAttachBackend(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	var bk endpointJoin
	var setFctList []libnetwork.EndpointOption
	err := json.Unmarshal(body, &bk)
	if err != nil {
		return nil, &responseStatus{Status: "Invalid body: " + err.Error(), StatusCode: http.StatusBadRequest}
	}

	epT, epBy := detectEndpointTarget(vars)
	sv, errRsp := findService(c, epT, epBy)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	sb, errRsp := findSandbox(c, bk.SandboxID, byID)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	for _, str := range bk.Aliases {
		name, alias, err := netutils.ParseAlias(str)
		if err != nil {
			return "", convertNetworkError(err)
		}
		setFctList = append(setFctList, libnetwork.CreateOptionAlias(name, alias))
	}

	err = sv.Join(sb, setFctList...)
	if err != nil {
		return nil, convertNetworkError(err)
	}

	return sb.Key(), &successResponse
}

func procDetachBackend(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	epT, epBy := detectEndpointTarget(vars)
	sv, errRsp := findService(c, epT, epBy)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	sb, errRsp := findSandbox(c, vars[urlSbID], byID)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	err := sv.Leave(sb)
	if err != nil {
		return nil, convertNetworkError(err)
	}

	return nil, &successResponse
}

/******************
 Sandbox interface
*******************/
func procGetSandbox(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	if epT, ok := vars[urlEpID]; ok {
		sv, errRsp := findService(c, epT, byID)
		if !errRsp.isOK() {
			return nil, endpointToService(errRsp)
		}
		return buildSandboxResource(sv.Info().Sandbox()), &successResponse
	}

	sbT, by := detectSandboxTarget(vars)
	sb, errRsp := findSandbox(c, sbT, by)
	if !errRsp.isOK() {
		return nil, errRsp
	}
	return buildSandboxResource(sb), &successResponse
}

type cndFnMkr func(string) cndFn
type cndFn func(libnetwork.Sandbox) bool

// list of (query type, condition function makers) couples
var cndMkrList = []struct {
	identifier string
	maker      cndFnMkr
}{
	{urlSbPID, func(id string) cndFn {
		return func(sb libnetwork.Sandbox) bool { return strings.HasPrefix(sb.ID(), id) }
	}},
	{urlCnID, func(id string) cndFn {
		return func(sb libnetwork.Sandbox) bool { return sb.ContainerID() == id }
	}},
	{urlCnPID, func(id string) cndFn {
		return func(sb libnetwork.Sandbox) bool { return strings.HasPrefix(sb.ContainerID(), id) }
	}},
}

func getQueryCondition(vars map[string]string) func(libnetwork.Sandbox) bool {
	for _, im := range cndMkrList {
		if val, ok := vars[im.identifier]; ok {
			return im.maker(val)
		}
	}
	return func(sb libnetwork.Sandbox) bool { return true }
}

func sandboxWalker(condition cndFn, list *[]*sandboxResource) libnetwork.SandboxWalker {
	return func(sb libnetwork.Sandbox) bool {
		if condition(sb) {
			*list = append(*list, buildSandboxResource(sb))
		}
		return false
	}
}

func procGetSandboxes(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	var list []*sandboxResource

	cnd := getQueryCondition(vars)
	c.WalkSandboxes(sandboxWalker(cnd, &list))

	return list, &successResponse
}

func procDeleteSandbox(c libnetwork.NetworkController, vars map[string]string, body []byte) (interface{}, *responseStatus) {
	sbT, by := detectSandboxTarget(vars)

	sb, errRsp := findSandbox(c, sbT, by)
	if !errRsp.isOK() {
		return nil, errRsp
	}

	err := sb.Delete()
	if err != nil {
		return nil, convertNetworkError(err)
	}

	return nil, &successResponse
}

/***********
  Utilities
************/
const (
	byID = iota
	byName
)

func detectNetworkTarget(vars map[string]string) (string, int) {
	if target, ok := vars[urlNwName]; ok {
		return target, byName
	}
	if target, ok := vars[urlNwID]; ok {
		return target, byID
	}
	// vars are populated from the URL, following cannot happen
	panic("Missing URL variable parameter for network")
}

func detectSandboxTarget(vars map[string]string) (string, int) {
	if target, ok := vars[urlSbID]; ok {
		return target, byID
	}
	// vars are populated from the URL, following cannot happen
	panic("Missing URL variable parameter for sandbox")
}

func detectEndpointTarget(vars map[string]string) (string, int) {
	if target, ok := vars[urlEpName]; ok {
		return target, byName
	}
	if target, ok := vars[urlEpID]; ok {
		return target, byID
	}
	// vars are populated from the URL, following cannot happen
	panic("Missing URL variable parameter for endpoint")
}

func findNetwork(c libnetwork.NetworkController, s string, by int) (libnetwork.Network, *responseStatus) {
	var (
		nw  libnetwork.Network
		err error
	)
	switch by {
	case byID:
		nw, err = c.NetworkByID(s)
	case byName:
		if s == "" {
			s = c.Config().Daemon.DefaultNetwork
		}
		nw, err = c.NetworkByName(s)
	default:
		panic(fmt.Sprintf("unexpected selector for network search: %d", by))
	}
	if err != nil {
		if _, ok := err.(types.NotFoundError); ok {
			return nil, &responseStatus{Status: "Resource not found: Network", StatusCode: http.StatusNotFound}
		}
		return nil, &responseStatus{Status: err.Error(), StatusCode: http.StatusBadRequest}
	}
	return nw, &successResponse
}

func findSandbox(c libnetwork.NetworkController, s string, by int) (libnetwork.Sandbox, *responseStatus) {
	var (
		sb  libnetwork.Sandbox
		err error
	)

	switch by {
	case byID:
		sb, err = c.SandboxByID(s)
	default:
		panic(fmt.Sprintf("unexpected selector for sandbox search: %d", by))
	}
	if err != nil {
		if _, ok := err.(types.NotFoundError); ok {
			return nil, &responseStatus{Status: "Resource not found: Sandbox", StatusCode: http.StatusNotFound}
		}
		return nil, &responseStatus{Status: err.Error(), StatusCode: http.StatusBadRequest}
	}
	return sb, &successResponse
}

func findEndpoint(c libnetwork.NetworkController, ns, es string, nwBy, epBy int) (libnetwork.Endpoint, *responseStatus) {
	nw, errRsp := findNetwork(c, ns, nwBy)
	if !errRsp.isOK() {
		return nil, errRsp
	}
	var (
		err error
		ep  libnetwork.Endpoint
	)
	switch epBy {
	case byID:
		ep, err = nw.EndpointByID(es)
	case byName:
		ep, err = nw.EndpointByName(es)
	default:
		panic(fmt.Sprintf("unexpected selector for endpoint search: %d", epBy))
	}
	if err != nil {
		if _, ok := err.(types.NotFoundError); ok {
			return nil, &responseStatus{Status: "Resource not found: Endpoint", StatusCode: http.StatusNotFound}
		}
		return nil, &responseStatus{Status: err.Error(), StatusCode: http.StatusBadRequest}
	}
	return ep, &successResponse
}

func findService(c libnetwork.NetworkController, svs string, svBy int) (libnetwork.Endpoint, *responseStatus) {
	for _, nw := range c.Networks() {
		var (
			ep  libnetwork.Endpoint
			err error
		)
		switch svBy {
		case byID:
			ep, err = nw.EndpointByID(svs)
		case byName:
			ep, err = nw.EndpointByName(svs)
		default:
			panic(fmt.Sprintf("unexpected selector for service search: %d", svBy))
		}
		if err == nil {
			return ep, &successResponse
		} else if _, ok := err.(types.NotFoundError); !ok {
			return nil, convertNetworkError(err)
		}
	}
	return nil, &responseStatus{Status: "Service not found", StatusCode: http.StatusNotFound}
}

func endpointToService(rsp *responseStatus) *responseStatus {
	rsp.Status = strings.Replace(rsp.Status, "endpoint", "service", -1)
	return rsp
}

func convertNetworkError(err error) *responseStatus {
	var code int
	switch err.(type) {
	case types.BadRequestError:
		code = http.StatusBadRequest
	case types.ForbiddenError:
		code = http.StatusForbidden
	case types.NotFoundError:
		code = http.StatusNotFound
	case types.TimeoutError:
		code = http.StatusRequestTimeout
	case types.NotImplementedError:
		code = http.StatusNotImplemented
	case types.NoServiceError:
		code = http.StatusServiceUnavailable
	case types.InternalError:
		code = http.StatusInternalServerError
	default:
		code = http.StatusInternalServerError
	}
	return &responseStatus{Status: err.Error(), StatusCode: code}
}

func writeJSON(w http.ResponseWriter, code int, v interface{}) error {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	return json.NewEncoder(w).Encode(v)
}
