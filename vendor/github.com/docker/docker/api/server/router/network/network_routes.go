package network

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/errors"
	"github.com/docker/docker/api/server/httputils"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/api/types/versions"
	"github.com/docker/libnetwork"
	"github.com/docker/libnetwork/networkdb"
)

var (
	// acceptedNetworkFilters is a list of acceptable filters
	acceptedNetworkFilters = map[string]bool{
		"driver": true,
		"type":   true,
		"name":   true,
		"id":     true,
		"label":  true,
		"scope":  true,
	}
)

func (n *networkRouter) getNetworksList(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	filter := r.Form.Get("filters")
	netFilters, err := filters.FromParam(filter)
	if err != nil {
		return err
	}

	if err := netFilters.Validate(acceptedNetworkFilters); err != nil {
		return err
	}

	list := []types.NetworkResource{}

	if nr, err := n.cluster.GetNetworks(); err == nil {
		list = append(list, nr...)
	}

	// Combine the network list returned by Docker daemon if it is not already
	// returned by the cluster manager
SKIP:
	for _, nw := range n.backend.GetNetworks() {
		for _, nl := range list {
			if nl.ID == nw.ID() {
				continue SKIP
			}
		}

		var nr *types.NetworkResource
		// Versions < 1.28 fetches all the containers attached to a network
		// in a network list api call. It is a heavy weight operation when
		// run across all the networks. Starting API version 1.28, this detailed
		// info is available for network specific GET API (equivalent to inspect)
		if versions.LessThan(httputils.VersionFromContext(ctx), "1.28") {
			nr = n.buildDetailedNetworkResources(nw, false)
		} else {
			nr = n.buildNetworkResource(nw)
		}
		list = append(list, *nr)
	}

	list, err = filterNetworks(list, netFilters)
	if err != nil {
		return err
	}
	return httputils.WriteJSON(w, http.StatusOK, list)
}

func (n *networkRouter) getNetwork(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	term := vars["id"]
	var (
		verbose bool
		err     error
	)
	if v := r.URL.Query().Get("verbose"); v != "" {
		if verbose, err = strconv.ParseBool(v); err != nil {
			err = fmt.Errorf("invalid value for verbose: %s", v)
			return errors.NewBadRequestError(err)
		}
	}
	scope := r.URL.Query().Get("scope")

	isMatchingScope := func(scope, term string) bool {
		if term != "" {
			return scope == term
		}
		return true
	}

	// In case multiple networks have duplicate names, return error.
	// TODO (yongtang): should we wrap with version here for backward compatibility?

	// First find based on full ID, return immediately once one is found.
	// If a network appears both in swarm and local, assume it is in local first

	// For full name and partial ID, save the result first, and process later
	// in case multiple records was found based on the same term
	listByFullName := map[string]types.NetworkResource{}
	listByPartialID := map[string]types.NetworkResource{}

	nw := n.backend.GetNetworks()
	for _, network := range nw {
		if network.ID() == term && isMatchingScope(network.Info().Scope(), scope) {
			return httputils.WriteJSON(w, http.StatusOK, *n.buildDetailedNetworkResources(network, verbose))
		}
		if network.Name() == term && isMatchingScope(network.Info().Scope(), scope) {
			// No need to check the ID collision here as we are still in
			// local scope and the network ID is unique in this scope.
			listByFullName[network.ID()] = *n.buildDetailedNetworkResources(network, verbose)
		}
		if strings.HasPrefix(network.ID(), term) && isMatchingScope(network.Info().Scope(), scope) {
			// No need to check the ID collision here as we are still in
			// local scope and the network ID is unique in this scope.
			listByPartialID[network.ID()] = *n.buildDetailedNetworkResources(network, verbose)
		}
	}

	nr, _ := n.cluster.GetNetworks()
	for _, network := range nr {
		if network.ID == term && isMatchingScope(network.Scope, scope) {
			return httputils.WriteJSON(w, http.StatusOK, network)
		}
		if network.Name == term && isMatchingScope(network.Scope, scope) {
			// Check the ID collision as we are in swarm scope here, and
			// the map (of the listByFullName) may have already had a
			// network with the same ID (from local scope previously)
			if _, ok := listByFullName[network.ID]; !ok {
				listByFullName[network.ID] = network
			}
		}
		if strings.HasPrefix(network.ID, term) && isMatchingScope(network.Scope, scope) {
			// Check the ID collision as we are in swarm scope here, and
			// the map (of the listByPartialID) may have already had a
			// network with the same ID (from local scope previously)
			if _, ok := listByPartialID[network.ID]; !ok {
				listByPartialID[network.ID] = network
			}
		}
	}

	// Find based on full name, returns true only if no duplicates
	if len(listByFullName) == 1 {
		for _, v := range listByFullName {
			return httputils.WriteJSON(w, http.StatusOK, v)
		}
	}
	if len(listByFullName) > 1 {
		return fmt.Errorf("network %s is ambiguous (%d matches found based on name)", term, len(listByFullName))
	}

	// Find based on partial ID, returns true only if no duplicates
	if len(listByPartialID) == 1 {
		for _, v := range listByPartialID {
			return httputils.WriteJSON(w, http.StatusOK, v)
		}
	}
	if len(listByPartialID) > 1 {
		return fmt.Errorf("network %s is ambiguous (%d matches found based on ID prefix)", term, len(listByPartialID))
	}

	return libnetwork.ErrNoSuchNetwork(term)
}

func (n *networkRouter) postNetworkCreate(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var create types.NetworkCreateRequest

	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	if err := httputils.CheckForJSON(r); err != nil {
		return err
	}

	if err := json.NewDecoder(r.Body).Decode(&create); err != nil {
		return err
	}

	if nws, err := n.cluster.GetNetworksByName(create.Name); err == nil && len(nws) > 0 {
		return libnetwork.NetworkNameError(create.Name)
	}

	nw, err := n.backend.CreateNetwork(create)
	if err != nil {
		var warning string
		if _, ok := err.(libnetwork.NetworkNameError); ok {
			// check if user defined CheckDuplicate, if set true, return err
			// otherwise prepare a warning message
			if create.CheckDuplicate {
				return libnetwork.NetworkNameError(create.Name)
			}
			warning = libnetwork.NetworkNameError(create.Name).Error()
		}

		if _, ok := err.(libnetwork.ManagerRedirectError); !ok {
			return err
		}
		id, err := n.cluster.CreateNetwork(create)
		if err != nil {
			return err
		}
		nw = &types.NetworkCreateResponse{
			ID:      id,
			Warning: warning,
		}
	}

	return httputils.WriteJSON(w, http.StatusCreated, nw)
}

func (n *networkRouter) postNetworkConnect(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var connect types.NetworkConnect
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	if err := httputils.CheckForJSON(r); err != nil {
		return err
	}

	if err := json.NewDecoder(r.Body).Decode(&connect); err != nil {
		return err
	}

	return n.backend.ConnectContainerToNetwork(connect.Container, vars["id"], connect.EndpointConfig)
}

func (n *networkRouter) postNetworkDisconnect(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	var disconnect types.NetworkDisconnect
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	if err := httputils.CheckForJSON(r); err != nil {
		return err
	}

	if err := json.NewDecoder(r.Body).Decode(&disconnect); err != nil {
		return err
	}

	return n.backend.DisconnectContainerFromNetwork(disconnect.Container, vars["id"], disconnect.Force)
}

func (n *networkRouter) deleteNetwork(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}
	if _, err := n.cluster.GetNetwork(vars["id"]); err == nil {
		if err = n.cluster.RemoveNetwork(vars["id"]); err != nil {
			return err
		}
		w.WriteHeader(http.StatusNoContent)
		return nil
	}
	if err := n.backend.DeleteNetwork(vars["id"]); err != nil {
		return err
	}
	w.WriteHeader(http.StatusNoContent)
	return nil
}

func (n *networkRouter) buildNetworkResource(nw libnetwork.Network) *types.NetworkResource {
	r := &types.NetworkResource{}
	if nw == nil {
		return r
	}

	info := nw.Info()
	r.Name = nw.Name()
	r.ID = nw.ID()
	r.Created = info.Created()
	r.Scope = info.Scope()
	r.Driver = nw.Type()
	r.EnableIPv6 = info.IPv6Enabled()
	r.Internal = info.Internal()
	r.Attachable = info.Attachable()
	r.Ingress = info.Ingress()
	r.Options = info.DriverOptions()
	r.Containers = make(map[string]types.EndpointResource)
	buildIpamResources(r, info)
	r.Labels = info.Labels()
	r.ConfigOnly = info.ConfigOnly()

	if cn := info.ConfigFrom(); cn != "" {
		r.ConfigFrom = network.ConfigReference{Network: cn}
	}

	peers := info.Peers()
	if len(peers) != 0 {
		r.Peers = buildPeerInfoResources(peers)
	}

	return r
}

func (n *networkRouter) buildDetailedNetworkResources(nw libnetwork.Network, verbose bool) *types.NetworkResource {
	if nw == nil {
		return &types.NetworkResource{}
	}

	r := n.buildNetworkResource(nw)
	epl := nw.Endpoints()
	for _, e := range epl {
		ei := e.Info()
		if ei == nil {
			continue
		}
		sb := ei.Sandbox()
		tmpID := e.ID()
		key := "ep-" + tmpID
		if sb != nil {
			key = sb.ContainerID()
		}

		r.Containers[key] = buildEndpointResource(tmpID, e.Name(), ei)
	}
	if !verbose {
		return r
	}
	services := nw.Info().Services()
	r.Services = make(map[string]network.ServiceInfo)
	for name, service := range services {
		tasks := []network.Task{}
		for _, t := range service.Tasks {
			tasks = append(tasks, network.Task{
				Name:       t.Name,
				EndpointID: t.EndpointID,
				EndpointIP: t.EndpointIP,
				Info:       t.Info,
			})
		}
		r.Services[name] = network.ServiceInfo{
			VIP:          service.VIP,
			Ports:        service.Ports,
			Tasks:        tasks,
			LocalLBIndex: service.LocalLBIndex,
		}
	}
	return r
}

func buildPeerInfoResources(peers []networkdb.PeerInfo) []network.PeerInfo {
	peerInfo := make([]network.PeerInfo, 0, len(peers))
	for _, peer := range peers {
		peerInfo = append(peerInfo, network.PeerInfo{
			Name: peer.Name,
			IP:   peer.IP,
		})
	}
	return peerInfo
}

func buildIpamResources(r *types.NetworkResource, nwInfo libnetwork.NetworkInfo) {
	id, opts, ipv4conf, ipv6conf := nwInfo.IpamConfig()

	ipv4Info, ipv6Info := nwInfo.IpamInfo()

	r.IPAM.Driver = id

	r.IPAM.Options = opts

	r.IPAM.Config = []network.IPAMConfig{}
	for _, ip4 := range ipv4conf {
		if ip4.PreferredPool == "" {
			continue
		}
		iData := network.IPAMConfig{}
		iData.Subnet = ip4.PreferredPool
		iData.IPRange = ip4.SubPool
		iData.Gateway = ip4.Gateway
		iData.AuxAddress = ip4.AuxAddresses
		r.IPAM.Config = append(r.IPAM.Config, iData)
	}

	if len(r.IPAM.Config) == 0 {
		for _, ip4Info := range ipv4Info {
			iData := network.IPAMConfig{}
			iData.Subnet = ip4Info.IPAMData.Pool.String()
			iData.Gateway = ip4Info.IPAMData.Gateway.IP.String()
			r.IPAM.Config = append(r.IPAM.Config, iData)
		}
	}

	hasIpv6Conf := false
	for _, ip6 := range ipv6conf {
		if ip6.PreferredPool == "" {
			continue
		}
		hasIpv6Conf = true
		iData := network.IPAMConfig{}
		iData.Subnet = ip6.PreferredPool
		iData.IPRange = ip6.SubPool
		iData.Gateway = ip6.Gateway
		iData.AuxAddress = ip6.AuxAddresses
		r.IPAM.Config = append(r.IPAM.Config, iData)
	}

	if !hasIpv6Conf {
		for _, ip6Info := range ipv6Info {
			if ip6Info.IPAMData.Pool == nil {
				continue
			}
			iData := network.IPAMConfig{}
			iData.Subnet = ip6Info.IPAMData.Pool.String()
			iData.Gateway = ip6Info.IPAMData.Gateway.String()
			r.IPAM.Config = append(r.IPAM.Config, iData)
		}
	}
}

func buildEndpointResource(id string, name string, info libnetwork.EndpointInfo) types.EndpointResource {
	er := types.EndpointResource{}

	er.EndpointID = id
	er.Name = name
	ei := info
	if ei == nil {
		return er
	}

	if iface := ei.Iface(); iface != nil {
		if mac := iface.MacAddress(); mac != nil {
			er.MacAddress = mac.String()
		}
		if ip := iface.Address(); ip != nil && len(ip.IP) > 0 {
			er.IPv4Address = ip.String()
		}

		if ipv6 := iface.AddressIPv6(); ipv6 != nil && len(ipv6.IP) > 0 {
			er.IPv6Address = ipv6.String()
		}
	}
	return er
}

func (n *networkRouter) postNetworksPrune(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	if err := httputils.ParseForm(r); err != nil {
		return err
	}

	pruneFilters, err := filters.FromParam(r.Form.Get("filters"))
	if err != nil {
		return err
	}

	pruneReport, err := n.backend.NetworksPrune(ctx, pruneFilters)
	if err != nil {
		return err
	}
	return httputils.WriteJSON(w, http.StatusOK, pruneReport)
}
