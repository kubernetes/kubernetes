package clc

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/types"
	// activating stack-dump code requires this
	// "runtime/debug"
)

//// clcProviderLB implements the LoadBalancer interface (from pkg/cloudprovider/cloud.go)
//	type LoadBalancer interface {
//		GetLoadBalancer(name, region string) (status *api.LoadBalancerStatus, exists bool, err error)
//
//		EnsureLoadBalancer(name, region string, loadBalancerIP net.IP, ports []*api.ServicePort,
//			hosts []string, serviceName types.NamespacedName, affinityType api.ServiceAffinity,
//			annotations ServiceAnnotation) (*api.LoadBalancerStatus, error)
//
//		UpdateLoadBalancer(name, region string, hosts []string) error
//		EnsureLoadBalancerDeleted(name, region string) error
//	}

type clcProviderLB struct {
	// nonpointer because CenturyLinkClient is an interface
	clcClient CenturyLinkClient // the one owned by CLCCloud

	// any other LB info could go here
}

const (
	K8S_LB_PREFIX = "Kubernetes:"
)

func makeProviderLB(clc CenturyLinkClient) *clcProviderLB {
	if clc == nil {
		return nil
	}

	return &clcProviderLB{
		clcClient: clc,
	}
}

// returns LBDetails, or nil if the name was simply not found, or error if something failed
func findLoadBalancerInstance(clcClient CenturyLinkClient, name, region string, reason string) (*LoadBalancerDetails, error) {
	// name is the Kubernetes-assigned name.  EnsureLoadBalancer assigns that to our LoadBalancerDetails.Name

	summaries, err := clcClient.listAllLB()
	if err != nil {
		glog.Info(fmt.Sprintf("CLC.findLoadBalancerInstance could not get LB list: dc=%s, name=%s, reason=%s, err=%s", region, name, reason, err.Error()))
		return nil, err
	}

	for _, lbSummary := range summaries {
		if (lbSummary.DataCenter == region) && ((lbSummary.Name == name) || (lbSummary.Name == (K8S_LB_PREFIX + name))) {
			ret, e := clcClient.inspectLB(lbSummary.DataCenter, lbSummary.LBID)
			if e != nil {
				glog.Info(fmt.Sprintf("CLC.findLoadBalancerInstance could not inspect LB: dc=%s, LBID=%s, err=%s", lbSummary.DataCenter, lbSummary.LBID, e.Error()))
				return nil, e
			}

			return ret, nil
		}
	}

	// not an error.  K asks for the LB on services that don't have one, just to verify that there isn't one.
	return nil, nil
}

// GetLoadBalancer returns whether the specified load balancer exists, and if so, what its status is.
// NB: status is a single Ingress spec, has nothing to do with operational status
func (clc *clcProviderLB) GetLoadBalancer(name, region string) (status *api.LoadBalancerStatus, exists bool, err error) {

	lb, err := findLoadBalancerInstance(clc.clcClient, name, region, "K8S.LB.GetLoadBalancer")
	if err != nil {
		return nil, false, err
	}

	if lb == nil { // not found is a legitimate case
		return nil, false, nil
	}

	return toStatus(lb.PublicIP), true, nil
}

// EnsureLoadBalancer creates a new load balancer 'name', or updates the existing one. Returns the status of the balancer
// For an LB identified by region,name (or created that way, with name=LBID returned) (and possibly desc=serviceName)
//	create a pool for every entry in ports, using serviceAffinity.  Equates api.ServicePort.Port to PoolDetails.IncomingPort
//	for every one of those pools, add a node list from the hosts array
func (clc *clcProviderLB) EnsureLoadBalancer(name, region string,
	loadBalancerIP string, // ignore this.  AWS actually returns error if it's non-nil
	ports []api.ServicePort, hosts []string, serviceName types.NamespacedName,
	affinityType api.ServiceAffinity,
	annotations map[string]string) (*api.LoadBalancerStatus, error) {

	glog.Info("CLC: inside EnsureLoadBalancer")

	lb, e := findLoadBalancerInstance(clc.clcClient, name, region, "K8S.LB.EnsureLoadBalancer")
	if e != nil { // couldn't talk to the datacenter
		return nil, e
	}

	if lb == nil { // make a new LB
		glog.Info(fmt.Sprintf("CLC.EnsureLoadBalancer: creating LB, dc=%s, name=%s", region, name))
		inf, e := clc.clcClient.createLB(region, K8S_LB_PREFIX+name, serviceName.String())
		if e != nil {
			glog.Info("CLC.EnsureLoadBalancer: failed to create new LB: err=%s", e.Error())
			return nil, e
		}

		glog.Info(fmt.Sprintf("CLC.EnsureLoadBalancer: created LB, ID=%s", inf.LBID))
		lb, e = clc.clcClient.inspectLB(region, inf.LBID)
		if e != nil {
			glog.Info(fmt.Sprintf("CLC.EnsureLoadBalancer: could not inspect new LB: dc=%s LBID=%s err=%s", region, inf.LBID, e.Error()))
			return nil, e
		}
	}

	// either way, we now have an LB that answers to (name,region).  Sanity-check its status
	if (lb.Status == "FAILED") || (lb.Status == "DELETED") { // definitely failed
		return nil, clcError(fmt.Sprintf("EnsureLoadBalancer: failed, lbid=%s", lb.LBID))

	} else if (lb.Status == "READY") || (lb.Status == "COMPLETE") {
		if lb.PublicIP == "" {
			return nil, clcError(fmt.Sprintf("EnsureLoadBalancer: no publicIP, lbid=%s", lb.LBID))
		}

		// else success, we have a good status and public IP, fall through to the config code below

	} else if (lb.Status == "UNDER_CONSTRUCTION") || (lb.Status == "UPDATING_CONFIGURATION") || (lb.Status == "ACTIVE") {
		// is returning an error correct here?
		return nil, clcError(fmt.Sprintf("EnsureLoadBalancer: delayed, lbid=%s", lb.LBID))

	} else { // ??
		return nil, clcError(fmt.Sprintf("EnsureLoadBalancer: bad status, lbid=%s, status=%s", lb.LBID, lb.Status))
	}

	existingPoolCount := len(lb.Pools) // now configure it with ports and hosts.
	desiredPoolCount := len(ports)

	addPorts := make([]api.ServicePort, 0, desiredPoolCount) // ServicePort specs to make new pools out of
	deletePools := make([]PoolDetails, 0, existingPoolCount) // unwanted existing PoolDetails to destroy

	fromPorts := make([]api.ServicePort, 0, desiredPoolCount) // existing port/pool pairs to adjust so they match
	toPools := make([]PoolDetails, 0, desiredPoolCount)

	for _, port := range ports {
		bMatched := false
		for _, pool := range lb.Pools {
			if port.Port == pool.IncomingPort { // use ServicePort.Port==PoolDetails.IncomingPort to match
				fromPorts = append(fromPorts, port)
				toPools = append(toPools, pool) // insert fromPorts/toPool as a pair only
				bMatched = true
				break
			}
		}

		if !bMatched {
			addPorts = append(addPorts, port)
		}
	}

	for _, pool := range lb.Pools {
		bMatched := false
		for _, port := range ports {
			if port.Port == pool.IncomingPort { // would have been sent to fromPorts/toPool above
				bMatched = true
				break
			}
		}

		if !bMatched {
			deletePools = append(deletePools, pool)
		}
	}

	for _, creationPort := range addPorts {
		desiredPool := makePoolDetailsFromServicePort(lb.LBID, &creationPort, hosts, affinityType)
		_, eCreate := clc.clcClient.createPool(lb.DataCenter, lb.LBID, desiredPool)
		if eCreate != nil {
			glog.Info(fmt.Sprintf("CLC.EnsureLoadBalancer: failed creating pool, dc=%s, LBID=%s, PoolID=%s, err=%s", lb.DataCenter, lb.LBID, desiredPool.PoolID, eCreate.Error()))
			return nil, eCreate
		}

		glog.Info(fmt.Sprintf("CLC.EnsureLoadBalancer: created pool, dc=%s, LBID=%s, PoolID=%s", lb.DataCenter, lb.LBID, desiredPool.PoolID))
	}

	for _, deletionPool := range deletePools {
		eDelete := clc.clcClient.deletePool(lb.DataCenter, lb.LBID, deletionPool.PoolID)
		if eDelete != nil {
			glog.Info(fmt.Sprintf("CLC.EnsureLoadBalancer: failed deleting pool, dc=%s, LBID=%s, PoolID=%s, err=%s", lb.DataCenter, lb.LBID, deletionPool.PoolID, eDelete.Error()))
			return nil, eDelete
		}

		glog.Info(fmt.Sprintf("CLC.EnsureLoadBalancer: deleted pool, dc=%s, LBID=%s, PoolID=%s", lb.DataCenter, lb.LBID, deletionPool.PoolID))
	}

	for idx := range fromPorts {
		desiredPort := &fromPorts[idx] // ServicePort, what K wants
		existingPool := &toPools[idx]  // PoolDetails, what CL has now

		desiredPool := makePoolDetailsFromServicePort(lb.LBID, desiredPort, hosts, affinityType)
		_, eConform := conformPoolDetails(clc.clcClient, lb.DataCenter, desiredPool, existingPool)

		if eConform != nil { // conformPoolDetails already did the logging
			return nil, eConform
		}
	}

	glog.Info("CLC.EnsureLoadBalancer returning successfully")
	return toStatus(lb.PublicIP), nil // ingress is the actual lb.PublicIP, not the one passed in to this func
}

func makePoolDetailsFromServicePort(lbid string, srcPort *api.ServicePort, hosts []string, affinity api.ServiceAffinity) *PoolDetails {
	persist := "none"
	if affinity == "ClientIP" { // K. calls it this
		persist = "source_ip" // CL calls it that
	}

	return &PoolDetails{
		PoolID:       "", // createPool to fill in
		LBID:         lbid,
		IncomingPort: srcPort.Port,
		Method:       "roundrobin",
		Persistence:  persist,
		TimeoutMS:    99999, // and what should the default be?
		Mode:         "tcp",
		Health:       &HealthCheck{UnhealthyThreshold: 2, HealthyThreshold: 2, IntervalSeconds: 5, TargetPort: srcPort.NodePort, Mode: "TCP"},
		Nodes:        makeNodeListFromHosts(hosts, srcPort.NodePort),
	}
}

func conformPoolDetails(clcClient CenturyLinkClient, dc string, desiredPool, existingPool *PoolDetails) (bool, error) {

	desiredPool.PoolID = existingPool.PoolID
	desiredPool.LBID = existingPool.LBID
	desiredPool.IncomingPort = existingPool.IncomingPort

	bMatch := true
	if (desiredPool.Method != existingPool.Method) || (desiredPool.Persistence != existingPool.Persistence) {
		bMatch = false
	} else if (desiredPool.TimeoutMS != existingPool.TimeoutMS) || (desiredPool.Mode != existingPool.Mode) {
		bMatch = false
	} else if len(desiredPool.Nodes) != len(existingPool.Nodes) {
		bMatch = false
	} else {
		for idx := range desiredPool.Nodes {
			if desiredPool.Nodes[idx].TargetIP != existingPool.Nodes[idx].TargetIP {
				bMatch = false
			} else if desiredPool.Nodes[idx].TargetPort != existingPool.Nodes[idx].TargetPort {
				bMatch = false
			}
		}
	}

	if bMatch {
		return false, nil // no changes made, no error
	}

	_, e := clcClient.updatePool(dc, desiredPool.LBID, desiredPool)
	glog.Info(fmt.Sprintf("CLC.EnsureLoadBalancer(conformPoolDetails) updated pool: dc=%s, LBID=%s, PoolID=%s", dc, desiredPool.LBID, desiredPool.PoolID))
	return true, e
}

func toStatus(ip string) *api.LoadBalancerStatus {
	var ingress api.LoadBalancerIngress
	ingress.IP = ip

	ret := api.LoadBalancerStatus{}
	ret.Ingress = []api.LoadBalancerIngress{ingress}

	return &ret
}

// UpdateLoadBalancer updates hosts under the specified load balancer.  For every pool, this rewrites the hosts list.
// We require that every pool must have a nonempty nodes list, deleting pools if necessary to enforce this.
func (clc *clcProviderLB) UpdateLoadBalancer(name, region string, hosts []string) error {
	glog.Info(fmt.Sprintf("CLC.UpdateLoadBalancer entry: dc=%s, name=%s", region, name))

	lb, e := findLoadBalancerInstance(clc.clcClient, name, region, "K8S.LB.UpdateLoadBalancer")
	if e != nil {
		return e // can't see it?  Can't update it.
	}

	if lb == nil {
		return clcError(fmt.Sprintf("UpdateLoadBalancer could not find instance: dc=%s, name=%s", region, name))
	}

	for _, pool := range lb.Pools {

		if (hosts == nil) || (len(hosts) == 0) { // must delete pool
			glog.Info(fmt.Sprintf("CLC.UpdateLoadBalancer deleting pool (no hosts): dc=%s, LBID=%s, PoolID=%s", lb.DataCenter, lb.LBID, pool.PoolID))
			err := clc.clcClient.deletePool(lb.DataCenter, lb.LBID, pool.PoolID)
			if err != nil {
				glog.Info("CLC.UpdateLoadBalancer could not delete pool: dc=%s, LBID=%s, PoolID=%s, err=%s", lb.DataCenter, lb.LBID, pool.PoolID, err.Error())
				return err // and punt on any other pools.  This LB is in bad shape now.
			}

		} else { // update hosts in the pool, using port number from the existing hosts

			if (pool.Nodes == nil) || (len(pool.Nodes) == 0) { // no nodes to get targetPort from
				glog.Info(fmt.Sprintf("CLC.UpdateLoadBalancer deleting pool (no port): dc=%s, LBID=%s, PoolID=%s", lb.DataCenter, lb.LBID, pool.PoolID))
				err := clc.clcClient.deletePool(lb.DataCenter, lb.LBID, pool.PoolID)
				if err != nil {
					glog.Info("CLC.UpdateLoadBalancer could not delete pool: dc=%s, LBID=%s, PoolID=%s, err=%s", lb.DataCenter, lb.LBID, pool.PoolID, err.Error())
					return err
				}
			} else { // normal case, draw targetPort from an existing node and rewrite the pool

				targetPort := pool.Nodes[0].TargetPort
				nodelist := makeNodeListFromHosts(hosts, targetPort)

				pool.Nodes = nodelist
				glog.Info(fmt.Sprintf("CLC.UpdateLoadBalancer updating pool: dc=%s, LBID=%s, PoolID=%s", lb.DataCenter, lb.LBID, pool.PoolID))
				_, err := clc.clcClient.updatePool(lb.DataCenter, lb.LBID, &pool)
				if err != nil {
					glog.Info("CLC.UpdateLoadBalancer could not update pool: dc=%s, LBID=%s, PoolID=%s, err=%s", lb.DataCenter, lb.LBID, pool.PoolID, err.Error())
					return err
				}
			}
		}
	}

	return nil
}

func makeNodeListFromHosts(hosts []string, portnum int) []PoolNode {
	nNodes := len(hosts)
	nodelist := make([]PoolNode, nNodes, nNodes)
	for idx, hostnode := range hosts {
		nodelist[idx] = PoolNode{
			TargetIP:   hostnode,
			TargetPort: portnum,
		}
	}

	return nodelist
}

// EnsureLoadBalancerDeleted deletes the specified load balancer if it
// exists, returning nil if the load balancer specified either didn't exist or
// was successfully deleted.
// This construction is useful because many cloud providers' load balancers
// have multiple underlying components, meaning a Get could say that the LB
// doesn't exist even if some part of it is still laying around.
func (clc *clcProviderLB) EnsureLoadBalancerDeleted(name, region string) error {
	glog.Info("CLC.EnsureLoadBalancerDeleted entry")

	lb, e := findLoadBalancerInstance(clc.clcClient, name, region, "K8S.LB.EnsureLoadBalancerDeleted")
	if e != nil {
		return clcError("EnsureLoadBalancerDeleted failed to search for instances")
	}

	if lb == nil {
		return nil // no such LB, so nothing to do
	}

	glog.Info(fmt.Sprintf("CLC.EnsureLoadBalancerDeleted deleting LB: dc=%s, LBID=%s", lb.DataCenter, lb.LBID))
	_, e = clc.clcClient.deleteLB(lb.DataCenter, lb.LBID)

	if e != nil {
		glog.Info(fmt.Sprintf("CLC.EnsureLoadBalancerDeleted could not delete LB: dc=%s, LBID=%s, err=%s", lb.DataCenter, lb.LBID, e.Error()))
		return e
	}

	return nil // success
}

//////////////// Notes about mapping the Kubernetes data model to the CLC LBAAS
//
// (selecting a particular LB)
// K:name != CL:LBID (also CL:LB.LBID)    (K. assigns its own names, CL assigns LBID)
// K:region <--> CL:datacenter ID
//             CL:accountAlias comes from login creds
//
// (properties of the LB)
// K:IP <--> CL:PublicIP
//             CL:Name (details unknown, needs to be unique)
// K:serviceName <--> CL:Description (probably)
// K:affinity(ClientIP or None) <--> CL:PoolDetails.persistence(source_ip or none)
// K:annotations(ignore)
// K:hosts (array of strings) <--> CL:NodeDetails.IP (same hosts[] for every PoolDetails.Nodes[])
// K:ports (array of ServicePort) <--> CL:Pools (array of PoolDetails)
//
// (properties of a K:ServicePort<-->CL:PoolDetails)
// K:Name <--> CL:PoolID
// K:Protocol(TCP or UDP) (always TCP)
// K:Port <--> CL:Port
// K:TargetPort(ignore)
// K:NodePort <--> CL:NodeDetails.PrivatePort
//              CL:method(roundrobin or leastconn) always roundrobin
//              CL:mode:(tcp or http) always tcp
//              CL:timeout
