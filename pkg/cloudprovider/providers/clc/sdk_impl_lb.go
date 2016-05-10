/*
Copyright 2014 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package clc

import (
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"
)

//// api use involves calls to both addresses
var clcServerapiV2 = "api.ctl.io"               // URL form  https://api.ctl.io/v2/<resource>/<accountAlias>
var clcServerLBBETA = "api.loadbalancer.ctl.io" // URL form  https://api.loadbalancer.ctl.io/<accountAlias>/<datacenter>/loadbalancers

//// auth methods
func implMakeCLC() CenturyLinkClient {
	return clcImpl{
		creds: MakeEmptyCreds(clcServerapiV2, "/v2/authentication/login"),
	}
}

//// clcImpl is the internal layer that knows what HTTP calls to make

type clcImpl struct { // implements CenturyLinkClient
	creds Credentials // make it once, never reassign it
}

func (clc clcImpl) GetCreds() Credentials {
	return clc.creds
}

//////////////// clc method: listAllDC()

type dcNamesJSON struct {
	ID   string `json:"id"`
	Name string `json:"name"` // NB: capitalizing Name is required in Go
}

func (clc clcImpl) listAllDC() ([]DataCenterName, error) {

	uri := fmt.Sprintf("/v2/datacenters/%s", clc.creds.GetAccount())
	dcret := make([]*dcNamesJSON, 0)

	_, err := simpleGET(clcServerapiV2, uri, clc.creds, &dcret)
	if err != nil {
		return nil, err
	}

	ret := make([]DataCenterName, len(dcret))
	for idx, dc := range dcret {
		ret[idx] = DataCenterName{DCID: strings.ToUpper(dc.ID), Name: dc.Name} // api returns ID in lowercase but requires it in upper
	}

	return ret, nil
}

//////////////// clc method: listAllLB()

type lbListingDetailsJSON struct {
	LBID        string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	PublicIP    string `json:"publicIPAddress"`
	//	PrivateIP string `json:"privateIPAddress"`
	//	Pools []lbPoolDetailsJSON // ??json format?
	//	Status string `json:"status"`
	//	AccountAlias string `json:"accountAlias"`
	DataCenter string `json:"dataCenter"`
	// omit keepalivedRouterId and version
}

type lbListingWrapperJSON struct {
	Values []lbListingDetailsJSON `json:"values"`
}

func (clc clcImpl) listAllLB() ([]LoadBalancerSummary, error) {

	uri := fmt.Sprintf("/%s/loadbalancers", clc.creds.GetAccount())
	apiret := &lbListingWrapperJSON{}

	_, err := simpleGET(clcServerLBBETA, uri, clc.creds, &apiret)
	if err != nil {
		return nil, err
	}

	ret := make([]LoadBalancerSummary, len(apiret.Values))
	for idx, lb := range apiret.Values {
		ret[idx] = LoadBalancerSummary{
			LBID:        lb.LBID,
			Name:        lb.Name,
			Description: lb.Description,
			PublicIP:    lb.PublicIP,
			DataCenter:  lb.DataCenter,
		}
	}

	return ret, nil
}

//////////////// clc method: createLB()

type linkJSON struct {
	Rel  string `json:"rel,omitempty"`
	Href string `json:"href,omitempty"`
	ID   string `json:"resourceId,omitempty"`
}

type apiLinks []linkJSON

type lbCreateRequestJSON struct {
	LBID           string   `json:"id"`
	Status         string   `json:"status"`
	Description    string   `json:"description"`
	RequestDate    int64    `json:"requestDate"`
	CompletionDate int64    `json:"completionDate"`
	Links          apiLinks `json:"links"`
}

func rawCreateLB(clcCreds Credentials, dc string, lbname string, desc string) (*LoadBalancerCreationInfo, error) {

	uri := fmt.Sprintf("/%s/%s/loadbalancers", clcCreds.GetAccount(), dc)
	apiret := &lbCreateRequestJSON{}

	body := fmt.Sprintf("{ \"name\":\"%s\", \"description\":\"%s\" }", lbname, desc)

	_, err := simplePOST(clcServerLBBETA, uri, clcCreds, body, apiret)

	if err != nil {
		return nil, err
	}

	return &LoadBalancerCreationInfo{
		DataCenter:  dc,
		LBID:        findLinkLB(&apiret.Links, "loadbalancer"),
		RequestTime: apiret.RequestDate,
	}, nil
}

func (clc clcImpl) createLB(dc string, lbname string, desc string) (*LoadBalancerCreationInfo, error) {
	timeStarted := time.Now()
	LB, err := rawCreateLB(clc.creds, dc, lbname, desc)
	if err != nil {
		return nil, err
	}

	deadline := time.Now().Add(2 * time.Minute)
	for { // infinite loop.  Wait until the newly created LB actually appears in the list.
		time.Sleep(2 * time.Second)

		list, listErr := clc.listAllLB()
		if listErr != nil {
			continue
		}

		for _, entry := range list {
			if (entry.DataCenter == dc) && (entry.LBID == LB.LBID) { // success
				glog.Info(fmt.Sprintf("createLB complete, duration=%d seconds", (time.Now().Sub(timeStarted) / time.Second)))
				return LB, nil
			}
		}

		if deadline.Before(time.Now()) {
			return nil, fmt.Errorf(fmt.Sprintf("createLB timed out, LBID=%s", LB.LBID))
		}
	}
}

//////////////// clc method: inspectLB()

type nodeJSON struct {
	TargetIP   string `json:"ipAddress"`
	TargetPort int    `json:"privatePort"`
}

type apiNodes []nodeJSON

type poolJSON struct {
	PoolID       string           `json:"id"`
	IncomingPort int              `json:"port"`
	Method       string           `json:"loadBalancingMethod"`
	Persistence  string           `json:"persistence"`
	TimeoutMS    int64            `json:"idleTimeout"`
	Mode         string           `json:"loadBalancingMode"`
	Health       *HealthCheckJSON `json:"healthCheck,omitempty"`
	Nodes        apiNodes         `json:"nodes"`
}

type apiPools []poolJSON

type lbDetailsJSON struct {
	LBID        string   `json:"id"`
	Status      string   `json:"status"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	PublicIP    string   `json:"publicIPAddress"`
	DataCenter  string   `json:"dataCenter"`
	Pools       apiPools `json:"pools"`
}

func (clc clcImpl) inspectLB(dc, lbid string) (*LoadBalancerDetails, error) {

	uri := fmt.Sprintf("/%s/%s/loadbalancers/%s", clc.creds.GetAccount(), dc, lbid)
	apiret := &lbDetailsJSON{}

	_, err := simpleGET(clcServerLBBETA, uri, clc.creds, apiret)
	if err != nil {
		return nil, err
	}

	var jsonPools []PoolDetails
	if apiret.Pools != nil {
		nPools := len(apiret.Pools)
		jsonPools = make([]PoolDetails, nPools, nPools)

		for idx, srcpool := range apiret.Pools {
			var jsonNodes []PoolNode
			if srcpool.Nodes != nil {
				nNodes := len(srcpool.Nodes)
				jsonNodes = make([]PoolNode, nNodes, nNodes)

				for idxNode, srcNode := range srcpool.Nodes {
					jsonNodes[idxNode] = PoolNode{
						TargetIP:   srcNode.TargetIP,
						TargetPort: srcNode.TargetPort,
					}
				}
			} else {
				jsonNodes = make([]PoolNode, 0, 0)
			}

			var poolHealth *HealthCheck
			if srcpool.Health != nil {
				poolHealth = &HealthCheck{
					UnhealthyThreshold: srcpool.Health.UnhealthyThreshold,
					HealthyThreshold:   srcpool.Health.HealthyThreshold,
					IntervalSeconds:    srcpool.Health.IntervalSeconds,
					TargetPort:         srcpool.Health.TargetPort,
					Mode:               srcpool.Health.Mode,
				}
			}

			jsonPools[idx] = PoolDetails{
				PoolID:       srcpool.PoolID,
				LBID:         apiret.LBID,
				IncomingPort: srcpool.IncomingPort,
				Method:       srcpool.Method,
				Health:       poolHealth,
				Persistence:  srcpool.Persistence,
				TimeoutMS:    srcpool.TimeoutMS,
				Mode:         srcpool.Mode,
				Nodes:        jsonNodes,
			}
		}
	} else {
		jsonPools = make([]PoolDetails, 0, 0)
	}

	return &LoadBalancerDetails{
		LBID:        apiret.LBID,
		Status:      apiret.Status,
		Name:        apiret.Name,
		Description: apiret.Description,
		PublicIP:    apiret.PublicIP,
		DataCenter:  apiret.DataCenter,
		Pools:       jsonPools,
	}, nil
}

//////////////// clc method: deleteLB()

// unneeded, since now deleteLB doesn't return any of this
type lbDeleteJSON struct {
	ID             string   `json:"id"`
	Status         string   `json:"status"`
	Description    string   `json:"description"`
	RequestTime    int64    `json:"requestDate"`
	CompletionTime int64    `json:"completionTime"`
	Links          apiLinks `json:"links"`
}

func findLinkLB(links *apiLinks, rel string) string {
	for _, link := range *links {
		if link.Rel == rel {
			return link.ID
		}
	}

	return "" // not found, consider returning err?
}

func (clc clcImpl) deleteLB(dc, lbid string) (bool, error) {

	uri := fmt.Sprintf("/%s/%s/loadbalancers/%s", clc.creds.GetAccount(), dc, lbid)
	apiret := &lbDeleteJSON{}

	code, err := simpleDELETE(clcServerLBBETA, uri, clc.creds, apiret)
	if err == nil { // ordinary success, LB was deleted
		return true, nil
	}

	if code == 404 { // was no such LB, which is the goal of a delete.  Call it success
		return false, nil // err=nil is what designates success, boolean return is how we got there
	}

	return false, err // bool return is meaningless, we may or may not have the LB
}

//////////////// clc method: createPool()

type NodeEntityJSON struct {
	TargetIP   string `json:"ipAddress"`
	TargetPort int    `json:"privatePort"`
}

type HealthCheckJSON struct {
	UnhealthyThreshold int    `json:"unhealthyThreshold"`
	HealthyThreshold   int    `json:"healthyThreshold"`
	IntervalSeconds    int    `json:"intervalSeconds"`
	TargetPort         int    `json:"targetPort"`
	Mode               string `json:"mode,omitempty"`
}

type PoolEntityJSON struct {
	PoolID       string           `json:"id,omitempty"` // createPool doesn't want to send an id
	IncomingPort int              `json:"port"`
	Method       string           `json:"loadBalancingMethod"`
	Health       *HealthCheckJSON `json:"healthCheck,omitEmpty"`
	Persistence  string           `json:"persistence"`
	TimeoutMS    int64            `json:"idleTimeout"`
	Mode         string           `json:"loadBalancingMode"`
	Nodes        []NodeEntityJSON `json:"nodes"`
}

func healthToJSON(src *HealthCheck) *HealthCheckJSON {
	if src == nil {
		return nil
	}

	return &HealthCheckJSON{
		UnhealthyThreshold: src.UnhealthyThreshold,
		HealthyThreshold:   src.HealthyThreshold,
		IntervalSeconds:    src.IntervalSeconds,
		TargetPort:         src.TargetPort,
		Mode:               src.Mode,
	}
}

func poolToJSON(pool *PoolDetails) *PoolEntityJSON {
	var jsonNodes []NodeEntityJSON
	if pool.Nodes != nil {
		nNodes := len(pool.Nodes)
		jsonNodes = make([]NodeEntityJSON, nNodes, nNodes)

		for idx, srcNode := range pool.Nodes {
			jsonNodes[idx] = NodeEntityJSON{
				TargetIP:   srcNode.TargetIP,
				TargetPort: srcNode.TargetPort,
			}
		}
	} else {
		jsonNodes = make([]NodeEntityJSON, 0, 0)
	}

	return &PoolEntityJSON{
		PoolID:       pool.PoolID,
		IncomingPort: pool.IncomingPort,
		Method:       pool.Method,
		Health:       healthToJSON(pool.Health),
		Persistence:  pool.Persistence,
		TimeoutMS:    pool.TimeoutMS,
		Mode:         pool.Mode,
		Nodes:        jsonNodes,
	}
}

type CreatePoolResponseJSON struct {
	RequestID      string   `json:"id"`
	Status         string   `json:"status"`
	Description    string   `json:"description"`
	RequestDate    int64    `json:"requestDate"`
	CompletionDate int64    `json:"completionDate"`
	Links          apiLinks `json:"links"` // Q: does the marshaling work if all we include is this one field?  All we need is links[rel="pool"].resourceID
}

func (clc clcImpl) createPool(dc, lbid string, newpool *PoolDetails) (*PoolCreationInfo, error) {
	uri := fmt.Sprintf("/%s/%s/loadbalancers/%s/pools", clc.creds.GetAccount(), dc, lbid)
	poolReq := poolToJSON(newpool)

	poolResp := &CreatePoolResponseJSON{}
	_, err := marshalledPOST(clcServerLBBETA, uri, clc.creds, poolReq, poolResp)
	if err != nil {
		return nil, err
	}

	poolID := findLinkLB(&poolResp.Links, "pool")
	if poolID == "" {
		return nil, clcError("could not determine ID of new pool")
	}

	// but clc.inspectPool fails if executed this quickly.  Return a PoolCreationInfo instead
	return &PoolCreationInfo{
		DataCenter:  dc,
		LBID:        lbid,
		PoolID:      poolID,
		RequestTime: poolResp.RequestDate,
	}, nil
}

//////////////// clc method: updatePool()
func (clc clcImpl) updatePool(dc, lbid string, newpool *PoolDetails) (*PoolDetails, error) {

	uri := fmt.Sprintf("/%s/%s/loadbalancers/%s/pools/%s", clc.creds.GetAccount(),
		dc, lbid, newpool.PoolID)

	updateReq := poolToJSON(newpool) // and ignore async-request return object
	_, err := marshalledPUT(clcServerLBBETA, uri, clc.creds, updateReq, nil)
	if err != nil {
		return nil, err
	}

	return clc.inspectPool(dc, lbid, newpool.PoolID)
}

//////////////// clc method: deletePool()
func (clc clcImpl) deletePool(dc, lbid string, poolID string) error {

	uri := fmt.Sprintf("/%s/%s/loadbalancers/%s/pools/%s", clc.creds.GetAccount(),
		dc, lbid, poolID)

	_, err := simpleDELETE(clcServerLBBETA, uri, clc.creds, nil)
	return err // no other return body
}

//////////////// clc method: inspectPool()
// not actually part of the LBAAS interface at this time.  Synthesized by returning just part of the inspectLB response

func (clc clcImpl) inspectPool(dc, lbid, poolid string) (*PoolDetails, error) {

	lbDetails, err := clc.inspectLB(dc, lbid)
	if err != nil {
		return nil, err
	}

	for _, p := range lbDetails.Pools {
		if p.PoolID == poolid {
			return &p, nil // success
		}
	}

	return nil, clcError("pool not found")
}

//////////////// end clc methods
