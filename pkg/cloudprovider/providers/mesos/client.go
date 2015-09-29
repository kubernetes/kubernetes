/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package mesos

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"sync"
	"time"

	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/detector"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
)

const defaultClusterName = "mesos"

var noLeadingMasterError = fmt.Errorf("there is no current leading master available to query")

type mesosClient struct {
	masterLock    sync.RWMutex
	master        string // host:port formatted address
	httpClient    *http.Client
	tr            *http.Transport
	initialMaster <-chan struct{} // signal chan, closes once an initial, non-nil master is found
	state         *stateCache
}

type slaveNode struct {
	hostname  string
	resources *api.NodeResources
}

type mesosState struct {
	clusterName string
	nodes       []*slaveNode
}

type stateCache struct {
	sync.Mutex
	expiresAt time.Time
	cached    *mesosState
	err       error
	ttl       time.Duration
	refill    func(context.Context) (*mesosState, error)
}

// reloadCache reloads the state cache if it has expired.
func (c *stateCache) reloadCache(ctx context.Context) {
	now := time.Now()
	c.Lock()
	defer c.Unlock()
	if c.expiresAt.Before(now) {
		log.V(4).Infof("Reloading cached Mesos state")
		c.cached, c.err = c.refill(ctx)
		c.expiresAt = now.Add(c.ttl)
	} else {
		log.V(4).Infof("Using cached Mesos state")
	}
}

// cachedState returns the cached Mesos state.
func (c *stateCache) cachedState(ctx context.Context) (*mesosState, error) {
	c.reloadCache(ctx)
	return c.cached, c.err
}

// clusterName returns the cached Mesos cluster name.
func (c *stateCache) clusterName(ctx context.Context) (string, error) {
	cached, err := c.cachedState(ctx)
	return cached.clusterName, err
}

// nodes returns the cached list of slave nodes.
func (c *stateCache) nodes(ctx context.Context) ([]*slaveNode, error) {
	cached, err := c.cachedState(ctx)
	return cached.nodes, err
}

func newMesosClient(
	md detector.Master,
	mesosHttpClientTimeout, stateCacheTTL time.Duration) (*mesosClient, error) {

	tr := &http.Transport{}
	httpClient := &http.Client{
		Transport: tr,
		Timeout:   mesosHttpClientTimeout,
	}
	return createMesosClient(md, httpClient, tr, stateCacheTTL)
}

func createMesosClient(
	md detector.Master,
	httpClient *http.Client,
	tr *http.Transport,
	stateCacheTTL time.Duration) (*mesosClient, error) {

	initialMaster := make(chan struct{})
	client := &mesosClient{
		httpClient:    httpClient,
		tr:            tr,
		initialMaster: initialMaster,
		state: &stateCache{
			ttl: stateCacheTTL,
		},
	}
	client.state.refill = client.pollMasterForState
	first := true
	if err := md.Detect(detector.OnMasterChanged(func(info *mesos.MasterInfo) {
		client.masterLock.Lock()
		defer client.masterLock.Unlock()
		if info == nil {
			client.master = ""
		} else if host := info.GetHostname(); host != "" {
			client.master = host
		} else {
			client.master = unpackIPv4(info.GetIp())
		}
		if len(client.master) > 0 {
			client.master = fmt.Sprintf("%s:%d", client.master, info.GetPort())
			if first {
				first = false
				close(initialMaster)
			}
		}
		log.Infof("cloud master changed to '%v'", client.master)
	})); err != nil {
		log.V(1).Infof("detector initialization failed: %v", err)
		return nil, err
	}
	return client, nil
}

func unpackIPv4(ip uint32) string {
	octets := make([]byte, 4, 4)
	binary.BigEndian.PutUint32(octets, ip)
	ipv4 := net.IP(octets)
	return ipv4.String()
}

// listSlaves returns a (possibly cached) list of slave nodes.
// Callers must not mutate the contents of the returned slice.
func (c *mesosClient) listSlaves(ctx context.Context) ([]*slaveNode, error) {
	return c.state.nodes(ctx)
}

// clusterName returns a (possibly cached) cluster name.
func (c *mesosClient) clusterName(ctx context.Context) (string, error) {
	return c.state.clusterName(ctx)
}

// pollMasterForState returns an array of slave nodes
func (c *mesosClient) pollMasterForState(ctx context.Context) (*mesosState, error) {
	// wait for initial master detection
	select {
	case <-c.initialMaster: // noop
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	master := func() string {
		c.masterLock.RLock()
		defer c.masterLock.RUnlock()
		return c.master
	}()
	if master == "" {
		return nil, noLeadingMasterError
	}

	//TODO(jdef) should not assume master uses http (what about https?)

	uri := fmt.Sprintf("http://%s/state.json", master)
	req, err := http.NewRequest("GET", uri, nil)
	if err != nil {
		return nil, err
	}
	var state *mesosState
	err = c.httpDo(ctx, req, func(res *http.Response, err error) error {
		if err != nil {
			return err
		}
		defer res.Body.Close()
		if res.StatusCode != 200 {
			return fmt.Errorf("HTTP request failed with code %d: %v", res.StatusCode, res.Status)
		}
		blob, err1 := ioutil.ReadAll(res.Body)
		if err1 != nil {
			return err1
		}
		log.V(3).Infof("Got mesos state, content length %v", len(blob))
		state, err1 = parseMesosState(blob)
		return err1
	})
	return state, err
}

func parseMesosState(blob []byte) (*mesosState, error) {
	type State struct {
		ClusterName string `json:"cluster"`
		Slaves      []*struct {
			Id        string                 `json:"id"`        // ex: 20150106-162714-3815890698-5050-2453-S2
			Pid       string                 `json:"pid"`       // ex: slave(1)@10.22.211.18:5051
			Hostname  string                 `json:"hostname"`  // ex: 10.22.211.18, or slave-123.nowhere.com
			Resources map[string]interface{} `json:"resources"` // ex: {"mem": 123, "ports": "[31000-3200]"}
		} `json:"slaves"`
	}
	state := &State{ClusterName: defaultClusterName}
	if err := json.Unmarshal(blob, state); err != nil {
		return nil, err
	}
	nodes := []*slaveNode{}
	for _, slave := range state.Slaves {
		if slave.Hostname == "" {
			continue
		}
		node := &slaveNode{hostname: slave.Hostname}
		cap := api.ResourceList{}
		if slave.Resources != nil && len(slave.Resources) > 0 {
			// attempt to translate CPU (cores) and memory (MB) resources
			if cpu, found := slave.Resources["cpus"]; found {
				if cpuNum, ok := cpu.(float64); ok {
					cap[api.ResourceCPU] = *resource.NewQuantity(int64(cpuNum), resource.DecimalSI)
				} else {
					log.Warningf("unexpected slave cpu resource type %T: %v", cpu, cpu)
				}
			} else {
				log.Warningf("slave failed to report cpu resource")
			}
			if mem, found := slave.Resources["mem"]; found {
				if memNum, ok := mem.(float64); ok {
					cap[api.ResourceMemory] = *resource.NewQuantity(int64(memNum), resource.BinarySI)
				} else {
					log.Warningf("unexpected slave mem resource type %T: %v", mem, mem)
				}
			} else {
				log.Warningf("slave failed to report mem resource")
			}
		}
		if len(cap) > 0 {
			node.resources = &api.NodeResources{
				Capacity: cap,
			}
			log.V(4).Infof("node %q reporting capacity %v", node.hostname, cap)
		}
		nodes = append(nodes, node)
	}

	result := &mesosState{
		clusterName: state.ClusterName,
		nodes:       nodes,
	}

	return result, nil
}

type responseHandler func(*http.Response, error) error

// httpDo executes an HTTP request in the given context, canceling an ongoing request if the context
// is canceled prior to completion of the request. hacked from https://blog.golang.org/context
func (c *mesosClient) httpDo(ctx context.Context, req *http.Request, f responseHandler) error {
	// Run the HTTP request in a goroutine and pass the response to f.
	ch := make(chan error, 1)
	go func() { ch <- f(c.httpClient.Do(req)) }()
	select {
	case <-ctx.Done():
		c.tr.CancelRequest(req)
		<-ch // Wait for f to return.
		return ctx.Err()
	case err := <-ch:
		return err
	}
}
