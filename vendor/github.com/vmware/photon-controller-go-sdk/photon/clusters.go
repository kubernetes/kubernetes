// Copyright (c) 2016 VMware, Inc. All Rights Reserved.
//
// This product is licensed to you under the Apache License, Version 2.0 (the "License").
// You may not use this product except in compliance with the License.
//
// This product may include a number of subcomponents with separate copyright notices and
// license terms. Your use of these subcomponents is subject to the terms and conditions
// of the subcomponent's license, as noted in the LICENSE file.

package photon

import (
	"bytes"
	"encoding/json"
)

// Contains functionality for clusters API.
type ClustersAPI struct {
	client *Client
}

var clusterUrl string = "/clusters/"

const ExtendedPropertyDNS string = "dns"
const ExtendedPropertyGateway string = "gateway"
const ExtendedPropertyNetMask string = "netmask"
const ExtendedPropertyMasterIP string = "master_ip"
const ExtendedPropertyContainerNetwork string = "container_network"
const ExtendedPropertyZookeeperIP1 string = "zookeeper_ip1"
const ExtendedPropertyZookeeperIP2 string = "zookeeper_ip2"
const ExtendedPropertyZookeeperIP3 string = "zookeeper_ip3"
const ExtendedPropertyETCDIP1 string = "etcd_ip1"
const ExtendedPropertyETCDIP2 string = "etcd_ip2"
const ExtendedPropertyETCDIP3 string = "etcd_ip3"
const ExtendedPropertySSHKey string = "ssh_key"
const ExtendedPropertyRegistryCACert string = "registry_ca_cert"
const ExtendedPropertyAdminPassword string = "admin_password"

// Deletes a cluster with specified ID.
func (api *ClustersAPI) Delete(id string) (task *Task, err error) {
	res, err := api.client.restClient.Delete(api.client.Endpoint+clusterUrl+id, api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Gets a cluster with the specified ID.
func (api *ClustersAPI) Get(id string) (cluster *Cluster, err error) {
	res, err := api.client.restClient.Get(api.client.Endpoint+clusterUrl+id, api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	res, err = getError(res)
	if err != nil {
		return
	}
	var result Cluster
	err = json.NewDecoder(res.Body).Decode(&result)
	return &result, nil
}

// Gets vms for cluster with the specified ID.
func (api *ClustersAPI) GetVMs(id string) (result *VMs, err error) {
	uri := api.client.Endpoint + clusterUrl + id + "/vms"
	res, err := api.client.restClient.GetList(api.client.Endpoint, uri, api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}

	result = &VMs{}
	err = json.Unmarshal(res, result)
	return
}

// Resize a cluster to specified count.
func (api *ClustersAPI) Resize(id string, resize *ClusterResizeOperation) (task *Task, err error) {
	body, err := json.Marshal(resize)
	if err != nil {
		return
	}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+clusterUrl+id+"/resize",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}

// Start a background process to recreate failed VMs in a cluster with the specified ID.
func (api *ClustersAPI) TriggerMaintenance(id string) (task *Task, err error) {
	body := []byte{}
	res, err := api.client.restClient.Post(
		api.client.Endpoint+clusterUrl+id+"/trigger_maintenance",
		"application/json",
		bytes.NewReader(body),
		api.client.options.TokenOptions.AccessToken)
	if err != nil {
		return
	}
	defer res.Body.Close()
	task, err = getTask(getError(res))
	return
}
