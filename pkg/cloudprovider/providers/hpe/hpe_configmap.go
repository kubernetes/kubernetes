/*
Copyright 2017 The Kubernetes Authors.

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

package hpe

import (
	"errors"
	"strings"

	"k8s.io/client-go/kubernetes"
	meta_v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	cg_v1 "k8s.io/client-go/kubernetes/typed/core/v1"
	v1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/rest"
)

type ncsConfigMapsInterface interface {
	createCfgMap(cfgMap *v1.ConfigMap) error
	getCfgMap(cfgMapName string) (*v1.ConfigMap, error)
	updateCfgMap(cfgMap *v1.ConfigMap) error
	deleteCfgMap(cfgMap *v1.ConfigMap) error
}

/* ncsConfigMaps: implements ncsConfigMapsInterface
   each instance 'ncsConfigMaps' on a given namespace with which
   it is initialized
*/

type ncsConfigMaps struct {
	kubeApiClient *cg_v1.CoreV1Client
	cfgMaps       cg_v1.ConfigMapInterface
	ns            string
}

/* init () : Performs the following
   1. In cluster configuration reading
   2. Using the configuration gets API connection
   3. Get cg_v1.ConfigMaps object which implements actual API for configmap CRUD calls.
      And these API will be invoked through ncsConfigMapsInterface APIs
*/
func (ncsCfgMap *ncsConfigMaps) init() error {
	config, err := rest.InClusterConfig()
	if err != nil {
		return err
	}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		return err
	}
	client := clientset.CoreV1Client
	ncsCfgMap.kubeApiClient = client
	ncsCfgMap.cfgMaps = client.ConfigMaps(ncsCfgMap.ns)
	return nil
}

/* createNewCfgMap: Creates configmap in ncsCfgMap.namespace */
func (ncsCfgMap *ncsConfigMaps) createNewCfgMap(newCfgMap *v1.ConfigMap) (*v1.ConfigMap, error) {
	return ncsCfgMap.cfgMaps.Create(newCfgMap)
}

/* getCfgMap:Fetches a configmap in ncsCfgMap.namespace with given name */
func (ncsCfgMap *ncsConfigMaps) getCfgMap(cfgMapName string) (*v1.ConfigMap, error) {
	//options := metav1.GetOptions{}
	//return ncsCfgMap.cfgMaps.Get(cfgMapName, options)
	return ncsCfgMap.cfgMaps.Get(cfgMapName, meta_v1.GetOptions{})
}

/* Updates a configmap in ncsCfgMap.namespace with given name and data */
func (ncsCfgMap *ncsConfigMaps) updateCfgMap(cfgMap *v1.ConfigMap) (*v1.ConfigMap, error) {
	return ncsCfgMap.cfgMaps.Update(cfgMap)
}

/* Deletes a configma map in thespace.*/
func (ncsCfgMap *ncsConfigMaps) deleteCfgMap(cfgMap *v1.ConfigMap) error {
	return nil
}
func (ncsCfgMap *ncsConfigMaps) getVrids(configMap_data map[string]string) (vrids []string) {
	for _, v := range configMap_data {
		substring := strings.Split(v, ",")
		vrids = append(vrids, substring[0])
	}
	return vrids
}
func (ncsCfgMap *ncsConfigMaps) createVridsExIPCfgMap(vrids []string, cfgmapname string, extIP string) (result *v1.ConfigMap) {
	values := make(map[string]string)
	for _, v := range vrids { // Initialize the map with vrid as key and value empty string
		values[v] = ""
	}
	values[vrids[0]] = extIP + ","
	result = &v1.ConfigMap{
		ObjectMeta: meta_v1.ObjectMeta{
			Name:      cfgmapname,
			Namespace: ncsCfgMap.ns,
		},
		Data: values,
	}
	return result
}
func (ncsCfgMap *ncsConfigMaps) updateVridsExIPCfgMap(cfgMap *v1.ConfigMap, vrids []string, extIP string) error {
	if len(cfgMap.Data) <= 0 {
		cfgMap.Data = make(map[string]string)
	}
	data := cfgMap.Data
	for _, v := range vrids {
		_, ok := data[v]
		//if particular v:vrid is not present in data then it means which is a for newly added node
		// Set the new vrid in map with empty value
		if !ok {
			data[v] = ""
		}
	}
	idx := ""
	/* Update external IP to a vrid which has least IP count */
	min := 1<<8 - 1 // Max unsigned int value
	for k, v := range data {
		l := len(strings.Split(v, ","))
		if l < min {
			min = l
			idx = k
		}
	}
	if len(idx) > 0 {
		data[idx] += extIP + ","
		return nil
	} else {
		return errors.New("Error : Failed to update external IP to vrid-to-ip-map")
	}
}

func (ncsCfgMap *ncsConfigMaps) createSvcExIPCfgMap(svcname string, cfgmapname string, extIP string) (result *v1.ConfigMap) {
	values := make(map[string]string)
	values[svcname] = extIP
	result = &v1.ConfigMap{
		ObjectMeta: meta_v1.ObjectMeta{
			Name:      cfgmapname,
			Namespace: ncsCfgMap.ns,
		},
		Data: values,
	}
	return result
}

func (ncsCfgMap *ncsConfigMaps) updateSvcExIPCfgMap(cfgMap *v1.ConfigMap, svcname string, extIP string) error {

	if len(cfgMap.Data) <= 0 {
		cfgMap.Data = make(map[string]string)
	}
	data := cfgMap.Data
	data[svcname] = extIP
	return nil
}