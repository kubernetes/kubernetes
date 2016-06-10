/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package keystone

import (
	"encoding/json"
	"errors"
	"io/ioutil"

	"github.com/golang/glog"
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	openstackutil "k8s.io/kubernetes/pkg/util/openstack"
)

type apiCallValidator struct {
	configPath string
}

func newAPICallValidator(configPath string) (*apiCallValidator, error) {

	return &apiCallValidator{configPath: configPath}, nil
}

func (a *apiCallValidator) support(token string) bool {
	return true
}

func (a *apiCallValidator) validate(token string) (string, string, []string, bool, error) {
	//FIXME kfox1111: To enhance performance, cache admin token/client.
	glog.V(3).Infof("Starting Keystone Token Auth.")
	cfg, provider, err := openstackutil.ConfigFileToProvider(a.configPath)
	if err != nil {
		return "", "", nil, false, errors.New("Internal error getting the Keystone provider")
	}

	glog.V(3).Infof("Getting OpenStack Identity Provider...")
	client, err := openstack.NewIdentityAdminV3(provider, gophercloud.EndpointOpts{
		Region: cfg.Global.Region,
	})
	if err != nil {
		return "", "", nil, false, err
	}
	response, err := client.Request("GET", client.ServiceURL("auth", "tokens"), gophercloud.RequestOpts{
		MoreHeaders: map[string]string{"X-Subject-Token": token},
		OkCodes:     []int{200, 203},
	})
	if err != nil {
		return "", "", nil, false, err
	}
	defer response.Body.Close()
	bodyBytes, err := ioutil.ReadAll(response.Body)
	if err != nil {
		glog.V(3).Infof("Cannot get HTTP response body from keystone token validate: %v", err)
		return "", "", nil, false, err
	}
	obj := struct {
		Token struct {
			User struct {
				Id string `json:"id"`
			} `json:"user"`
			Project struct {
				Id string `json:"id"`
			} `json:"project"`
			Roles []struct {
				Name string `json:"name"`
			} `json:"roles"`
		} `json:"token"`
	}{}
	err = json.Unmarshal(bodyBytes, &obj)
	if err != nil {
		return "", "", nil, false, err
	}
	var roles []string
	if obj.Token.Roles != nil && len(obj.Token.Roles) > 0 {
		roles = make([]string, len(obj.Token.Roles))
		for i := 0; i < len(obj.Token.Roles); i++ {
			roles[i] = obj.Token.Roles[i].Name
		}
	} else {
		roles = make([]string, 0)
	}
	valid := response.StatusCode == 200 || response.StatusCode == 203
	return obj.Token.User.Id, obj.Token.Project.Id, roles, valid, nil
}
