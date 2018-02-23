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

package util

import (
	"fmt"
	"net"
	"strconv"

	"k8s.io/apimachinery/pkg/util/validation"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
)

// GetMasterEndpoint returns a properly formatted Master Endpoint
// or passes the error from GetMasterHostPort.
func GetMasterEndpoint(cfg *kubeadmapi.MasterConfiguration) (string, error) {

	hostPort, err := GetMasterHostPort(cfg)
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("https://%s", hostPort), nil
}

// GetMasterHostPort returns a properly formatted Master IP/port pair or error
// if the IP address can not be parsed or port is outside the valid TCP range.
func GetMasterHostPort(cfg *kubeadmapi.MasterConfiguration) (string, error) {
	var masterIP string
	if len(cfg.API.ControlPlaneEndpoint) > 0 {
		errs := validation.IsDNS1123Subdomain(cfg.API.ControlPlaneEndpoint)
		if len(errs) > 0 {
			return "", fmt.Errorf("error parsing `ControlPlaneEndpoint` to valid dns subdomain with errors: %s", errs)
		}
		masterIP = cfg.API.ControlPlaneEndpoint
	} else {
		ip := net.ParseIP(cfg.API.AdvertiseAddress)
		if ip == nil {
			return "", fmt.Errorf("error parsing address %s", cfg.API.AdvertiseAddress)
		}
		masterIP = ip.String()
	}

	if cfg.API.BindPort < 0 || cfg.API.BindPort > 65535 {
		return "", fmt.Errorf("api server port must be between 0 and 65535")
	}

	hostPort := net.JoinHostPort(masterIP, strconv.Itoa(int(cfg.API.BindPort)))
	return hostPort, nil
}
