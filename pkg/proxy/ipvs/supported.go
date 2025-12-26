//go:build linux
// +build linux

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

package ipvs

import (
	"context"
	"fmt"

	"k8s.io/klog/v2"
	utilipvs "k8s.io/kubernetes/pkg/proxy/ipvs/util"
	netutils "k8s.io/utils/net"
)

// CanUseIPVSProxier checks if we can use the ipvs Proxier.
// The ipset version and the scheduler are checked. If any virtual servers (VS)
// already exist with the configured scheduler, we just return. Otherwise
// we check if a dummy VS can be configured with the configured scheduler.
// Kernel modules will be loaded automatically if necessary.
func CanUseIPVSProxier(ctx context.Context, ipvs utilipvs.Interface, ipsetver IPSetVersioner, scheduler string) error {
	logger := klog.FromContext(ctx)
	// BUG: https://github.com/moby/ipvs/issues/27
	// If ipvs is not compiled into the kernel no error is returned and handle==nil.
	// This in turn causes ipvs.GetVirtualServers and ipvs.AddVirtualServer
	// to return ok (err==nil). If/when this bug is fixed parameter "ipvs" will be nil
	// if ipvs is not supported by the kernel. Until then a re-read work-around is used.
	if ipvs == nil {
		return fmt.Errorf("ipvs not supported by the kernel")
	}

	// Check ipset version
	versionString, err := ipsetver.GetVersion()
	if err != nil {
		return fmt.Errorf("error getting ipset version, error: %w", err)
	}
	if !checkMinVersion(versionString) {
		return fmt.Errorf("ipset version: %s is less than min required version: %s", versionString, MinIPSetCheckVersion)
	}

	if scheduler == "" {
		scheduler = defaultScheduler
	}

	// If any virtual server (VS) using the scheduler exist we skip the checks.
	vservers, err := ipvs.GetVirtualServers()
	if err != nil {
		logger.Error(err, "Can't read the ipvs")
		return err
	}
	logger.V(5).Info("Virtual Servers", "count", len(vservers))
	if len(vservers) > 0 {
		// This is most likely a kube-proxy re-start. We know that ipvs works
		// and if any VS uses the configured scheduler, we are done.
		for _, vs := range vservers {
			if vs.Scheduler == scheduler {
				logger.V(5).Info("VS exist, Skipping checks")
				return nil
			}
		}
		logger.V(5).Info("No existing VS uses the configured scheduler", "scheduler", scheduler)
	}

	// Try to insert a dummy VS with the passed scheduler.
	// We should use a VIP address that is not used on the node.
	// An address "198.51.100.0" from the TEST-NET-2 rage in https://datatracker.ietf.org/doc/html/rfc5737
	// is used. These addresses are reserved for documentation. If the user is using
	// this address for a VS anyway we *will* mess up, but that would be an invalid configuration.
	// If the user have configured the address to an interface on the node (but not a VS)
	// then traffic will temporary be routed to ipvs during the probe and dropped.
	// The later case is also and invalid configuration, but the traffic impact will be minor.
	// This should not be a problem if users honors reserved addresses, but cut/paste
	// from documentation is not unheard of, so the restriction to not use the TEST-NET-2 range
	// must be documented.
	vs := utilipvs.VirtualServer{
		Address:   netutils.ParseIPSloppy("198.51.100.0"),
		Protocol:  "TCP",
		Port:      20000,
		Scheduler: scheduler,
	}
	if err := ipvs.AddVirtualServer(&vs); err != nil {
		logger.Error(err, "Could not create dummy VS", "scheduler", scheduler)
		return err
	}

	// To overcome the BUG described above we check that the VS is *really* added.
	vservers, err = ipvs.GetVirtualServers()
	if err != nil {
		logger.Error(err, "ipvs.GetVirtualServers")
		return err
	}
	logger.V(5).Info("Virtual Servers after adding dummy", "count", len(vservers))
	if len(vservers) == 0 {
		logger.Info("Dummy VS not created", "scheduler", scheduler)
		return fmt.Errorf("ipvs not supported") // This is a BUG work-around
	}
	logger.V(5).Info("Dummy VS created", "vs", vs)

	if err := ipvs.DeleteVirtualServer(&vs); err != nil {
		logger.Error(err, "Could not delete dummy VS")
		return err
	}

	return nil
}
