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

package gce

import (
	"fmt"

	computealpha "google.golang.org/api/compute/v0.alpha"
	compute "google.golang.org/api/compute/v1"

	"github.com/golang/glog"
)

type addressManager struct {
	logPrefix  string
	svc        CloudAddressService
	targetIP   string
	alphaGate  *AlphaFeatureGate
	region     string
	subnetURL  string
	tryRelease bool
	addrParams addressParams
}

type addressParams struct {
	name        string
	desc        string
	requestedIP string
	currentIP   string
	addressType lbScheme
	netTier     NetworkTier
}

func newAddressParams(name, desc, requestedIP, currentIP string, addressType lbScheme, netTier NetworkTier) addressParams {
	return addressParams{
		name:        name,
		desc:        desc,
		requestedIP: requestedIP,
		currentIP:   currentIP,
		addressType: addressType,
		netTier:     netTier,
	}
}

func newAddressManager(svc CloudAddressService, region, subnetURL string, addrParams addressParams, alphaGate *AlphaFeatureGate) *addressManager {
	targetIP := addrParams.requestedIP
	if targetIP == "" {
		targetIP = addrParams.currentIP
	}

	return &addressManager{
		svc:        svc,
		logPrefix:  fmt.Sprintf("AddressManager(%q)", addrParams.name),
		region:     region,
		addrParams: addrParams,
		targetIP:   targetIP,
		alphaGate:  alphaGate,
		subnetURL:  subnetURL,
		tryRelease: true,
	}
}

// HoldAddress will ensure that the IP is reserved by an address - either owned by the controller or
// by a user. If the address is not the addressParams.name, then it's assumed to be a user's address.
// The string returned is the reserved IP address.
// Note: If CurrentIP is provided, it must be a *valid* IP (correct network tier).
func (am *addressManager) HoldAddress() (string, error) {
	// Retrieve the address that we use for this load balancer (by name).
	// The address could be reserving an undesired IP; therefore, it would need to be deleted.
	if reservedIP, err := am.getControllerOwned(); err != nil {
		return "", err
	} else if reservedIP != "" {
		return reservedIP, nil
	}

	// If a forwarding rule exists or user provided loadBalancerIP,
	// check if the user owns the IP.
	if am.targetIP != "" {
		if reservedIP, err := am.getUserOwned(); err != nil {
			return "", err
		} else if reservedIP != "" {
			return reservedIP, nil
		}

		// If loadBalancerIP was set, but no address was found, warn user.
		if am.addrParams.requestedIP != "" {
			// TODO: Warn user that they don't own the IP which they've requested.
		}
	}

	// Try reserving the IP with controller-owned address name
	return am.reserveAddr()
}

// ReleaseAddress releases the address if it's owned by the controller.
func (am *addressManager) ReleaseAddress() error {
	if !am.tryRelease {
		glog.V(4).Infof("%v: not attempting release of address %q.", am.logPrefix, am.targetIP)
		return nil
	}

	glog.V(4).Infof("%v: releasing address %q holding IP %q", am.logPrefix, am.addrParams.name, am.targetIP)
	return am.releaseAddress()
}

func (am *addressManager) getControllerOwned() (string, error) {
	addr, err := am.svc.GetRegionAddress(am.addrParams.name, am.region)
	if ignoreNotFound(err) != nil {
		return "", err
	}

	if addr == nil {
		return "", nil
	}

	// If address exists, check if the address had the expected attributes.
	validationMsg, err := am.validateAddress(addr)
	if err != nil {
		return "", err
	}

	if validationMsg == "" {
		glog.V(4).Infof("%v: address %q already reserves IP %q Type %q. No further action required.", am.logPrefix, addr.Name, addr.Address, addr.AddressType)
		return addr.Address, nil
	}

	glog.V(2).Infof("%v: deleting existing address because %v", am.logPrefix, validationMsg)
	return "", am.releaseAddress()
}

// getUserOwned attempts to retrieve an address by the current or requested IP. Existance of an addr
// which passes validation means the controller can rely on this address to hold the IP.
func (am *addressManager) getUserOwned() (string, error) {
	addr, err := am.svc.GetRegionAddressByIP(am.region, am.targetIP)
	if ignoreNotFound(err) != nil {
		return "", fmt.Errorf("failed to get address by IP %q after reservation attempt, err: %v", am.targetIP, err)
	}

	if addr == nil {
		return "", nil
	}

	validationMsg, err := am.validateAddress(addr)
	if err != nil {
		return "", err
	}

	// If the address is actually controller owned, either delete or use.
	// This should not happen in practice because the controller's address was not found earlier.
	// Possible that another service controller is running.
	if am.isManagedAddress(addr) {
		if validationMsg != "" {
			glog.Warning("%v: address %q unexpectedly existed with IP %q but is invalid because %q. Deleting address.", am.logPrefix, addr.Name, am.targetIP, validationMsg)
			return "", am.releaseAddress()
		}
		// The address with this name is checked at the beginning of 'HoldAddress()', but for some reason
		// it was re-created by this point. May be possible that two controllers are running.
		glog.Warning("%v: address %q unexpectedly existed with IP %q.", am.logPrefix, addr.Name, am.targetIP)
		return addr.Address, nil
	}

	// If validation succeeded, use this address owned by the user.
	if validationMsg == "" {
		glog.V(4).Infof("%v: IP %q is reserved by address %q, description: %q", am.logPrefix, am.targetIP, addr.Name, addr.Description)
		am.tryRelease = false
		return addr.Address, nil
	}

	// If the user requested this IP, raise an error that the address is unusable.
	if am.addrParams.requestedIP == addr.Address {
		return "", fmt.Errorf("IP %q is reserved by address %q but is invalid because %q", addr.Address, addr.Name, validationMsg)
	}

	// At this point, we know:
	//  - Address is owned by the user
	//  - Address failed validation
	//  - loadBalancerIP is empty; therefore, addr.IP = forwarding rule's IP
	// This situation should not happen since the AddressManager assumes the controller has deleted
	// incompatible forwarding rules earlier.
	glog.V(4).Infof("%v: IP %q is reserved by address %q but is invalid because %q", addr.Address, addr.Name, validationMsg)
	return "", nil
}

func (am *addressManager) reserveAddr() (string, error) {
	glog.V(4).Infof("%v: reserving IP %q Type %q", am.logPrefix, am.targetIP, am.addrParams.addressType)

	var err error
	switch am.addrParams.netTier {
	case NetworkTierPremium:
		a := &compute.Address{
			Name:        am.addrParams.name,
			Description: am.addrParams.desc,
			Address:     am.targetIP,
			AddressType: string(am.addrParams.addressType),
			Subnetwork:  am.subnetURL,
		}
		err = am.svc.ReserveRegionAddress(a, am.region)
	default:
		a := &computealpha.Address{
			Name:        am.addrParams.name,
			Description: am.addrParams.desc,
			Address:     am.targetIP,
			AddressType: string(am.addrParams.addressType),
			Subnetwork:  am.subnetURL,
			NetworkTier: am.addrParams.netTier.ToGCEValue(),
		}
		err = am.svc.ReserveAlphaRegionAddress(a, am.region)
	}

	if err != nil {
		// Status BadRequest or Conflict may occur if the IP is already reserved or the name
		// is used. By this point the address manager has checked for both of these conditions.
		return "", fmt.Errorf("failed to reserve address %q with IP %q, err: %v", am.addrParams.name, am.targetIP, err)
	}

	if am.targetIP != "" {
		glog.V(4).Infof("%v: successfully reserved IP %q with name %q", am.logPrefix, am.targetIP, am.addrParams.name)
		return am.targetIP, nil
	}

	addr, err := am.svc.GetRegionAddress(am.addrParams.name, am.region)
	if err != nil {
		return "", err
	}
	return addr.Address, nil
}

func (am *addressManager) releaseAddress() error {
	// Controller only ever tries to unreserve the address named with the load balancer's name.
	if err := am.svc.DeleteRegionAddress(am.addrParams.name, am.region); err != nil {
		if isNotFound(err) {
			glog.Warningf("%v: address %q was not found during delete. Ignoring.", am.logPrefix, am.addrParams.name)
			return nil
		}

		return err
	}

	glog.V(4).Infof("%v: successfully released IP %q named %q", am.logPrefix, am.targetIP, am.addrParams.name)
	return nil
}

func (am *addressManager) validateAddress(addr *compute.Address) (string, error) {
	// Check IP
	if am.targetIP != "" && am.targetIP != addr.Address {
		return fmt.Sprintf("expected IP %q, actual: %q", am.targetIP, addr.Address), nil
	}

	// Check Type
	if addr.AddressType != string(am.addrParams.addressType) {
		return fmt.Sprintf("expected address type %q, actual: %q", am.addrParams.addressType, addr.AddressType), nil
	}

	// If relevant, check network tier
	if am.addrParams.netTier != NetworkTier("") {
		t, err := am.getNetworkTierFromAddress(addr.Name, am.region)
		if err != nil {
			return "", err
		}
		if t != am.addrParams.netTier {
			return fmt.Sprintf("expected network tier %q, actual: %q", am.addrParams.netTier.ToGCEValue(), t.ToGCEValue()), nil
		}
	}

	return "", nil
}

func (am *addressManager) getNetworkTierFromAddress(name, region string) (NetworkTier, error) {
	if am.alphaGate == nil || !am.alphaGate.Enabled(AlphaFeatureNetworkTiers) {
		return NetworkTierDefault, nil
	}

	addr, err := am.svc.GetAlphaRegionAddress(name, region)
	if err != nil {
		if isForbidden(err) {
			return NetworkTierDefault, nil
		}
		return NetworkTier(""), err
	}
	return NetworkTierGCEValueToType(addr.NetworkTier), nil
}

func (am *addressManager) isManagedAddress(addr *compute.Address) bool {
	return addr.Name == am.addrParams.name
}

func ensureAddressDeleted(svc CloudAddressService, name, region string) error {
	return ignoreNotFound(svc.DeleteRegionAddress(name, region))
}
