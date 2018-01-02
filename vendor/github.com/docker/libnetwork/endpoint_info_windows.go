// +build windows

package libnetwork

import "fmt"

func (ep *endpoint) DriverInfo() (map[string]interface{}, error) {
	ep, err := ep.retrieveFromStore()
	if err != nil {
		return nil, err
	}

	var gwDriverInfo map[string]interface{}
	if sb, ok := ep.getSandbox(); ok {
		if gwep := sb.getEndpointInGWNetwork(); gwep != nil && gwep.ID() != ep.ID() {

			gwDriverInfo, err = gwep.DriverInfo()
			if err != nil {
				return nil, err
			}
		}
	}

	n, err := ep.getNetworkFromStore()
	if err != nil {
		return nil, fmt.Errorf("could not find network in store for driver info: %v", err)
	}

	driver, err := n.driver(true)
	if err != nil {
		return nil, fmt.Errorf("failed to get driver info: %v", err)
	}

	epInfo, err := driver.EndpointOperInfo(n.ID(), ep.ID())
	if err != nil {
		return nil, err
	}

	if epInfo != nil {
		epInfo["GW_INFO"] = gwDriverInfo
		return epInfo, nil
	}

	return gwDriverInfo, nil
}
