/*
 *
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package client

import (
	"fmt"

	xdspb "github.com/envoyproxy/go-control-plane/envoy/api/v2"
	httppb "github.com/envoyproxy/go-control-plane/envoy/config/filter/network/http_connection_manager/v2"
	"github.com/golang/protobuf/ptypes"
)

// handleLDSResponse processes an LDS response received from the xDS server. On
// receipt of a good response, it also invokes the registered watcher callback.
func (v2c *v2Client) handleLDSResponse(resp *xdspb.DiscoveryResponse) error {
	v2c.mu.Lock()
	defer v2c.mu.Unlock()

	wi := v2c.watchMap[ldsURL]
	if wi == nil {
		return fmt.Errorf("xds: no LDS watcher found when handling LDS response: %+v", resp)
	}

	routeName := ""
	for _, r := range resp.GetResources() {
		var resource ptypes.DynamicAny
		if err := ptypes.UnmarshalAny(r, &resource); err != nil {
			return fmt.Errorf("xds: failed to unmarshal resource in LDS response: %v", err)
		}
		lis, ok := resource.Message.(*xdspb.Listener)
		if !ok {
			return fmt.Errorf("xds: unexpected resource type: %T in LDS response", resource.Message)
		}
		if lis.GetName() != wi.target[0] {
			// We ignore listeners we are not watching for because LDS is
			// special in the sense that there is only one resource we are
			// interested in, and this resource does not change over the
			// lifetime of the v2Client. So, we don't have to cache other
			// listeners which we are not interested in.
			continue
		}
		var err error
		routeName, err = getRouteConfigNameFromListener(lis)
		if err != nil {
			return err
		}
	}

	var err error
	if routeName == "" {
		err = fmt.Errorf("xds: LDS target %s not found in received response %+v", wi.target, resp)
	}
	wi.stopTimer()
	wi.callback.(ldsCallback)(ldsUpdate{routeName: routeName}, err)
	return nil
}

// getRouteConfigNameFromListener checks if the provided Listener proto meets
// the expected criteria. If so, it returns a non-empty routeConfigName.
func getRouteConfigNameFromListener(lis *xdspb.Listener) (string, error) {
	if lis.GetApiListener() == nil {
		return "", fmt.Errorf("xds: no api_listener field in LDS response %+v", lis)
	}
	var apiAny ptypes.DynamicAny
	if err := ptypes.UnmarshalAny(lis.GetApiListener().GetApiListener(), &apiAny); err != nil {
		return "", fmt.Errorf("xds: failed to unmarshal api_listner in LDS response: %v", err)
	}
	apiLis, ok := apiAny.Message.(*httppb.HttpConnectionManager)
	if !ok {
		return "", fmt.Errorf("xds: unexpected api_listener type: %T in LDS response", apiAny.Message)
	}
	switch apiLis.RouteSpecifier.(type) {
	case *httppb.HttpConnectionManager_Rds:
		name := apiLis.GetRds().GetRouteConfigName()
		if name == "" {
			return "", fmt.Errorf("xds: empty route_config_name in LDS response: %+v", lis)
		}
		return name, nil
	case *httppb.HttpConnectionManager_RouteConfig:
		// TODO: Add support for specifying the RouteConfiguration inline
		// in the LDS response.
		return "", fmt.Errorf("xds: LDS response contains RDS config inline. Not supported for now: %+v", apiLis)
	case nil:
		return "", fmt.Errorf("xds: no RouteSpecifier in received LDS response: %+v", apiLis)
	default:
		return "", fmt.Errorf("xds: unsupported type %T for RouteSpecifier in received LDS response", apiLis.RouteSpecifier)
	}
}
