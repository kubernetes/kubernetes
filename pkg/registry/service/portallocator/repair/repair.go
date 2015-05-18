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

package repair

import (
	"fmt"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/pool"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/service"
	"strconv"
	"strings"
)

type PortAllocatorPoolRepair struct {
	registry service.Registry
}

func (p *PortAllocatorPoolRepair) ListAllOwnedKeys() ([]pool.PoolOwner, error) {
	ctx := api.WithNamespace(api.NewDefaultContext(), api.NamespaceAll)
	list, err := p.registry.ListServices(ctx)
	if err != nil {
		return nil, fmt.Errorf("unable to refresh the services: %v", err)
	}

	poolOwners := []pool.PoolOwner{}
	for i := range list.Items {
		poolOwners = append(poolOwners, toPoolOwner(&list.Items[i]))
	}

	return poolOwners, nil
}

func (p *PortAllocatorPoolRepair) GetOwnedKeys(ownerId string) (*pool.PoolOwner, error) {
	tokens := strings.SplitN(ownerId, "/", 2)
	if len(tokens) != 2 {
		return nil, fmt.Errorf("invalid service id: %s", ownerId)
	}

	ctx := api.WithNamespace(api.NewDefaultContext(), tokens[0])
	svc, err := p.registry.GetService(ctx, tokens[1])
	if err != nil {
		return nil, fmt.Errorf("unable to refresh the services: %v", err)
	}

	if svc == nil {
		return nil, nil
	}

	poolOwner := toPoolOwner(svc)
	return &poolOwner, nil
}

func toPoolOwner(svc *api.Service) pool.PoolOwner {
	var poolOwner pool.PoolOwner
	poolOwner.Owner = svc.Namespace + "/" + svc.Name
	poolOwner.Keys = []string{}
	//	resourceVersion, err := strconv.ParseUint(svc.ResourceVersion, 10, 64)
	//	if err != nil {
	//		panic("unexpected ResourceVersion format: " + svc.ResourceVersion)
	//	}
	//	poolOwner.ResourceVersion = resourceVersion
	for _, port := range svc.Spec.Ports {
		if port.NodePort != 0 {
			poolOwner.Keys = append(poolOwner.Keys, strconv.Itoa(port.NodePort))
		}
	}
	return poolOwner
}
