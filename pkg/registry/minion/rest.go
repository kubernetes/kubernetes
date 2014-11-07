/*
Copyright 2014 Google Inc. All rights reserved.

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

package minion

import (
	"errors"
	"fmt"
	"net"
	"strconv"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master/ports"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// REST implements the RESTStorage interface, backed by a MinionRegistry.
type REST struct {
	registry Registry
}

// NewREST returns a new REST.
func NewREST(m Registry) *REST {
	return &REST{
		registry: m,
	}
}

var ErrDoesNotExist = errors.New("The requested resource does not exist.")
var ErrNotHealty = errors.New("The requested minion is not healthy.")

func (rs *REST) Create(ctx api.Context, obj runtime.Object) (<-chan apiserver.RESTResult, error) {
	minion, ok := obj.(*api.Minion)
	if !ok {
		return nil, fmt.Errorf("not a minion: %#v", obj)
	}
	if minion.Name == "" {
		return nil, fmt.Errorf("ID should not be empty: %#v", minion)
	}

	minion.CreationTimestamp = util.Now()

	return apiserver.MakeAsync(func() (runtime.Object, error) {
		err := rs.registry.CreateMinion(ctx, minion)
		if err != nil {
			return nil, err
		}
		minionName := minion.Name
		minion, err := rs.registry.GetMinion(ctx, minionName)
		if err == ErrNotHealty {
			return rs.toApiMinion(minionName), nil
		}
		if minion == nil {
			return nil, ErrDoesNotExist
		}
		if err != nil {
			return nil, err
		}
		return minion, nil
	}), nil
}

func (rs *REST) Delete(ctx api.Context, id string) (<-chan apiserver.RESTResult, error) {
	minion, err := rs.registry.GetMinion(ctx, id)
	if minion == nil {
		return nil, ErrDoesNotExist
	}
	if err != nil {
		return nil, err
	}
	return apiserver.MakeAsync(func() (runtime.Object, error) {
		return &api.Status{Status: api.StatusSuccess}, rs.registry.DeleteMinion(ctx, id)
	}), nil
}

func (rs *REST) Get(ctx api.Context, id string) (runtime.Object, error) {
	minion, err := rs.registry.GetMinion(ctx, id)
	if minion == nil {
		return nil, ErrDoesNotExist
	}
	return minion, err
}

func (rs *REST) List(ctx api.Context, label, field labels.Selector) (runtime.Object, error) {
	return rs.registry.ListMinions(ctx)
}

func (rs *REST) New() runtime.Object {
	return &api.Minion{}
}

func (rs *REST) Update(ctx api.Context, minion runtime.Object) (<-chan apiserver.RESTResult, error) {
	return nil, fmt.Errorf("Minions can only be created (inserted) and deleted.")
}

func (rs *REST) toApiMinion(name string) *api.Minion {
	return &api.Minion{ObjectMeta: api.ObjectMeta{Name: name}}
}

// ResourceLocation returns a URL to which one can send traffic for the specified minion.
func (rs *REST) ResourceLocation(ctx api.Context, id string) (string, error) {
	minion, err := rs.registry.GetMinion(ctx, id)
	if err != nil {
		return "", err
	}
	host := minion.HostIP
	if host == "" {
		host = minion.Name
	}
	// TODO: Minion webservers should be secure!
	return "http://" + net.JoinHostPort(host, strconv.Itoa(ports.KubeletPort)), nil
}
