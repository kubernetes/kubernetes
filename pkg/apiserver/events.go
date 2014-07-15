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

package apiserver

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
)

type EventStore interface {
	ListEvents() ([]api.Event, error)
	ListEventsForPod(podID string) ([]api.Event, error)
}

type EtcdEventStore struct {
	etcdHelper tools.EtcdHelper
}

func NewEtcdEventStore(client tools.EtcdGetSet) EventStore {
	return &EtcdEventStore{
		etcdHelper: tools.EtcdHelper{Client: client},
	}
}

const eventBase = "/events"

func (e *EtcdEventStore) ListEvents() ([]api.Event, error) {
	key := eventBase
	keys, err := e.etcdHelper.ListChildren(key)
	if err != nil {
		if tools.IsEtcdNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	var result []api.Event
	for _, childKey := range keys {
		var subList []api.Event
		if err = e.etcdHelper.ExtractList(key+"/"+childKey, &subList); err != nil {
			if !tools.IsEtcdNotFound(err) {
				return nil, err
			}
		}
		result = append(result, subList...)
	}
	return result, nil
}

func (e *EtcdEventStore) ListEventsForPod(podID string) ([]api.Event, error) {
	var result []api.Event
	err := e.etcdHelper.ExtractList(eventBase+"/"+podID, &result)
	if tools.IsEtcdNotFound(err) {
		err = nil
	}
	return result, err
}
