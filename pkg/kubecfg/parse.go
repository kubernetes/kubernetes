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

package kubecfg

import (
	"fmt"
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

var storageToType = map[string]reflect.Type{
	"pods":                   reflect.TypeOf(api.Pod{}),
	"services":               reflect.TypeOf(api.Service{}),
	"replicationControllers": reflect.TypeOf(api.ReplicationController{}),
}

// Takes input 'data' as either json or yaml, checks that it parses as the
// appropriate object type, and returns json for sending to the API or an
// error.
func ToWireFormat(data []byte, storage string) ([]byte, error) {
	prototypeType, found := storageToType[storage]
	if !found {
		return nil, fmt.Errorf("unknown storage type: %v", storage)
	}

	obj := reflect.New(prototypeType).Interface()
	err := api.DecodeInto(data, obj)
	if err != nil {
		return nil, err
	}
	return api.Encode(obj)
}
