package cloudcfg

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"gopkg.in/v1/yaml"
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

	// Try parsing as json and yaml
	parsed_json := reflect.New(prototypeType).Interface()
	json_err := json.Unmarshal(data, parsed_json)
	parsed_yaml := reflect.New(prototypeType).Interface()
	yaml_err := yaml.Unmarshal(data, parsed_yaml)

	if json_err != nil && yaml_err != nil {
		return nil, fmt.Errorf("Could not parse input as json (error: %v) or yaml (error: %v", json_err, yaml_err)
	}

	if json_err != nil {
		return json.Marshal(parsed_json)
	}
	return json.Marshal(parsed_yaml)
}
