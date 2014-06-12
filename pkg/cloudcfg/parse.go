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

	// Try parsing json
	json_out, json_err := tryJSON(data, reflect.New(prototypeType).Interface())
	if json_err == nil {
		return json_out, json_err
	}

	// Try parsing yaml
	yaml_out, yaml_err := tryYAML(data, reflect.New(prototypeType).Interface())
	if yaml_err != nil {
		return nil, fmt.Errorf("Could not parse input as json (error: %v) or yaml (error: %v", json_err, yaml_err)
	}
	return yaml_out, yaml_err
}

func tryJSON(data []byte, obj interface{}) ([]byte, error) {
	err := json.Unmarshal(data, obj)
	if err != nil {
		return nil, err
	}
	return json.Marshal(obj)
}

func tryYAML(data []byte, obj interface{}) ([]byte, error) {
	err := yaml.Unmarshal(data, obj)
	if err != nil {
		return nil, err
	}
	return json.Marshal(obj)
}
