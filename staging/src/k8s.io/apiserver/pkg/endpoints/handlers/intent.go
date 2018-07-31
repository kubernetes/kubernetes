package handlers

import (
	"fmt"

	"github.com/ghodss/yaml"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/json"
)

// TODO(apelisse): workflowId needs to be passed as a query
// param/header, and a better defaulting needs to be defined too.
const workflowID = "default"

func saveNewIntent(data map[string]interface{}, workflow string, dst runtime.Object) error {
	j, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to serialize json: %v", err)
	}

	accessor, err := meta.Accessor(dst)
	if err != nil {
		return fmt.Errorf("couldn't get accessor: %v", err)
	}
	m := accessor.GetLastApplied()
	if m == nil {
		m = make(map[string]string)
	}
	m[workflow] = string(j)
	accessor.SetLastApplied(m)
	return nil
}

func getNewIntent(data []byte) (map[string]interface{}, error) {
	patch := make(map[string]interface{})
	if err := yaml.Unmarshal(data, &patch); err != nil {
		return nil, fmt.Errorf("couldn't unmarshal patch object: %v (patch: %v)", err, string(data))
	}
	return patch, nil
}
