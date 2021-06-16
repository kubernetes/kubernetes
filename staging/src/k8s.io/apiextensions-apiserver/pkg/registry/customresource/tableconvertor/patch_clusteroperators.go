package tableconvertor

import (
	"encoding/json"
	"io"
	"reflect"

	configv1 "github.com/openshift/api/config/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/rest"
)

var clusterOperatorGVK = schema.GroupVersionKind{configv1.GroupName, "v1", "ClusterOperator"}

func withClusterOperatorColumns(c *convertor, gvk schema.GroupVersionKind) rest.TableConvertor {
	if gvk != clusterOperatorGVK {
		return c
	}

	c.headers = append(c.headers, metav1.TableColumnDefinition{
		Name:        "Message",
		Type:        "string",
		Description: "A message describing the status of the operator",
		Priority:    0,
	})
	c.additionalColumns = append(c.additionalColumns, clusterOperatorConditionMessage{})

	return c
}

type clusterOperatorConditionMessage struct {
}

func (c clusterOperatorConditionMessage) FindResults(data interface{}) ([][]reflect.Value, error) {
	obj := data.(map[string]interface{})
	unstructuredConds, _, _ := unstructured.NestedFieldNoCopy(obj, "status", "conditions")
	var conds []configv1.ClusterOperatorStatusCondition
	bs, err := json.Marshal(unstructuredConds)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(bs, &conds); err != nil {
		return nil, err
	}

	var available, degraded, progressing *configv1.ClusterOperatorStatusCondition
	for i := range conds {
		cond := &conds[i]
		switch {
		case cond.Type == configv1.OperatorAvailable && cond.Status == configv1.ConditionFalse:
			available = cond
		case cond.Type == configv1.OperatorDegraded && cond.Status == configv1.ConditionTrue:
			degraded = cond
		case cond.Type == configv1.OperatorProgressing && cond.Status == configv1.ConditionTrue:
			progressing = cond
		}
	}

	mostCritical := progressing
	if degraded != nil {
		mostCritical = degraded
	}
	if available != nil {
		mostCritical = available
	}

	if mostCritical != nil {
		if len(mostCritical.Message) > 0 {
			return [][]reflect.Value{{reflect.ValueOf(mostCritical.Message)}}, nil
		}
		if len(mostCritical.Reason) > 0 {
			return [][]reflect.Value{{reflect.ValueOf(mostCritical.Reason)}}, nil
		}
	}

	return nil, nil
}

func (c clusterOperatorConditionMessage) PrintResults(wr io.Writer, results []reflect.Value) error {
	first := true
	for _, r := range results {
		if !first {
			wr.Write([]byte("; ")) // should never happen as we only return one result
		}
		if _, err := wr.Write([]byte(r.String())); err != nil {
			return err
		}
		first = false
	}

	return nil
}
