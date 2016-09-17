package watch

import (
	"fmt"
	"io"
	"sync"

	consulapi "github.com/hashicorp/consul/api"
)

// WatchPlan is the parsed version of a watch specification. A watch provides
// the details of a query, which generates a view into the Consul data store.
// This view is watched for changes and a handler is invoked to take any
// appropriate actions.
type WatchPlan struct {
	Datacenter string
	Token      string
	Type       string
	Exempt     map[string]interface{}

	Func      WatchFunc
	Handler   HandlerFunc
	LogOutput io.Writer

	address    string
	client     *consulapi.Client
	lastIndex  uint64
	lastResult interface{}

	stop     bool
	stopCh   chan struct{}
	stopLock sync.Mutex
}

// WatchFunc is used to watch for a diff
type WatchFunc func(*WatchPlan) (uint64, interface{}, error)

// HandlerFunc is used to handle new data
type HandlerFunc func(uint64, interface{})

// Parse takes a watch query and compiles it into a WatchPlan or an error
func Parse(params map[string]interface{}) (*WatchPlan, error) {
	return ParseExempt(params, nil)
}

// ParseExempt takes a watch query and compiles it into a WatchPlan or an error
// Any exempt parameters are stored in the Exempt map
func ParseExempt(params map[string]interface{}, exempt []string) (*WatchPlan, error) {
	plan := &WatchPlan{
		stopCh: make(chan struct{}),
	}

	// Parse the generic parameters
	if err := assignValue(params, "datacenter", &plan.Datacenter); err != nil {
		return nil, err
	}
	if err := assignValue(params, "token", &plan.Token); err != nil {
		return nil, err
	}
	if err := assignValue(params, "type", &plan.Type); err != nil {
		return nil, err
	}

	// Ensure there is a watch type
	if plan.Type == "" {
		return nil, fmt.Errorf("Watch type must be specified")
	}

	// Look for a factory function
	factory := watchFuncFactory[plan.Type]
	if factory == nil {
		return nil, fmt.Errorf("Unsupported watch type: %s", plan.Type)
	}

	// Get the watch func
	fn, err := factory(params)
	if err != nil {
		return nil, err
	}
	plan.Func = fn

	// Remove the exempt parameters
	if len(exempt) > 0 {
		plan.Exempt = make(map[string]interface{})
		for _, ex := range exempt {
			val, ok := params[ex]
			if ok {
				plan.Exempt[ex] = val
				delete(params, ex)
			}
		}
	}

	// Ensure all parameters are consumed
	if len(params) != 0 {
		var bad []string
		for key := range params {
			bad = append(bad, key)
		}
		return nil, fmt.Errorf("Invalid parameters: %v", bad)
	}
	return plan, nil
}

// assignValue is used to extract a value ensuring it is a string
func assignValue(params map[string]interface{}, name string, out *string) error {
	if raw, ok := params[name]; ok {
		val, ok := raw.(string)
		if !ok {
			return fmt.Errorf("Expecting %s to be a string", name)
		}
		*out = val
		delete(params, name)
	}
	return nil
}

// assignValueBool is used to extract a value ensuring it is a bool
func assignValueBool(params map[string]interface{}, name string, out *bool) error {
	if raw, ok := params[name]; ok {
		val, ok := raw.(bool)
		if !ok {
			return fmt.Errorf("Expecting %s to be a boolean", name)
		}
		*out = val
		delete(params, name)
	}
	return nil
}
