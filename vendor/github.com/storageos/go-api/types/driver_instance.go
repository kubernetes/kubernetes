package types

import "encoding/gob"

// DriverInstance is used to define an instance of a storage capacity driver.
type DriverInstance struct {

	// Instance unique ID.
	// Read Only: true
	ID string `json:"id"`

	// Instance name.
	Name string `json:"name"`

	// Instance description.
	Description string `json:"description"`

	// Flag describing whether the template is active.
	// Default: false
	Active bool `json:"active"`

	// Config is JSON struct that is passed directly to the driver.  There is no
	// specific format, and the driver is responsible for validation.
	Config interface{} `json:"config"`

	// Labels define a list of labels that describe the driver instance.  These
	// are inherited from the pool when the driver instance is created.
	Labels []string `json:"labels"`

	// ControllerName specifies the controller that this instance is running on.
	ControllerName string `json:"controllerName"`

	// PoolID refers to the pool that this driver instance relates to.
	PoolID string `json:"poolID"`

	// DriverName specifies which capacity driver this is an instance of.
	DriverName string `json:"driverName"`

	// CapacityStats tracks that capacity usage of this driver instance on the
	// current controller.
	CapacityStats CapacityStats `json:"capacityStats"`
}

// DriverInstances is a collection of Driver instance objects.
type DriverInstances []*DriverInstance

func init() {
	gob.Register(DriverInstance{})
	gob.Register([]interface{}{})
}

// Find an instance matching the parameters.
func (i *DriverInstances) Find(pool string, driver string, controller string) *DriverInstance {

	for _, inst := range *i {
		if inst.PoolID == pool && inst.DriverName == driver && inst.ControllerName == controller {
			return inst
		}
	}
	return nil
}

// Add a new instance to the list of instances.
func (i *DriverInstances) Add(new *DriverInstance) {

	for _, inst := range *i {
		// Skip if it already exists
		if inst.PoolID == new.PoolID && inst.DriverName == new.DriverName && inst.ControllerName == new.ControllerName {
			return
		}
	}
	*i = append(*i, new)
}

// Remove an instance to the list of instances.
func (i *DriverInstances) Remove(id string) {

	// TODO: not working
	// for ndx, inst := range *i {
	// 	if inst.ID == id {
	// 		// splice out the item to remove
	// 		*i = append(*i[:ndx], *i[ndx+1:]...)
	// 		return
	// 	}
	// }
}
