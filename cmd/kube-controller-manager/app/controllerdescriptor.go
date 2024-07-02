package app

import (
	"context"
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-base/featuregate"
	"k8s.io/controller-manager/controller"
)

// NewControllerDescriptors is a public map of named controller groups (you can start more than one in an init func)
// paired to their ControllerDescriptor wrapper object that includes InitFunc.
// This allows for structured downstream composition and subdivision.
func NewControllerDescriptors() map[string]*ControllerDescriptor {
	return DefaultControllerDescRegistry.Registered()
}

var DefaultControllerDescRegistry ControllerDescriptorRegistry = &controllerDescriptorRegistry{
	controllers: make(map[string]*ControllerDescriptor),
	alias:       sets.NewString(),
}

type ControllerDescriptorRegistry interface {
	Register(descriptor *ControllerDescriptor)
	Registered() map[string]*ControllerDescriptor
}

var _ ControllerDescriptorRegistry = &controllerDescriptorRegistry{}

type controllerDescriptorRegistry struct {
	sync.Mutex
	controllers     map[string]*ControllerDescriptor
	controllerNames sets.String
	alias           sets.String
}

func (c *controllerDescriptorRegistry) Register(controllerDesc *ControllerDescriptor) {
	c.Lock()
	defer c.Unlock()
	if controllerDesc == nil {
		panic("received nil controller for a registration")
	}
	name := controllerDesc.Name()
	if len(name) == 0 {
		panic("received controller without a name for a registration")
	}
	if _, found := c.controllers[name]; found {
		panic(fmt.Sprintf("controller name %q was registered twice", name))
	}
	if found := c.alias.Has(name); found {
		panic(fmt.Sprintf("controller %q has a duplicate alias", name))
	}
	if controllerDesc.GetInitFunc() == nil {
		panic(fmt.Sprintf("controller %q does not have an init function", name))
	}

	if duplicateAlias := c.alias.Intersection(sets.NewString(controllerDesc.GetAliases()...)); duplicateAlias.Len() > 0 {
		panic(fmt.Sprintf("controller %q has a duplicate alias %q", name, duplicateAlias.List()))
	}
	c.alias.Insert(controllerDesc.GetAliases()...)
	c.controllerNames.Insert(name)
	c.controllers[name] = controllerDesc
}

func (c *controllerDescriptorRegistry) Registered() map[string]*ControllerDescriptor {
	c.Lock()
	defer c.Unlock()
	out := make(map[string]*ControllerDescriptor, len(c.controllers))
	for k, v := range c.controllers {
		out[k] = v
	}
	return out
}

// InitFunc is used to launch a particular controller. It returns a controller
// that can optionally implement other interfaces so that the controller manager
// can support the requested features.
// The returned controller may be nil, which will be considered an anonymous controller
// that requests no additional features from the controller manager.
// Any error returned will cause the controller process to `Fatal`
// The bool indicates whether the controller was enabled.
type InitFunc func(ctx context.Context, controllerContext ControllerContext, controllerName string) (controller controller.Interface, enabled bool, err error)

type ControllerDescriptor struct {
	name                      string
	initFunc                  InitFunc
	requiredFeatureGates      []featuregate.Feature
	aliases                   []string
	isDisabledByDefault       bool
	isCloudProviderController bool
	requiresSpecialHandling   bool
}

func (r *ControllerDescriptor) Name() string {
	return r.name
}

func (r *ControllerDescriptor) GetInitFunc() InitFunc {
	return r.initFunc
}

func (r *ControllerDescriptor) GetRequiredFeatureGates() []featuregate.Feature {
	return append([]featuregate.Feature(nil), r.requiredFeatureGates...)
}

// GetAliases returns aliases to ensure backwards compatibility and should never be removed!
// Only addition of new aliases is allowed, and only when a canonical name is changed (please see CHANGE POLICY of controller names)
func (r *ControllerDescriptor) GetAliases() []string {
	return append([]string(nil), r.aliases...)
}

func (r *ControllerDescriptor) IsDisabledByDefault() bool {
	return r.isDisabledByDefault
}

func (r *ControllerDescriptor) IsCloudProviderController() bool {
	return r.isCloudProviderController
}

// RequiresSpecialHandling should return true only in a special non-generic controllers like ServiceAccountTokenController
func (r *ControllerDescriptor) RequiresSpecialHandling() bool {
	return r.requiresSpecialHandling
}
