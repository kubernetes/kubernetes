package types

import (
	"encoding/json"
	"errors"

	"k8s.io/kubernetes/pkg/api/resource"
)

var (
	ErrDefaultTrue     = errors.New("default must be false")
	ErrDefaultRequired = errors.New("default must be true")
	ErrRequestNonEmpty = errors.New("request not supported by this resource, must be empty")
)

const (
	ResourceBlockBandwidthName   = "resource/block-bandwidth"
	ResourceBlockIOPSName        = "resource/block-iops"
	ResourceCPUName              = "resource/cpu"
	ResourceMemoryName           = "resource/memory"
	ResourceNetworkBandwidthName = "resource/network-bandwidth"
)

func init() {
	AddIsolatorValueConstructor(ResourceBlockBandwidthName, NewResourceBlockBandwidth)
	AddIsolatorValueConstructor(ResourceBlockIOPSName, NewResourceBlockIOPS)
	AddIsolatorValueConstructor(ResourceCPUName, NewResourceCPU)
	AddIsolatorValueConstructor(ResourceMemoryName, NewResourceMemory)
	AddIsolatorValueConstructor(ResourceNetworkBandwidthName, NewResourceNetworkBandwidth)
}

func NewResourceBlockBandwidth() IsolatorValue {
	return &ResourceBlockBandwidth{}
}
func NewResourceBlockIOPS() IsolatorValue {
	return &ResourceBlockIOPS{}
}
func NewResourceCPU() IsolatorValue {
	return &ResourceCPU{}
}
func NewResourceNetworkBandwidth() IsolatorValue {
	return &ResourceNetworkBandwidth{}
}
func NewResourceMemory() IsolatorValue {
	return &ResourceMemory{}
}

type Resource interface {
	Limit() *resource.Quantity
	Request() *resource.Quantity
	Default() bool
}

type ResourceBase struct {
	val resourceValue
}

type resourceValue struct {
	Default bool               `json:"default"`
	Request *resource.Quantity `json:"request"`
	Limit   *resource.Quantity `json:"limit"`
}

func (r ResourceBase) Limit() *resource.Quantity {
	return r.val.Limit
}
func (r ResourceBase) Request() *resource.Quantity {
	return r.val.Request
}
func (r ResourceBase) Default() bool {
	return r.val.Default
}

func (r *ResourceBase) UnmarshalJSON(b []byte) error {
	return json.Unmarshal(b, &r.val)
}

func (r ResourceBase) AssertValid() error {
	return nil
}

type ResourceBlockBandwidth struct {
	ResourceBase
}

func (r ResourceBlockBandwidth) AssertValid() error {
	if r.Default() != true {
		return ErrDefaultRequired
	}
	if r.Request() != nil {
		return ErrRequestNonEmpty
	}
	return nil
}

type ResourceBlockIOPS struct {
	ResourceBase
}

func (r ResourceBlockIOPS) AssertValid() error {
	if r.Default() != true {
		return ErrDefaultRequired
	}
	if r.Request() != nil {
		return ErrRequestNonEmpty
	}
	return nil
}

type ResourceCPU struct {
	ResourceBase
}

func (r ResourceCPU) AssertValid() error {
	if r.Default() != false {
		return ErrDefaultTrue
	}
	return nil
}

type ResourceMemory struct {
	ResourceBase
}

func (r ResourceMemory) AssertValid() error {
	if r.Default() != false {
		return ErrDefaultTrue
	}
	return nil
}

type ResourceNetworkBandwidth struct {
	ResourceBase
}

func (r ResourceNetworkBandwidth) AssertValid() error {
	if r.Default() != true {
		return ErrDefaultRequired
	}
	if r.Request() != nil {
		return ErrRequestNonEmpty
	}
	return nil
}
