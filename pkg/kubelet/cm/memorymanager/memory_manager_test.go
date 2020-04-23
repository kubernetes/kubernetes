package memorymanager

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	info "github.com/google/cadvisor/info/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	hugepages2M = "hugepages-2Mi"
	hugepages1G = "hugepages-1Gi"
)

type nodeResources map[v1.ResourceName]resource.Quantity

// validateReservedMemory
func TestValidatePreReservedMemory(t *testing.T) {
	const msgNotEqual = "the total amount of memory of type \"%s\" is not equal to the value determined by Node Allocatable feature"
	testCases := []struct {
		description                string
		nodeAllocatableReservation v1.ResourceList
		preReservedMemory          map[int]map[v1.ResourceName]resource.Quantity
		expectedError              string
	}{
		{
			"Node Allocatable not set, pre-reserved not set",
			v1.ResourceList{},
			map[int]map[v1.ResourceName]resource.Quantity{},
			"",
		},
		{
			"Node Allocatable set to zero, pre-reserved set to zero",
			v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(0, resource.DecimalSI)},
			map[int]map[v1.ResourceName]resource.Quantity{
				0: nodeResources{v1.ResourceMemory: *resource.NewQuantity(0, resource.DecimalSI)},
			},
			"",
		},
		{
			"Node Allocatable not set (equal zero), pre-reserved set",
			v1.ResourceList{},
			map[int]map[v1.ResourceName]resource.Quantity{
				0: nodeResources{v1.ResourceMemory: *resource.NewQuantity(12, resource.DecimalSI)},
			},
			fmt.Sprintf(msgNotEqual, v1.ResourceMemory),
		},
		{
			"Node Allocatable set, pre-reserved not set",
			v1.ResourceList{hugepages2M: *resource.NewQuantity(5, resource.DecimalSI)},
			map[int]map[v1.ResourceName]resource.Quantity{},
			fmt.Sprintf(msgNotEqual, hugepages2M),
		},
		{
			"Pre-reserved not equal to Node Allocatable",
			v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(5, resource.DecimalSI)},
			map[int]map[v1.ResourceName]resource.Quantity{
				0: nodeResources{v1.ResourceMemory: *resource.NewQuantity(12, resource.DecimalSI)},
			},
			fmt.Sprintf(msgNotEqual, v1.ResourceMemory),
		},
		{
			"Pre-reserved total equal to Node Allocatable",
			v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(17, resource.DecimalSI),
				hugepages2M: *resource.NewQuantity(77, resource.DecimalSI),
				hugepages1G: *resource.NewQuantity(13, resource.DecimalSI)},
			map[int]map[v1.ResourceName]resource.Quantity{
				0: nodeResources{v1.ResourceMemory: *resource.NewQuantity(12, resource.DecimalSI),
					hugepages2M: *resource.NewQuantity(70, resource.DecimalSI),
					hugepages1G: *resource.NewQuantity(13, resource.DecimalSI)},
				1: nodeResources{v1.ResourceMemory: *resource.NewQuantity(5, resource.DecimalSI),
					hugepages2M: *resource.NewQuantity(7, resource.DecimalSI)},
			},
			"",
		},
		{
			"Pre-reserved total hugapages-2M not equal to Node Allocatable",
			v1.ResourceList{v1.ResourceMemory: *resource.NewQuantity(17, resource.DecimalSI),
				hugepages2M: *resource.NewQuantity(14, resource.DecimalSI),
				hugepages1G: *resource.NewQuantity(13, resource.DecimalSI)},
			map[int]map[v1.ResourceName]resource.Quantity{
				0: nodeResources{v1.ResourceMemory: *resource.NewQuantity(12, resource.DecimalSI),
					hugepages2M: *resource.NewQuantity(70, resource.DecimalSI),
					hugepages1G: *resource.NewQuantity(13, resource.DecimalSI)},
				1: nodeResources{v1.ResourceMemory: *resource.NewQuantity(5, resource.DecimalSI),
					hugepages2M: *resource.NewQuantity(7, resource.DecimalSI)},
			},
			fmt.Sprintf(msgNotEqual, hugepages2M),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			err := validateReservedMemory(tc.nodeAllocatableReservation, tc.preReservedMemory)
			if strings.TrimSpace(tc.expectedError) != "" {
				assert.Error(t, err)
				assert.Equal(t, err.Error(), tc.expectedError)
			}
		})
	}
}

func TestConvertPreReserved(t *testing.T) {
	machineInfo := info.MachineInfo{
		Topology: []info.Node{
			info.Node{Id: 0},
			info.Node{Id: 1},
		},
	}

	testCases := []struct {
		description      string
		reserved         map[int]map[v1.ResourceName]resource.Quantity
		reservedExpected reservedMemory
		expectedError    string
	}{
		{
			"Empty",
			map[int]map[v1.ResourceName]resource.Quantity{},
			reservedMemory{
				0: map[v1.ResourceName]uint64{},
				1: map[v1.ResourceName]uint64{},
			},
			"",
		},
		{
			"Single NUMA node is pre-reserved",
			map[int]map[v1.ResourceName]resource.Quantity{
				0: nodeResources{v1.ResourceMemory: *resource.NewQuantity(12, resource.DecimalSI),
					hugepages2M: *resource.NewQuantity(70, resource.DecimalSI),
					hugepages1G: *resource.NewQuantity(13, resource.DecimalSI)},
			},
			reservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 12,
					hugepages2M:       70,
					hugepages1G:       13,
				},
				1: map[v1.ResourceName]uint64{},
			},
			"",
		},
		{
			"Both NUMA nodes are pre-reserved",
			map[int]map[v1.ResourceName]resource.Quantity{
				0: nodeResources{v1.ResourceMemory: *resource.NewQuantity(12, resource.DecimalSI),
					hugepages2M: *resource.NewQuantity(70, resource.DecimalSI),
					hugepages1G: *resource.NewQuantity(13, resource.DecimalSI)},
				1: nodeResources{v1.ResourceMemory: *resource.NewQuantity(5, resource.DecimalSI),
					hugepages2M: *resource.NewQuantity(7, resource.DecimalSI)},
			},
			reservedMemory{
				0: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 12,
					hugepages2M:       70,
					hugepages1G:       13,
				},
				1: map[v1.ResourceName]uint64{
					v1.ResourceMemory: 5,
					hugepages2M:       7,
				},
			},
			"",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			reserved, _ := convertReserved(&machineInfo, tc.reserved)
			if !reflect.DeepEqual(reserved, tc.reservedExpected) {
				t.Errorf("got %v, expected %v", reserved, tc.reservedExpected)
			}
		})
	}
}
