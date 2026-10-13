/*
Copyright The Kubernetes Authors.

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

package deviceattribute

import (
	"fmt"
	"io/fs"
	"path/filepath"
	"slices"
	"strconv"
	"strings"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/utils/cpuset"
)

// AttributeForm selects whether the numaNode attribute is published as a
// scalar IntValue or a list IntValues.
type AttributeForm int

const (
	// ScalarAttribute produces a scalar IntValue with the physical NUMA node.
	// This form has no feature-gate dependency.
	ScalarAttribute AttributeForm = iota
	// ListAttribute produces an IntValues list. The first element is the
	// physical NUMA node; additional elements are same-socket nodes at the
	// minimum ACPI SLIT distance. Falls back to a single-element list when
	// SLIT distances are unavailable. Requires DRAListTypeAttributes to be
	// enabled in the cluster.
	ListAttribute
)

// maxNUMANodes is the upper bound for valid NUMA node IDs.
// Linux defines MAX_NUMNODES = (1 << CONFIG_NODES_SHIFT) in
// include/linux/nodemask_types.h (as of v7.2), with CONFIG_NODES_SHIFT
// capped at 10 across all architectures (x86, arm64, riscv).
const maxNUMANodes = 1024

// GetNUMANodeAttributeByPCIBusID returns the numaNode attribute for a PCI
// device, reading the physical NUMA node from sysfs numa_node.
//
// attrForm selects the form of the attribute and MUST reflect the cluster's
// DRAListTypeAttributes state. A driver cannot detect that state (feature gates
// are per-process), so the operator configures it on the driver, consistent
// with what the apiserver and scheduler have enabled. The helper renders
// whichever form it is told; it does not detect anything.
//
// If ListAttribute is used but DRAListTypeAttributes is off in the cluster, the
// apiserver drops the list value and rejects the resulting value-less
// attribute; the driver should treat that publish failure as fatal rather than
// adapting.
func GetNUMANodeAttributeByPCIBusID(pciBusID string, attrForm AttributeForm, mods ...MachineModifier) (DeviceAttribute, error) {
	var mc machine
	initDefaultMachine(&mc)
	for _, mod := range mods {
		mod(&mc)
	}

	if err := verifyPCIBDFFormat(pciBusID); err != nil {
		return DeviceAttribute{}, err
	}

	physicalNode, err := readNUMANode(mc, pciBusID)
	if err != nil {
		return DeviceAttribute{}, err
	}

	// sysfs reports numa_node = -1 for a device with no NUMA affinity. Such a
	// device must not publish numaNode: a value of -1 would match every other
	// NUMA-less device under matchAttribute set intersection. Return an error so
	// the driver omits the attribute instead.
	if physicalNode < 0 {
		return DeviceAttribute{}, fmt.Errorf("PCI device %s has no NUMA affinity (numa_node=%d); do not publish the numaNode attribute for it", pciBusID, physicalNode)
	}

	if attrForm == ListAttribute {
		return makeNUMANodeListAttribute(mc, physicalNode), nil
	}
	return makeNUMANodeScalarAttribute(physicalNode), nil
}

// GetNUMANodeAttribute returns the numaNode attribute for a device that already
// knows its NUMA node (for example CPU and memory devices). See
// GetNUMANodeAttributeByPCIBusID for the meaning of attrForm.
//
// numaNode must be a valid (non-negative) NUMA node. A negative value indicates
// no NUMA affinity, for which the attribute must not be published (it would
// match every other NUMA-less device), so an error is returned.
func GetNUMANodeAttribute(numaNode int, attrForm AttributeForm, mods ...MachineModifier) (DeviceAttribute, error) {
	if numaNode < 0 {
		return DeviceAttribute{}, fmt.Errorf("invalid NUMA node %d: do not publish the numaNode attribute for a device with no NUMA affinity", numaNode)
	}
	if numaNode >= maxNUMANodes {
		return DeviceAttribute{}, fmt.Errorf("invalid NUMA node %d: exceeds maximum supported (%d)", numaNode, maxNUMANodes-1)
	}

	if attrForm == ScalarAttribute {
		return makeNUMANodeScalarAttribute(numaNode), nil
	}

	var mc machine
	initDefaultMachine(&mc)
	for _, mod := range mods {
		mod(&mc)
	}

	return makeNUMANodeListAttribute(mc, numaNode), nil
}

func makeNUMANodeScalarAttribute(numaNode int) DeviceAttribute {
	value := int64(numaNode)
	return DeviceAttribute{
		Name:  StandardDeviceAttributeNUMANode,
		Value: resourceapi.DeviceAttribute{IntValue: &value},
	}
}

// GetNUMANodeForCPU returns the NUMA node ID for a given CPU core by reading
// the /sys/devices/system/cpu/cpuX/nodeY symlink.
func GetNUMANodeForCPU(cpuID int, mods ...MachineModifier) (int, error) {
	var mc machine
	initDefaultMachine(&mc)
	for _, mod := range mods {
		mod(&mc)
	}

	pattern := filepath.Join("devices", "system", "cpu", fmt.Sprintf("cpu%d", cpuID), "node[0-9]*")
	matches, err := fs.Glob(mc.sysfs, pattern)
	if err != nil {
		return -1, fmt.Errorf("failed to find NUMA node for CPU %d: %w", cpuID, err)
	}
	if len(matches) == 0 {
		return -1, fmt.Errorf("CPU %d not found in any NUMA node", cpuID)
	}

	nodeDir := filepath.Base(matches[0])
	nodeNum, err := strconv.Atoi(strings.TrimPrefix(nodeDir, "node"))
	if err != nil {
		return -1, fmt.Errorf("failed to parse NUMA node number from %s: %w", nodeDir, err)
	}
	return nodeNum, nil
}

// makeNUMANodeListAttribute builds the list-form numaNode attribute. The caller
// must pass a non-negative physicalNode; negative nodes (no NUMA affinity) are
// rejected by the exported helpers before reaching here.
func makeNUMANodeListAttribute(mc machine, physicalNode int) DeviceAttribute {
	equidistant, err := getEquidistantNUMANodes(mc, physicalNode)
	if err != nil || len(equidistant) <= 1 {
		return DeviceAttribute{
			Name:  StandardDeviceAttributeNUMANode,
			Value: resourceapi.DeviceAttribute{IntValues: []int64{int64(physicalNode)}},
		}
	}

	// getEquidistantNUMANodes always includes the physical node itself in the
	// returned set, so drop it here and prepend it to keep it first.
	result := []int64{int64(physicalNode)}
	for _, n := range equidistant {
		if n != physicalNode {
			result = append(result, int64(n))
		}
	}

	return DeviceAttribute{
		Name:  StandardDeviceAttributeNUMANode,
		Value: resourceapi.DeviceAttribute{IntValues: result},
	}
}

func readSysfsInt(mc machine, path string) (int, error) {
	data, err := fs.ReadFile(mc.sysfs, path)
	if err != nil {
		return 0, err
	}
	return strconv.Atoi(strings.TrimSpace(string(data)))
}

func readNUMANode(mc machine, pciBusID string) (int, error) {
	numaNodePath := filepath.Join("bus", "pci", "devices", pciBusID, "numa_node")
	node, err := readSysfsInt(mc, numaNodePath)
	if err != nil {
		return -1, fmt.Errorf("failed to read NUMA node for PCI Bus ID %s: %w", pciBusID, err)
	}
	return node, nil
}

// getEquidistantNUMANodes reads the SLIT distance row for the given NUMA node
// and returns all same-socket nodes at the minimum non-self distance, plus
// the node itself.
func getEquidistantNUMANodes(mc machine, node int) ([]int, error) {
	if node < 0 {
		return nil, fmt.Errorf("invalid NUMA node %d", node)
	}

	distPath := filepath.Join("devices", "system", "node", fmt.Sprintf("node%d", node), "distance")
	data, err := fs.ReadFile(mc.sysfs, distPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read SLIT distances for NUMA node %d: %w", node, err)
	}

	fields := strings.Fields(strings.TrimSpace(string(data)))
	if len(fields) == 0 {
		return nil, fmt.Errorf("empty distance file for NUMA node %d", node)
	}

	if node >= len(fields) {
		return nil, fmt.Errorf("NUMA node %d out of range for distance table with %d entries", node, len(fields))
	}

	distances := make([]int, len(fields))
	for i, f := range fields {
		d, err := strconv.Atoi(f)
		if err != nil {
			return nil, fmt.Errorf("failed to parse distance value %q for NUMA node %d: %w", f, node, err)
		}
		distances[i] = d
	}

	// Find minimum non-self distance
	minDist := -1
	for i, d := range distances {
		if i == node {
			continue
		}
		if minDist == -1 || d < minDist {
			minDist = d
		}
	}

	if minDist == -1 {
		return []int{node}, nil
	}

	reportedSocket, haveSocket := getSocketForNUMANode(mc, node)

	var nodes []int
	for i, d := range distances {
		if i == node {
			nodes = append(nodes, i)
			continue
		}
		if d != minDist {
			continue
		}
		if haveSocket {
			// Exclude a candidate only when its socket is known and differs.
			// If the candidate's socket cannot be determined (for example a
			// CPU-less memory node with no cpulist), it is included rather than
			// dropped. This is a deliberate over-approximation: it is safer to
			// include a possibly-same-socket node than to drop a genuinely
			// same-socket one.
			candidateSocket, ok := getSocketForNUMANode(mc, i)
			if ok && candidateSocket != reportedSocket {
				continue
			}
		}
		nodes = append(nodes, i)
	}

	slices.Sort(nodes)
	return nodes, nil
}

func getSocketForNUMANode(mc machine, node int) (int, bool) {
	cpulistPath := filepath.Join("devices", "system", "node", fmt.Sprintf("node%d", node), "cpulist")
	data, err := fs.ReadFile(mc.sysfs, cpulistPath)
	if err != nil {
		return -1, false
	}

	cpus, err := cpuset.Parse(strings.TrimSpace(string(data)))
	if err != nil || cpus.Size() == 0 {
		return -1, false
	}

	pkgPath := filepath.Join("devices", "system", "cpu", fmt.Sprintf("cpu%d", cpus.UnsortedList()[0]), "topology", "physical_package_id")
	socketID, err := readSysfsInt(mc, pkgPath)
	if err != nil {
		return -1, false
	}

	return socketID, true
}
