package resource

// Name is the name identifying various resources in a List.
type Name string

// List is a set of (resource name, quantity) pairs.
type List map[Name]Quantity

// Returns string version of Name.
func (self Name) String() string {
	return string(self)
}

// A set of canonical resource types
// TODO remove these helpers, they invert type

// Resource names must be not more than 63 characters, consisting of upper- or lower-case alphanumeric characters,
// with the -, _, and . characters allowed anywhere, except the first or last character.
// The default convention, matching that for annotations, is to use lower-case names, with dashes, rather than
// camel case, separating compound words.
// Fully-qualified resource typenames are constructed from a DNS-style subdomain, followed by a slash `/` and a name.
const (
	// CPU, in cores. (500m = .5 cores)
	ResourceCPU Name = "cpu"
	// Memory, in bytes. (500Gi = 500GiB = 500 * 1024 * 1024 * 1024)
	ResourceMemory Name = "memory"
	// Volume size, in bytes (e,g. 5Gi = 5GiB = 5 * 1024 * 1024 * 1024)
	ResourceStorage Name = "storage"
	// NVIDIA GPU, in devices. Alpha, might change: although fractional and allowing values >1, only one whole device per node is assigned.
	ResourceNvidiaGPU Name = "alpha.kubernetes.io/nvidia-gpu"
	// Pods, number
	ResourcePods Name = "pods"
)

// Returns the CPU limit if specified.
func (self *List) Cpu() *Quantity {
	if val, ok := (*self)[ResourceCPU]; ok {
		return &val
	}
	return &Quantity{Format: DecimalSI}
}

// Returns the Memory limit if specified.
func (self *List) Memory() *Quantity {
	if val, ok := (*self)[ResourceMemory]; ok {
		return &val
	}
	return &Quantity{Format: BinarySI}
}

func (self *List) Pods() *Quantity {
	if val, ok := (*self)[ResourcePods]; ok {
		return &val
	}
	return &Quantity{}
}

func (self *List) NvidiaGPU() *Quantity {
	if val, ok := (*self)[ResourceNvidiaGPU]; ok {
		return &val
	}
	return &Quantity{}
}
