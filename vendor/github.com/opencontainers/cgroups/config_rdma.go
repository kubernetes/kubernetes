package cgroups

// LinuxRdma for Linux cgroup 'rdma' resource management (Linux 4.11)
type LinuxRdma struct {
	// Maximum number of HCA handles that can be opened. Default is "no limit".
	HcaHandles *uint32 `json:"hca_handles,omitempty"`
	// Maximum number of HCA objects that can be created. Default is "no limit".
	HcaObjects *uint32 `json:"hca_objects,omitempty"`
}
