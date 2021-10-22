package hypervisors

import (
	"encoding/json"
	"fmt"
	"strconv"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Topology represents a CPU Topology.
type Topology struct {
	Sockets int `json:"sockets"`
	Cores   int `json:"cores"`
	Threads int `json:"threads"`
}

// CPUInfo represents CPU information of the hypervisor.
type CPUInfo struct {
	Vendor   string   `json:"vendor"`
	Arch     string   `json:"arch"`
	Model    string   `json:"model"`
	Features []string `json:"features"`
	Topology Topology `json:"topology"`
}

// Service represents a Compute service running on the hypervisor.
type Service struct {
	Host           string `json:"host"`
	ID             string `json:"-"`
	DisabledReason string `json:"disabled_reason"`
}

func (r *Service) UnmarshalJSON(b []byte) error {
	type tmp Service
	var s struct {
		tmp
		ID interface{} `json:"id"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = Service(s.tmp)

	// OpenStack Compute service returns ID in string representation since
	// 2.53 microversion API (Pike release).
	switch t := s.ID.(type) {
	case int:
		r.ID = strconv.Itoa(t)
	case float64:
		r.ID = strconv.Itoa(int(t))
	case string:
		r.ID = t
	default:
		return fmt.Errorf("ID has unexpected type: %T", t)
	}

	return nil
}

// Hypervisor represents a hypervisor in the OpenStack cloud.
type Hypervisor struct {
	// A structure that contains cpu information like arch, model, vendor,
	// features and topology.
	CPUInfo CPUInfo `json:"-"`

	// The current_workload is the number of tasks the hypervisor is responsible
	// for. This will be equal or greater than the number of active VMs on the
	// system (it can be greater when VMs are being deleted and the hypervisor is
	// still cleaning up).
	CurrentWorkload int `json:"current_workload"`

	// Status of the hypervisor, either "enabled" or "disabled".
	Status string `json:"status"`

	// State of the hypervisor, either "up" or "down".
	State string `json:"state"`

	// DiskAvailableLeast is the actual free disk on this hypervisor,
	// measured in GB.
	DiskAvailableLeast int `json:"disk_available_least"`

	// HostIP is the hypervisor's IP address.
	HostIP string `json:"host_ip"`

	// FreeDiskGB is the free disk remaining on the hypervisor, measured in GB.
	FreeDiskGB int `json:"-"`

	// FreeRAMMB is the free RAM in the hypervisor, measured in MB.
	FreeRamMB int `json:"free_ram_mb"`

	// HypervisorHostname is the hostname of the hypervisor.
	HypervisorHostname string `json:"hypervisor_hostname"`

	// HypervisorType is the type of hypervisor.
	HypervisorType string `json:"hypervisor_type"`

	// HypervisorVersion is the version of the hypervisor.
	HypervisorVersion int `json:"-"`

	// ID is the unique ID of the hypervisor.
	ID string `json:"-"`

	// LocalGB is the disk space in the hypervisor, measured in GB.
	LocalGB int `json:"-"`

	// LocalGBUsed is the used disk space of the  hypervisor, measured in GB.
	LocalGBUsed int `json:"local_gb_used"`

	// MemoryMB is the total memory of the hypervisor, measured in MB.
	MemoryMB int `json:"memory_mb"`

	// MemoryMBUsed is the used memory of the hypervisor, measured in MB.
	MemoryMBUsed int `json:"memory_mb_used"`

	// RunningVMs is the The number of running vms on the hypervisor.
	RunningVMs int `json:"running_vms"`

	// Service is the service this hypervisor represents.
	Service Service `json:"service"`

	// VCPUs is the total number of vcpus on the hypervisor.
	VCPUs int `json:"vcpus"`

	// VCPUsUsed is the number of used vcpus on the hypervisor.
	VCPUsUsed int `json:"vcpus_used"`
}

func (r *Hypervisor) UnmarshalJSON(b []byte) error {
	type tmp Hypervisor
	var s struct {
		tmp
		ID                interface{} `json:"id"`
		CPUInfo           interface{} `json:"cpu_info"`
		HypervisorVersion interface{} `json:"hypervisor_version"`
		FreeDiskGB        interface{} `json:"free_disk_gb"`
		LocalGB           interface{} `json:"local_gb"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = Hypervisor(s.tmp)

	// Newer versions return the CPU info as the correct type.
	// Older versions return the CPU info as a string and need to be
	// unmarshalled by the json parser.
	var tmpb []byte

	switch t := s.CPUInfo.(type) {
	case string:
		tmpb = []byte(t)
	case map[string]interface{}:
		tmpb, err = json.Marshal(t)
		if err != nil {
			return err
		}
	default:
		return fmt.Errorf("CPUInfo has unexpected type: %T", t)
	}

	if len(tmpb) != 0 {
		err = json.Unmarshal(tmpb, &r.CPUInfo)
		if err != nil {
			return err
		}
	}

	// These fields may be returned as a scientific notation, so they need
	// converted to int.
	switch t := s.HypervisorVersion.(type) {
	case int:
		r.HypervisorVersion = t
	case float64:
		r.HypervisorVersion = int(t)
	default:
		return fmt.Errorf("Hypervisor version has unexpected type: %T", t)
	}

	switch t := s.FreeDiskGB.(type) {
	case int:
		r.FreeDiskGB = t
	case float64:
		r.FreeDiskGB = int(t)
	default:
		return fmt.Errorf("Free disk GB has unexpected type: %T", t)
	}

	switch t := s.LocalGB.(type) {
	case int:
		r.LocalGB = t
	case float64:
		r.LocalGB = int(t)
	default:
		return fmt.Errorf("Local GB has unexpected type: %T", t)
	}

	// OpenStack Compute service returns ID in string representation since
	// 2.53 microversion API (Pike release).
	switch t := s.ID.(type) {
	case int:
		r.ID = strconv.Itoa(t)
	case float64:
		r.ID = strconv.Itoa(int(t))
	case string:
		r.ID = t
	default:
		return fmt.Errorf("ID has unexpected type: %T", t)
	}

	return nil
}

// HypervisorPage represents a single page of all Hypervisors from a List
// request.
type HypervisorPage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a HypervisorPage is empty.
func (page HypervisorPage) IsEmpty() (bool, error) {
	va, err := ExtractHypervisors(page)
	return len(va) == 0, err
}

// ExtractHypervisors interprets a page of results as a slice of Hypervisors.
func ExtractHypervisors(p pagination.Page) ([]Hypervisor, error) {
	var h struct {
		Hypervisors []Hypervisor `json:"hypervisors"`
	}
	err := (p.(HypervisorPage)).ExtractInto(&h)
	return h.Hypervisors, err
}

type HypervisorResult struct {
	gophercloud.Result
}

// Extract interprets any HypervisorResult as a Hypervisor, if possible.
func (r HypervisorResult) Extract() (*Hypervisor, error) {
	var s struct {
		Hypervisor Hypervisor `json:"hypervisor"`
	}
	err := r.ExtractInto(&s)
	return &s.Hypervisor, err
}

// Statistics represents a summary statistics for all enabled
// hypervisors over all compute nodes in the OpenStack cloud.
type Statistics struct {
	// The number of hypervisors.
	Count int `json:"count"`

	// The current_workload is the number of tasks the hypervisor is responsible for
	CurrentWorkload int `json:"current_workload"`

	// The actual free disk on this hypervisor(in GB).
	DiskAvailableLeast int `json:"disk_available_least"`

	// The free disk remaining on this hypervisor(in GB).
	FreeDiskGB int `json:"free_disk_gb"`

	// The free RAM in this hypervisor(in MB).
	FreeRamMB int `json:"free_ram_mb"`

	// The disk in this hypervisor(in GB).
	LocalGB int `json:"local_gb"`

	// The disk used in this hypervisor(in GB).
	LocalGBUsed int `json:"local_gb_used"`

	// The memory of this hypervisor(in MB).
	MemoryMB int `json:"memory_mb"`

	// The memory used in this hypervisor(in MB).
	MemoryMBUsed int `json:"memory_mb_used"`

	// The total number of running vms on all hypervisors.
	RunningVMs int `json:"running_vms"`

	// The number of vcpu in this hypervisor.
	VCPUs int `json:"vcpus"`

	// The number of vcpu used in this hypervisor.
	VCPUsUsed int `json:"vcpus_used"`
}

type StatisticsResult struct {
	gophercloud.Result
}

// Extract interprets any StatisticsResult as a Statistics, if possible.
func (r StatisticsResult) Extract() (*Statistics, error) {
	var s struct {
		Stats Statistics `json:"hypervisor_statistics"`
	}
	err := r.ExtractInto(&s)
	return &s.Stats, err
}

// Uptime represents uptime and additional info for a specific hypervisor.
type Uptime struct {
	// The hypervisor host name provided by the Nova virt driver.
	// For the Ironic driver, it is the Ironic node uuid.
	HypervisorHostname string `json:"hypervisor_hostname"`

	// The id of the hypervisor.
	ID string `json:"-"`

	// The state of the hypervisor. One of up or down.
	State string `json:"state"`

	// The status of the hypervisor. One of enabled or disabled.
	Status string `json:"status"`

	// The total uptime of the hypervisor and information about average load.
	Uptime string `json:"uptime"`
}

func (r *Uptime) UnmarshalJSON(b []byte) error {
	type tmp Uptime
	var s struct {
		tmp
		ID interface{} `json:"id"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = Uptime(s.tmp)

	// OpenStack Compute service returns ID in string representation since
	// 2.53 microversion API (Pike release).
	switch t := s.ID.(type) {
	case int:
		r.ID = strconv.Itoa(t)
	case float64:
		r.ID = strconv.Itoa(int(t))
	case string:
		r.ID = t
	default:
		return fmt.Errorf("ID has unexpected type: %T", t)
	}

	return nil
}

type UptimeResult struct {
	gophercloud.Result
}

// Extract interprets any UptimeResult as a Uptime, if possible.
func (r UptimeResult) Extract() (*Uptime, error) {
	var s struct {
		Uptime Uptime `json:"hypervisor"`
	}
	err := r.ExtractInto(&s)
	return &s.Uptime, err
}
