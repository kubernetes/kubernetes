package introspection

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type introspectionResult struct {
	gophercloud.Result
}

// Extract interprets any introspectionResult as an Introspection, if possible.
func (r introspectionResult) Extract() (*Introspection, error) {
	var s Introspection
	err := r.ExtractInto(&s)
	return &s, err
}

// ExtractInto will extract a response body into an Introspection struct.
func (r introspectionResult) ExtractInto(v interface{}) error {
	return r.Result.ExtractIntoStructPtr(v, "")
}

// ExtractIntrospectionsInto will extract a collection of introspectResult pages into a
// slice of Introspection entities.
func ExtractIntrospectionsInto(r pagination.Page, v interface{}) error {
	return r.(IntrospectionPage).Result.ExtractIntoSlicePtr(v, "introspection")
}

// ExtractIntrospections interprets the results of a single page from a
// ListIntrospections() call, producing a slice of Introspection entities.
func ExtractIntrospections(r pagination.Page) ([]Introspection, error) {
	var s []Introspection
	err := ExtractIntrospectionsInto(r, &s)
	return s, err
}

// IntrospectionPage abstracts the raw results of making a ListIntrospections()
// request against the Inspector API. As OpenStack extensions may freely alter
// the response bodies of structures returned to the client, you may only safely
// access the data provided through the ExtractIntrospections call.
type IntrospectionPage struct {
	pagination.LinkedPageBase
}

// Introspection represents an introspection in the OpenStack Bare Metal Introspector API.
type Introspection struct {
	// Whether introspection is finished
	Finished bool `json:"finished"`

	// State of the introspection
	State string `json:"state"`

	// Error message, can be null; "Canceled by operator" in case introspection was aborted
	Error string `json:"error"`

	// UUID of the introspection
	UUID string `json:"uuid"`

	// UTC ISO8601 timestamp
	StartedAt time.Time `json:"-"`

	// UTC ISO8601 timestamp or null
	FinishedAt time.Time `json:"-"`

	// Link to the introspection URL
	Links []interface{} `json:"links"`
}

// IsEmpty returns true if a page contains no Introspection results.
func (r IntrospectionPage) IsEmpty() (bool, error) {
	s, err := ExtractIntrospections(r)
	return len(s) == 0, err
}

// NextPageURL uses the response's embedded link reference to navigate to the
// next page of results.
func (r IntrospectionPage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"introspection_links"`
	}
	err := r.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// UnmarshalJSON trie to convert values for started_at and finished_at from the
// json response into RFC3339 standard. Since Introspection API can remove the
// Z from the format, if the conversion fails, it falls back to an RFC3339
// with no Z format supported by gophercloud.
func (r *Introspection) UnmarshalJSON(b []byte) error {
	type tmp Introspection
	var s struct {
		tmp
		StartedAt  string `json:"started_at"`
		FinishedAt string `json:"finished_at"`
	}

	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}

	*r = Introspection(s.tmp)

	if s.StartedAt != "" {
		t, err := time.Parse(time.RFC3339, s.StartedAt)
		if err != nil {
			t, err = time.Parse(gophercloud.RFC3339NoZ, s.StartedAt)
			if err != nil {
				return err
			}
		}
		r.StartedAt = t
	}

	if s.FinishedAt != "" {
		t, err := time.Parse(time.RFC3339, s.FinishedAt)
		if err != nil {
			t, err = time.Parse(gophercloud.RFC3339NoZ, s.FinishedAt)
			if err != nil {
				return err
			}
		}
		r.FinishedAt = t
	}

	return nil
}

// GetIntrospectionStatusResult is the response from a GetIntrospectionStatus operation.
// Call its Extract method to interpret it as an Introspection.
type GetIntrospectionStatusResult struct {
	introspectionResult
}

// StartResult is the response from a StartIntrospection operation.
// Call its ExtractErr method to determine if the call succeeded or failed.
type StartResult struct {
	gophercloud.ErrResult
}

// AbortResult is the response from an AbortIntrospection operation.
// Call its ExtractErr method to determine if the call succeeded or failed.
type AbortResult struct {
	gophercloud.ErrResult
}

// Data represents the full introspection data collected.
// The format and contents of the stored data depends on the ramdisk used
// and plugins enabled both in the ramdisk and in inspector itself.
// This structure has been provided for basic compatibility but it
// will need extensions
type Data struct {
	AllInterfaces map[string]BaseInterfaceType `json:"all_interfaces"`
	BootInterface string                       `json:"boot_interface"`
	CPUArch       string                       `json:"cpu_arch"`
	CPUs          int                          `json:"cpus"`
	Error         string                       `json:"error"`
	Interfaces    map[string]BaseInterfaceType `json:"interfaces"`
	Inventory     InventoryType                `json:"inventory"`
	IPMIAddress   string                       `json:"ipmi_address"`
	LocalGB       int                          `json:"local_gb"`
	MACs          []string                     `json:"macs"`
	MemoryMB      int                          `json:"memory_mb"`
	RootDisk      RootDiskType                 `json:"root_disk"`
}

// Sub Types defined under Data and deeper in the structure

type BaseInterfaceType struct {
	ClientID      string                 `json:"client_id"`
	IP            string                 `json:"ip"`
	MAC           string                 `json:"mac"`
	PXE           bool                   `json:"pxe"`
	LLDPProcessed map[string]interface{} `json:"lldp_processed"`
}

type BootInfoType struct {
	CurrentBootMode string `json:"current_boot_mode"`
	PXEInterface    string `json:"pxe_interface"`
}

type CPUType struct {
	Architecture string   `json:"architecture"`
	Count        int      `json:"count"`
	Flags        []string `json:"flags"`
	Frequency    string   `json:"frequency"`
	ModelName    string   `json:"model_name"`
}

type LLDPTLVType struct {
	Type  int
	Value string
}

type InterfaceType struct {
	BIOSDevName string        `json:"biosdevname"`
	ClientID    string        `json:"client_id"`
	HasCarrier  bool          `json:"has_carrier"`
	IPV4Address string        `json:"ipv4_address"`
	IPV6Address string        `json:"ipv6_address"`
	LLDP        []LLDPTLVType `json:"lldp"`
	MACAddress  string        `json:"mac_address"`
	Name        string        `json:"name"`
	Product     string        `json:"product"`
	Vendor      string        `json:"vendor"`
}

type InventoryType struct {
	BmcAddress   string           `json:"bmc_address"`
	Boot         BootInfoType     `json:"boot"`
	CPU          CPUType          `json:"cpu"`
	Disks        []RootDiskType   `json:"disks"`
	Interfaces   []InterfaceType  `json:"interfaces"`
	Memory       MemoryType       `json:"memory"`
	SystemVendor SystemVendorType `json:"system_vendor"`
}

type MemoryType struct {
	PhysicalMb int `json:"physical_mb"`
	Total      int `json:"total"`
}

type RootDiskType struct {
	Hctl               string `json:"hctl"`
	Model              string `json:"model"`
	Name               string `json:"name"`
	Rotational         bool   `json:"rotational"`
	Serial             string `json:"serial"`
	Size               int    `json:"size"`
	Vendor             string `json:"vendor"`
	Wwn                string `json:"wwn"`
	WwnVendorExtension string `json:"wwn_vendor_extension"`
	WwnWithExtension   string `json:"wwn_with_extension"`
}

type SystemVendorType struct {
	Manufacturer string `json:"manufacturer"`
	ProductName  string `json:"product_name"`
	SerialNumber string `json:"serial_number"`
}

// UnmarshalJSON interprets an LLDP TLV [key, value] pair as an LLDPTLVType structure
func (r *LLDPTLVType) UnmarshalJSON(data []byte) error {
	var list []interface{}
	if err := json.Unmarshal(data, &list); err != nil {
		return err
	}

	if len(list) != 2 {
		return fmt.Errorf("Invalid LLDP TLV key-value pair")
	}

	fieldtype, ok := list[0].(float64)
	if !ok {
		return fmt.Errorf("LLDP TLV key is not number")
	}

	value, ok := list[1].(string)
	if !ok {
		return fmt.Errorf("LLDP TLV value is not string")
	}

	r.Type = int(fieldtype)
	r.Value = value
	return nil
}

// Extract interprets any IntrospectionDataResult as IntrospectionData, if possible.
func (r DataResult) Extract() (*Data, error) {
	var s Data
	err := r.ExtractInto(&s)
	return &s, err
}

// DataResult represents the response from a GetIntrospectionData operation.
// Call its Extract method to interpret it as a Data.
type DataResult struct {
	gophercloud.Result
}

// ApplyDataResult is the response from an ApplyData operation.
// Call its ExtractErr method to determine if the call succeeded or failed.
type ApplyDataResult struct {
	gophercloud.ErrResult
}
