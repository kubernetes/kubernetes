package extendedserverattributes

// ServerAttributesExt represents basic OS-EXT-SRV-ATTR server response fields.
// You should use extract methods from microversions.go to retrieve additional
// fields.
type ServerAttributesExt struct {
	Host               string `json:"OS-EXT-SRV-ATTR:host"`
	InstanceName       string `json:"OS-EXT-SRV-ATTR:instance_name"`
	HypervisorHostname string `json:"OS-EXT-SRV-ATTR:hypervisor_hostname"`
}
