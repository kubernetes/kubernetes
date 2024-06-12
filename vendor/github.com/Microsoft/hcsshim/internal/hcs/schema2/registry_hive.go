package hcsschema

// NOTE: manually added

type RegistryHive string

// List of RegistryHive
const (
	RegistryHive_SYSTEM   RegistryHive = "System"
	RegistryHive_SOFTWARE RegistryHive = "Software"
	RegistryHive_SECURITY RegistryHive = "Security"
	RegistryHive_SAM      RegistryHive = "Sam"
)
