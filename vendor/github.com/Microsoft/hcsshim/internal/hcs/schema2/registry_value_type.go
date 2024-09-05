package hcsschema

// NOTE: manually added

type RegistryValueType string

// List of RegistryValueType
const (
	RegistryValueType_NONE            RegistryValueType = "None"
	RegistryValueType_STRING          RegistryValueType = "String"
	RegistryValueType_EXPANDED_STRING RegistryValueType = "ExpandedString"
	RegistryValueType_MULTI_STRING    RegistryValueType = "MultiString"
	RegistryValueType_BINARY          RegistryValueType = "Binary"
	RegistryValueType_D_WORD          RegistryValueType = "DWord"
	RegistryValueType_Q_WORD          RegistryValueType = "QWord"
	RegistryValueType_CUSTOM_TYPE     RegistryValueType = "CustomType"
)
