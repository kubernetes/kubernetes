package v1alpha1

// Statusz is a struct used for versioned statusz endpoint.
type Statusz struct {
	// APIVersion of the statusz endpoint
	APIVersion string `json:"apiVersion"`
	// ComponentName of the component that is running
	ComponentName string `json:"componentName"`
	// StartTime of the component
	StartTime string `json:"startTime"`
	// UpTime of the component
	UpTime string `json:"upTime"`
	// GoVersion of the component
	GoVersion string `json:"goVersion"`
	// BinaryVersion of the component
	BinaryVersion string `json:"binaryVersion"`
	// EmulationVersion of the component
	EmulationVersion string `json:"emulationVersion,omitempty"`
	// DeprecationMessage is a banner that announces the deprecation of an endpoint.
	// We would want to use this field to announce the deprecation of a version of the statusz endpoint.
	// For example, we could set this field to a string that contains a message to the user, such as:
	// "This version of the statusz endpoint is deprecated. Please use v2 instead."
	DeprecationMessage string `json:"deprecationMessage,omitempty"`
	// Paths of the component
	Paths []string `json:"paths"`
}
