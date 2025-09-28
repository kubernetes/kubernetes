package v1alpha1

import (
	"encoding/json"
)

// StatuszData is the data required to populate the statusz endpoint.
type StatuszData struct {
	ComponentName      string
	StartTime          string
	UpTime             string
	GoVersion          string
	BinaryVersion      string
	EmulationVersion   string
	DeprecationMessage string
	Paths              []string
}

// PopulateStatuszDataV1Alpha1 populates the statusz data for v1alpha1.
func PopulateStatuszDataV1Alpha1(data *StatuszData) ([]byte, error) {
	statusz := &Statusz{
		APIVersion:         "v1alpha1",
		ComponentName:      data.ComponentName,
		StartTime:          data.StartTime,
		UpTime:             data.UpTime,
		GoVersion:          data.GoVersion,
		BinaryVersion:      data.BinaryVersion,
		EmulationVersion:   data.EmulationVersion,
		DeprecationMessage: data.DeprecationMessage,
		Paths:              data.Paths,
	}
	return json.Marshal(statusz)
}
