package v1

import (
	"encoding/json"
)

func PopulateStatuszDataV1(componentName, startTime, upTime, goVersion, binaryVersion, emulationVersion string, paths []string) ([]byte, error) {
	status := Statusz{
		APIVersion:       "v1",
		ComponentName:    componentName,
		StartTime:        startTime,
		UpTime:           upTime,
		GoVersion:        goVersion,
		BinaryVersion:    binaryVersion,
		EmulationVersion: emulationVersion,
		Paths:            paths,
	}

	return json.Marshal(status)
}
