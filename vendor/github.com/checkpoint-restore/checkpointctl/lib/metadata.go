// SPDX-License-Identifier: Apache-2.0

package metadata

import (
	"encoding/json"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/pkg/errors"
)

type CheckpointedPods struct {
	PodUID                 string                   `json:"io.kubernetes.pod.uid,omitempty"`
	ID                     string                   `json:"SandboxID,omitempty"`
	Name                   string                   `json:"io.kubernetes.pod.name,omitempty"`
	TerminationGracePeriod int64                    `json:"io.kubernetes.pod.terminationGracePeriod,omitempty"`
	Namespace              string                   `json:"io.kubernetes.pod.namespace,omitempty"`
	ConfigSource           string                   `json:"kubernetes.io/config.source,omitempty"`
	ConfigSeen             string                   `json:"kubernetes.io/config.seen,omitempty"`
	Manager                string                   `json:"io.container.manager,omitempty"`
	Containers             []CheckpointedContainers `json:"Containers"`
	HostIP                 string                   `json:"hostIP,omitempty"`
	PodIP                  string                   `json:"podIP,omitempty"`
	PodIPs                 []string                 `json:"podIPs,omitempty"`
}

type CheckpointedContainers struct {
	Name                      string `json:"io.kubernetes.container.name,omitempty"`
	ID                        string `json:"id,omitempty"`
	TerminationMessagePath    string `json:"io.kubernetes.container.terminationMessagePath,omitempty"`
	TerminationMessagePolicy  string `json:"io.kubernetes.container.terminationMessagePolicy,omitempty"`
	RestartCounter            int32  `json:"io.kubernetes.container.restartCount,omitempty"`
	TerminationMessagePathUID string `json:"terminationMessagePathUID,omitempty"`
	Image                     string `json:"Image"`
}

type CheckpointMetadata struct {
	Version          int `json:"version"`
	CheckpointedPods []CheckpointedPods
}

const (
	CheckpointedPodsFile = "checkpointed.pods"
)

func ReadCheckpointedPods(checkpointsDirectory string) (error, *CheckpointMetadata, string) {
	checkpointMetadataPath := filepath.Join(checkpointsDirectory, CheckpointedPodsFile)

	checkpointMetadataFile, err := os.Open(checkpointMetadataPath)
	if err != nil {
		return errors.Wrapf(err, "Error opening %q", checkpointMetadataPath), nil, checkpointMetadataPath
	}
	defer checkpointMetadataFile.Close()

	var checkpointMetadata CheckpointMetadata
	checkpointMetadataJSON, err := ioutil.ReadAll(checkpointMetadataFile)
	if err != nil {
		return errors.Wrapf(err, "Error reading from %q", checkpointMetadataPath), nil, checkpointMetadataPath
	}

	if err := json.Unmarshal(checkpointMetadataJSON, &checkpointMetadata); err != nil {
		return errors.Wrapf(err, "Error unmarshalling JSON from %q", checkpointMetadataPath), nil, checkpointMetadataPath
	}

	return nil, &checkpointMetadata, checkpointMetadataPath
}

// WriteJSONFile marshalls and writes the given data to a JSON file
func WriteJSONFile(v interface{}, dir, file string) error {
	fileJSON, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return errors.Wrapf(err, "Error marshalling JSON")
	}
	file = filepath.Join(dir, file)
	if err := ioutil.WriteFile(file, fileJSON, 0o644); err != nil {
		return errors.Wrapf(err, "Error writing to %q", file)
	}

	return nil
}

func WriteKubeletCheckpointsMetadata(checkpointMetadata *CheckpointMetadata, dir string) error {
	return WriteJSONFile(checkpointMetadata, dir, CheckpointedPodsFile)
}
