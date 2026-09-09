/*
Copyright 2024 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package gce

import (
	"encoding/json"
	"fmt"
	"os/exec"
)

type gceImage struct {
	CreationTimestamp string `json:"creationTimestamp"`
	Family            string `json:"family"`
	ID                string `json:"id"`
	Name              string `json:"name"`
}

type gceMetadata struct {
	Fingerprint string             `json:"fingerprint"`
	Kind        string             `json:"kind"`
	Items       []gceMetadataItems `json:"items,omitempty"`
}

type gceMetadataItems struct {
	Key   string `json:"key,omitempty"`
	Value string `json:"value,omitempty"`
}

type gceAccessConfigs struct {
	Type  string `json:"type"`
	Name  string `json:"name"`
	NatIP string `json:"natIP,omitempty"`
}

type gceNetworkInterfaces struct {
	AccessConfigs []gceAccessConfigs `json:"accessConfigs"`
}

type gceInstance struct {
	CreationTimestamp       string                 `json:"creationTimestamp"`
	Description             string                 `json:"description"`
	Fingerprint             string                 `json:"fingerprint"`
	ID                      string                 `json:"id"`
	KeyRevocationActionType string                 `json:"keyRevocationActionType"`
	Kind                    string                 `json:"kind"`
	LabelFingerprint        string                 `json:"labelFingerprint"`
	LastStartTimestamp      string                 `json:"lastStartTimestamp"`
	MachineType             string                 `json:"machineType"`
	Metadata                gceMetadata            `json:"metadata"`
	Name                    string                 `json:"name"`
	NetworkInterfaces       []gceNetworkInterfaces `json:"networkInterfaces"`
	Status                  string                 `json:"status"`
}

type projectInfo struct {
	CommonInstanceMetadata struct {
		Fingerprint string `json:"fingerprint"`
		Items       []struct {
			Key   string `json:"key"`
			Value string `json:"value"`
		} `json:"items"`
		Kind string `json:"kind"`
	} `json:"commonInstanceMetadata"`
	CreationTimestamp     string `json:"creationTimestamp"`
	DefaultNetworkTier    string `json:"defaultNetworkTier"`
	DefaultServiceAccount string `json:"defaultServiceAccount"`
	ID                    string `json:"id"`
}

func runGCPCommandWithZone(args ...string) ([]byte, error) {
	if zone != nil && len(*zone) > 0 {
		args = append(args, "--zone="+*zone)
	}
	return runGCPCommand(args...)
}

func runGCPCommandWithZones(args ...string) ([]byte, error) {
	if zone != nil && len(*zone) > 0 {
		args = append(args, "--zones="+*zone+",")
	}
	return runGCPCommand(args...)
}

func runGCPCommand(args ...string) ([]byte, error) {
	if project != nil && len(*project) > 0 {
		args = append(args, "--project="+*project)
	}
	return runGCPCommandNoProject(args...)
}

func runGCPCommandNoProject(args ...string) ([]byte, error) {
	bytes, err := exec.Command("gcloud", args...).Output()
	if err != nil {
		var message string
		if ee, ok := err.(*exec.ExitError); ok {
			message = fmt.Sprintf("%v\n%v", ee, string(ee.Stderr))
		} else {
			message = fmt.Sprintf("%v", err)
		}
		return nil, fmt.Errorf("unable to run gcloud command\n %s \n %w", message, err)
	}
	return bytes, nil
}

func getGCEInstance(host string) (*gceInstance, error) {
	data, err := runGCPCommandWithZone("compute", "instances", "describe", host, "--format=json")
	if err != nil {
		return nil, fmt.Errorf("failed to describe instance in project %q: %w", *project, err)
	}

	var gceHost gceInstance
	err = json.Unmarshal(data, &gceHost)
	if err != nil {
		return nil, fmt.Errorf("failed to parse instance: %w", err)
	}
	return &gceHost, nil
}

func (g *GCERunner) getSerialOutput(host string) (string, error) {
	data, err := runGCPCommandWithZone("compute", "instances", "get-serial-port-output", "--port=1", host)
	if err != nil {
		return "", fmt.Errorf("failed to describe instance in project %q: %w", *project, err)
	}
	return string(data), nil
}
