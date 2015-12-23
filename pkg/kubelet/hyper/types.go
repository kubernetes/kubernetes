/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package hyper

const (
	StatusRunning = "running"
	StatusPending = "pending"
	StatusFailed  = "failed"
	StatusSuccess = "succeeded"
)

type HyperImage struct {
	repository  string
	tag         string
	imageID     string
	createdAt   int64
	virtualSize int64
}

// Container JSON Data Structure
type ContainerPort struct {
	Name          string `json:"name"`
	HostPort      int    `json:"hostPort"`
	ContainerPort int    `json:"containerPort"`
	Protocol      string `json:"protocol"`
	HostIP        string `json:"hostIP"`
}

type EnvironmentVar struct {
	Env   string `json:"env"`
	Value string `json:"value"`
}

type VolumeMount struct {
	Name      string `json:"name"`
	ReadOnly  bool   `json:"readOnly"`
	MountPath string `json:"mountPath"`
}

type WaitingStatus struct {
	Reason string `json:"reason"`
}

type RunningStatus struct {
	StartedAt string `json:"startedAt"`
}

type TermStatus struct {
	ExitCode   int    `json:"exitCode"`
	Reason     string `json:"reason"`
	Message    string `json:"message"`
	StartedAt  string `json:"startedAt"`
	FinishedAt string `json:"finishedAt"`
}

type ContainerStatus struct {
	Name        string        `json:"name"`
	ContainerID string        `json:"containerID"`
	Phase       string        `json:"phase"`
	Waiting     WaitingStatus `json:"waiting"`
	Running     RunningStatus `json:"running"`
	Terminated  TermStatus    `json:"terminated"`
}

// Pod JSON Data Structure
type Container struct {
	Name            string           `json:"name"`
	ContainerID     string           `json:"containerID"`
	Image           string           `json:"image"`
	ImageID         string           `json:"imageID"`
	Commands        []string         `json:"commands"`
	Args            []string         `json:"args"`
	Workdir         string           `json:"workingDir"`
	Ports           []ContainerPort  `json:"ports"`
	Environment     []EnvironmentVar `json:"env"`
	Volume          []VolumeMount    `json:"volumeMounts"`
	ImagePullPolicy string           `json:"imagePullPolicy"`
}

type RBDVolumeSource struct {
	Monitors []string `json:"monitors"`
	Image    string   `json:"image"`
	FsType   string   `json:"fsType"`
	Pool     string   `json:"pool"`
	User     string   `json:"user"`
	Keyring  string   `json:"keyring"`
	ReadOnly bool     `json:"readOnly"`
}

type PodVolume struct {
	Name     string          `json:"name"`
	HostPath string          `json:"source"`
	Driver   string          `json:"driver"`
	Rbd      RBDVolumeSource `json:"rbd"`
}

type PodSpec struct {
	Volumes    []PodVolume `json:"volumes"`
	Containers []Container `json:"containers"`
}

type PodStatus struct {
	Phase     string            `json:"phase"`
	Message   string            `json:"message"`
	Reason    string            `json:"reason"`
	HostIP    string            `json:"hostIP"`
	PodIP     []string          `json:"podIP"`
	StartTime string            `json:"startTime"`
	Status    []ContainerStatus `json:"containerStatus"`
}

type PodInfo struct {
	Kind       string    `json:"kind"`
	ApiVersion string    `json:"apiVersion"`
	Vm         string    `json:"vm"`
	Spec       PodSpec   `json:"spec"`
	Status     PodStatus `json:"status"`
}

type HyperPod struct {
	podID   string
	podName string
	vmName  string
	status  string
	podInfo PodInfo
}

type HyperContainer struct {
	containerID string
	name        string
	podID       string
	status      string
}
