// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rkt

import "github.com/coreos/rkt/networking/netinfo"

// AppState defines the state of the app.
type AppState string

const (
	AppStateUnknown AppState = "unknown"
	AppStateCreated AppState = "created"
	AppStateRunning AppState = "running"
	AppStateExited  AppState = "exited"
)

type (
	// Mount defines the mount point.
	Mount struct {
		// Name of the mount.
		Name string `json:"name"`
		// Container path of the mount.
		ContainerPath string `json:"container_path"`
		// Host path of the mount.
		HostPath string `json:"host_path"`
		// Whether the mount is read-only.
		ReadOnly bool `json:"read_only"`
		// TODO(yifan): What about 'SelinuxRelabel bool'?
	}

	// App defines the app object.
	App struct {
		// Name of the app.
		Name string `json:"name"`
		// State of the app, can be created, running, exited, or unknown.
		State AppState `json:"state"`
		// Creation time of the container, nanoseconds since epoch.
		CreatedAt *int64 `json:"created_at,omitempty"`
		// Start time of the container, nanoseconds since epoch.
		StartedAt *int64 `json:"started_at,omitempty"`
		// Finish time of the container, nanoseconds since epoch.
		FinishedAt *int64 `json:"finished_at,omitempty"`
		// Exit code of the container.
		ExitCode *int32 `json:"exit_code,omitempty"`
		// Image ID of the container.
		ImageID string `json:"image_id"`
		// Mount points of the container.
		Mounts []*Mount `json:"mounts,omitempty"`
		// User annotations of the container.
		UserAnnotations map[string]string `json:"user_annotations,omitempty"`
		// User labels of the container.
		UserLabels map[string]string `json:"user_labels,omitempty"`
	}

	// Pod defines the pod object.
	Pod struct {
		// UUID of the pod.
		UUID string `json:"name"`
		// State of the pod, all valid values are defined in pkg/pod/pods.go.
		State string `json:"state"`
		// Networks are the information of the networks.
		Networks []netinfo.NetInfo `json:"networks,omitempty"`
		// AppNames are the names of the apps.
		AppNames []string `json:"app_names,omitempty"`
		// The start time of the pod.
		StartedAt *int64 `json:"started_at,omitempty"`
		// UserAnnotations are the pod user annotations.
		UserAnnotations map[string]string `json:"user_annotations,omitempty"`
		// UserLabels are the pod user labels.
		UserLabels map[string]string `json:"user_labels,omitempty"`
	}

	ImageListEntry struct {
		// ID is the Image ID for this image
		ID string `json:"id"`
		// Name is the name of this image, such as example.com/some/image
		Name string `json:"name"`
		// ImportTime indicates when this image was imported in nanoseconds
		// since the unix epoch
		ImportTime int64 `json:"import_time"`
		// LastUsedTime indicates when was last used in nanoseconds since the
		// unix epoch
		LastUsedTime int64 `json:"last_used_time"`
		// Size is the size of this image in bytes
		Size int64 `json:"size"`
	}
)
