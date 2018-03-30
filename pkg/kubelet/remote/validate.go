/*
Copyright 2018 The Kubernetes Authors.

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

package remote

import (
	"errors"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/runtime/v1alpha2"
)

func validateImageSpec(image *runtimeapi.ImageSpec) error {
	if image.Image == "" {
		return errors.New("imageSpec.Image not set")
	}

	return nil
}

func validateImageStatus(status *runtimeapi.ImageStatusResponse) error {
	if status.Image == nil {
		return nil
	}

	if status.Image.Id == "" {
		return errors.New("imageStatusResponse.Image.Id not set")
	}

	if status.Image.Size_ == 0 {
		return errors.New("imageStatusResponse.Image.Size not set")
	}

	return nil
}

func validatePortForwardRequest(req *runtimeapi.PortForwardRequest) error {
	if req == nil {
		return errors.New("portForwardRequest not set")
	}

	if req.PodSandboxId == "" {
		return errors.New("portForwardRequest.PodSandboxId not set")
	}

	return nil
}

func validateAttachRequest(req *runtimeapi.AttachRequest) error {
	if req == nil {
		return errors.New("attachRequest not set")
	}

	if req.ContainerId == "" {
		return errors.New("attachRequest.ContainerId not set")
	}

	if !req.Stdin && !req.Stdout && !req.Stderr {
		return errors.New("one of stdin, stdout and stderr MUST be set")
	}

	if req.Tty && req.Stderr {
		return errors.New("stderr shoudn't be set when tty is set")
	}

	return nil
}

func validateExecRequest(req *runtimeapi.ExecRequest) error {
	if req == nil {
		return errors.New("execRequest not set")
	}

	if req.ContainerId == "" {
		return errors.New("execRequest.ContainerId not set")
	}

	if len(req.Cmd) == 0 {
		return errors.New("execRequest.Cmd not set")
	}

	if !req.Stdin && !req.Stdout && !req.Stderr {
		return errors.New("one of stdin, stdout and stderr MUST be set")
	}

	if req.Tty && req.Stderr {
		return errors.New("stderr shoudn't be set when tty is set")
	}

	return nil
}

func validatePodSandboxConfig(config *runtimeapi.PodSandboxConfig) error {
	if config.Metadata == nil {
		return errors.New("podSandboxConfig.Metadata not set")
	}

	if config.Metadata.Name == "" {
		return errors.New("podSandboxConfig.Metadata.Name not set")
	}

	if config.Metadata.Namespace == "" {
		return errors.New("podSandboxConfig.Metadata.Namespace not set")
	}

	if config.Metadata.Uid == "" {
		return errors.New("podSandboxConfig.Metadata.Uid not set")
	}

	if config.LogDirectory == "" {
		return errors.New("podSandboxConfig.LogDirectory not set")
	}

	if config.Linux != nil {
		if config.Linux.CgroupParent == "" {
			return errors.New("podSandboxConfig.Linux.CgroupParent not set")
		}
	}

	return nil
}

func validateContainerConfig(config *runtimeapi.ContainerConfig) error {
	if config.Metadata == nil {
		return errors.New("containerConfig.Metadata not set")
	}

	if config.Metadata.Name == "" {
		return errors.New("containerConfig.Metadata.Name not set")
	}

	if config.Image == nil || validateImageSpec(config.Image) != nil {
		return errors.New("containerConfig.Image not set")
	}

	if config.LogPath == "" {
		return errors.New("containerConfig.LogPath not set")
	}

	return nil
}
