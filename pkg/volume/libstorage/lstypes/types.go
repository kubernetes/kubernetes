/*
Copyright 2016 The Kubernetes Authors.

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

package lstypes

type Service struct {
	Name     string   `json:"name"`
	Driver   Driver   `json:"driver,omitempty"`
	Instance Instance `json:"instance"`
}

type Instance struct {
	InstanceID IID `json:"instanceID"`
}

type Driver struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

type Executor struct {
	Name   string `json:"name"`
	MD5Sum string `json:"md5checksum"`
	Size   int64  `json:"size"`
}

type IID struct {
	ID string `json:"id"`
}

type Attachment struct {
	InstanceID IID    `json:"instanceID"`
	VolumeID   string `json:"volumeID"`
	DeviceName string `json:"deviceName"`
}

type Volume struct {
	ID          string        `json:"id"`
	Name        string        `json:"name"`
	Size        int64         `json:"size,omitempty"`
	Attachments []*Attachment `json:"attachments,omitempty"`
}

type Client interface {
	Volumes(attachments bool) ([]*Volume, error)
	FindVolume(name string) (*Volume, error)
	Volume(id string) (*Volume, error)
	CreateVolume(name string, size int64) (*Volume, error)
	AttachVolume(id string) (string, error)
	DetachVolume(name string) (*Volume, error)
	DeleteVolume(name string) error
	IID() (string, error)
	LocalDevs() (string, error)
	WaitForAttachedDevice(string) (string, error)
	WaitForDetachedDevice(string) error
}
