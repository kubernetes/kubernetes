/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package importx

import (
	"context"
	"encoding/json"
	"flag"
	"os"

	"github.com/vmware/govmomi/ovf"
	"github.com/vmware/govmomi/vim25/types"
)

type Property struct {
	types.KeyValue
	Spec *ovf.Property `json:",omitempty"`
}

type Network struct {
	Name    string
	Network string
}

type Options struct {
	AllDeploymentOptions []string `json:",omitempty"`
	Deployment           string   `json:",omitempty"`

	AllDiskProvisioningOptions []string `json:",omitempty"`
	DiskProvisioning           string

	AllIPAllocationPolicyOptions []string `json:",omitempty"`
	IPAllocationPolicy           string

	AllIPProtocolOptions []string `json:",omitempty"`
	IPProtocol           string

	PropertyMapping []Property `json:",omitempty"`

	NetworkMapping []Network `json:",omitempty"`

	Annotation string `json:",omitempty"`

	MarkAsTemplate bool
	PowerOn        bool
	InjectOvfEnv   bool
	WaitForIP      bool
	Name           *string
}

type OptionsFlag struct {
	Options Options

	path string
}

func newOptionsFlag(ctx context.Context) (*OptionsFlag, context.Context) {
	return &OptionsFlag{}, ctx
}

func (flag *OptionsFlag) Register(ctx context.Context, f *flag.FlagSet) {
	f.StringVar(&flag.path, "options", "", "Options spec file path for VM deployment")
}

func (flag *OptionsFlag) Process(ctx context.Context) error {
	if len(flag.path) == 0 {
		return nil
	}

	var err error
	in := os.Stdin

	if flag.path != "-" {
		in, err = os.Open(flag.path)
		if err != nil {
			return err
		}
		defer in.Close()
	}

	return json.NewDecoder(in).Decode(&flag.Options)
}
