/*
Copyright (c) 2015-2016 VMware, Inc. All Rights Reserved.

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
	"fmt"
	"path"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/ovf"
	"github.com/vmware/govmomi/vim25/types"
)

var (
	allDiskProvisioningOptions   = []string{"thin", "monolithicSparse", "monolithicFlat", "twoGbMaxExtentSparse", "twoGbMaxExtentFlat", "seSparse", "eagerZeroedThick", "thick", "sparse", "flat"}
	allIPAllocationPolicyOptions = []string{"dhcpPolicy", "transientPolicy", "fixedPolicy", "fixedAllocatedPolicy"}
	allIPProtocolOptions         = []string{"IPv4", "IPv6"}
)

type spec struct {
	*ArchiveFlag

	verbose bool
}

func init() {
	cli.Register("import.spec", &spec{})
}

func (cmd *spec) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ArchiveFlag, ctx = newArchiveFlag(ctx)
	cmd.ArchiveFlag.Register(ctx, f)

	f.BoolVar(&cmd.verbose, "verbose", false, "Verbose spec output")
}

func (cmd *spec) Process(ctx context.Context) error {
	if err := cmd.ArchiveFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *spec) Usage() string {
	return "PATH_TO_OVF_OR_OVA"
}

func (cmd *spec) Run(ctx context.Context, f *flag.FlagSet) error {
	fpath := ""
	args := f.Args()
	if len(args) == 1 {
		fpath = f.Arg(0)
	}

	if len(fpath) > 0 {
		switch path.Ext(fpath) {
		case ".ovf":
			cmd.Archive = &FileArchive{path: fpath}
		case "", ".ova":
			cmd.Archive = &TapeArchive{path: fpath}
			fpath = "*.ovf"
		default:
			return fmt.Errorf("invalid file extension %s", path.Ext(fpath))
		}
	}

	return cmd.Spec(fpath)
}

func (cmd *spec) Map(e *ovf.Envelope) (res []Property) {
	if e == nil || e.VirtualSystem == nil {
		return nil
	}

	for _, p := range e.VirtualSystem.Product {
		for i, v := range p.Property {
			if v.UserConfigurable == nil || !*v.UserConfigurable {
				continue
			}

			d := ""
			if v.Default != nil {
				d = *v.Default
			}

			// vSphere only accept True/False as boolean values for some reason
			if v.Type == "boolean" {
				d = strings.Title(d)
			}

			// From OVF spec, section 9.5.1:
			// key-value-env = [class-value "."] key-value-prod ["." instance-value]
			k := v.Key
			if p.Class != nil {
				k = fmt.Sprintf("%s.%s", *p.Class, k)
			}
			if p.Instance != nil {
				k = fmt.Sprintf("%s.%s", k, *p.Instance)
			}

			np := Property{KeyValue: types.KeyValue{Key: k, Value: d}}
			if cmd.verbose {
				np.Spec = &p.Property[i]
			}

			res = append(res, np)
		}
	}

	return
}

func (cmd *spec) Spec(fpath string) error {
	e := &ovf.Envelope{}
	if fpath != "" {
		d, err := cmd.ReadOvf(fpath)
		if err != nil {
			return err
		}

		if e, err = cmd.ReadEnvelope(d); err != nil {
			return err
		}
	}

	var deploymentOptions []string
	if e.DeploymentOption != nil && e.DeploymentOption.Configuration != nil {
		// add default first
		for _, c := range e.DeploymentOption.Configuration {
			if c.Default != nil && *c.Default {
				deploymentOptions = append(deploymentOptions, c.ID)
			}
		}

		for _, c := range e.DeploymentOption.Configuration {
			if c.Default == nil || !*c.Default {
				deploymentOptions = append(deploymentOptions, c.ID)
			}
		}
	}

	o := Options{
		DiskProvisioning:   allDiskProvisioningOptions[0],
		IPAllocationPolicy: allIPAllocationPolicyOptions[0],
		IPProtocol:         allIPProtocolOptions[0],
		MarkAsTemplate:     false,
		PowerOn:            false,
		WaitForIP:          false,
		InjectOvfEnv:       false,
		PropertyMapping:    cmd.Map(e),
	}

	if deploymentOptions != nil {
		o.Deployment = deploymentOptions[0]
	}

	if e.VirtualSystem != nil && e.VirtualSystem.Annotation != nil {
		for _, a := range e.VirtualSystem.Annotation {
			o.Annotation += a.Annotation
		}
	}

	if e.Network != nil {
		for _, net := range e.Network.Networks {
			o.NetworkMapping = append(o.NetworkMapping, Network{net.Name, ""})
		}
	}

	if cmd.verbose {
		if deploymentOptions != nil {
			o.AllDeploymentOptions = deploymentOptions
		}
		o.AllDiskProvisioningOptions = allDiskProvisioningOptions
		o.AllIPAllocationPolicyOptions = allIPAllocationPolicyOptions
		o.AllIPProtocolOptions = allIPProtocolOptions
	}

	j, err := json.Marshal(&o)
	if err != nil {
		return err
	}

	fmt.Println(string(j))
	return nil
}
