/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

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

package version

import (
	"context"
	"flag"
	"fmt"
	"log"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

var gitVersion string

type version struct {
	*flags.EmptyFlag

	require string
}

func init() {
	// Check that git tag in the release builds match the hardcoded version
	if gitVersion != "" && gitVersion[1:] != flags.Version {
		log.Panicf("version mismatch: git=%s vs govc=%s", gitVersion[1:], flags.Version)
	}

	cli.Register("version", &version{})
}

func (cmd *version) Register(ctx context.Context, f *flag.FlagSet) {
	f.StringVar(&cmd.require, "require", "", "Require govc version >= this value")
}

func (cmd *version) Run(ctx context.Context, f *flag.FlagSet) error {
	if cmd.require != "" {
		v, err := flags.ParseVersion(flags.Version)
		if err != nil {
			panic(err)
		}

		rv, err := flags.ParseVersion(cmd.require)
		if err != nil {
			return fmt.Errorf("failed to parse required version '%s': %s", cmd.require, err)
		}

		if !rv.Lte(v) {
			return fmt.Errorf("version %s or higher is required, this is version %s", cmd.require, flags.Version)
		}
	}

	fmt.Printf("govc %s\n", flags.Version)
	return nil
}
