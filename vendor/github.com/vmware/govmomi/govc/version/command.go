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
	"flag"
	"fmt"

	"golang.org/x/net/context"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

var gitVersion string

type version struct {
	*flags.EmptyFlag
}

func init() {
	if gitVersion == "" {
		gitVersion = "unknown"
	}

	cli.Register("version", &version{})
}

func (c *version) Run(ctx context.Context, f *flag.FlagSet) error {
	fmt.Printf("govc %s\n", gitVersion)
	return nil
}
