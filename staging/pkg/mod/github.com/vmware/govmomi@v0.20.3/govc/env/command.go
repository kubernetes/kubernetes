/*
Copyright (c) 2016 VMware, Inc. All Rights Reserved.

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

package env

import (
	"context"
	"flag"
	"fmt"
	"io"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
)

type env struct {
	*flags.OutputFlag
	*flags.ClientFlag

	extra bool
}

func init() {
	cli.Register("env", &env{})
}

func (cmd *env) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.OutputFlag, ctx = flags.NewOutputFlag(ctx)
	cmd.OutputFlag.Register(ctx, f)

	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	f.BoolVar(&cmd.extra, "x", false, "Output variables for each GOVC_URL component")
}

func (cmd *env) Process(ctx context.Context) error {
	if err := cmd.OutputFlag.Process(ctx); err != nil {
		return err
	}
	if err := cmd.ClientFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *env) Description() string {
	return `Output the environment variables for this client.

If credentials are included in the url, they are split into separate variables.
Useful as bash scripting helper to parse GOVC_URL.`
}

func (cmd *env) Run(ctx context.Context, f *flag.FlagSet) error {
	env := envResult(cmd.ClientFlag.Environ(cmd.extra))

	if f.NArg() > 1 {
		return flag.ErrHelp
	}

	// Option to just output the value, example use:
	// password=$(govc env GOVC_PASSWORD)
	if f.NArg() == 1 {
		var output []string

		prefix := fmt.Sprintf("%s=", f.Arg(0))

		for _, e := range env {
			if strings.HasPrefix(e, prefix) {
				output = append(output, e[len(prefix):])
				break
			}
		}

		return cmd.WriteResult(envResult(output))
	}

	return cmd.WriteResult(env)
}

type envResult []string

func (r envResult) Write(w io.Writer) error {
	for _, e := range r {
		fmt.Println(e)
	}

	return nil
}
