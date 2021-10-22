/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package session

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/session"
)

type logout struct {
	*flags.ClientFlag
}

func init() {
	cli.Register("session.logout", &logout{})
}

func (cmd *logout) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)
}

func (cmd *logout) Process(ctx context.Context) error {
	return cmd.ClientFlag.Process(ctx)
}

func (cmd *logout) Description() string {
	return `Logout the current session.

By default, govc commands persist sessions and do not logout unless '-persist-session=false' is set.
The session.logout command can be used to end the current persisted session.
The session.rm command can be used to remove sessions other than the current session.

Examples:
  govc session.logout`
}

func (cmd *logout) Run(ctx context.Context, f *flag.FlagSet) error {
	c, err := cmd.Client()
	if err != nil {
		return err
	}

	return session.NewManager(c).Logout(ctx)
}
