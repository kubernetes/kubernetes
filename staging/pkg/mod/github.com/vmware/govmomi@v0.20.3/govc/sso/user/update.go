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

package user

import (
	"context"
	"flag"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/ssoadmin"
	"github.com/vmware/govmomi/ssoadmin/types"
)

type update struct {
	userDetails
}

func init() {
	cli.Register("sso.user.update", &update{})
}

func (cmd *update) Description() string {
	return `Update SSO users.

Examples:
  govc sso.user.update -C "$(cat cert.pem)" NAME
  govc sso.user.update -p password NAME`
}

// merge uses the current value of a field if the corresponding flag was not set.
// Otherwise the API call would set the current fields to empty strings.
func merge(src *string, current string) {
	if *src == "" {
		*src = current
	}
}

func (cmd *update) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}
	id := f.Arg(0)

	return withClient(ctx, cmd.ClientFlag, func(c *ssoadmin.Client) error {
		user, err := c.FindUser(ctx, id)
		if err != nil {
			return err
		}

		if user.Kind == "person" {
			current, cerr := c.FindPersonUser(ctx, id)
			if cerr != nil {
				return cerr
			}

			merge(&cmd.AdminPersonDetails.Description, current.Details.Description)
			merge(&cmd.FirstName, current.Details.FirstName)
			merge(&cmd.LastName, current.Details.LastName)
			merge(&cmd.EmailAddress, current.Details.EmailAddress)
			if err := c.UpdatePersonUser(ctx, id, cmd.AdminPersonDetails); err != nil {
				return err
			}

			if cmd.password != "" {
				return c.ResetPersonPassword(ctx, id, cmd.password)
			}

			return nil
		}

		if cmd.actas != nil {
			action := c.GrantWSTrustRole
			if *cmd.actas == false {
				action = c.RevokeWSTrustRole
			}
			if _, err := action(ctx, user.Id, types.RoleActAsUser); err != nil {
				return err
			}
		}

		if cmd.role != "" {
			if _, err := c.SetRole(ctx, user.Id, cmd.role); err != nil {
				return err
			}
		}

		cmd.solution.Certificate = cmd.Certificate()
		cmd.solution.Description = cmd.AdminPersonDetails.Description

		current, cerr := c.FindSolutionUser(ctx, id)
		if cerr != nil {
			return cerr
		}

		merge(&cmd.solution.Certificate, current.Details.Certificate)
		merge(&cmd.solution.Description, current.Details.Description)
		return c.UpdateSolutionUser(ctx, id, cmd.solution)
	})
}
