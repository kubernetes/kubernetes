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
	"encoding/base64"
	"encoding/pem"
	"flag"
	"os"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/ssoadmin"
	"github.com/vmware/govmomi/ssoadmin/types"
	"github.com/vmware/govmomi/sts"
	"github.com/vmware/govmomi/vim25/soap"
)

func withClient(ctx context.Context, cmd *flags.ClientFlag, f func(*ssoadmin.Client) error) error {
	vc, err := cmd.Client()
	if err != nil {
		return err
	}

	c, err := ssoadmin.NewClient(ctx, vc)
	if err != nil {
		return err
	}

	// SSO admin server has its own session manager, so the govc persisted session cookies cannot
	// be used to authenticate.  There is no SSO token persistence in govc yet, so just use an env
	// var for now.  If no GOVC_LOGIN_TOKEN is set, issue a new token.
	token := os.Getenv("GOVC_LOGIN_TOKEN")
	header := soap.Header{
		Security: &sts.Signer{
			Certificate: vc.Certificate(),
			Token:       token,
		},
	}

	if token == "" {
		tokens, cerr := sts.NewClient(ctx, vc)
		if cerr != nil {
			return cerr
		}

		req := sts.TokenRequest{
			Certificate: vc.Certificate(),
			Userinfo:    cmd.Userinfo(),
		}

		header.Security, cerr = tokens.Issue(ctx, req)
		if cerr != nil {
			return cerr
		}
	}

	if err = c.Login(c.WithHeader(ctx, header)); err != nil {
		return err
	}
	defer c.Logout(ctx)

	return f(c)
}

type userDetails struct {
	*flags.ClientFlag

	types.AdminPersonDetails
	password string
	solution types.AdminSolutionDetails
	actas    *bool
	role     string
}

func (cmd *userDetails) Usage() string {
	return "NAME"
}

func (cmd *userDetails) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.ClientFlag, ctx = flags.NewClientFlag(ctx)
	cmd.ClientFlag.Register(ctx, f)

	f.StringVar(&cmd.Description, "d", "", "User description")
	f.StringVar(&cmd.EmailAddress, "m", "", "Email address")
	f.StringVar(&cmd.FirstName, "f", "", "First name")
	f.StringVar(&cmd.LastName, "l", "", "Last name")
	f.StringVar(&cmd.password, "p", "", "Password")
	f.StringVar(&cmd.solution.Certificate, "C", "", "Certificate for solution user")
	f.Var(flags.NewOptionalBool(&cmd.actas), "A", "ActAsUser role for solution user WSTrust")
	f.StringVar(&cmd.role, "R", "", "Role for solution user (RegularUser|Administrator)")
}

func (cmd *userDetails) Certificate() string {
	block, _ := pem.Decode([]byte(cmd.solution.Certificate))
	if block != nil {
		return base64.StdEncoding.EncodeToString(block.Bytes)
	}
	return cmd.solution.Certificate
}

type create struct {
	userDetails
}

func init() {
	cli.Register("sso.user.create", &create{})
}

func (cmd *create) Description() string {
	return `Create SSO users.

Examples:
  govc sso.user.create -C "$(cat cert.pem)" -A -R Administrator NAME # solution user
  govc sso.user.create -p password NAME # person user`
}

func (cmd *create) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		return flag.ErrHelp
	}
	id := f.Arg(0)
	person := cmd.solution.Certificate == ""
	if person {
		if cmd.password == "" {
			return flag.ErrHelp
		}
	} else {
		if cmd.password != "" {
			return flag.ErrHelp
		}
	}

	return withClient(ctx, cmd.ClientFlag, func(c *ssoadmin.Client) error {
		if person {
			return c.CreatePersonUser(ctx, id, cmd.AdminPersonDetails, cmd.password)
		}

		cmd.solution.Certificate = cmd.Certificate()
		cmd.solution.Description = cmd.AdminPersonDetails.Description

		if err := c.CreateSolutionUser(ctx, id, cmd.solution); err != nil {
			return err
		}

		p := types.PrincipalId{Name: id, Domain: c.Domain}

		if cmd.role != "" {
			if _, err := c.SetRole(ctx, p, cmd.role); err != nil {
				return err
			}
		}

		if cmd.actas != nil && *cmd.actas {
			if _, err := c.GrantWSTrustRole(ctx, p, types.RoleActAsUser); err != nil {
				return err
			}
		}

		return nil
	})
}
