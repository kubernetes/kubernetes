/*
Copyright 2017 The Kubernetes Authors.

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

package cmd

import (
	"fmt"
	"io"
	"strings"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack"
	tokens3 "github.com/rackspace/gophercloud/openstack/identity/v3/tokens"
	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

func NewCmdKeystone(f cmdutil.Factory, out io.Writer) *cobra.Command {
	keystone_long := templates.LongDesc(`
		Perform operations with Keystone.`)

	keystone_example := templates.Examples(`
		  # issue keystone token
		  kubectl keystone token issue

		  # revoke keysone tokens
		  kubectl keystone token revoke TOKEN1 TOKEN2`)

	cmd := &cobra.Command{
		Use:     "keystone token (issue|revoke)",
		Aliases: []string{"ks"},
		Short:   "Perform operations with Keystone",
		Long:    keystone_long,
		Example: keystone_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}

	cmd.AddCommand(NewCmdKeysoneToken(f, out))
	return cmd
}

func NewCmdKeysoneToken(f cmdutil.Factory, out io.Writer) *cobra.Command {
	token_long := templates.LongDesc(`
		Issue new token or revoke existing tokens.`)

	token_example := templates.Examples(`
		  # issue new keystone token
		  kubectl keystone token issue

		  # revoke existing keysone tokens
		  kubectl keystone token revoke TOKEN1 TOKEN2`)

	cmd := &cobra.Command{
		Use:     "token (issue|revoke)",
		Short:   "Issue new token or revoke existing tokens",
		Long:    token_long,
		Example: token_example,
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}

	cmd.AddCommand(NewCmdKeystoneTokenIssue(f, out))
	cmd.AddCommand(NewCmdKeystoneTokenRevoke(f, out))
	return cmd
}

func NewCmdKeystoneTokenIssue(f cmdutil.Factory, out io.Writer) *cobra.Command {
	issue_token_long := templates.LongDesc(`
		Issue new token.

		Find more information at http://docs.openstack.org/admin-guide/identity-tokens.html

		Related environment variables when performing requests to keystone:

			* OS_AUTH_URL: keystone auth url (v2 or v3(preferred))
			* OS_USERNAME: keystone user name
			* OS_USERID: keystone user id
			* OS_PASSWORD: keystone password
			* OS_TENANT_ID: keystone tenant/project id
			* OS_TENANT_NAME: keystone tenant/project name
			* OS_DOMAIN_ID: keystone domain id (only for keystone v3 use)
			* OS_DOMAIN_NAME: keystone domain name (only for keystone v3 use)
			* OS_TOKEN: keystone token

		It will check that (1) OS_AUTH_URL is set, (2) OS_USERNAME, OS_USERID, or OS_TOKEN is set,
		(3) OS_PASSWORD or OS_TOKEN is set. Other environment variables are optional,
		you can set them to genereate a scoped token.`)

	issue_token_example := templates.Examples(`
	  # Issue new token (scoped or unscoped)
	  kubectl keystone token issue`)

	cmd := &cobra.Command{
		Use:     "issue",
		Short:   "Issue new token",
		Long:    issue_token_long,
		Example: issue_token_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := KeystoneTokenIssue(f, out, cmd, args)
			cmdutil.CheckErr(err)
		},
	}

	return cmd
}

// KeystoneTokenIssue implements the behavior to issue keystone token
func KeystoneTokenIssue(f cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	options, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return fmt.Errorf("failed to read keystone env vars: %s", err)
	}

	client, err := openstack.AuthenticatedClient(options)
	if err != nil {
		return fmt.Errorf("keystone authentication failed: %s", err)
	}

	fmt.Fprintf(out, "issued keystone token: %s.\n", client.TokenID)
	return nil
}

func NewCmdKeystoneTokenRevoke(f cmdutil.Factory, out io.Writer) *cobra.Command {
	revoke_token_ong := templates.LongDesc(`
		Revoke existing tokens.

		Find more information at http://docs.openstack.org/admin-guide/identity-tokens.html

		Related environment variables when performing requests to keystone:

			* OS_AUTH_URL: keystone auth url (v2 or v3(preferred))
			* OS_USERNAME: keystone user name
			* OS_USERID: keystone user id
			* OS_PASSWORD: keystone password
			* OS_TENANT_ID: keystone tenant/project id
			* OS_TENANT_NAME: keystone tenant/project name
			* OS_DOMAIN_ID: keystone domain id (only for keystone v3 use)
			* OS_DOMAIN_NAME: keystone domain name (only for keystone v3 use)
			* OS_TOKEN: keystone token

		It will check that (1) OS_AUTH_URL is set, (2) OS_USERNAME, OS_USERID, or OS_TOKEN is set,
		(3) OS_PASSWORD or OS_TOKEN is set. Other environment variables are also important,
		you have to set them to get authorized to revoke tokens.`)

	revoke_token_example := templates.Examples(`
	  # Revoke existing scoped or unscoped tokens
	  kubectl keystone token revoke TOKEN1 TOKEN2`)

	cmd := &cobra.Command{
		Use:     "revoke [TOKEN...]",
		Short:   "Revoke existing tokens",
		Long:    revoke_token_ong,
		Example: revoke_token_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := KeystoneTokenRevoke(f, out, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	return cmd
}

// KeystoneTokenRevoke implements the behavior to issue keystone token
func KeystoneTokenRevoke(f cmdutil.Factory, out io.Writer, cmd *cobra.Command, args []string) error {
	if len(args) == 0 {
		return cmdutil.UsageError(cmd, "TOKEN-ID is required")
	}
	tokens := args

	options, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return fmt.Errorf("failed to read keystone env vars: %s", err)
	}

	client, err := openstack.AuthenticatedClient(options)
	if err != nil {
		return fmt.Errorf("keystone authentication failed: %s", err)
	}

	// client.IdentityEndpoint has already been normalized during above authentication
	if strings.HasSuffix(client.IdentityEndpoint, "/v3/") {
		v3Client := openstack.NewIdentityV3(client)
		for _, token := range tokens {
			res := tokens3.Revoke(v3Client, token)
			if res.Err != nil {
				return res.Err
			}
		}
	} else if strings.HasSuffix(client.IdentityEndpoint, "/v2.0/") {
		v2Client := openstack.NewIdentityV2(client)
		for _, token := range tokens {
			err := revokeV2Token(v2Client, token)
			if err != nil {
				return err
			}
		}
	}

	fmt.Fprintf(out, "keystone tokens have already been revoked.\n")
	return nil
}

func revokeV2Token(client *gophercloud.ServiceClient, token string) error {
	_, err := client.Delete(client.ServiceURL("tokens", token), &gophercloud.RequestOpts{})
	return err
}
