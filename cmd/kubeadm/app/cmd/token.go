/*
Copyright 2014 The Kubernetes Authors.

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
	"path"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubemaster "k8s.io/kubernetes/cmd/kubeadm/app/master"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubectl"
)

func NewCmdToken(out io.Writer, errW io.Writer) *cobra.Command {
	tokenCmd := &cobra.Command{
		Use:   "token",
		Short: "Manage bootstrap tokens",
	}

	var tokenDuration time.Duration
	tokenSecret := &kubeadmapi.Secrets{}
	createCmd := &cobra.Command{
		Use:   "create",
		Short: "Create bootstrap tokens on the server.",
		Run: func(tokenCmd *cobra.Command, args []string) {
			err := RunCreateToken(out, tokenCmd, tokenDuration, tokenSecret)
			kubeadmutil.CheckErr(err)
		},
	}
	createCmd.PersistentFlags().DurationVar(&tokenDuration,
		"ttl", kubeadmutil.DefaultTokenDuration, "The duration before the token is automatically deleted.")
	createCmd.PersistentFlags().StringVar(
		&tokenSecret.GivenToken, "token", "",
		"Shared secret used to secure cluster bootstrap. If none is provided, one will be generated for you.",
	)
	tokenCmd.AddCommand(createCmd)

	listCmd := &cobra.Command{
		Use:   "list",
		Short: "List bootstrap tokens on the server.",
		Run: func(tokenCmd *cobra.Command, args []string) {
			err := RunListTokens(out, errW, tokenCmd)
			kubeadmutil.CheckErr(err)
		},
	}
	tokenCmd.AddCommand(listCmd)

	deleteCmd := &cobra.Command{
		Use:   "delete",
		Short: "Delete bootstrap tokens on the server.",
		Run: func(tokenCmd *cobra.Command, args []string) {
			err := RunDeleteToken(out, tokenCmd, args[0])
			kubeadmutil.CheckErr(err)
		},
	}
	tokenCmd.AddCommand(deleteCmd)

	return tokenCmd
}

// RunCreateToken generates a new bootstrap token and stores it as a secret on the server.
func RunCreateToken(out io.Writer, cmd *cobra.Command, tokenDuration time.Duration, tokenSecret *kubeadmapi.Secrets) error {
	client, err := kubemaster.CreateClientFromFile(path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, "admin.conf"))
	if err != nil {
		return err
	}

	err = kubeadmutil.GenerateTokenIfNeeded(tokenSecret)
	if err != nil {
		return err
	}

	err = kubeadmutil.UpdateOrCreateToken(client, tokenSecret, tokenDuration)
	if err != nil {
		return err
	}
	fmt.Fprintln(out, tokenSecret.GivenToken)

	return nil
}

// RunListTokens lists details on all existing bootstrap tokens on the server.
func RunListTokens(out io.Writer, errW io.Writer, cmd *cobra.Command) error {
	client, err := kubemaster.CreateClientFromFile(path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, "admin.conf"))
	if err != nil {
		return err
	}

	tokenSelector := fields.SelectorFromSet(
		map[string]string{
			api.SecretTypeField: string(api.SecretTypeBootstrapToken),
		},
	)
	listOptions := api.ListOptions{
		FieldSelector: tokenSelector,
	}

	results, err := client.Secrets(api.NamespaceSystem).List(listOptions)
	if err != nil {
		return fmt.Errorf("<cmd/token> failed to list bootstrap tokens [%v]", err)
	}

	w := tabwriter.NewWriter(out, 10, 4, 3, ' ', 0)
	fmt.Fprintln(w, "ID\tTOKEN\tEXPIRATION")
	for _, secret := range results.Items {
		tokenId, ok := secret.Data["token-id"]
		if !ok {
			fmt.Fprintf(errW, "<cmd/token> bootstrap token has no token-id data: %s\n", secret.Name)
			continue
		}

		tokenSecret, ok := secret.Data["token-secret"]
		if !ok {
			fmt.Fprintf(errW, "<cmd/token> bootstrap token has no token-secret data: %s\n", secret.Name)
			continue
		}
		token := fmt.Sprintf("%s.%s", tokenId, tokenSecret)

		// Expiration time is optional, if not specified this implies the token
		// never expires.
		expires := "<never>"
		secretExpiration, ok := secret.Data["expiration"]
		if ok {
			expireTime, err := time.Parse(time.RFC3339, string(secretExpiration))
			if err != nil {
				return fmt.Errorf("<cmd/token> error parsing expiry time [%v]", err)
			}
			expires = kubectl.ShortHumanDuration(expireTime.Sub(time.Now()))
		}
		fmt.Fprintf(w, "%s\t%s\t%s\n", tokenId, token, expires)
	}
	w.Flush()

	return nil
}

// RunDeleteToken removes a bootstrap token from the server.
func RunDeleteToken(out io.Writer, cmd *cobra.Command, tokenId string) error {
	client, err := kubemaster.CreateClientFromFile(path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, "admin.conf"))
	if err != nil {
		return err
	}

	tokenSecretName := fmt.Sprintf("%s%s", kubeadmutil.BootstrapTokenSecretPrefix, tokenId)
	if err := client.Secrets(api.NamespaceSystem).Delete(tokenSecretName, nil); err != nil {
		return fmt.Errorf("<cmd/token> failed to delete bootstrap token [%v]", err)
	}
	fmt.Fprintf(out, "<cmd/token> bootstrap token deleted: %s\n", tokenId)

	return nil
}
