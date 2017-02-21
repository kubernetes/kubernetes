/*
Copyright 2016 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"
	"path"
	"text/tabwriter"
	"time"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/client-go/pkg/api"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	tokenphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/token"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
	"k8s.io/kubernetes/pkg/kubectl"
)

func NewCmdToken(out io.Writer, errW io.Writer) *cobra.Command {

	tokenCmd := &cobra.Command{
		Use:   "token",
		Short: "Manage bootstrap tokens.",

		// Without this callback, if a user runs just the "token"
		// command without a subcommand, or with an invalid subcommand,
		// cobra will print usage information, but still exit cleanly.
		// We want to return an error code in these cases so that the
		// user knows that their command was invalid.
		RunE: func(cmd *cobra.Command, args []string) error {
			if len(args) < 1 {
				return errors.New("missing subcommand; 'token' is not meant to be run on its own")
			} else {
				return fmt.Errorf("invalid subcommand: %s", args[0])
			}
		},
	}

	var token string
	var tokenDuration time.Duration
	createCmd := &cobra.Command{
		Use:   "create",
		Short: "Create bootstrap tokens on the server.",
		Run: func(tokenCmd *cobra.Command, args []string) {
			err := RunCreateToken(out, tokenCmd, tokenDuration, token)
			kubeadmutil.CheckErr(err)
		},
	}
	createCmd.PersistentFlags().DurationVar(&tokenDuration,
		"ttl", kubeadmconstants.DefaultTokenDuration, "The duration before the token is automatically deleted. 0 means 'never expires'.")
	createCmd.PersistentFlags().StringVar(
		&token, "token", "",
		"Shared secret used to secure cluster bootstrap. If none is provided, one will be generated for you.",
	)
	tokenCmd.AddCommand(createCmd)

	tokenCmd.AddCommand(NewCmdTokenGenerate(out))

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
			if len(args) < 1 {
				kubeadmutil.CheckErr(fmt.Errorf("missing subcommand; 'token delete' is missing token of form [\"^([a-z0-9]{6})$\"]"))
			}
			err := RunDeleteToken(out, tokenCmd, args[0])
			kubeadmutil.CheckErr(err)
		},
	}
	tokenCmd.AddCommand(deleteCmd)

	return tokenCmd
}

func NewCmdTokenGenerate(out io.Writer) *cobra.Command {
	return &cobra.Command{
		Use:   "generate",
		Short: "Generate and print a bootstrap token, but do not create it on the server.",
		Long: dedent.Dedent(`
			This command will print out a randomly-generated bootstrap token that can be used with
			the "init" and "join" commands.

			You don't have to use this command in order to generate a token, you can do so
			yourself as long as it's in the format "<6 characters>:<16 characters>". This
			command is provided for convenience to generate tokens in that format.

			You can also use "kubeadm init" without specifying a token, and it will
			generate and print one for you.
		`),
		Run: func(cmd *cobra.Command, args []string) {
			err := RunGenerateToken(out)
			kubeadmutil.CheckErr(err)
		},
	}
}

// RunCreateToken generates a new bootstrap token and stores it as a secret on the server.
func RunCreateToken(out io.Writer, cmd *cobra.Command, tokenDuration time.Duration, token string) error {
	client, err := kubeconfigutil.ClientSetFromFile(path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.AdminKubeConfigFileName))
	if err != nil {
		return err
	}

	parsedID, parsedSecret, err := tokenutil.ParseToken(token)
	if err != nil {
		return err
	}
	td := &kubeadmapi.TokenDiscovery{ID: parsedID, Secret: parsedSecret}

	err = tokenphase.UpdateOrCreateToken(client, td, tokenDuration)
	if err != nil {
		return err
	}
	fmt.Fprintln(out, tokenutil.BearerToken(td))

	return nil
}

func RunGenerateToken(out io.Writer) error {
	td := &kubeadmapi.TokenDiscovery{}
	err := tokenutil.GenerateToken(td)
	if err != nil {
		return err
	}

	fmt.Fprintln(out, tokenutil.BearerToken(td))
	return nil
}

// RunListTokens lists details on all existing bootstrap tokens on the server.
func RunListTokens(out io.Writer, errW io.Writer, cmd *cobra.Command) error {
	client, err := kubeconfigutil.ClientSetFromFile(path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.AdminKubeConfigFileName))
	if err != nil {
		return err
	}

	tokenSelector := fields.SelectorFromSet(
		map[string]string{
			api.SecretTypeField: string(bootstrapapi.SecretTypeBootstrapToken),
		},
	)
	listOptions := metav1.ListOptions{
		FieldSelector: tokenSelector.String(),
	}

	results, err := client.Secrets(metav1.NamespaceSystem).List(listOptions)
	if err != nil {
		return fmt.Errorf("failed to list bootstrap tokens [%v]", err)
	}

	w := tabwriter.NewWriter(out, 10, 4, 3, ' ', 0)
	fmt.Fprintln(w, "ID\tTOKEN\tTTL")
	for _, secret := range results.Items {
		tokenId, ok := secret.Data[bootstrapapi.BootstrapTokenIDKey]
		if !ok {
			fmt.Fprintf(errW, "[token] bootstrap token has no token-id data: %s\n", secret.Name)
			continue
		}

		tokenSecret, ok := secret.Data[bootstrapapi.BootstrapTokenSecretKey]
		if !ok {
			fmt.Fprintf(errW, "[token] bootstrap token has no token-secret data: %s\n", secret.Name)
			continue
		}
		td := &kubeadmapi.TokenDiscovery{ID: string(tokenId), Secret: string(tokenSecret)}

		// Expiration time is optional, if not specified this implies the token
		// never expires.
		expires := "<never>"
		secretExpiration, ok := secret.Data[bootstrapapi.BootstrapTokenExpirationKey]
		if ok {
			expireTime, err := time.Parse(time.RFC3339, string(secretExpiration))
			if err != nil {
				return fmt.Errorf("error parsing expiration time [%v]", err)
			}
			expires = kubectl.ShortHumanDuration(expireTime.Sub(time.Now()))
		}
		fmt.Fprintf(w, "%s\t%s\t%s\n", tokenId, tokenutil.BearerToken(td), expires)
	}
	w.Flush()

	return nil
}

// RunDeleteToken removes a bootstrap token from the server.
func RunDeleteToken(out io.Writer, cmd *cobra.Command, tokenId string) error {
	if err := tokenutil.ParseTokenID(tokenId); err != nil {
		return err
	}

	client, err := kubeconfigutil.ClientSetFromFile(path.Join(kubeadmapi.GlobalEnvParams.KubernetesDir, kubeadmconstants.AdminKubeConfigFileName))
	if err != nil {
		return err
	}

	tokenSecretName := fmt.Sprintf("%s%s", bootstrapapi.BootstrapTokenSecretPrefix, tokenId)
	if err := client.Secrets(metav1.NamespaceSystem).Delete(tokenSecretName, nil); err != nil {
		return fmt.Errorf("failed to delete bootstrap token [%v]", err)
	}
	fmt.Fprintf(out, "[token] bootstrap token deleted: %s\n", tokenId)

	return nil
}
