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
	"sort"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/pkg/api/v1"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	tokenphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/token"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	tokenutil "k8s.io/kubernetes/cmd/kubeadm/app/util/token"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
	"k8s.io/kubernetes/pkg/printers"
)

func NewCmdToken(out io.Writer, errW io.Writer) *cobra.Command {

	var kubeConfigFile string
	tokenCmd := &cobra.Command{
		Use:   "token",
		Short: "Manage bootstrap tokens.",
		Long: dedent.Dedent(`
			This command will manage Bootstrap Token for you.
			  Please note this usage of this command is optional, and mostly for advanced users.

			In short, Bootstrap Tokens are used for establishing bidirectional trust between a client and a server.
			A Bootstrap Token can be used when a client (for example a node that's about to join the cluster) needs
			to trust the server it is talking to. Then a Bootstrap Token with the "signing" usage can be used.
			Bootstrap Tokens can also function as a way to allow short-lived authentication to the API Server
			(the token serves as a way for the API Server to trust the client), for example for doing the TLS Bootstrap.

			What is a Bootstrap Token more exactly?
			 - It is a Secret in the kube-system namespace of type "bootstrap.kubernetes.io/token".
			 - A Bootstrap Token must be of the form "[a-z0-9]{6}.[a-z0-9]{16}"; the former part is the public Token ID,
			   and the latter is the Token Secret, which must be kept private at all circumstances.
			 - The name of the Secret must be named "bootstrap-token-(token-id)".

			You can read more about Bootstrap Tokens in this proposal:

			  https://github.com/kubernetes/community/blob/master/contributors/design-proposals/bootstrap-discovery.md
		`),

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

	tokenCmd.PersistentFlags().StringVar(&kubeConfigFile,
		"kubeconfig", "/etc/kubernetes/admin.conf", "The KubeConfig file to use for talking to the cluster")

	var usages []string
	var tokenDuration time.Duration
	var description string
	createCmd := &cobra.Command{
		Use:   "create [token]",
		Short: "Create bootstrap tokens on the server.",
		Long: dedent.Dedent(`
			This command will create a Bootstrap Token for you.
			You can specify the usages for this token, the time to live and an optional human friendly description.

			The [token] is the actual token to write.
			This should be a securely generated random token of the form "[a-z0-9]{6}.[a-z0-9]{16}".
			If no [token] is given, kubeadm will generate a random token instead.
		`),
		Run: func(tokenCmd *cobra.Command, args []string) {
			token := ""
			if len(args) != 0 {
				token = args[0]
			}
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = RunCreateToken(out, client, token, tokenDuration, usages, description)
			kubeadmutil.CheckErr(err)
		},
	}
	createCmd.Flags().DurationVar(&tokenDuration,
		"ttl", kubeadmconstants.DefaultTokenDuration, "The duration before the token is automatically deleted. 0 means 'never expires'.")
	createCmd.Flags().StringSliceVar(&usages,
		"usages", kubeadmconstants.DefaultTokenUsages, "The ways in which this token can be used. Valid options: [signing,authentication].")
	createCmd.Flags().StringVar(&description,
		"description", "", "A human friendly description of how this token is used.")
	tokenCmd.AddCommand(createCmd)

	tokenCmd.AddCommand(NewCmdTokenGenerate(out))

	listCmd := &cobra.Command{
		Use:   "list",
		Short: "List bootstrap tokens on the server.",
		Long: dedent.Dedent(`
			This command will list all Bootstrap Tokens for you.
		`),
		Run: func(tokenCmd *cobra.Command, args []string) {
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = RunListTokens(out, errW, client)
			kubeadmutil.CheckErr(err)
		},
	}
	tokenCmd.AddCommand(listCmd)

	deleteCmd := &cobra.Command{
		Use:   "delete [token-value]",
		Short: "Delete bootstrap tokens on the server.",
		Long: dedent.Dedent(`
			This command will delete a given Bootstrap Token for you.

			The [token-value] is the full Token of the form "[a-z0-9]{6}.[a-z0-9]{16}" or the
			Token ID of the form "[a-z0-9]{6}" to delete.
		`),
		Run: func(tokenCmd *cobra.Command, args []string) {
			if len(args) < 1 {
				kubeadmutil.CheckErr(fmt.Errorf("missing subcommand; 'token delete' is missing token of form [%q]", tokenutil.TokenIDRegexpString))
			}
			client, err := kubeconfigutil.ClientSetFromFile(kubeConfigFile)
			kubeadmutil.CheckErr(err)

			err = RunDeleteToken(out, client, args[0])
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
			yourself as long as it's in the format "[a-z0-9]{6}.[a-z0-9]{16}". This
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
func RunCreateToken(out io.Writer, client *clientset.Clientset, token string, tokenDuration time.Duration, usages []string, description string) error {

	if len(token) == 0 {
		var err error
		token, err = tokenutil.GenerateToken()
		if err != nil {
			return err
		}
	} else {
		_, _, err := tokenutil.ParseToken(token)
		if err != nil {
			return err
		}
	}

	// TODO: Validate usages here so we don't allow something unsupported
	err := tokenphase.CreateNewToken(client, token, tokenDuration, usages, description)
	if err != nil {
		return err
	}

	fmt.Fprintln(out, token)
	return nil
}

// RunGenerateToken just generates a random token for the user
func RunGenerateToken(out io.Writer) error {
	token, err := tokenutil.GenerateToken()
	if err != nil {
		return err
	}

	fmt.Fprintln(out, token)
	return nil
}

// RunListTokens lists details on all existing bootstrap tokens on the server.
func RunListTokens(out io.Writer, errW io.Writer, client *clientset.Clientset) error {
	// First, build our selector for bootstrap tokens only
	tokenSelector := fields.SelectorFromSet(
		map[string]string{
			api.SecretTypeField: string(bootstrapapi.SecretTypeBootstrapToken),
		},
	)
	listOptions := metav1.ListOptions{
		FieldSelector: tokenSelector.String(),
	}

	secrets, err := client.CoreV1().Secrets(metav1.NamespaceSystem).List(listOptions)
	if err != nil {
		return fmt.Errorf("failed to list bootstrap tokens [%v]", err)
	}

	w := tabwriter.NewWriter(out, 10, 4, 3, ' ', 0)
	fmt.Fprintln(w, "TOKEN\tTTL\tEXPIRES\tUSAGES\tDESCRIPTION")
	for _, secret := range secrets.Items {
		tokenId := getSecretString(&secret, bootstrapapi.BootstrapTokenIDKey)
		if len(tokenId) == 0 {
			fmt.Fprintf(errW, "bootstrap token has no token-id data: %s\n", secret.Name)
			continue
		}

		// enforce the right naming convention
		if secret.Name != fmt.Sprintf("%s%s", bootstrapapi.BootstrapTokenSecretPrefix, tokenId) {
			fmt.Fprintf(errW, "bootstrap token name is not of the form '%s(token-id)': %s\n", bootstrapapi.BootstrapTokenSecretPrefix, secret.Name)
			continue
		}

		tokenSecret := getSecretString(&secret, bootstrapapi.BootstrapTokenSecretKey)
		if len(tokenSecret) == 0 {
			fmt.Fprintf(errW, "bootstrap token has no token-secret data: %s\n", secret.Name)
			continue
		}
		td := &kubeadmapi.TokenDiscovery{ID: tokenId, Secret: tokenSecret}

		// Expiration time is optional, if not specified this implies the token
		// never expires.
		ttl := "<forever>"
		expires := "<never>"
		secretExpiration := getSecretString(&secret, bootstrapapi.BootstrapTokenExpirationKey)
		if len(secretExpiration) > 0 {
			expireTime, err := time.Parse(time.RFC3339, secretExpiration)
			if err != nil {
				fmt.Fprintf(errW, "can't parse expiration time of bootstrap token %s\n", secret.Name)
				continue
			}
			ttl = printers.ShortHumanDuration(expireTime.Sub(time.Now()))
			expires = expireTime.Format(time.RFC3339)
		}

		usages := []string{}
		for k, v := range secret.Data {
			// Skip all fields that don't include this prefix
			if !strings.Contains(k, bootstrapapi.BootstrapTokenUsagePrefix) {
				continue
			}
			// Skip those that don't have this usage set to true
			if string(v) != "true" {
				continue
			}
			usages = append(usages, strings.TrimPrefix(k, bootstrapapi.BootstrapTokenUsagePrefix))
		}
		sort.Strings(usages)
		usageString := strings.Join(usages, ",")
		if len(usageString) == 0 {
			usageString = "<none>"
		}

		description := getSecretString(&secret, bootstrapapi.BootstrapTokenDescriptionKey)
		if len(description) == 0 {
			description = "<none>"
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", tokenutil.BearerToken(td), ttl, expires, usageString, description)
	}
	w.Flush()
	return nil
}

// RunDeleteToken removes a bootstrap token from the server.
func RunDeleteToken(out io.Writer, client *clientset.Clientset, tokenIdOrToken string) error {
	// Assume the given first argument is a token id and try to parse it
	tokenId := tokenIdOrToken
	if err := tokenutil.ParseTokenID(tokenIdOrToken); err != nil {
		if tokenId, _, err = tokenutil.ParseToken(tokenIdOrToken); err != nil {
			return fmt.Errorf("given token or token id %q didn't match pattern [%q] or [%q]", tokenIdOrToken, tokenutil.TokenIDRegexpString, tokenutil.TokenRegexpString)
		}
	}

	tokenSecretName := fmt.Sprintf("%s%s", bootstrapapi.BootstrapTokenSecretPrefix, tokenId)
	if err := client.CoreV1().Secrets(metav1.NamespaceSystem).Delete(tokenSecretName, nil); err != nil {
		return fmt.Errorf("failed to delete bootstrap token [%v]", err)
	}
	fmt.Fprintf(out, "bootstrap token with id %q deleted\n", tokenId)
	return nil
}

func getSecretString(secret *v1.Secret, key string) string {
	if secret.Data == nil {
		return ""
	}
	if val, ok := secret.Data[key]; ok {
		return string(val)
	}
	return ""
}
