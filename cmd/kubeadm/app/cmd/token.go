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
	"os"
	"path"
	"text/tabwriter"
	"time"

	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/kubectl"
)

const (
	bootstrapTokenSecretPrefix = "bootstrap-token-"
	defaultTokenDuration       = time.Duration(8) * time.Hour
)

func NewCmdToken(out io.Writer) *cobra.Command {
	tokenCmd := &cobra.Command{
		Use:   "token",
		Short: "Manage bootstrap tokens",
	}

	var tokenDuration time.Duration
	tokenSecret := &kubeadmapi.Secrets{}
	createCmd := &cobra.Command{
		Use:   "create",
		Short: "Create discovery tokens on the server.",
		Run: func(tokenCmd *cobra.Command, args []string) {
			err := RunCreateToken(out, tokenCmd, tokenDuration, tokenSecret)
			kubeadmutil.CheckErr(err)
		},
	}
	createCmd.PersistentFlags().DurationVar(&tokenDuration,
		"duration", defaultTokenDuration, "Duration the token should be valid")
	createCmd.PersistentFlags().StringVar(
		&tokenSecret.GivenToken, "token", "",
		"Shared secret used to secure cluster bootstrap; if none is provided, one will be generated for you",
	)
	tokenCmd.AddCommand(createCmd)

	listCmd := &cobra.Command{
		Use:   "list",
		Short: "List discovery tokens on the server.",
		Run: func(tokenCmd *cobra.Command, args []string) {
			err := RunListTokens(out, tokenCmd)
			kubeadmutil.CheckErr(err)
		},
	}
	tokenCmd.AddCommand(listCmd)

	deleteCmd := &cobra.Command{
		Use:   "delete",
		Short: "Delete discovery tokens on the server.",
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
	client, err := createAPIClient()
	if err != nil {
		return err
	}

	err = kubeadmutil.GenerateTokenIfNeeded(tokenSecret)
	if err != nil {
		return err
	}

	secret := &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name: fmt.Sprintf("%s%s", bootstrapTokenSecretPrefix, tokenSecret.TokenID),
		},
		Type: api.SecretTypeBootstrapToken,
		Data: encodeTokenSecretData(tokenSecret, tokenDuration),
	}
	if _, err := client.Secrets(api.NamespaceSystem).Create(secret); err != nil {
		return fmt.Errorf("<cmd/token> failed to bootstrap token [%v]", err)
	}
	fmt.Printf("<cmd/token> Token secret created: %s\n", tokenSecret.GivenToken)

	return nil
}

func encodeTokenSecretData(tokenSecret *kubeadmapi.Secrets, duration time.Duration) map[string][]byte {
	var (
		data = map[string][]byte{}
	)

	data["token-id"] = []byte(tokenSecret.TokenID)
	data["token-secret"] = []byte(tokenSecret.BearerToken)

	t := time.Now()
	t = t.Add(duration)
	data["expiration"] = []byte(t.Format(time.RFC3339))

	return data
}

func createAPIClient() (*clientset.Clientset, error) {
	envParams := kubeadmapi.GetEnvParams()
	adminKubeconfig, err := clientcmd.LoadFromFile(path.Join(envParams["kubernetes_dir"], "admin.conf"))
	if err != nil {
		return nil, fmt.Errorf("<cmd/token> failed to load admin kubeconfig [%v]", err)
	}
	adminClientConfig, err := clientcmd.NewDefaultClientConfig(
		*adminKubeconfig,
		&clientcmd.ConfigOverrides{},
	).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("<cmd/token> failed to create API client configuration [%v]", err)
	}

	client, err := clientset.NewForConfig(adminClientConfig)
	if err != nil {
		return nil, fmt.Errorf("<cmd/token> failed to create API client [%v]", err)
	}
	return client, nil
}

// RunListTokens lists details on all existing bootstrap tokens on the server.
func RunListTokens(out io.Writer, cmd *cobra.Command) error {
	client, err := createAPIClient()
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

	w := tabwriter.NewWriter(os.Stdout, 10, 4, 3, ' ', 0)
	fmt.Fprintln(w, "ID\tTOKEN\tEXPIRATION")
	for _, secret := range results.Items {
		tokenId := secret.Data["token-id"]
		token := fmt.Sprintf("%s.%s", tokenId, secret.Data["token-secret"])
		expireTime, err := time.Parse(time.RFC3339, string(secret.Data["expiration"]))
		if err != nil {
			return fmt.Errorf("<cmd/token> error parsing expiry time [%v]", err)
		}
		expires := kubectl.ShortHumanDuration(expireTime.Sub(time.Now()))
		fmt.Fprintf(w, "%s\t%s\t%s\n", tokenId, token, expires)
	}
	w.Flush()

	return nil
}

// RunDeleteToken removes a bootstrap token from the server.
func RunDeleteToken(out io.Writer, cmd *cobra.Command, tokenId string) error {
	client, err := createAPIClient()
	if err != nil {
		return err
	}

	tokenSecretName := fmt.Sprintf("%s%s", bootstrapTokenSecretPrefix, tokenId)
	if err := client.Secrets(api.NamespaceSystem).Delete(tokenSecretName, nil); err != nil {
		return fmt.Errorf("<cmd/token> failed to delete bootstrap token [%v]", err)
	}
	fmt.Printf("<cmd/token> bootstrap token deleted: %s\n", tokenId)

	return nil
}
