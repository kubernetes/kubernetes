package cmd

import (
	"io"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	utilflag "k8s.io/kubernetes/pkg/util/flag"
)

func NewCmdLogin(f *cmdutil.Factory, pathOptions *clientcmd.PathOptions, out io.Writer) *cobra.Command {

	cmd := &cobra.Command{
		Use:   "login",
		Short: "login Log in to the cluster.",
		Long:  `login obtains the neccesary credentials to make authenticated requests to the cluster. The type of log in depends on the 'auth-provider' type in your configuration, and will likely require some type of user interaction.`,

		Run: func(cmd *cobra.Command, args []string) {
			err := RunLogin(out, cmd, f, pathOptions, args)
			cmdutil.CheckErr(err)
		},
	}

	return cmd
}

func RunLogin(out io.Writer, cmd *cobra.Command, f *cmdutil.Factory, pathOptions *clientcmd.PathOptions, args []string) error {

	flags := pflag.NewFlagSet("", pflag.ContinueOnError)
	flags.SetNormalizeFunc(utilflag.WarnWordSepNormalizeFunc) // Warn for "_" flags

	cmdCliCfg := cmdutil.DefaultClientConfig(flags)
	cliCfg, err := cmdCliCfg.ClientConfig()
	if err != nil {
		return err
	}

	auth, err := restclient.GetAuthProvider(cliCfg.Host,
		cliCfg.AuthProvider,
		cliCfg.AuthConfigPersister)
	if err != nil {
		return err
	}

	return auth.Login()
}
