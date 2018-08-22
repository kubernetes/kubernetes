package cmd

import (
	"fmt"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/clientcmd/api"

	"k8s.io/cli-runtime/pkg/genericclioptions"
)

var (
	namespace_example = `
	# view the current namespace in your KUBECONFIG
	%[1]s ns

	# view all of the namespaces in use by contexts in your KUBECONFIG
	%[1] ns --list

	# switch your current-context to one that contains the desired namespace
	%[1]s ns foo
`
)

type NamespaceOptions struct {
	configFlags *genericclioptions.ConfigFlags

	rawConfig      api.Config
	listNamespaces bool
	args           []string

	genericclioptions.IOStreams
}

func NewNamespaceOptions(streams genericclioptions.IOStreams) *NamespaceOptions {
	return &NamespaceOptions{
		configFlags: genericclioptions.NewConfigFlags(),

		IOStreams: streams,
	}
}

func NewCmdNamespace(streams genericclioptions.IOStreams) *cobra.Command {
	o := NewNamespaceOptions(streams)

	cmd := &cobra.Command{
		Use:          "ns [new-namespace] [flags]",
		Short:        "View or set the current namespace",
		Example:      namespace_example,
		SilenceUsage: true,
		RunE: func(c *cobra.Command, args []string) error {
			if err := o.Complete(args); err != nil {
				return err
			}
			if err := o.Validate(); err != nil {
				return err
			}
			if err := o.Run(); err != nil {
				return err
			}

			return nil
		},
	}

	cmd.Flags().BoolVar(&o.listNamespaces, "list", o.listNamespaces, "if true, print the list of all namespaces in the current KUBECONFIG")
	o.configFlags.AddFlags(cmd.Flags())

	return cmd
}

func (o *NamespaceOptions) Complete(args []string) error {
	var err error
	o.rawConfig, err = o.configFlags.ToRawKubeConfigLoader().RawConfig()
	if err != nil {
		return err
	}

	o.args = args
	return nil
}

func (o *NamespaceOptions) Validate() error {
	if len(o.rawConfig.CurrentContext) == 0 {
		return fmt.Errorf("no context is currently set in your configuration")
	}
	if len(o.args) > 1 {
		return fmt.Errorf("either one or no arguments are allowed")
	}

	return nil
}

func (o *NamespaceOptions) Run() error {
	if len(o.args) > 0 && len(o.args[0]) > 0 {
		return o.setNamespace(o.args[0])
	}

	namespaces := map[string]bool{}

	for name, c := range o.rawConfig.Contexts {
		if !o.listNamespaces && name == o.rawConfig.CurrentContext {
			if len(c.Namespace) == 0 {
				return fmt.Errorf("no namespace is set for your current context: %q", name)
			}

			fmt.Fprintf(o.Out, "%s\n", c.Namespace)
			return nil
		}

		// skip if dealing with a namespace we have already seen
		// or if the namespace for the current context is empty
		if len(c.Namespace) == 0 {
			continue
		}
		if namespaces[c.Namespace] {
			continue
		}

		namespaces[c.Namespace] = true
	}

	if !o.listNamespaces {
		return fmt.Errorf("unable to find information for the current namespace in your configuration")
	}

	for n := range namespaces {
		fmt.Fprintf(o.Out, "%s\n", n)
	}

	return nil
}

func (o *NamespaceOptions) setNamespace(newNamespace string) error {
	if len(newNamespace) == 0 {
		return fmt.Errorf("a non-empty namespace must be provided")
	}

	existingCtx, ok := o.rawConfig.Contexts[o.rawConfig.CurrentContext]
	if !ok {
		return fmt.Errorf("unable to gather information about the current context")
	}

	if existingCtx.Namespace == newNamespace {
		fmt.Fprintf(o.Out, "already using namespace %q\n", newNamespace)
		return nil
	}

	// determine if a context exists for the new namespace
	existingCtxName := ""
	for name, c := range o.rawConfig.Contexts {
		if c.Namespace != newNamespace || c.Cluster != existingCtx.Cluster || c.AuthInfo != existingCtx.AuthInfo {
			continue
		}

		existingCtxName = name
		break
	}

	if len(existingCtxName) == 0 {
		newCtx := api.NewContext()
		newCtx.AuthInfo = existingCtx.AuthInfo
		newCtx.Cluster = existingCtx.Cluster
		newCtx.Namespace = newNamespace

		newCtxName := newNamespace
		if len(existingCtx.Cluster) > 0 {
			newCtxName = fmt.Sprintf("%s/%s", newCtxName, existingCtx.Cluster)
		}
		if len(existingCtx.AuthInfo) > 0 {
			cleanAuthInfo := strings.Split(existingCtx.AuthInfo, "/")[0]
			newCtxName = fmt.Sprintf("%s/%s", newCtxName, cleanAuthInfo)
		}

		o.rawConfig.Contexts[newCtxName] = newCtx
		existingCtxName = newCtxName
	}

	configAccess := clientcmd.NewDefaultPathOptions()
	o.rawConfig.CurrentContext = existingCtxName

	if err := clientcmd.ModifyConfig(configAccess, o.rawConfig, true); err != nil {
		return err
	}

	fmt.Fprintf(o.Out, "namespace changed to %q\n", newNamespace)
	return nil
}
