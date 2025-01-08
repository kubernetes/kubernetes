package cmdlist

import (
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/openshift-eng/openshift-tests-extension/pkg/extension"
	"github.com/openshift-eng/openshift-tests-extension/pkg/flags"
)

func NewListCommand(registry *extension.Registry) *cobra.Command {
	opts := struct {
		componentFlags     *flags.ComponentFlags
		suiteFlags         *flags.SuiteFlags
		outputFlags        *flags.OutputFlags
		environmentalFlags *flags.EnvironmentalFlags
	}{
		suiteFlags:         flags.NewSuiteFlags(),
		componentFlags:     flags.NewComponentFlags(),
		outputFlags:        flags.NewOutputFlags(),
		environmentalFlags: flags.NewEnvironmentalFlags(),
	}

	// Tests
	listTestsCmd := &cobra.Command{
		Use:          "tests",
		Short:        "List available tests",
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			ext := registry.Get(opts.componentFlags.Component)
			if ext == nil {
				return fmt.Errorf("component not found: %s", opts.componentFlags.Component)
			}

			// Find suite, if specified
			var foundSuite *extension.Suite
			var err error
			if opts.suiteFlags.Suite != "" {
				foundSuite, err = ext.GetSuite(opts.suiteFlags.Suite)
				if err != nil {
					return err
				}
			}

			// Filter for suite
			specs := ext.GetSpecs()
			if foundSuite != nil {
				specs, err = specs.Filter(foundSuite.Qualifiers)
				if err != nil {
					return err
				}
			}

			specs, err = specs.FilterByEnvironment(*opts.environmentalFlags)
			if err != nil {
				return err
			}

			data, err := opts.outputFlags.Marshal(specs)
			if err != nil {
				return err
			}
			fmt.Fprintf(os.Stdout, "%s\n", string(data))
			return nil
		},
	}
	opts.suiteFlags.BindFlags(listTestsCmd.Flags())
	opts.componentFlags.BindFlags(listTestsCmd.Flags())
	opts.environmentalFlags.BindFlags(listTestsCmd.Flags())
	opts.outputFlags.BindFlags(listTestsCmd.Flags())

	// Suites
	listSuitesCommand := &cobra.Command{
		Use:          "suites",
		Short:        "List available suites",
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			ext := registry.Get(opts.componentFlags.Component)
			if ext == nil {
				return fmt.Errorf("component not found: %s", opts.componentFlags.Component)
			}

			suites := ext.Suites

			data, err := opts.outputFlags.Marshal(suites)
			if err != nil {
				return err
			}
			fmt.Fprintf(os.Stdout, "%s\n", string(data))
			return nil
		},
	}
	opts.componentFlags.BindFlags(listSuitesCommand.Flags())
	opts.outputFlags.BindFlags(listSuitesCommand.Flags())

	// Components
	listComponentsCmd := &cobra.Command{
		Use:          "components",
		Short:        "List available components",
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			var components []*extension.Component
			registry.Walk(func(e *extension.Extension) {
				components = append(components, &e.Component)
			})

			data, err := opts.outputFlags.Marshal(components)
			if err != nil {
				return err
			}
			fmt.Fprintf(os.Stdout, "%s\n", string(data))
			return nil
		},
	}
	opts.outputFlags.BindFlags(listComponentsCmd.Flags())

	var listCmd = &cobra.Command{
		Use:   "list [subcommand]",
		Short: "List items",
		RunE: func(cmd *cobra.Command, args []string) error {
			return listTestsCmd.RunE(cmd, args)
		},
	}
	opts.suiteFlags.BindFlags(listCmd.Flags())
	opts.componentFlags.BindFlags(listCmd.Flags())
	opts.outputFlags.BindFlags(listCmd.Flags())
	opts.environmentalFlags.BindFlags(listCmd.Flags())
	listCmd.AddCommand(listTestsCmd, listComponentsCmd, listSuitesCommand)

	return listCmd
}
