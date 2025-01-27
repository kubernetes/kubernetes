package cmdrun

import (
	"fmt"
	"os"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"

	"github.com/openshift-eng/openshift-tests-extension/pkg/extension"
	"github.com/openshift-eng/openshift-tests-extension/pkg/extension/extensiontests"
	"github.com/openshift-eng/openshift-tests-extension/pkg/flags"
)

func NewRunSuiteCommand(registry *extension.Registry) *cobra.Command {
	opts := struct {
		componentFlags   *flags.ComponentFlags
		outputFlags      *flags.OutputFlags
		concurrencyFlags *flags.ConcurrencyFlags
	}{
		componentFlags:   flags.NewComponentFlags(),
		outputFlags:      flags.NewOutputFlags(),
		concurrencyFlags: flags.NewConcurrencyFlags(),
	}

	cmd := &cobra.Command{
		Use:          "run-suite NAME",
		Short:        "Run a group of tests by suite",
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			ext := registry.Get(opts.componentFlags.Component)
			if ext == nil {
				return fmt.Errorf("component not found: %s", opts.componentFlags.Component)
			}
			if len(args) != 1 {
				return fmt.Errorf("must specify one suite name")
			}

			w, err := extensiontests.NewResultWriter(os.Stdout, extensiontests.ResultFormat(opts.outputFlags.Output))
			if err != nil {
				return err
			}
			defer w.Flush()

			suite, err := ext.GetSuite(args[0])
			if err != nil {
				return errors.Wrapf(err, "couldn't find suite: %s", args[0])
			}

			specs, err := ext.GetSpecs().Filter(suite.Qualifiers)
			if err != nil {
				return errors.Wrap(err, "couldn't filter specs")
			}

			return specs.Run(w, opts.concurrencyFlags.MaxConcurency)
		},
	}
	opts.componentFlags.BindFlags(cmd.Flags())
	opts.outputFlags.BindFlags(cmd.Flags())
	opts.concurrencyFlags.BindFlags(cmd.Flags())

	return cmd
}
