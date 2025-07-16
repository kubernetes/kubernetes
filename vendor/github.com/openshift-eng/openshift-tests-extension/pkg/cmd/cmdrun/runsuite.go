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
		junitPath        string
	}{
		componentFlags:   flags.NewComponentFlags(),
		outputFlags:      flags.NewOutputFlags(),
		concurrencyFlags: flags.NewConcurrencyFlags(),
		junitPath:        "",
	}

	cmd := &cobra.Command{
		Use: "run-suite NAME",
		Short: "Run a group of tests by suite. This is more limited than origin, and intended for light local " +
			"development use. Orchestration parameters, scheduling, isolation, etc are not obeyed, and Ginkgo tests are executed serially.",
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			ext := registry.Get(opts.componentFlags.Component)
			if ext == nil {
				return fmt.Errorf("component not found: %s", opts.componentFlags.Component)
			}
			if len(args) != 1 {
				return fmt.Errorf("must specify one suite name")
			}
			suite, err := ext.GetSuite(args[0])
			if err != nil {
				return errors.Wrapf(err, "couldn't find suite: %s", args[0])
			}

			compositeWriter := extensiontests.NewCompositeResultWriter()
			defer func() {
				if err = compositeWriter.Flush(); err != nil {
					fmt.Fprintf(os.Stderr, "failed to write results: %v\n", err)
				}
			}()

			// JUnit writer if needed
			if opts.junitPath != "" {
				junitWriter, err := extensiontests.NewJUnitResultWriter(opts.junitPath, suite.Name)
				if err != nil {
					return errors.Wrap(err, "couldn't create junit writer")
				}
				compositeWriter.AddWriter(junitWriter)
			}

			// JSON writer
			jsonWriter, err := extensiontests.NewJSONResultWriter(os.Stdout,
				extensiontests.ResultFormat(opts.outputFlags.Output))
			if err != nil {
				return err
			}
			compositeWriter.AddWriter(jsonWriter)

			specs, err := ext.GetSpecs().Filter(suite.Qualifiers)
			if err != nil {
				return errors.Wrap(err, "couldn't filter specs")
			}

			return specs.Run(compositeWriter, opts.concurrencyFlags.MaxConcurency)
		},
	}
	opts.componentFlags.BindFlags(cmd.Flags())
	opts.outputFlags.BindFlags(cmd.Flags())
	opts.concurrencyFlags.BindFlags(cmd.Flags())
	cmd.Flags().StringVarP(&opts.junitPath, "junit-path", "j", opts.junitPath, "write results to junit XML")

	return cmd
}
