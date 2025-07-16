package cmdimages

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/spf13/cobra"

	"github.com/openshift-eng/openshift-tests-extension/pkg/extension"
	"github.com/openshift-eng/openshift-tests-extension/pkg/flags"
)

func NewImagesCommand(registry *extension.Registry) *cobra.Command {
	componentFlags := flags.NewComponentFlags()

	cmd := &cobra.Command{
		Use:          "images",
		Short:        "List test images",
		SilenceUsage: true,
		RunE: func(cmd *cobra.Command, args []string) error {
			extension := registry.Get(componentFlags.Component)
			if extension == nil {
				return fmt.Errorf("couldn't find the component %q", componentFlags.Component)
			}
			images, err := json.Marshal(extension.Images)
			if err != nil {
				return err
			}
			fmt.Fprintf(os.Stdout, "%s\n", images)
			return nil
		},
	}
	componentFlags.BindFlags(cmd.Flags())
	return cmd
}
