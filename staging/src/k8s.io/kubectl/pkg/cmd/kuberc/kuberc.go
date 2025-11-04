/*
Copyright The Kubernetes Authors.

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

package kuberc

import (
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"
	"k8s.io/kubectl/pkg/config/v1beta1"
	"sigs.k8s.io/yaml"

	"k8s.io/cli-runtime/pkg/genericiooptions"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

var (
	kubercLong = templates.LongDesc(i18n.T(`
		Manage user preferences (kuberc) file.

		The kuberc file allows you to customize your kubectl experience.`))

	kubercExample = templates.Examples(i18n.T(`
		# View the current kuberc configuration
		kubectl alpha kuberc view

		# Set a default value for a command flag
		kubectl alpha kuberc set --section defaults --command get --option output=wide

		# Create an alias for a command
		kubectl alpha kuberc set --section aliases --name getn --command get --prependarg nodes --option output=wide`))
)

// NewCmdKubeRC creates a command object for the "kuberc" action, and adds all child commands to it.
func NewCmdKubeRC(streams genericiooptions.IOStreams) *cobra.Command {
	cmd := &cobra.Command{
		Use:                   "kuberc SUBCOMMAND",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Manage kuberc configuration files"),
		Long:                  kubercLong,
		Example:               kubercExample,
		Run:                   cmdutil.DefaultSubCommandRun(streams.ErrOut),
	}

	cmd.AddCommand(NewCmdKubeRCView(streams))
	cmd.AddCommand(NewCmdKubeRCSet(streams))

	return cmd
}

// SavePreference saves the preference to the kuberc file
func SavePreference(pref *v1beta1.Preference, file string, out io.Writer) error {
	dir := filepath.Dir(file)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", dir, err)
	}

	data, err := yaml.Marshal(pref)
	if err != nil {
		return fmt.Errorf("failed to marshal preferences: %w", err)
	}

	if err := os.WriteFile(file, data, 0644); err != nil {
		return fmt.Errorf("failed to write kuberc file: %w", err)
	}

	fmt.Fprintf(out, "Updated %s\n", file) // nolint:errcheck
	return nil
}
