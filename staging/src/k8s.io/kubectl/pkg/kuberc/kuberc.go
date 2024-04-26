package kuberc

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
	"k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/config/v1alpha1"
	"k8s.io/kubectl/pkg/scheme"
)

const RecommendedKubeRCFileName = ".kuberc"

var (
	RecommendedConfigDir  = filepath.Join(homedir.HomeDir(), clientcmd.RecommendedHomeDir)
	RecommendedKubeRCFile = filepath.Join(RecommendedConfigDir, RecommendedKubeRCFileName)
)

// PreferencesHandler is responsible for setting default flags
// arguments based on user's kuberc configuration.
type PreferencesHandler interface {
	InjectOverrides(rootCmd *cobra.Command, args []string)
	InjectAliases(rootCmd *cobra.Command, args []string)
}

// Preferences stores the kuberc file coming either from environment variable
// or file from set in flag or the default kuberc path.
type Preferences struct {
	KubeRC string
}

// NewPreferences returns initialized Prefrences object.
func NewPreferences() *Preferences {
	return &Preferences{}
}

// AddFlags adds kuberc related flags into the command.
func (p *Preferences) AddFlags(flags *pflag.FlagSet) {
	if util.KubeRC.IsEnabled() {
		flags.StringVar(&p.KubeRC, "kuberc", p.KubeRC, "Path to the kuberc file to use for preferences.")
	}
}

// InjectOverrides injects the default flags defined in kuberc file.
func (p *Preferences) InjectOverrides(rootCmd *cobra.Command, args []string, errOut io.Writer) {
	kuberc, err := p.getPreferences()
	if err != nil {
		fmt.Fprintf(errOut, "kuberc error %v\n", err)
		return
	}

	if kuberc == nil {
		return
	}

	args = args[1:]
	cmd, _, err := rootCmd.Find(args)
	if err != nil {
		fmt.Fprintf(errOut, "could not find command %q\n", args)
		return
	}

	for _, c := range kuberc.Spec.Overrides {
		if c.Command != cmd.Name() {
			continue
		}

		for _, fl := range c.Flags {
			err = cmd.Flags().Set(fmt.Sprintf("%s", fl.Name), fl.Default)
			if err != nil {
				fmt.Fprintf(errOut, "could not apply value %s to flag %s in command %s\n", fl.Default, fl.Name, c.Command)
				return
			}
		}

		// TODO do we really need to parse in here?
		/*if err = cmd.Flags().Parse(args); err != nil {
			// return without raising any error because
			// real command execution will catch this invalid request
			return
		}*/
	}
}

func (p *Preferences) InjectAliases(rootCmd *cobra.Command, args []string, errOut io.Writer) {
	kuberc, err := p.getPreferences()
	if err != nil {
		fmt.Fprintf(errOut, "kuberc error %v\n", err)
		return
	}

	if kuberc == nil {
		return
	}

	for _, alias := range kuberc.Spec.Aliases {
		commands := strings.Split(alias.Command, " ")
		cmd, flags, err := rootCmd.Find(commands)
		if err != nil {
			fmt.Fprintf(errOut, "Command %q not found to set alias %q: %v\n", alias.Command, alias.Name, flags)
			continue
		}
		// do not allow shadowing built-ins
		if _, _, err := rootCmd.Find([]string{alias.Name}); err == nil {
			fmt.Fprintf(errOut, "Setting alias %q to a built-in command is not supported\n", alias.Name)
			continue
		}
		cmd.Aliases = append(cmd.Aliases, alias.Name)
		cmd.Flags().Parse(alias.Arguments)
	}
}

// getPreferences returns v1alpha1.KubeRCConfiguration.
// If users sets kuberc file explicitly in --kuberc flag, it has the highest
// priority. If not specified, it looks for in KUBERC environment variable.
// If KUBERC is also not set, it falls back to default .kuberc file at the same location
// where kubeconfig's defaults are residing in.
func (p *Preferences) getPreferences() (*v1alpha1.Preferences, error) {
	if !util.KubeRC.IsEnabled() {
		return nil, nil
	}

	kubeRCFile := RecommendedKubeRCFile
	explicitly := false
	if p.KubeRC != "" {
		// TODO not working. Parseflags and get explicit kuberc path
		kubeRCFile = p.KubeRC
		explicitly = true
	}

	if kubeRCFile == "" && os.Getenv("KUBERC") != "" {
		kubeRCFile = os.Getenv("KUBERC")
		explicitly = true
	}

	kubeRCBytes, err := os.ReadFile(kubeRCFile)
	if err != nil {
		if os.IsNotExist(err) && !explicitly {
			return nil, nil
		}
		return nil, err
	}

	decoded, err := runtime.Decode(scheme.Codecs.UniversalDecoder(v1alpha1.SchemeGroupVersion), kubeRCBytes)
	if err != nil {
		return nil, err
	}
	return decoded.(*v1alpha1.Preferences), nil

}
