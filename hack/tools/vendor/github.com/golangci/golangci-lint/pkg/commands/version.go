package commands

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"github.com/golangci/golangci-lint/pkg/config"
)

type jsonVersion struct {
	Version string `json:"version"`
	Commit  string `json:"commit"`
	Date    string `json:"date"`
}

func (e *Executor) initVersionConfiguration(cmd *cobra.Command) {
	fs := cmd.Flags()
	fs.SortFlags = false // sort them as they are defined here
	initVersionFlagSet(fs, e.cfg)
}

func initVersionFlagSet(fs *pflag.FlagSet, cfg *config.Config) {
	// Version config
	vc := &cfg.Version
	fs.StringVar(&vc.Format, "format", "", wh("The version's format can be: 'short', 'json'"))
}

func (e *Executor) initVersion() {
	versionCmd := &cobra.Command{
		Use:   "version",
		Short: "Version",
		RunE: func(cmd *cobra.Command, _ []string) error {
			switch strings.ToLower(e.cfg.Version.Format) {
			case "short":
				fmt.Println(e.version)
			case "json":
				ver := jsonVersion{
					Version: e.version,
					Commit:  e.commit,
					Date:    e.date,
				}
				data, err := json.Marshal(&ver)
				if err != nil {
					return err
				}
				fmt.Println(string(data))
			default:
				fmt.Printf("golangci-lint has version %s built from %s on %s\n", e.version, e.commit, e.date)
			}
			return nil
		},
	}

	e.rootCmd.AddCommand(versionCmd)
	e.initVersionConfiguration(versionCmd)
}
