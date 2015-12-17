package run

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/BurntSushi/toml"
)

// PrintConfigCommand represents the command executed by "influxd config".
type PrintConfigCommand struct {
	Stdin  io.Reader
	Stdout io.Writer
	Stderr io.Writer
}

// NewPrintConfigCommand return a new instance of PrintConfigCommand.
func NewPrintConfigCommand() *PrintConfigCommand {
	return &PrintConfigCommand{
		Stdin:  os.Stdin,
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
}

// Run parses and prints the current config loaded.
func (cmd *PrintConfigCommand) Run(args ...string) error {
	// Parse command flags.
	fs := flag.NewFlagSet("", flag.ContinueOnError)
	configPath := fs.String("config", "", "")
	hostname := fs.String("hostname", "", "")
	fs.Usage = func() { fmt.Fprintln(cmd.Stderr, printConfigUsage) }
	if err := fs.Parse(args); err != nil {
		return err
	}

	// Parse config from path.
	config, err := cmd.parseConfig(*configPath)
	if err != nil {
		return fmt.Errorf("parse config: %s", err)
	}

	// Apply any environment variables on top of the parsed config
	if err := config.ApplyEnvOverrides(); err != nil {
		return fmt.Errorf("apply env config: %v", err)
	}

	// Override config properties.
	if *hostname != "" {
		config.Meta.Hostname = *hostname
	}

	// Validate the configuration.
	if err := config.Validate(); err != nil {
		return fmt.Errorf("%s. To generate a valid configuration file run `influxd config > influxdb.generated.conf`", err)
	}

	toml.NewEncoder(cmd.Stdout).Encode(config)
	fmt.Fprint(cmd.Stdout, "\n")

	return nil
}

// ParseConfig parses the config at path.
// Returns a demo configuration if path is blank.
func (cmd *PrintConfigCommand) parseConfig(path string) (*Config, error) {
	if path == "" {
		return NewDemoConfig()
	}

	config := NewConfig()
	if _, err := toml.DecodeFile(path, &config); err != nil {
		return nil, err
	}
	return config, nil
}

var printConfigUsage = `usage: config

	config displays the default configuration
`
