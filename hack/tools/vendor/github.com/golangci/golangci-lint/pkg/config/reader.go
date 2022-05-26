package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/mitchellh/go-homedir"
	"github.com/spf13/viper"

	"github.com/golangci/golangci-lint/pkg/exitcodes"
	"github.com/golangci/golangci-lint/pkg/fsutils"
	"github.com/golangci/golangci-lint/pkg/logutils"
	"github.com/golangci/golangci-lint/pkg/sliceutil"
)

type FileReader struct {
	log            logutils.Log
	cfg            *Config
	commandLineCfg *Config
}

func NewFileReader(toCfg, commandLineCfg *Config, log logutils.Log) *FileReader {
	return &FileReader{
		log:            log,
		cfg:            toCfg,
		commandLineCfg: commandLineCfg,
	}
}

func (r *FileReader) Read() error {
	// XXX: hack with double parsing for 2 purposes:
	// 1. to access "config" option here.
	// 2. to give config less priority than command line.

	configFile, err := r.parseConfigOption()
	if err != nil {
		if err == errConfigDisabled {
			return nil
		}

		return fmt.Errorf("can't parse --config option: %s", err)
	}

	if configFile != "" {
		viper.SetConfigFile(configFile)

		// Assume YAML if the file has no extension.
		if filepath.Ext(configFile) == "" {
			viper.SetConfigType("yaml")
		}
	} else {
		r.setupConfigFileSearch()
	}

	return r.parseConfig()
}

func (r *FileReader) parseConfig() error {
	if err := viper.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); ok {
			return nil
		}

		return fmt.Errorf("can't read viper config: %s", err)
	}

	usedConfigFile := viper.ConfigFileUsed()
	if usedConfigFile == "" {
		return nil
	}

	usedConfigFile, err := fsutils.ShortestRelPath(usedConfigFile, "")
	if err != nil {
		r.log.Warnf("Can't pretty print config file path: %s", err)
	}
	r.log.Infof("Used config file %s", usedConfigFile)
	usedConfigDir := filepath.Dir(usedConfigFile)
	if usedConfigDir, err = filepath.Abs(usedConfigDir); err != nil {
		return fmt.Errorf("can't get config directory")
	}
	r.cfg.cfgDir = usedConfigDir

	if err := viper.Unmarshal(r.cfg); err != nil {
		return fmt.Errorf("can't unmarshal config by viper: %s", err)
	}

	if err := r.validateConfig(); err != nil {
		return fmt.Errorf("can't validate config: %s", err)
	}

	if r.cfg.InternalTest { // just for testing purposes: to detect config file usage
		fmt.Fprintln(logutils.StdOut, "test")
		os.Exit(exitcodes.Success)
	}

	return nil
}

func (r *FileReader) validateConfig() error {
	c := r.cfg
	if len(c.Run.Args) != 0 {
		return errors.New("option run.args in config isn't supported now")
	}

	if c.Run.CPUProfilePath != "" {
		return errors.New("option run.cpuprofilepath in config isn't allowed")
	}

	if c.Run.MemProfilePath != "" {
		return errors.New("option run.memprofilepath in config isn't allowed")
	}

	if c.Run.TracePath != "" {
		return errors.New("option run.tracepath in config isn't allowed")
	}

	if c.Run.IsVerbose {
		return errors.New("can't set run.verbose option with config: only on command-line")
	}
	for i, rule := range c.Issues.ExcludeRules {
		if err := rule.Validate(); err != nil {
			return fmt.Errorf("error in exclude rule #%d: %v", i, err)
		}
	}
	if len(c.Severity.Rules) > 0 && c.Severity.Default == "" {
		return errors.New("can't set severity rule option: no default severity defined")
	}
	for i, rule := range c.Severity.Rules {
		if err := rule.Validate(); err != nil {
			return fmt.Errorf("error in severity rule #%d: %v", i, err)
		}
	}
	if err := c.LintersSettings.Govet.Validate(); err != nil {
		return fmt.Errorf("error in govet config: %v", err)
	}
	return nil
}

func getFirstPathArg() string {
	args := os.Args

	// skip all args ([golangci-lint, run/linters]) before files/dirs list
	for len(args) != 0 {
		if args[0] == "run" {
			args = args[1:]
			break
		}

		args = args[1:]
	}

	// find first file/dir arg
	firstArg := "./..."
	for _, arg := range args {
		if !strings.HasPrefix(arg, "-") {
			firstArg = arg
			break
		}
	}

	return firstArg
}

func (r *FileReader) setupConfigFileSearch() {
	firstArg := getFirstPathArg()
	absStartPath, err := filepath.Abs(firstArg)
	if err != nil {
		r.log.Warnf("Can't make abs path for %q: %s", firstArg, err)
		absStartPath = filepath.Clean(firstArg)
	}

	// start from it
	var curDir string
	if fsutils.IsDir(absStartPath) {
		curDir = absStartPath
	} else {
		curDir = filepath.Dir(absStartPath)
	}

	// find all dirs from it up to the root
	configSearchPaths := []string{"./"}

	for {
		configSearchPaths = append(configSearchPaths, curDir)
		newCurDir := filepath.Dir(curDir)
		if curDir == newCurDir || newCurDir == "" {
			break
		}
		curDir = newCurDir
	}

	// find home directory for global config
	if home, err := homedir.Dir(); err != nil {
		r.log.Warnf("Can't get user's home directory: %s", err.Error())
	} else if !sliceutil.Contains(configSearchPaths, home) {
		configSearchPaths = append(configSearchPaths, home)
	}

	r.log.Infof("Config search paths: %s", configSearchPaths)
	viper.SetConfigName(".golangci")
	for _, p := range configSearchPaths {
		viper.AddConfigPath(p)
	}
}

var errConfigDisabled = errors.New("config is disabled by --no-config")

func (r *FileReader) parseConfigOption() (string, error) {
	cfg := r.commandLineCfg
	if cfg == nil {
		return "", nil
	}

	configFile := cfg.Run.Config
	if cfg.Run.NoConfig && configFile != "" {
		return "", fmt.Errorf("can't combine option --config and --no-config")
	}

	if cfg.Run.NoConfig {
		return "", errConfigDisabled
	}

	configFile, err := homedir.Expand(configFile)
	if err != nil {
		return "", fmt.Errorf("failed to expand configuration path")
	}

	return configFile, nil
}
