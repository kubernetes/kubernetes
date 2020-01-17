/*
Copyright 2018 The Kubernetes Authors.

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

package viperconfig

import (
	"flag"
	"fmt"
	"github.com/pkg/errors"
	"path/filepath"

	"github.com/spf13/viper"
)

// ViperizeFlags checks whether a configuration file was specified,
// reads it, and updates the configuration variables in the specified
// flag set accordingly. Must be called after framework.HandleFlags()
// and before framework.AfterReadingAllFlags().
//
// The logic is so that a required configuration file must be present. If empty,
// the optional configuration file is used instead, unless also empty.
//
// Files can be specified with just a base name ("e2e", matches "e2e.json/yaml/..." in
// the current directory) or with path and suffix.
func ViperizeFlags(requiredConfig, optionalConfig string, flags *flag.FlagSet) error {
	viperConfig := optionalConfig
	required := false
	if requiredConfig != "" {
		viperConfig = requiredConfig
		required = true
	}
	if viperConfig == "" {
		return nil
	}
	viper.SetConfigName(filepath.Base(viperConfig))
	viper.AddConfigPath(filepath.Dir(viperConfig))
	wrapError := func(err error) error {
		if err == nil {
			return nil
		}
		errorPrefix := fmt.Sprintf("viper config %q", viperConfig)
		actualFile := viper.ConfigFileUsed()
		if actualFile != "" && actualFile != viperConfig {
			errorPrefix = fmt.Sprintf("%s = %q", errorPrefix, actualFile)
		}
		return errors.Wrap(err, errorPrefix)
	}

	if err := viper.ReadInConfig(); err != nil {
		// If the user specified a file suffix, the Viper won't
		// find the file because it always appends its known set
		// of file suffices. Therefore try once more without
		// suffix.
		ext := filepath.Ext(viperConfig)
		if _, ok := err.(viper.ConfigFileNotFoundError); ok && ext != "" {
			viper.SetConfigName(filepath.Base(viperConfig[0 : len(viperConfig)-len(ext)]))
			err = viper.ReadInConfig()
		}
		if err != nil {
			// If a config was required, then parsing must
			// succeed. This catches syntax errors and
			// "file not found". Unfortunately error
			// messages are sometimes hard to understand,
			// so try to help the user a bit.
			switch err.(type) {
			case viper.ConfigFileNotFoundError:
				if required {
					return wrapError(errors.New("not found"))
				}
				// Proceed without config.
				return nil
			case viper.UnsupportedConfigError:
				if required {
					return wrapError(errors.New("not using a supported file format"))
				}
				// Proceed without config.
				return nil
			default:
				// Something isn't right in the file.
				return wrapError(err)
			}
		}
	}

	// Update all flag values not already set with values found
	// via Viper. We do this ourselves instead of calling
	// something like viper.Unmarshal(&TestContext) because we
	// want to support all values, regardless where they are
	// stored.
	return wrapError(viperUnmarshal(flags))
}

// viperUnmarshall updates all flags with the corresponding values found
// via Viper, regardless whether the flag value is stored in TestContext, some other
// context or a local variable.
func viperUnmarshal(flags *flag.FlagSet) error {
	var result error
	set := make(map[string]bool)

	// Determine which values were already set explicitly via
	// flags. Those we don't overwrite because command line
	// flags have a higher priority.
	flags.Visit(func(f *flag.Flag) {
		set[f.Name] = true
	})

	flags.VisitAll(func(f *flag.Flag) {
		if result != nil ||
			set[f.Name] ||
			!viper.IsSet(f.Name) {
			return
		}

		// In contrast to viper.Unmarshal(), values
		// that have the wrong type (for example, a
		// list instead of a plain string) will not
		// trigger an error here. This could be fixed
		// by checking the type ourselves, but
		// probably isn't worth the effort.
		//
		// "%v" correctly turns bool, int, strings into
		// the representation expected by flag, so those
		// can be used in config files. Plain strings
		// always work there, just as on the command line.
		str := fmt.Sprintf("%v", viper.Get(f.Name))
		if err := f.Value.Set(str); err != nil {
			result = fmt.Errorf("setting option %q from config file value: %s", f.Name, err)
		}
	})

	return result
}
