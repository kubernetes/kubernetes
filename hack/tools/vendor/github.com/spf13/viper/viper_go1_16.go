//go:build go1.16 && finder
// +build go1.16,finder

package viper

import (
	"fmt"

	"github.com/spf13/afero"
)

// Search all configPaths for any config file.
// Returns the first path that exists (and is a config file).
func (v *Viper) findConfigFile() (string, error) {
	finder := finder{
		paths:            v.configPaths,
		fileNames:        []string{v.configName},
		extensions:       SupportedExts,
		withoutExtension: v.configType != "",
	}

	file, err := finder.Find(afero.NewIOFS(v.fs))
	if err != nil {
		return "", err
	}

	if file == "" {
		return "", ConfigFileNotFoundError{v.configName, fmt.Sprintf("%s", v.configPaths)}
	}

	return file, nil
}
