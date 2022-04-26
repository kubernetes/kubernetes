// Copyright © 2014 Steve Francia <spf@spf13.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

// Viper is a application configuration system.
// It believes that applications can be configured a variety of ways
// via flags, ENVIRONMENT variables, configuration files retrieved
// from the file system, or a remote key/value store.

package viper

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"unicode"

	"github.com/spf13/cast"
)

// ConfigParseError denotes failing to parse configuration file.
type ConfigParseError struct {
	err error
}

// Error returns the formatted configuration error.
func (pe ConfigParseError) Error() string {
	return fmt.Sprintf("While parsing config: %s", pe.err.Error())
}

// toCaseInsensitiveValue checks if the value is a  map;
// if so, create a copy and lower-case the keys recursively.
func toCaseInsensitiveValue(value interface{}) interface{} {
	switch v := value.(type) {
	case map[interface{}]interface{}:
		value = copyAndInsensitiviseMap(cast.ToStringMap(v))
	case map[string]interface{}:
		value = copyAndInsensitiviseMap(v)
	}

	return value
}

// copyAndInsensitiviseMap behaves like insensitiviseMap, but creates a copy of
// any map it makes case insensitive.
func copyAndInsensitiviseMap(m map[string]interface{}) map[string]interface{} {
	nm := make(map[string]interface{})

	for key, val := range m {
		lkey := strings.ToLower(key)
		switch v := val.(type) {
		case map[interface{}]interface{}:
			nm[lkey] = copyAndInsensitiviseMap(cast.ToStringMap(v))
		case map[string]interface{}:
			nm[lkey] = copyAndInsensitiviseMap(v)
		default:
			nm[lkey] = v
		}
	}

	return nm
}

func insensitiviseMap(m map[string]interface{}) {
	for key, val := range m {
		switch val.(type) {
		case map[interface{}]interface{}:
			// nested map: cast and recursively insensitivise
			val = cast.ToStringMap(val)
			insensitiviseMap(val.(map[string]interface{}))
		case map[string]interface{}:
			// nested map: recursively insensitivise
			insensitiviseMap(val.(map[string]interface{}))
		}

		lower := strings.ToLower(key)
		if key != lower {
			// remove old key (not lower-cased)
			delete(m, key)
		}
		// update map
		m[lower] = val
	}
}

func absPathify(logger Logger, inPath string) string {
	logger.Info("trying to resolve absolute path", "path", inPath)

	if inPath == "$HOME" || strings.HasPrefix(inPath, "$HOME"+string(os.PathSeparator)) {
		inPath = userHomeDir() + inPath[5:]
	}

	inPath = os.ExpandEnv(inPath)

	if filepath.IsAbs(inPath) {
		return filepath.Clean(inPath)
	}

	p, err := filepath.Abs(inPath)
	if err == nil {
		return filepath.Clean(p)
	}

	logger.Error(fmt.Errorf("could not discover absolute path: %w", err).Error())

	return ""
}

func stringInSlice(a string, list []string) bool {
	for _, b := range list {
		if b == a {
			return true
		}
	}
	return false
}

func userHomeDir() string {
	if runtime.GOOS == "windows" {
		home := os.Getenv("HOMEDRIVE") + os.Getenv("HOMEPATH")
		if home == "" {
			home = os.Getenv("USERPROFILE")
		}
		return home
	}
	return os.Getenv("HOME")
}

func safeMul(a, b uint) uint {
	c := a * b
	if a > 1 && b > 1 && c/b != a {
		return 0
	}
	return c
}

// parseSizeInBytes converts strings like 1GB or 12 mb into an unsigned integer number of bytes
func parseSizeInBytes(sizeStr string) uint {
	sizeStr = strings.TrimSpace(sizeStr)
	lastChar := len(sizeStr) - 1
	multiplier := uint(1)

	if lastChar > 0 {
		if sizeStr[lastChar] == 'b' || sizeStr[lastChar] == 'B' {
			if lastChar > 1 {
				switch unicode.ToLower(rune(sizeStr[lastChar-1])) {
				case 'k':
					multiplier = 1 << 10
					sizeStr = strings.TrimSpace(sizeStr[:lastChar-1])
				case 'm':
					multiplier = 1 << 20
					sizeStr = strings.TrimSpace(sizeStr[:lastChar-1])
				case 'g':
					multiplier = 1 << 30
					sizeStr = strings.TrimSpace(sizeStr[:lastChar-1])
				default:
					multiplier = 1
					sizeStr = strings.TrimSpace(sizeStr[:lastChar])
				}
			}
		}
	}

	size := cast.ToInt(sizeStr)
	if size < 0 {
		size = 0
	}

	return safeMul(uint(size), multiplier)
}

// deepSearch scans deep maps, following the key indexes listed in the
// sequence "path".
// The last value is expected to be another map, and is returned.
//
// In case intermediate keys do not exist, or map to a non-map value,
// a new map is created and inserted, and the search continues from there:
// the initial map "m" may be modified!
func deepSearch(m map[string]interface{}, path []string) map[string]interface{} {
	for _, k := range path {
		m2, ok := m[k]
		if !ok {
			// intermediate key does not exist
			// => create it and continue from there
			m3 := make(map[string]interface{})
			m[k] = m3
			m = m3
			continue
		}
		m3, ok := m2.(map[string]interface{})
		if !ok {
			// intermediate key is a value
			// => replace with a new map
			m3 = make(map[string]interface{})
			m[k] = m3
		}
		// continue search from here
		m = m3
	}
	return m
}
