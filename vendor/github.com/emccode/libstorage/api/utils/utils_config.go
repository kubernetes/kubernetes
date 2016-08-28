package utils

import (
	"fmt"
	"strings"

	"github.com/akutz/gofig"
)

func isSet(
	config gofig.Config,
	key string,
	roots ...string) bool {

	for _, r := range roots {
		rk := strings.Replace(key, "libstorage.", fmt.Sprintf("%s.", r), 1)
		if config.IsSet(rk) {
			return true
		}
	}

	if config.IsSet(key) {
		return true
	}

	return false
}

func getString(
	config gofig.Config,
	key string,
	roots ...string) string {

	var val string

	for _, r := range roots {
		rk := strings.Replace(key, "libstorage.", fmt.Sprintf("%s.", r), 1)
		if val = config.GetString(rk); val != "" {
			return val
		}
	}

	val = config.GetString(key)
	if val != "" {
		return val
	}

	return ""
}

func getBool(
	config gofig.Config,
	key string,
	roots ...string) bool {

	for _, r := range roots {
		rk := strings.Replace(key, "libstorage.", fmt.Sprintf("%s.", r), 1)
		if config.IsSet(rk) {
			return config.GetBool(rk)
		}
	}

	if config.IsSet(key) {
		return config.GetBool(key)
	}

	return false
}
