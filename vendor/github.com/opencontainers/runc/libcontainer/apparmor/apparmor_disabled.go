// +build !apparmor !linux

package apparmor

import (
	"errors"
)

var ErrApparmorNotEnabled = errors.New("apparmor: config provided but apparmor not supported")

func IsEnabled() bool {
	return false
}

func ApplyProfile(name string) error {
	if name != "" {
		return ErrApparmorNotEnabled
	}
	return nil
}
