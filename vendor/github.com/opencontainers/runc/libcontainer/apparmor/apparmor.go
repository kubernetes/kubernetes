package apparmor

import "errors"

var (
	// IsEnabled returns true if apparmor is enabled for the host.
	IsEnabled = isEnabled

	// ApplyProfile will apply the profile with the specified name to the process after
	// the next exec. It is only supported on Linux and produces an ErrApparmorNotEnabled
	// on other platforms.
	ApplyProfile = applyProfile

	// ErrApparmorNotEnabled indicates that AppArmor is not enabled or not supported.
	ErrApparmorNotEnabled = errors.New("apparmor: config provided but apparmor not supported")
)
