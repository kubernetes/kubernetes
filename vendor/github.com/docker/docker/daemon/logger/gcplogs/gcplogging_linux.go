// +build linux

package gcplogs

import (
	"os"

	"github.com/docker/docker/dockerversion"
	"github.com/docker/docker/pkg/homedir"
	"github.com/sirupsen/logrus"
)

// ensureHomeIfIAmStatic ensure $HOME to be set if dockerversion.IAmStatic is "true".
// See issue #29344: gcplogs segfaults (static binary)
// If HOME is not set, logging.NewClient() will call os/user.Current() via oauth2/google.
// However, in static binary, os/user.Current() leads to segfault due to a glibc issue that won't be fixed
// in a short term. (golang/go#13470, https://sourceware.org/bugzilla/show_bug.cgi?id=19341)
// So we forcibly set HOME so as to avoid call to os/user/Current()
func ensureHomeIfIAmStatic() error {
	// Note: dockerversion.IAmStatic and homedir.GetStatic() is only available for linux.
	// So we need to use them in this gcplogging_linux.go rather than in gcplogging.go
	if dockerversion.IAmStatic == "true" && os.Getenv("HOME") == "" {
		home, err := homedir.GetStatic()
		if err != nil {
			return err
		}
		logrus.Warnf("gcplogs requires HOME to be set for static daemon binary. Forcibly setting HOME to %s.", home)
		os.Setenv("HOME", home)
	}
	return nil
}
