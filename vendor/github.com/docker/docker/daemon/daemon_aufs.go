// +build !exclude_graphdriver_aufs,linux

package daemon

import (
	"github.com/Sirupsen/logrus"
	"github.com/docker/docker/daemon/graphdriver"
	"github.com/docker/docker/daemon/graphdriver/aufs"
)

// Given the graphdriver ad, if it is aufs, then migrate it.
// If aufs driver is not built, this func is a noop.
func migrateIfAufs(driver graphdriver.Driver, root string) error {
	if ad, ok := driver.(*aufs.Driver); ok {
		logrus.Debugf("Migrating existing containers")
		if err := ad.Migrate(root, setupInitLayer); err != nil {
			return err
		}
	}
	return nil
}
