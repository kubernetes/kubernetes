// +build windows

package libnetwork

import (
	"github.com/docker/libnetwork/etchosts"
)

// Stub implementations for DNS related functions

func (sb *sandbox) startResolver(bool) {
}

func (sb *sandbox) setupResolutionFiles() error {
	return nil
}

func (sb *sandbox) restorePath() {
}

func (sb *sandbox) updateHostsFile(ifaceIP string) error {
	return nil
}

func (sb *sandbox) addHostsEntries(recs []etchosts.Record) {

}

func (sb *sandbox) deleteHostsEntries(recs []etchosts.Record) {

}

func (sb *sandbox) updateDNS(ipv6Enabled bool) error {
	return nil
}
