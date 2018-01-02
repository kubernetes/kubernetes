// +build !windows

package libnetwork

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/docker/libnetwork/etchosts"
	"github.com/docker/libnetwork/resolvconf"
	"github.com/docker/libnetwork/resolvconf/dns"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

const (
	defaultPrefix = "/var/lib/docker/network/files"
	dirPerm       = 0755
	filePerm      = 0644
)

func (sb *sandbox) startResolver(restore bool) {
	sb.resolverOnce.Do(func() {
		var err error
		sb.resolver = NewResolver(resolverIPSandbox, true, sb.Key(), sb)
		defer func() {
			if err != nil {
				sb.resolver = nil
			}
		}()

		// In the case of live restore container is already running with
		// right resolv.conf contents created before. Just update the
		// external DNS servers from the restored sandbox for embedded
		// server to use.
		if !restore {
			err = sb.rebuildDNS()
			if err != nil {
				logrus.Errorf("Updating resolv.conf failed for container %s, %q", sb.ContainerID(), err)
				return
			}
		}
		sb.resolver.SetExtServers(sb.extDNS)

		if err = sb.osSbox.InvokeFunc(sb.resolver.SetupFunc(0)); err != nil {
			logrus.Errorf("Resolver Setup function failed for container %s, %q", sb.ContainerID(), err)
			return
		}

		if err = sb.resolver.Start(); err != nil {
			logrus.Errorf("Resolver Start failed for container %s, %q", sb.ContainerID(), err)
		}
	})
}

func (sb *sandbox) setupResolutionFiles() error {
	if err := sb.buildHostsFile(); err != nil {
		return err
	}

	if err := sb.updateParentHosts(); err != nil {
		return err
	}

	if err := sb.setupDNS(); err != nil {
		return err
	}

	return nil
}

func (sb *sandbox) buildHostsFile() error {
	if sb.config.hostsPath == "" {
		sb.config.hostsPath = defaultPrefix + "/" + sb.id + "/hosts"
	}

	dir, _ := filepath.Split(sb.config.hostsPath)
	if err := createBasePath(dir); err != nil {
		return err
	}

	// This is for the host mode networking
	if sb.config.originHostsPath != "" {
		if err := copyFile(sb.config.originHostsPath, sb.config.hostsPath); err != nil && !os.IsNotExist(err) {
			return types.InternalErrorf("could not copy source hosts file %s to %s: %v", sb.config.originHostsPath, sb.config.hostsPath, err)
		}
		return nil
	}

	extraContent := make([]etchosts.Record, 0, len(sb.config.extraHosts))
	for _, extraHost := range sb.config.extraHosts {
		extraContent = append(extraContent, etchosts.Record{Hosts: extraHost.name, IP: extraHost.IP})
	}

	return etchosts.Build(sb.config.hostsPath, "", sb.config.hostName, sb.config.domainName, extraContent)
}

func (sb *sandbox) updateHostsFile(ifaceIP string) error {
	if ifaceIP == "" {
		return nil
	}

	if sb.config.originHostsPath != "" {
		return nil
	}

	// User might have provided a FQDN in hostname or split it across hostname
	// and domainname.  We want the FQDN and the bare hostname.
	fqdn := sb.config.hostName
	mhost := sb.config.hostName
	if sb.config.domainName != "" {
		fqdn = fmt.Sprintf("%s.%s", fqdn, sb.config.domainName)
	}

	parts := strings.SplitN(fqdn, ".", 2)
	if len(parts) == 2 {
		mhost = fmt.Sprintf("%s %s", fqdn, parts[0])
	}

	extraContent := []etchosts.Record{{Hosts: mhost, IP: ifaceIP}}

	sb.addHostsEntries(extraContent)
	return nil
}

func (sb *sandbox) addHostsEntries(recs []etchosts.Record) {
	if err := etchosts.Add(sb.config.hostsPath, recs); err != nil {
		logrus.Warnf("Failed adding service host entries to the running container: %v", err)
	}
}

func (sb *sandbox) deleteHostsEntries(recs []etchosts.Record) {
	if err := etchosts.Delete(sb.config.hostsPath, recs); err != nil {
		logrus.Warnf("Failed deleting service host entries to the running container: %v", err)
	}
}

func (sb *sandbox) updateParentHosts() error {
	var pSb Sandbox

	for _, update := range sb.config.parentUpdates {
		sb.controller.WalkSandboxes(SandboxContainerWalker(&pSb, update.cid))
		if pSb == nil {
			continue
		}
		if err := etchosts.Update(pSb.(*sandbox).config.hostsPath, update.ip, update.name); err != nil {
			return err
		}
	}

	return nil
}

func (sb *sandbox) restorePath() {
	if sb.config.resolvConfPath == "" {
		sb.config.resolvConfPath = defaultPrefix + "/" + sb.id + "/resolv.conf"
	}
	sb.config.resolvConfHashFile = sb.config.resolvConfPath + ".hash"
	if sb.config.hostsPath == "" {
		sb.config.hostsPath = defaultPrefix + "/" + sb.id + "/hosts"
	}
}

func (sb *sandbox) setExternalResolvers(content []byte, addrType int, checkLoopback bool) {
	servers := resolvconf.GetNameservers(content, addrType)
	for _, ip := range servers {
		hostLoopback := false
		if checkLoopback {
			hostLoopback = dns.IsIPv4Localhost(ip)
		}
		sb.extDNS = append(sb.extDNS, extDNSEntry{
			IPStr:        ip,
			HostLoopback: hostLoopback,
		})
	}
}

func (sb *sandbox) setupDNS() error {
	var newRC *resolvconf.File

	if sb.config.resolvConfPath == "" {
		sb.config.resolvConfPath = defaultPrefix + "/" + sb.id + "/resolv.conf"
	}

	sb.config.resolvConfHashFile = sb.config.resolvConfPath + ".hash"

	dir, _ := filepath.Split(sb.config.resolvConfPath)
	if err := createBasePath(dir); err != nil {
		return err
	}

	// This is for the host mode networking
	if sb.config.originResolvConfPath != "" {
		if err := copyFile(sb.config.originResolvConfPath, sb.config.resolvConfPath); err != nil {
			if !os.IsNotExist(err) {
				return fmt.Errorf("could not copy source resolv.conf file %s to %s: %v", sb.config.originResolvConfPath, sb.config.resolvConfPath, err)
			}
			logrus.Infof("%s does not exist, we create an empty resolv.conf for container", sb.config.originResolvConfPath)
			if err := createFile(sb.config.resolvConfPath); err != nil {
				return err
			}
		}
		return nil
	}

	currRC, err := resolvconf.Get()
	if err != nil {
		if !os.IsNotExist(err) {
			return err
		}
		// it's ok to continue if /etc/resolv.conf doesn't exist, default resolvers (Google's Public DNS)
		// will be used
		currRC = &resolvconf.File{}
		logrus.Infof("/etc/resolv.conf does not exist")
	}

	if len(sb.config.dnsList) > 0 || len(sb.config.dnsSearchList) > 0 || len(sb.config.dnsOptionsList) > 0 {
		var (
			err            error
			dnsList        = resolvconf.GetNameservers(currRC.Content, types.IP)
			dnsSearchList  = resolvconf.GetSearchDomains(currRC.Content)
			dnsOptionsList = resolvconf.GetOptions(currRC.Content)
		)
		if len(sb.config.dnsList) > 0 {
			dnsList = sb.config.dnsList
		}
		if len(sb.config.dnsSearchList) > 0 {
			dnsSearchList = sb.config.dnsSearchList
		}
		if len(sb.config.dnsOptionsList) > 0 {
			dnsOptionsList = sb.config.dnsOptionsList
		}
		newRC, err = resolvconf.Build(sb.config.resolvConfPath, dnsList, dnsSearchList, dnsOptionsList)
		if err != nil {
			return err
		}
		// After building the resolv.conf from the user config save the
		// external resolvers in the sandbox. Note that --dns 127.0.0.x
		// config refers to the loopback in the container namespace
		sb.setExternalResolvers(newRC.Content, types.IPv4, false)
	} else {
		// If the host resolv.conf file has 127.0.0.x container should
		// use the host restolver for queries. This is supported by the
		// docker embedded DNS server. Hence save the external resolvers
		// before filtering it out.
		sb.setExternalResolvers(currRC.Content, types.IPv4, true)

		// Replace any localhost/127.* (at this point we have no info about ipv6, pass it as true)
		if newRC, err = resolvconf.FilterResolvDNS(currRC.Content, true); err != nil {
			return err
		}
		// No contention on container resolv.conf file at sandbox creation
		if err := ioutil.WriteFile(sb.config.resolvConfPath, newRC.Content, filePerm); err != nil {
			return types.InternalErrorf("failed to write unhaltered resolv.conf file content when setting up dns for sandbox %s: %v", sb.ID(), err)
		}
	}

	// Write hash
	if err := ioutil.WriteFile(sb.config.resolvConfHashFile, []byte(newRC.Hash), filePerm); err != nil {
		return types.InternalErrorf("failed to write resolv.conf hash file when setting up dns for sandbox %s: %v", sb.ID(), err)
	}

	return nil
}

func (sb *sandbox) updateDNS(ipv6Enabled bool) error {
	var (
		currHash string
		hashFile = sb.config.resolvConfHashFile
	)

	// This is for the host mode networking
	if sb.config.originResolvConfPath != "" {
		return nil
	}

	if len(sb.config.dnsList) > 0 || len(sb.config.dnsSearchList) > 0 || len(sb.config.dnsOptionsList) > 0 {
		return nil
	}

	currRC, err := resolvconf.GetSpecific(sb.config.resolvConfPath)
	if err != nil {
		if !os.IsNotExist(err) {
			return err
		}
	} else {
		h, err := ioutil.ReadFile(hashFile)
		if err != nil {
			if !os.IsNotExist(err) {
				return err
			}
		} else {
			currHash = string(h)
		}
	}

	if currHash != "" && currHash != currRC.Hash {
		// Seems the user has changed the container resolv.conf since the last time
		// we checked so return without doing anything.
		//logrus.Infof("Skipping update of resolv.conf file with ipv6Enabled: %t because file was touched by user", ipv6Enabled)
		return nil
	}

	// replace any localhost/127.* and remove IPv6 nameservers if IPv6 disabled.
	newRC, err := resolvconf.FilterResolvDNS(currRC.Content, ipv6Enabled)
	if err != nil {
		return err
	}
	err = ioutil.WriteFile(sb.config.resolvConfPath, newRC.Content, 0644)
	if err != nil {
		return err
	}

	// write the new hash in a temp file and rename it to make the update atomic
	dir := path.Dir(sb.config.resolvConfPath)
	tmpHashFile, err := ioutil.TempFile(dir, "hash")
	if err != nil {
		return err
	}
	if err = tmpHashFile.Chmod(filePerm); err != nil {
		tmpHashFile.Close()
		return err
	}
	_, err = tmpHashFile.Write([]byte(newRC.Hash))
	if err1 := tmpHashFile.Close(); err == nil {
		err = err1
	}
	if err != nil {
		return err
	}
	return os.Rename(tmpHashFile.Name(), hashFile)
}

// Embedded DNS server has to be enabled for this sandbox. Rebuild the container's
// resolv.conf by doing the following
// - Add only the embedded server's IP to container's resolv.conf
// - If the embedded server needs any resolv.conf options add it to the current list
func (sb *sandbox) rebuildDNS() error {
	currRC, err := resolvconf.GetSpecific(sb.config.resolvConfPath)
	if err != nil {
		return err
	}

	if len(sb.extDNS) == 0 {
		sb.setExternalResolvers(currRC.Content, types.IPv4, false)
	}
	var (
		dnsList        = []string{sb.resolver.NameServer()}
		dnsOptionsList = resolvconf.GetOptions(currRC.Content)
		dnsSearchList  = resolvconf.GetSearchDomains(currRC.Content)
	)

	// external v6 DNS servers has to be listed in resolv.conf
	dnsList = append(dnsList, resolvconf.GetNameservers(currRC.Content, types.IPv6)...)

	// If the user config and embedded DNS server both have ndots option set,
	// remember the user's config so that unqualified names not in the docker
	// domain can be dropped.
	resOptions := sb.resolver.ResolverOptions()

dnsOpt:
	for _, resOpt := range resOptions {
		if strings.Contains(resOpt, "ndots") {
			for _, option := range dnsOptionsList {
				if strings.Contains(option, "ndots") {
					parts := strings.Split(option, ":")
					if len(parts) != 2 {
						return fmt.Errorf("invalid ndots option %v", option)
					}
					if num, err := strconv.Atoi(parts[1]); err != nil {
						return fmt.Errorf("invalid number for ndots option %v", option)
					} else if num > 0 {
						sb.ndotsSet = true
						break dnsOpt
					}
				}
			}
		}
	}

	dnsOptionsList = append(dnsOptionsList, resOptions...)

	_, err = resolvconf.Build(sb.config.resolvConfPath, dnsList, dnsSearchList, dnsOptionsList)
	return err
}

func createBasePath(dir string) error {
	return os.MkdirAll(dir, dirPerm)
}

func createFile(path string) error {
	var f *os.File

	dir, _ := filepath.Split(path)
	err := createBasePath(dir)
	if err != nil {
		return err
	}

	f, err = os.Create(path)
	if err == nil {
		f.Close()
	}

	return err
}

func copyFile(src, dst string) error {
	sBytes, err := ioutil.ReadFile(src)
	if err != nil {
		return err
	}
	return ioutil.WriteFile(dst, sBytes, filePerm)
}
