package migration

// This package provides a set of functions to help handle migrations of CoreDNS Corefiles to be compatible with new
// versions of CoreDNS. The task of upgrading CoreDNS is the responsibility of a variety of Kubernetes management tools
// (e.g. kubeadm and others), and the precise behavior may be different for each one. This library abstracts some basic
// helper functions that make this easier to implement.

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/coredns/corefile-migration/migration/corefile"
)

// Deprecated returns a list of deprecation notifications affecting the given Corefile.  Notifications are returned for
// any deprecated, removed, or ignored plugins/directives present in the Corefile.  Notifications are also returned for
// any new default plugins that would be added in a migration.
func Deprecated(fromCoreDNSVersion, toCoreDNSVersion, corefileStr string) ([]Notice, error) {
	if fromCoreDNSVersion == toCoreDNSVersion {
		return nil, nil
	}
	return getStatus(fromCoreDNSVersion, toCoreDNSVersion, corefileStr, SevAll)
}

// Unsupported returns a list notifications of plugins/options that are not handled supported by this migration tool,
// but may still be valid in CoreDNS.
func Unsupported(fromCoreDNSVersion, toCoreDNSVersion, corefileStr string) ([]Notice, error) {
	if fromCoreDNSVersion == toCoreDNSVersion {
		return nil, nil
	}
	return getStatus(fromCoreDNSVersion, toCoreDNSVersion, corefileStr, SevUnsupported)
}

func getStatus(fromCoreDNSVersion, toCoreDNSVersion, corefileStr, status string) ([]Notice, error) {
	err := ValidUpMigration(fromCoreDNSVersion, toCoreDNSVersion)
	if err != nil {
		return nil, err
	}
	cf, err := corefile.New(corefileStr)
	if err != nil {
		return nil, err
	}
	notices := []Notice{}
	v := fromCoreDNSVersion
	for {
		if fromCoreDNSVersion != toCoreDNSVersion {
			v = Versions[v].nextVersion
		}
		for _, s := range cf.Servers {
			for _, p := range s.Plugins {
				vp, present := Versions[v].plugins[p.Name]
				if status == SevUnsupported && !present {
					notices = append(notices, Notice{Plugin: p.Name, Severity: status, Version: v})
					continue
				}
				if !present {
					continue
				}
				if vp.status != "" && vp.status != SevNewDefault && status != SevUnsupported {
					notices = append(notices, Notice{
						Plugin:     p.Name,
						Severity:   vp.status,
						Version:    v,
						ReplacedBy: vp.replacedBy,
						Additional: vp.additional,
					})
					continue
				}
				for _, o := range p.Options {
					vo, present := matchOption(o.Name, Versions[v].plugins[p.Name])
					if status == SevUnsupported {
						if present {
							continue
						}
						notices = append(notices, Notice{
							Plugin:   p.Name,
							Option:   o.Name,
							Severity: status,
							Version:  v,
						})
						continue
					}
					if !present {
						continue
					}
					if vo.status != "" && vo.status != SevNewDefault {
						notices = append(notices, Notice{Plugin: p.Name, Option: o.Name, Severity: vo.status, Version: v})
						continue
					}
				}
				if status != SevUnsupported {
				CheckForNewOptions:
					for name, vo := range Versions[v].plugins[p.Name].namedOptions {
						if vo.status != SevNewDefault {
							continue
						}
						for _, o := range p.Options {
							if name == o.Name {
								continue CheckForNewOptions
							}
						}
						notices = append(notices, Notice{Plugin: p.Name, Option: name, Severity: SevNewDefault, Version: v})
					}
				}
			}
			if status != SevUnsupported {
			CheckForNewPlugins:
				for name, vp := range Versions[v].plugins {
					if vp.status != SevNewDefault {
						continue
					}
					for _, p := range s.Plugins {
						if name == p.Name {
							continue CheckForNewPlugins
						}
					}
					notices = append(notices, Notice{Plugin: name, Option: "", Severity: SevNewDefault, Version: v})
				}
			}
		}
		if v == toCoreDNSVersion {
			break
		}
	}
	return notices, nil
}

// Migrate returns the Corefile converted to toCoreDNSVersion, or an error if it cannot.  This function only accepts
// a forward migration, where the destination version is => the start version.
// If deprecations is true, deprecated plugins/options will be migrated as soon as they are deprecated.
// If deprecations is false, deprecated plugins/options will be migrated only once they become removed or ignored.
func Migrate(fromCoreDNSVersion, toCoreDNSVersion, corefileStr string, deprecations bool) (string, error) {
	if fromCoreDNSVersion == toCoreDNSVersion {
		return corefileStr, nil
	}
	err := ValidUpMigration(fromCoreDNSVersion, toCoreDNSVersion)
	if err != nil {
		return "", err
	}
	cf, err := corefile.New(corefileStr)
	if err != nil {
		return "", err
	}
	v := fromCoreDNSVersion
	for {
		v = Versions[v].nextVersion

		// apply any global corefile level pre-processing
		if Versions[v].preProcess != nil {
			cf, err = Versions[v].preProcess(cf)
			if err != nil {
				return "", err
			}
		}

		newSrvs := []*corefile.Server{}
		for _, s := range cf.Servers {
			newPlugs := []*corefile.Plugin{}
			for _, p := range s.Plugins {
				vp, present := Versions[v].plugins[p.Name]
				if !present {
					newPlugs = append(newPlugs, p)
					continue
				}
				if !deprecations && vp.status == SevDeprecated {
					newPlugs = append(newPlugs, p)
					continue
				}
				newOpts := []*corefile.Option{}
				for _, o := range p.Options {
					vo, present := matchOption(o.Name, Versions[v].plugins[p.Name])
					if !present {
						newOpts = append(newOpts, o)
						continue
					}
					if !deprecations && vo.status == SevDeprecated {
						newOpts = append(newOpts, o)
						continue
					}
					if vo.action == nil {
						newOpts = append(newOpts, o)
						continue
					}
					o, err := vo.action(o)
					if err != nil {
						return "", err
					}
					if o == nil {
						// remove option
						continue
					}
					newOpts = append(newOpts, o)
				}
				if vp.action != nil {
					p, err := vp.action(p)
					if err != nil {
						return "", err
					}
					if p == nil {
						// remove plugin, skip options processing
						continue
					}
				}
				newPlug := &corefile.Plugin{
					Name:    p.Name,
					Args:    p.Args,
					Options: newOpts,
				}
			CheckForNewOptions:
				for name, vo := range Versions[v].plugins[p.Name].namedOptions {
					if vo.status != SevNewDefault {
						continue
					}
					for _, o := range p.Options {
						if name == o.Name {
							continue CheckForNewOptions
						}
					}
					newPlug, err = vo.add(newPlug)
					if err != nil {
						return "", err
					}
				}

				newPlugs = append(newPlugs, newPlug)
			}
			newSrv := &corefile.Server{
				DomPorts: s.DomPorts,
				Plugins:  newPlugs,
			}
		CheckForNewPlugins:
			for name, vp := range Versions[v].plugins {
				if vp.status != SevNewDefault {
					continue
				}
				for _, p := range s.Plugins {
					if name == p.Name {
						continue CheckForNewPlugins
					}
				}
				newSrv, err = vp.add(newSrv)
				if err != nil {
					return "", err
				}
			}

			newSrvs = append(newSrvs, newSrv)
		}

		cf = &corefile.Corefile{Servers: newSrvs}

		// apply any global corefile level post processing
		if Versions[v].postProcess != nil {
			cf, err = Versions[v].postProcess(cf)
			if err != nil {
				return "", err
			}
		}

		if v == toCoreDNSVersion {
			break
		}
	}
	return cf.ToString(), nil
}

// MigrateDown returns the Corefile converted to toCoreDNSVersion, or an error if it cannot. This function only accepts
// a downward migration, where the destination version is <= the start version.
func MigrateDown(fromCoreDNSVersion, toCoreDNSVersion, corefileStr string) (string, error) {
	if fromCoreDNSVersion == toCoreDNSVersion {
		return corefileStr, nil
	}
	err := validDownMigration(fromCoreDNSVersion, toCoreDNSVersion)
	if err != nil {
		return "", err
	}
	cf, err := corefile.New(corefileStr)
	if err != nil {
		return "", err
	}
	v := fromCoreDNSVersion
	for {
		newSrvs := []*corefile.Server{}
		for _, s := range cf.Servers {
			newPlugs := []*corefile.Plugin{}
			for _, p := range s.Plugins {
				vp, present := Versions[v].plugins[p.Name]
				if !present {
					newPlugs = append(newPlugs, p)
					continue
				}
				if vp.downAction == nil {
					newPlugs = append(newPlugs, p)
					continue
				}
				p, err := vp.downAction(p)
				if err != nil {
					return "", err
				}
				if p == nil {
					// remove plugin, skip options processing
					continue
				}

				newOpts := []*corefile.Option{}
				for _, o := range p.Options {
					vo, present := matchOption(o.Name, Versions[v].plugins[p.Name])
					if !present {
						newOpts = append(newOpts, o)
						continue
					}
					if vo.downAction == nil {
						newOpts = append(newOpts, o)
						continue
					}
					o, err := vo.downAction(o)
					if err != nil {
						return "", err
					}
					if o == nil {
						// remove option
						continue
					}
					newOpts = append(newOpts, o)
				}
				newPlug := &corefile.Plugin{
					Name:    p.Name,
					Args:    p.Args,
					Options: newOpts,
				}
				newPlugs = append(newPlugs, newPlug)
			}
			newSrv := &corefile.Server{
				DomPorts: s.DomPorts,
				Plugins:  newPlugs,
			}
			newSrvs = append(newSrvs, newSrv)
		}

		cf = &corefile.Corefile{Servers: newSrvs}

		if v == toCoreDNSVersion {
			break
		}
		v = Versions[v].priorVersion
	}
	return cf.ToString(), nil
}

// Default returns true if the Corefile is the default for a given version of Kubernetes.
// Or, if k8sVersion is empty, Default returns true if the Corefile is the default for any version of Kubernetes.
func Default(k8sVersion, corefileStr string) bool {
	cf, err := corefile.New(corefileStr)
	if err != nil {
		return false
	}
NextVersion:
	for _, v := range Versions {
		for _, release := range v.k8sReleases {
			if k8sVersion != "" && k8sVersion != release {
				continue
			}
		}
		defCf, err := corefile.New(v.defaultConf)
		if err != nil {
			continue
		}
		// check corefile against k8s release default
		if len(cf.Servers) != len(defCf.Servers) {
			continue NextVersion
		}
		for _, s := range cf.Servers {
			defS, found := s.FindMatch(defCf.Servers)
			if !found {
				continue NextVersion
			}
			if len(s.Plugins) != len(defS.Plugins) {
				continue NextVersion
			}
			for _, p := range s.Plugins {
				defP, found := p.FindMatch(defS.Plugins)
				if !found {
					continue NextVersion
				}
				if len(p.Options) != len(defP.Options) {
					continue NextVersion
				}
				for _, o := range p.Options {
					_, found := o.FindMatch(defP.Options)
					if !found {
						continue NextVersion
					}
				}
			}
		}
		return true
	}
	return false
}

// Released returns true if dockerImageSHA matches any released image of CoreDNS.
func Released(dockerImageSHA string) bool {
	for _, v := range Versions {
		if v.dockerImageSHA == dockerImageSHA {
			return true
		}
	}
	return false
}

// VersionFromSHA returns the version string matching the dockerImageSHA.
func VersionFromSHA(dockerImageSHA string) (string, error) {
	for vStr, v := range Versions {
		if v.dockerImageSHA == dockerImageSHA {
			return vStr, nil
		}
	}
	return "", errors.New("sha unsupported")
}

// ValidVersions returns a list of all versions defined
func ValidVersions() []string {
	var vStrs []string
	for vStr := range Versions {
		vStrs = append(vStrs, vStr)
	}
	sort.Slice(vStrs, func(i, j int) bool {
		iSegs := strings.Split(vStrs[i], ".")
		jSegs := strings.Split(vStrs[j], ".")
		for k, iSeg := range iSegs {
			if iSeg == jSegs[k] {
				continue
			}
			iInt, err := strconv.Atoi(iSeg)
			if err != nil {
				panic(err)
			}
			jInt, err := strconv.Atoi(jSegs[k])
			if err != nil {
				panic(err)
			}
			return iInt < jInt
		}
		return false
	})

	return vStrs
}

func ValidUpMigration(fromCoreDNSVersion, toCoreDNSVersion string) error {

	err := validateVersion(fromCoreDNSVersion)
	if err != nil {
		return err
	}
	if fromCoreDNSVersion == toCoreDNSVersion {
		return nil
	}
	for next := Versions[fromCoreDNSVersion].nextVersion; next != ""; next = Versions[next].nextVersion {
		if next != toCoreDNSVersion {
			continue
		}
		return nil
	}
	return fmt.Errorf("cannot migrate up to '%v' from '%v'", toCoreDNSVersion, fromCoreDNSVersion)
}

func validateVersion(fromCoreDNSVersion string) error {
	if _, ok := Versions[fromCoreDNSVersion]; !ok {
		return fmt.Errorf("start version '%v' not supported", fromCoreDNSVersion)
	}
	return nil
}

func validDownMigration(fromCoreDNSVersion, toCoreDNSVersion string) error {
	err := validateVersion(fromCoreDNSVersion)
	if err != nil {
		return err
	}
	for prior := Versions[fromCoreDNSVersion].priorVersion; prior != ""; prior = Versions[prior].priorVersion {
		if prior != toCoreDNSVersion {
			continue
		}
		return nil
	}
	return fmt.Errorf("cannot migrate down to '%v' from '%v'", toCoreDNSVersion, fromCoreDNSVersion)
}

func matchOption(oName string, p plugin) (*option, bool) {
	o, exists := p.namedOptions[oName]
	if exists {
		o.name = oName
		return &o, exists
	}
	for pattern, o := range p.patternOptions {
		matched, err := regexp.MatchString(pattern, oName)
		if err != nil {
			continue
		}
		if matched {
			o.name = oName
			return &o, true
		}
	}
	return nil, false
}
