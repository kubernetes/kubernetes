// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package flags implements command-line flag parsing.
package flags

import (
	"flag"
	"fmt"
	"net/url"
	"os"
	"strings"

	"github.com/coreos/etcd/pkg/transport"
	"github.com/coreos/pkg/capnslog"
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd/pkg", "flags")
)

// DeprecatedFlag encapsulates a flag that may have been previously valid but
// is now deprecated. If a DeprecatedFlag is set, an error occurs.
type DeprecatedFlag struct {
	Name string
}

func (f *DeprecatedFlag) Set(_ string) error {
	return fmt.Errorf(`flag "-%s" is no longer supported.`, f.Name)
}

func (f *DeprecatedFlag) String() string {
	return ""
}

// IgnoredFlag encapsulates a flag that may have been previously valid but is
// now ignored. If an IgnoredFlag is set, a warning is printed and
// operation continues.
type IgnoredFlag struct {
	Name string
}

// IsBoolFlag is defined to allow the flag to be defined without an argument
func (f *IgnoredFlag) IsBoolFlag() bool {
	return true
}

func (f *IgnoredFlag) Set(s string) error {
	plog.Warningf(`flag "-%s" is no longer supported - ignoring.`, f.Name)
	return nil
}

func (f *IgnoredFlag) String() string {
	return ""
}

// SetFlagsFromEnv parses all registered flags in the given flagset,
// and if they are not already set it attempts to set their values from
// environment variables. Environment variables take the name of the flag but
// are UPPERCASE, have the prefix "ETCD_", and any dashes are replaced by
// underscores - for example: some-flag => ETCD_SOME_FLAG
func SetFlagsFromEnv(fs *flag.FlagSet) error {
	var err error
	alreadySet := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) {
		alreadySet[flagToEnv(f.Name)] = true
	})
	usedEnvKey := make(map[string]bool)
	fs.VisitAll(func(f *flag.Flag) {
		key := flagToEnv(f.Name)
		if !alreadySet[key] {
			val := os.Getenv(key)
			if val != "" {
				usedEnvKey[key] = true
				if serr := fs.Set(f.Name, val); serr != nil {
					err = fmt.Errorf("invalid value %q for %s: %v", val, key, serr)
				}
				plog.Infof("recognized and used environment variable %s=%s", key, val)
			}
		}
	})

	for _, env := range os.Environ() {
		kv := strings.SplitN(env, "=", 2)
		if len(kv) != 2 {
			plog.Warningf("found invalid env %s", env)
		}
		if usedEnvKey[kv[0]] {
			continue
		}
		if alreadySet[kv[0]] {
			plog.Infof("recognized environment variable %s, but unused: shadowed by corresponding flag ", kv[0])
			continue
		}
		if strings.HasPrefix(env, "ETCD_") {
			plog.Warningf("unrecognized environment variable %s", env)
		}
	}

	return err
}

func flagToEnv(name string) string {
	return "ETCD_" + strings.ToUpper(strings.Replace(name, "-", "_", -1))
}

// SetBindAddrFromAddr sets the value of bindAddr flag from the value
// of addr flag. Both flags' Value must be of type IPAddressPort. If the
// bindAddr flag is set and the addr flag is unset, it will set bindAddr to
// [::]:port of addr. Otherwise, it keeps the original values.
func SetBindAddrFromAddr(fs *flag.FlagSet, bindAddrFlagName, addrFlagName string) {
	if IsSet(fs, bindAddrFlagName) || !IsSet(fs, addrFlagName) {
		return
	}
	addr := *fs.Lookup(addrFlagName).Value.(*IPAddressPort)
	addr.IP = "::"
	if err := fs.Set(bindAddrFlagName, addr.String()); err != nil {
		plog.Panicf("unexpected flags set error: %v", err)
	}
}

// URLsFromFlags decides what URLs should be using two different flags
// as datasources. The first flag's Value must be of type URLs, while
// the second must be of type IPAddressPort. If both of these flags
// are set, an error will be returned. If only the first flag is set,
// the underlying url.URL objects will be returned unmodified. If the
// second flag happens to be set, the underlying IPAddressPort will be
// converted to a url.URL and returned. The Scheme of the returned
// url.URL will be http unless the provided TLSInfo object is non-empty.
// If neither of the flags have been explicitly set, the default value
// of the first flag will be returned unmodified.
func URLsFromFlags(fs *flag.FlagSet, urlsFlagName string, addrFlagName string, tlsInfo transport.TLSInfo) ([]url.URL, error) {
	visited := make(map[string]struct{})
	fs.Visit(func(f *flag.Flag) {
		visited[f.Name] = struct{}{}
	})

	_, urlsFlagIsSet := visited[urlsFlagName]
	_, addrFlagIsSet := visited[addrFlagName]

	if addrFlagIsSet {
		if urlsFlagIsSet {
			return nil, fmt.Errorf("Set only one of flags -%s and -%s", urlsFlagName, addrFlagName)
		}

		addr := *fs.Lookup(addrFlagName).Value.(*IPAddressPort)
		addrURL := url.URL{Scheme: "http", Host: addr.String()}
		if !tlsInfo.Empty() {
			addrURL.Scheme = "https"
		}
		return []url.URL{addrURL}, nil
	}

	return []url.URL(*fs.Lookup(urlsFlagName).Value.(*URLsValue)), nil
}

func IsSet(fs *flag.FlagSet, name string) bool {
	set := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			set = true
		}
	})
	return set
}
