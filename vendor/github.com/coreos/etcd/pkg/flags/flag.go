// Copyright 2015 The etcd Authors
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

	"github.com/coreos/pkg/capnslog"
	"github.com/spf13/pflag"
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "pkg/flags")
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
// are UPPERCASE, have the given prefix  and any dashes are replaced by
// underscores - for example: some-flag => ETCD_SOME_FLAG
func SetFlagsFromEnv(prefix string, fs *flag.FlagSet) error {
	var err error
	alreadySet := make(map[string]bool)
	fs.Visit(func(f *flag.Flag) {
		alreadySet[FlagToEnv(prefix, f.Name)] = true
	})
	usedEnvKey := make(map[string]bool)
	fs.VisitAll(func(f *flag.Flag) {
		err = setFlagFromEnv(fs, prefix, f.Name, usedEnvKey, alreadySet, true)
	})

	verifyEnv(prefix, usedEnvKey, alreadySet)

	return err
}

// SetPflagsFromEnv is similar to SetFlagsFromEnv. However, the accepted flagset type is pflag.FlagSet
// and it does not do any logging.
func SetPflagsFromEnv(prefix string, fs *pflag.FlagSet) error {
	var err error
	alreadySet := make(map[string]bool)
	usedEnvKey := make(map[string]bool)
	fs.VisitAll(func(f *pflag.Flag) {
		if f.Changed {
			alreadySet[FlagToEnv(prefix, f.Name)] = true
		}
		if serr := setFlagFromEnv(fs, prefix, f.Name, usedEnvKey, alreadySet, false); serr != nil {
			err = serr
		}
	})
	return err
}

// FlagToEnv converts flag string to upper-case environment variable key string.
func FlagToEnv(prefix, name string) string {
	return prefix + "_" + strings.ToUpper(strings.Replace(name, "-", "_", -1))
}

func verifyEnv(prefix string, usedEnvKey, alreadySet map[string]bool) {
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
		if strings.HasPrefix(env, prefix+"_") {
			plog.Warningf("unrecognized environment variable %s", env)
		}
	}
}

type flagSetter interface {
	Set(fk string, fv string) error
}

func setFlagFromEnv(fs flagSetter, prefix, fname string, usedEnvKey, alreadySet map[string]bool, log bool) error {
	key := FlagToEnv(prefix, fname)
	if !alreadySet[key] {
		val := os.Getenv(key)
		if val != "" {
			usedEnvKey[key] = true
			if serr := fs.Set(fname, val); serr != nil {
				return fmt.Errorf("invalid value %q for %s: %v", val, key, serr)
			}
			if log {
				plog.Infof("recognized and used environment variable %s=%s", key, val)
			}
		}
	}
	return nil
}

// URLsFromFlag returns a slices from url got from the flag.
func URLsFromFlag(fs *flag.FlagSet, urlsFlagName string) []url.URL {
	return []url.URL(*fs.Lookup(urlsFlagName).Value.(*URLsValue))
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
