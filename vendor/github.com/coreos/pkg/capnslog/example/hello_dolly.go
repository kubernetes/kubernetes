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

package main

import (
	"flag"
	oldlog "log"

	"github.com/coreos/pkg/capnslog"
)

var logLevel = capnslog.INFO
var log = capnslog.NewPackageLogger("github.com/coreos/pkg/capnslog/cmd", "main")
var dlog = capnslog.NewPackageLogger("github.com/coreos/pkg/capnslog/cmd", "dolly")

func init() {
	flag.Var(&logLevel, "log-level", "Global log level.")
}

func main() {
	rl := capnslog.MustRepoLogger("github.com/coreos/pkg/capnslog/cmd")

	// We can parse the log level configs from the command line
	flag.Parse()
	if flag.NArg() > 1 {
		cfg, err := rl.ParseLogLevelConfig(flag.Arg(1))
		if err != nil {
			log.Fatal(err)
		}
		rl.SetLogLevel(cfg)
		log.Infof("Setting output to %s", flag.Arg(1))
	}

	// Send some messages at different levels to the different packages
	dlog.Infof("Hello Dolly")
	dlog.Warningf("Well hello, Dolly")
	log.Errorf("It's so nice to have you back where you belong")
	dlog.Debugf("You're looking swell, Dolly")
	dlog.Tracef("I can tell, Dolly")

	// We also have control over the built-in "log" package.
	capnslog.SetGlobalLogLevel(logLevel)
	oldlog.Println("You're still glowin', you're still crowin', you're still lookin' strong")
	log.Fatalf("Dolly'll never go away again")
}
