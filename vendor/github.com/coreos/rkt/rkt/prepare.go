// Copyright 2015 The rkt Authors
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

//+build linux

package main

import (
	"os"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/lock"
	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/coreos/rkt/pkg/user"
	"github.com/coreos/rkt/rkt/image"
	"github.com/coreos/rkt/stage0"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/coreos/rkt/store/treestore"
	"github.com/spf13/cobra"
)

var (
	cmdPrepare = &cobra.Command{
		Use:   "prepare [--volume=name,kind=host,...] [--mount volume=VOL,target=PATH] [--quiet] IMAGE [-- image-args...[---]]...",
		Short: "Prepare to run image(s) in a pod in rkt",
		Long: `Prepares the pod and returns its UUID for a subsequent call to run-prepared.

Prepare is useful to get the UUID of the pod before running or when preparing
takes a long time.

IMAGE should be a string referencing an image; either a hash, local file on
disk, or URL. They will be checked in that order and the first match will be
used.

An "--" may be used to inhibit rkt prepare's parsing of subsequent arguments,
which will instead be appended to the preceding image app's exec arguments. End
the image arguments with a lone "---" to resume argument parsing.`,
		Run: ensureSuperuser(runWrapper(runPrepare)),
	}
	flagQuiet bool
)

func init() {
	cmdRkt.AddCommand(cmdPrepare)

	addStage1ImageFlags(cmdPrepare.Flags())
	cmdPrepare.Flags().Var(&flagPorts, "port", "ports to expose on the host (requires contained network). Syntax: --port=NAME:HOSTPORT")
	cmdPrepare.Flags().BoolVar(&flagQuiet, "quiet", false, "suppress superfluous output on stdout, print only the UUID on success")
	cmdPrepare.Flags().BoolVar(&flagInheritEnv, "inherit-env", false, "inherit all environment variables not set by apps")
	cmdPrepare.Flags().BoolVar(&flagNoOverlay, "no-overlay", false, "disable overlay filesystem")
	cmdPrepare.Flags().BoolVar(&flagPrivateUsers, "private-users", false, "run within user namespaces.")
	cmdPrepare.Flags().Var(&flagExplicitEnv, "set-env", "environment variable to set for all the apps in the form key=value, this will be overriden by --environment")
	cmdPrepare.Flags().Var(&flagEnvFromFile, "set-env-file", "the path to an environment variables file")
	cmdPrepare.Flags().BoolVar(&flagStoreOnly, "store-only", false, "use only available images in the store (do not discover or download from remote URLs)")
	cmdPrepare.Flags().MarkDeprecated("store-only", "please use --pull-policy=never")
	cmdPrepare.Flags().BoolVar(&flagNoStore, "no-store", false, "fetch images ignoring the local store")
	cmdPrepare.Flags().MarkDeprecated("no-store", "please use --pull-policy=update")
	cmdPrepare.Flags().StringVar(&flagPullPolicy, "pull-policy", image.PullPolicyNew, "when to pull an image")
	cmdPrepare.Flags().StringVar(&flagPodManifest, "pod-manifest", "", "the path to the pod manifest. If it's non-empty, then only '--quiet' and '--no-overlay' will have effect")
	cmdPrepare.Flags().Var((*appsVolume)(&rktApps), "volume", "volumes to make available in the pod")

	// per-app flags
	cmdPrepare.Flags().Var((*appAsc)(&rktApps), "signature", "local signature file to use in validating the preceding image")
	addAppFlags(cmdPrepare)
	addIsolatorFlags(cmdPrepare, true)

	// Disable interspersed flags to stop parsing after the first non flag
	// argument. This is need to permit to correctly handle
	// multiple "IMAGE -- imageargs ---"  options
	cmdPrepare.Flags().SetInterspersed(false)
}

func runPrepare(cmd *cobra.Command, args []string) (exit int) {
	var err error
	origStdout := os.Stdout
	privateUsers := user.NewBlankUidRange()
	if flagQuiet {
		if os.Stdout, err = os.Open("/dev/null"); err != nil {
			stderr.PrintE("unable to open /dev/null", err)
			return 254
		}
	}

	if flagStoreOnly && flagNoStore {
		stderr.Print("both --store-only and --no-store specified")
		return 254
	}
	if flagStoreOnly {
		flagPullPolicy = image.PullPolicyNever
	}
	if flagNoStore {
		flagPullPolicy = image.PullPolicyUpdate
	}

	if flagPrivateUsers {
		if !common.SupportsUserNS() {
			stderr.Print("--private-users is not supported, kernel compiled without user namespace support")
			return 254
		}
		privateUsers.SetRandomUidRange(user.DefaultRangeCount)
	}

	if err = parseApps(&rktApps, args, cmd.Flags(), true); err != nil {
		stderr.PrintE("error parsing app image arguments", err)
		return 254
	}

	if len(flagPodManifest) > 0 && (rktApps.Count() > 0 ||
		(*appsVolume)(&rktApps).String() != "" || (*appMount)(&rktApps).String() != "" ||
		len(flagPorts) > 0 || flagPullPolicy == image.PullPolicyNever ||
		flagPullPolicy == image.PullPolicyUpdate || flagInheritEnv ||
		!flagExplicitEnv.IsEmpty() || !flagEnvFromFile.IsEmpty()) {
		stderr.Print("conflicting flags set with --pod-manifest (see --help)")
		return 254
	}

	if rktApps.Count() < 1 && len(flagPodManifest) == 0 {
		stderr.Print("must provide at least one image or specify the pod manifest")
		return 254
	}

	s, err := imagestore.NewStore(storeDir())
	if err != nil {
		stderr.PrintE("cannot open store", err)
		return 254
	}

	ts, err := treestore.NewStore(treeStoreDir(), s)
	if err != nil {
		stderr.PrintE("cannot open treestore", err)
		return 254
	}

	config, err := getConfig()
	if err != nil {
		stderr.PrintE("cannot get configuration", err)
		return 254
	}

	s1img, err := getStage1Hash(s, ts, config)
	if err != nil {
		stderr.Error(err)
		return 254
	}

	fn := &image.Finder{
		S:                  s,
		Ts:                 ts,
		Ks:                 getKeystore(),
		Headers:            config.AuthPerHost,
		DockerAuth:         config.DockerCredentialsPerRegistry,
		InsecureFlags:      globalFlags.InsecureFlags,
		Debug:              globalFlags.Debug,
		TrustKeysFromHTTPS: globalFlags.TrustKeysFromHTTPS,

		PullPolicy: flagPullPolicy,
		WithDeps:   true,
	}
	if err := fn.FindImages(&rktApps); err != nil {
		stderr.PrintE("error finding images", err)
		return 254
	}

	p, err := pkgPod.NewPod(getDataDir())
	if err != nil {
		stderr.PrintE("error creating new pod", err)
		return 254
	}

	cfg := stage0.CommonConfig{
		DataDir:     getDataDir(),
		Store:       s,
		TreeStore:   ts,
		Stage1Image: *s1img,
		UUID:        p.UUID,
		Debug:       globalFlags.Debug,
	}

	ovlOk := true
	if err := common.PathSupportsOverlay(getDataDir()); err != nil {
		if oerr, ok := err.(common.ErrOverlayUnsupported); ok {
			stderr.Printf("disabling overlay support: %q", oerr.Error())
			ovlOk = false
		} else {
			stderr.PrintE("error determining overlay support", err)
			return 254
		}
	}

	pcfg := stage0.PrepareConfig{
		CommonConfig:       &cfg,
		UseOverlay:         !flagNoOverlay && ovlOk,
		PrivateUsers:       privateUsers,
		SkipTreeStoreCheck: globalFlags.InsecureFlags.SkipOnDiskCheck(),
	}

	if len(flagPodManifest) > 0 {
		pcfg.PodManifest = flagPodManifest
	} else {
		pcfg.Ports = []types.ExposedPort(flagPorts)
		pcfg.InheritEnv = flagInheritEnv
		pcfg.ExplicitEnv = flagExplicitEnv.Strings()
		pcfg.EnvFromFile = flagEnvFromFile.Strings()
		pcfg.Apps = &rktApps
	}

	if globalFlags.Debug {
		stage0.InitDebug()
	}

	keyLock, err := lock.SharedKeyLock(lockDir(), common.PrepareLock)
	if err != nil {
		stderr.PrintE("cannot get shared prepare lock", err)
		return 254
	}
	if err = stage0.Prepare(pcfg, p.Path(), p.UUID); err != nil {
		stderr.PrintE("error setting up stage0", err)
		keyLock.Close()
		return 254
	}
	keyLock.Close()

	if err := p.Sync(); err != nil {
		stderr.PrintE("error syncing pod data", err)
		return 254
	}

	if err := p.ToPrepared(); err != nil {
		stderr.PrintE("error transitioning to prepared", err)
		return 254
	}

	os.Stdout = origStdout // restore output in case of --quiet
	stdout.Printf("%s", p.UUID.String())

	return 0
}
