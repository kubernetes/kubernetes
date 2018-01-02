// Copyright 2014 The rkt Authors
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
	"runtime"

	"github.com/appc/spec/schema/types"
	"github.com/coreos/rkt/common/apps"
	"github.com/coreos/rkt/rkt/image"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/coreos/rkt/store/treestore"

	"github.com/spf13/cobra"
)

const (
	defaultOS   = runtime.GOOS
	defaultArch = runtime.GOARCH
)

var (
	cmdFetch = &cobra.Command{
		Use:   "fetch IMAGE_URL...",
		Short: "Fetch image(s) and store them in the local store",
		Long: `Locates and downloads remote ACIs and their attached signatures.

If the ACI is available in the local store, the image will not be fetched
again.`,
		Run: runWrapper(runFetch),
	}
	flagFullHash bool
	// We can't have different defaults for a given flag variable shared across
	// subcommands, so we can't use pullPolicyUpdate here
	flagPullPolicyDefaultUpdate string
)

func init() {
	// Disable interspersed flags to stop parsing after the first non flag
	// argument. All the subsequent parsing will be done by parseApps.
	// This is needed to correctly handle multiple IMAGE --signature=sigfile options
	cmdFetch.Flags().SetInterspersed(false)

	cmdFetch.Flags().Var((*appAsc)(&rktApps), "signature", "local signature file to use in validating the preceding image")
	cmdFetch.Flags().BoolVar(&flagStoreOnly, "store-only", false, "use only available images in the store (do not discover or download from remote URLs)")
	cmdFetch.Flags().MarkDeprecated("store-only", "please use --pull-policy=never")
	cmdFetch.Flags().BoolVar(&flagNoStore, "no-store", false, "fetch images ignoring the local store")
	cmdFetch.Flags().MarkDeprecated("no-store", "please use --pull-policy=update")
	cmdFetch.Flags().BoolVar(&flagFullHash, "full", false, "print the full image hash after fetching")
	cmdFetch.Flags().StringVar(&flagPullPolicyDefaultUpdate, "pull-policy", image.PullPolicyUpdate, "when to pull an image")

	cmdRkt.AddCommand(cmdFetch)

	// Hide image fetch option in command list
	cmdImageFetch := *cmdFetch
	cmdImageFetch.Hidden = true
	cmdImage.AddCommand(&cmdImageFetch)
}

func runFetch(cmd *cobra.Command, args []string) (exit int) {
	if err := parseApps(&rktApps, args, cmd.Flags(), false); err != nil {
		stderr.PrintE("unable to parse arguments", err)
		return 254
	}

	if rktApps.Count() < 1 {
		stderr.Print("must provide at least one image")
		return 254
	}

	// flagPullPolicy defaults to new regardless of subcommand, so we use a
	// different variable for the flag on fetch and then set it here
	flagPullPolicy = flagPullPolicyDefaultUpdate

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

	ks := getKeystore()
	config, err := getConfig()
	if err != nil {
		stderr.PrintE("cannot get configuration", err)
		return 254
	}
	ft := &image.Fetcher{
		S:                  s,
		Ts:                 ts,
		Ks:                 ks,
		Headers:            config.AuthPerHost,
		DockerAuth:         config.DockerCredentialsPerRegistry,
		InsecureFlags:      globalFlags.InsecureFlags,
		Debug:              globalFlags.Debug,
		TrustKeysFromHTTPS: globalFlags.TrustKeysFromHTTPS,

		PullPolicy: flagPullPolicy,
		WithDeps:   true,
	}

	err = ft.FetchImages(&rktApps)
	if err != nil {
		stderr.Error(err)
		return 254
	}
	err = rktApps.Walk(func(app *apps.App) error {
		hash := app.ImageID.String()
		if !flagFullHash {
			hash = types.ShortHash(hash)
		}
		stdout.Print(hash)
		return nil
	})
	if err != nil {
		stderr.Error(err)
		return 254
	}

	return
}
