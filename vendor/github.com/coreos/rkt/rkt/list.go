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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/lastditch"
	"github.com/appc/spec/schema/types"
	lib "github.com/coreos/rkt/lib"
	"github.com/coreos/rkt/networking/netinfo"
	pkgPod "github.com/coreos/rkt/pkg/pod"
	"github.com/dustin/go-humanize"
	"github.com/hashicorp/errwrap"
	"github.com/spf13/cobra"
)

var (
	cmdList = &cobra.Command{
		Use:   "list",
		Short: "List pods",
		Long:  `Displays a table of information about the pods.`,
		Run:   runWrapper(runList),
	}
	flagNoLegend   bool
	flagFullOutput bool
	flagFormat     outputFormat
)

func init() {
	cmdRkt.AddCommand(cmdList)
	cmdList.Flags().BoolVar(&flagNoLegend, "no-legend", false, "suppress a legend with the list")
	cmdList.Flags().BoolVar(&flagFullOutput, "full", false, "use long output format")
	cmdList.Flags().Var(&flagFormat, "format", "choose the output format, allowed format includes 'json', 'json-pretty'. If empty, then the result is printed as key value pairs")
}

func runList(cmd *cobra.Command, args []string) int {
	var errors []error
	tabBuffer := new(bytes.Buffer)
	tabOut := getTabOutWithWriter(tabBuffer)

	if !flagNoLegend && flagFormat == outputFormatTabbed {
		if flagFullOutput {
			fmt.Fprintf(tabOut, "UUID\tAPP\tIMAGE NAME\tIMAGE ID\tSTATE\tCREATED\tSTARTED\tNETWORKS\n")
		} else {
			fmt.Fprintf(tabOut, "UUID\tAPP\tIMAGE NAME\tSTATE\tCREATED\tSTARTED\tNETWORKS\n")
		}
	}

	var pods []*lib.Pod

	if err := pkgPod.WalkPods(getDataDir(), pkgPod.IncludeMostDirs, func(p *pkgPod.Pod) {
		if flagFormat != outputFormatTabbed {
			pod, err := lib.NewPodFromInternalPod(p)
			if err != nil {
				errors = append(errors, err)
			} else {
				pods = append(pods, pod)
			}
			return
		}

		var pm schema.PodManifest
		var err error

		if p.PodManifestAvailable() {
			// TODO(vc): we should really hold a shared lock here to prevent gc of the pod
			_, manifest, err := p.PodManifest()
			if err != nil {
				errors = append(errors, newPodListReadError(p, err))
				return
			}
			pm = *manifest
		}

		type printedApp struct {
			uuid    string
			appName string
			imgName string
			imgID   string
			state   string
			nets    string
			created string
			started string
		}

		var appsToPrint []printedApp
		uuid := p.UUID.String()
		state := p.State()
		nets := fmtNets(p.Nets)

		created, err := p.CreationTime()
		if err != nil {
			errors = append(errors, errwrap.Wrap(fmt.Errorf("unable to get creation time for pod %q", uuid), err))
		}
		var createdStr string
		if flagFullOutput {
			createdStr = created.Format(defaultTimeLayout)
		} else {
			createdStr = humanize.Time(created)
		}

		started, err := p.StartTime()
		if err != nil {
			errors = append(errors, errwrap.Wrap(fmt.Errorf("unable to get start time for pod %q", uuid), err))
		}
		var startedStr string
		if !started.IsZero() {
			if flagFullOutput {
				startedStr = started.Format(defaultTimeLayout)
			} else {
				startedStr = humanize.Time(started)
			}
		}

		if !flagFullOutput {
			uuid = uuid[:8]
		}
		if len(pm.Apps) == 0 {
			appsToPrint = append(appsToPrint, printedApp{
				uuid:    uuid,
				appName: "-",
				imgName: "-",
				imgID:   "-",
				state:   state,
				nets:    nets,
				created: createdStr,
				started: startedStr,
			})
		}
		for _, app := range pm.Apps {
			imageName, err := getImageName(p, app.Name)
			if err != nil {
				errors = append(errors, newPodListLoadImageManifestError(p, err))
				imageName = "--"
			}

			var imageID string
			if flagFullOutput {
				imageID = app.Image.ID.String()[:19]
			}

			appsToPrint = append(appsToPrint, printedApp{
				uuid:    uuid,
				appName: app.Name.String(),
				imgName: imageName,
				imgID:   imageID,
				state:   state,
				nets:    nets,
				created: createdStr,
				started: startedStr,
			})
			// clear those variables so they won't be
			// printed for another apps in the pod as they
			// are actually describing a pod, not an app
			uuid = ""
			state = ""
			nets = ""
			createdStr = ""
			startedStr = ""
		}
		// if we reached that point, then it means that the
		// pod and all its apps are valid, so they can be
		// printed
		for _, app := range appsToPrint {
			if flagFullOutput {
				fmt.Fprintf(tabOut, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", app.uuid, app.appName, app.imgName, app.imgID, app.state, app.created, app.started, app.nets)
			} else {
				fmt.Fprintf(tabOut, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", app.uuid, app.appName, app.imgName, app.state, app.created, app.started, app.nets)
			}
		}

	}); err != nil {
		stderr.PrintE("failed to get pod handles", err)
		return 254
	}

	switch flagFormat {
	case outputFormatTabbed:
		tabOut.Flush()
		stdout.Print(tabBuffer)
	case outputFormatJSON:
		result, err := json.Marshal(pods)
		if err != nil {
			stderr.PrintE("error marshaling the pods", err)
			return 254
		}
		stdout.Print(string(result))
	case outputFormatPrettyJSON:
		result, err := json.MarshalIndent(pods, "", "\t")
		if err != nil {
			stderr.PrintE("error marshaling the pods", err)
			return 254
		}
		stdout.Print(string(result))
	}

	if len(errors) > 0 {
		printErrors(errors, "listing pods")
	}

	return 0
}

func newPodListReadError(p *pkgPod.Pod, err error) error {
	lines := []string{
		fmt.Sprintf("Unable to read pod %s manifest:", p.UUID.String()),
		fmt.Sprintf("  %v", err),
	}
	return fmt.Errorf("%s", strings.Join(lines, "\n"))
}

func newPodListZeroAppsError(p *pkgPod.Pod) error {
	return fmt.Errorf("pod %s contains zero apps", p.UUID.String())
}

func newPodListLoadImageManifestError(p *pkgPod.Pod, err error) error {
	return errwrap.Wrap(fmt.Errorf("pod %s ImageManifest could not be loaded", p.UUID.String()), err)
}

func appLine(app lastditch.RuntimeApp) string {
	return fmt.Sprintf("App: %q from image %q (%s)",
		app.Name, app.Image.Name, app.Image.ID)
}

func fmtNets(nis []netinfo.NetInfo) string {
	var parts []string
	for _, ni := range nis {
		// there will be IPv6 support soon so distinguish between v4 and v6
		parts = append(parts, fmt.Sprintf("%v:ip4=%v", ni.NetName, ni.IP))
	}
	return strings.Join(parts, ", ")
}

func getImageName(p *pkgPod.Pod, appName types.ACName) (string, error) {
	aim, err := p.AppImageManifest(appName.String())
	if err != nil {
		return "", errwrap.Wrap(errors.New("problem retrieving ImageManifests from pod"), err)
	}

	imageName := aim.Name.String()
	if version, ok := aim.Labels.Get("version"); ok {
		imageName = fmt.Sprintf("%s:%s", imageName, version)
	}

	return imageName, nil
}
