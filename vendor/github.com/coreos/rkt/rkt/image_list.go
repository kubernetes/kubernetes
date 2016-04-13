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

package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	rktflag "github.com/coreos/rkt/rkt/flag"
	"github.com/coreos/rkt/store"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/lastditch"
	"github.com/dustin/go-humanize"
	"github.com/spf13/cobra"
)

const (
	defaultTimeLayout = "2006-01-02 15:04:05.999 -0700 MST"

	id         = "id"
	name       = "name"
	importTime = "import time"
	lastUsed   = "last used"
	size       = "size"
	latest     = "latest"
)

// Convenience methods for formatting fields
func l(s string) string {
	return strings.ToLower(strings.Replace(s, " ", "", -1))
}
func u(s string) string {
	return strings.ToUpper(s)
}

var (
	// map of valid fields and related header name
	ImagesFieldHeaderMap = map[string]string{
		l(id):         u(id),
		l(name):       u(name),
		l(importTime): u(importTime),
		l(lastUsed):   u(lastUsed),
		l(latest):     u(latest),
		l(size):       u(size),
	}

	// map of valid sort fields containing the mapping between the provided field name
	// and the related aciinfo's field name.
	ImagesFieldAciInfoMap = map[string]string{
		l(id):         "blobkey",
		l(name):       l(name),
		l(importTime): l(importTime),
		l(lastUsed):   l(lastUsed),
		l(latest):     l(latest),
		l(size):       l(size),
	}

	ImagesSortableFields = map[string]struct{}{
		l(name):       struct{}{},
		l(importTime): struct{}{},
		l(lastUsed):   struct{}{},
		l(size):       struct{}{},
	}
)

type ImagesSortAsc bool

func (isa *ImagesSortAsc) Set(s string) error {
	switch strings.ToLower(s) {
	case "asc":
		*isa = true
	case "desc":
		*isa = false
	default:
		return fmt.Errorf("wrong sort order")
	}
	return nil
}

func (isa *ImagesSortAsc) String() string {
	if *isa {
		return "asc"
	}
	return "desc"
}

func (isa *ImagesSortAsc) Type() string {
	return "imagesSortAsc"
}

var (
	cmdImageList = &cobra.Command{
		Use:   "list",
		Short: "List images in the local store",
		Long:  `Optionally, allows the user to specify the fields and sort order.`,
		Run:   runWrapper(runImages),
	}
	flagImagesFields     *rktflag.OptionList
	flagImagesSortFields *rktflag.OptionList
	flagImagesSortAsc    ImagesSortAsc
)

func init() {
	sortFields := []string{l(name), l(importTime), l(lastUsed), l(size)}
	fields := append([]string{l(id)}, append(sortFields, l(latest))...)

	// Set defaults
	var err error
	flagImagesFields, err = rktflag.NewOptionList(fields, strings.Join(fields, ","))
	if err != nil {
		stderr.FatalE("", err)
	}
	flagImagesSortFields, err = rktflag.NewOptionList(sortFields, l(importTime))
	if err != nil {
		stderr.FatalE("", err)
	}
	flagImagesSortAsc = true

	cmdImage.AddCommand(cmdImageList)
	cmdImageList.Flags().Var(flagImagesFields, "fields", fmt.Sprintf(`comma-separated list of fields to display. Accepted values: %s`,
		flagImagesFields.PermissibleString()))
	cmdImageList.Flags().Var(flagImagesSortFields, "sort", fmt.Sprintf(`sort the output according to the provided comma-separated list of fields. Accepted values: %s`,
		flagImagesSortFields.PermissibleString()))
	cmdImageList.Flags().Var(&flagImagesSortAsc, "order", `choose the sorting order if at least one sort field is provided (--sort). Accepted values: "asc", "desc"`)
	cmdImageList.Flags().BoolVar(&flagNoLegend, "no-legend", false, "suppress a legend with the list")
	cmdImageList.Flags().BoolVar(&flagFullOutput, "full", false, "use long output format")
}

func runImages(cmd *cobra.Command, args []string) int {
	var errors []error
	tabBuffer := new(bytes.Buffer)
	tabOut := getTabOutWithWriter(tabBuffer)

	if !flagNoLegend {
		var headerFields []string
		for _, f := range flagImagesFields.Options {
			headerFields = append(headerFields, ImagesFieldHeaderMap[f])
		}
		fmt.Fprintf(tabOut, "%s\n", strings.Join(headerFields, "\t"))
	}

	s, err := store.NewStore(getDataDir())
	if err != nil {
		stderr.PrintE("cannot open store", err)
		return 1
	}

	var sortAciinfoFields []string
	for _, f := range flagImagesSortFields.Options {
		sortAciinfoFields = append(sortAciinfoFields, ImagesFieldAciInfoMap[f])
	}
	aciInfos, err := s.GetAllACIInfos(sortAciinfoFields, bool(flagImagesSortAsc))
	if err != nil {
		stderr.PrintE("unable to get aci infos", err)
		return 1
	}

	for _, aciInfo := range aciInfos {
		imj, err := s.GetImageManifestJSON(aciInfo.BlobKey)
		if err != nil {
			// ignore aciInfo with missing image manifest as it can be deleted in the meantime
			continue
		}
		var im *schema.ImageManifest
		if err = json.Unmarshal(imj, &im); err != nil {
			errors = append(errors, newImgListLoadError(err, imj, aciInfo.BlobKey))
			continue
		}
		version, ok := im.Labels.Get("version")
		var fieldValues []string
		for _, f := range flagImagesFields.Options {
			fieldValue := ""
			switch f {
			case l(id):
				hashKey := aciInfo.BlobKey
				if !flagFullOutput {
					// The short hash form is [HASH_ALGO]-[FIRST 12 CHAR]
					// For example, sha512-123456789012
					pos := strings.Index(hashKey, "-")
					trimLength := pos + 13
					if pos > 0 && trimLength < len(hashKey) {
						hashKey = hashKey[:trimLength]
					}
				}
				fieldValue = hashKey
			case l(name):
				fieldValue = aciInfo.Name
				if ok {
					fieldValue = fmt.Sprintf("%s:%s", fieldValue, version)
				}
			case l(importTime):
				if flagFullOutput {
					fieldValue = aciInfo.ImportTime.Format(defaultTimeLayout)
				} else {
					fieldValue = humanize.Time(aciInfo.ImportTime)
				}
			case l(lastUsed):
				if flagFullOutput {
					fieldValue = aciInfo.LastUsed.Format(defaultTimeLayout)
				} else {
					fieldValue = humanize.Time(aciInfo.LastUsed)
				}
			case l(size):
				totalSize := aciInfo.Size + aciInfo.TreeStoreSize
				if flagFullOutput {
					fieldValue = fmt.Sprintf("%d", totalSize)
				} else {
					fieldValue = humanize.IBytes(uint64(totalSize))
				}
			case l(latest):
				fieldValue = fmt.Sprintf("%t", aciInfo.Latest)
			}
			fieldValues = append(fieldValues, fieldValue)

		}
		fmt.Fprintf(tabOut, "%s\n", strings.Join(fieldValues, "\t"))
	}

	if len(errors) > 0 {
		sep := "----------------------------------------"
		stderr.Printf("%d error(s) encountered when listing images:", len(errors))
		stderr.Print(sep)
		for _, err := range errors {
			stderr.Error(err)
			stderr.Print(sep)
		}
		stderr.Print("misc:")
		stderr.Printf("  rkt's appc version: %s", schema.AppContainerVersion)
		// make a visible break between errors and the listing
		stderr.Print("")
	}
	tabOut.Flush()
	stdout.Print(tabBuffer.String())
	return 0
}

func newImgListLoadError(err error, imj []byte, blobKey string) error {
	var lines []string
	im := lastditch.ImageManifest{}
	imErr := im.UnmarshalJSON(imj)
	if imErr == nil {
		lines = append(lines, fmt.Sprintf("Unable to load manifest of image %s (spec version %s) because it is invalid:", im.Name, im.ACVersion))
		lines = append(lines, fmt.Sprintf("  %v", err))
	} else {
		lines = append(lines, "Unable to load manifest of an image because it is invalid:")
		lines = append(lines, fmt.Sprintf("  %v", err))
		lines = append(lines, "  Also, failed to get any information about invalid image manifest:")
		lines = append(lines, fmt.Sprintf("    %v", imErr))
	}
	lines = append(lines, "ID of the invalid image:")
	lines = append(lines, fmt.Sprintf("  %s", blobKey))
	return fmt.Errorf("%s", strings.Join(lines, "\n"))
}
