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
	"time"

	rktlib "github.com/coreos/rkt/lib"
	rktflag "github.com/coreos/rkt/pkg/flag"
	"github.com/coreos/rkt/store/imagestore"
	"github.com/dustin/go-humanize"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/lastditch"
	"github.com/spf13/cobra"
)

const (
	id         = "id"
	name       = "name"
	importTime = "import time"
	lastUsed   = "last used"
	size       = "size"
	latest     = "latest"
)

const defaultTimeLayout = "2006-01-02 15:04:05.999 -0700 MST"

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
		l(name):       {},
		l(importTime): {},
		l(lastUsed):   {},
		l(size):       {},
	}
)

type printableImage struct {
	rktlib.ImageListEntry

	format outputFormat
	full   bool
}

func (pi *printableImage) MarshalJSON() ([]byte, error) {
	return json.Marshal(pi.ImageListEntry)
}

type outputFormat int

const (
	outputFormatTabbed = iota
	outputFormatJSON
	outputFormatPrettyJSON
)

func (pi *printableImage) imageSize() string {
	if pi.format != outputFormatTabbed || pi.full {
		return fmt.Sprintf("%d", pi.Size)
	}
	return humanize.IBytes(uint64(pi.Size))
}

func (pi *printableImage) importTime() string {
	t := time.Unix(0, pi.ImportTime)
	if pi.format != outputFormatTabbed || pi.full {
		return t.Format(defaultTimeLayout)
	}
	return humanize.Time(t)
}

func (pi *printableImage) lastUsedTime() string {
	t := time.Unix(0, pi.LastUsedTime)
	if pi.format != outputFormatTabbed || pi.full {
		return t.Format(defaultTimeLayout)
	}
	return humanize.Time(t)
}

func (pi *printableImage) attributes(fields *rktflag.OptionList) []string {
	if fields == nil {
		return []string{pi.ID, pi.Name, pi.imageSize(), pi.importTime(), pi.lastUsedTime()}
	}
	optionMapping := map[string]string{
		l(id):         pi.ID,
		l(name):       pi.Name,
		l(size):       pi.imageSize(),
		l(importTime): pi.importTime(),
		l(lastUsed):   pi.lastUsedTime(),
	}
	attrs := []string{}
	for _, f := range fields.Options {
		if a, ok := optionMapping[f]; ok {
			attrs = append(attrs, a)
		}
	}
	return attrs
}

func (pi *printableImage) printableString(fields *rktflag.OptionList) string {
	return strings.Join(pi.attributes(fields), "\t")
}

func (e *outputFormat) Set(s string) error {
	switch s {
	case "":
		*e = outputFormatTabbed
	case "json":
		*e = outputFormatJSON
	case "json-pretty":
		*e = outputFormatPrettyJSON
	default:
		return fmt.Errorf("Invalid format option: %s", s)
	}
	return nil
}

func (s *outputFormat) String() string {
	switch int(*s) {
	case outputFormatJSON:
		return "json"
	case outputFormatPrettyJSON:
		return "json-pretty"
	default:
		return ""
	}
}

func (s *outputFormat) Type() string {
	return "outputFormat"
}

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
	flagImageFormat      outputFormat
)

func init() {
	sortFields := []string{l(name), l(importTime), l(lastUsed), l(size)}

	fields := []string{l(id), l(name), l(size), l(importTime), l(lastUsed)}

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
	cmdImageList.Flags().Var(&flagImageFormat, "format", fmt.Sprintf("choose the output format, allowed format includes 'json', 'json-pretty'. If empty, then the result is printed as key value pairs"))
}

func runImages(cmd *cobra.Command, args []string) int {
	var errors []error
	tabBuffer := new(bytes.Buffer)
	tabOut := getTabOutWithWriter(tabBuffer)

	s, err := imagestore.NewStore(storeDir())
	if err != nil {
		stderr.PrintE("cannot open store", err)
		return 254
	}

	remotes, err := s.GetAllRemotes()
	if err != nil {
		stderr.PrintE("unable to get remotes", err)
		return 254
	}

	remoteMap := make(map[string]*imagestore.Remote)
	for _, r := range remotes {
		remoteMap[r.BlobKey] = r
	}

	var sortAciinfoFields []string
	for _, f := range flagImagesSortFields.Options {
		sortAciinfoFields = append(sortAciinfoFields, ImagesFieldAciInfoMap[f])
	}
	aciInfos, err := s.GetAllACIInfos(sortAciinfoFields, bool(flagImagesSortAsc))
	if err != nil {
		stderr.PrintE("unable to get aci infos", err)
		return 254
	}

	var imagesToPrint []printableImage

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

		imageID := aciInfo.BlobKey
		imageName := aciInfo.Name
		if ok {
			imageName = fmt.Sprintf("%s:%s", imageName, version)
		}

		totalSize := aciInfo.Size + aciInfo.TreeStoreSize

		if !flagFullOutput && flagImageFormat == outputFormatTabbed {
			imageID = trimImageID(imageID)
		}

		imagesToPrint = append(imagesToPrint, printableImage{
			ImageListEntry: rktlib.ImageListEntry{
				ID:           imageID,
				Name:         imageName,
				ImportTime:   aciInfo.ImportTime.UnixNano(),
				LastUsedTime: aciInfo.LastUsed.UnixNano(),
				Size:         totalSize,
			},
			format: flagImageFormat,
			full:   flagFullOutput,
		})
	}

	switch flagImageFormat {
	case outputFormatTabbed:
		if !flagNoLegend {
			var headerFields []string
			for _, f := range flagImagesFields.Options {
				headerFields = append(headerFields, ImagesFieldHeaderMap[f])
			}
			fmt.Fprintf(tabOut, "%s\n", strings.Join(headerFields, "\t"))
		}
		for _, image := range imagesToPrint {
			fmt.Fprintf(tabOut, "%s\n", image.printableString(flagImagesFields))
		}
	case outputFormatJSON:
		result, err := json.Marshal(imagesToPrint)
		if err != nil {
			errors = append(errors, err)
		} else {
			fmt.Fprintf(tabOut, "%s\n", result)
		}
	case outputFormatPrettyJSON:
		result, err := json.MarshalIndent(imagesToPrint, "", "\t")
		if err != nil {
			errors = append(errors, err)
		} else {
			fmt.Fprintf(tabOut, "%s\n", result)
		}
	}

	if len(errors) > 0 {
		printErrors(errors, "listing images")
	}

	tabOut.Flush()
	stdout.Print(tabBuffer.String())
	return 0
}

func trimImageID(imageID string) string {
	// The short hash form is [HASH_ALGO]-[FIRST 12 CHAR]
	// For example, sha512-123456789012
	pos := strings.Index(imageID, "-")
	trimLength := pos + 13
	if pos > 0 && trimLength < len(imageID) {
		imageID = imageID[:trimLength]
	}
	return imageID
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
