/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package set

import (
	"errors"
	"regexp"
	"sort"
	"strings"

	"github.com/spf13/cobra"
	"sigs.k8s.io/kustomize/pkg/commands/kustfile"
	"sigs.k8s.io/kustomize/pkg/fs"
	"sigs.k8s.io/kustomize/pkg/image"
)

type setImageOptions struct {
	imageMap map[string]image.Image
}

var pattern = regexp.MustCompile("^(.*):([a-zA-Z0-9._-]*)$")

// errors

var (
	errImageNoArgs      = errors.New("no image specified")
	errImageInvalidArgs = errors.New(`invalid format of image, use one of the following options:
- <image>=<newimage>:<newtag>
- <image>=<newimage>@<newtag>
- <image>=<newimage>
- <image>:<newtag>
- <image>@<digest>`)
)

const separator = "="

// newCmdSetImage sets the new names, tags or digests for images in the kustomization.
func newCmdSetImage(fsys fs.FileSystem) *cobra.Command {
	var o setImageOptions

	cmd := &cobra.Command{
		Use:   "image",
		Short: `Sets images and their new names, new tags or digests in the kustomization file`,
		Example: `
The command
  set image postgres=eu.gcr.io/my-project/postgres:latest my-app=my-registry/my-app@sha256:24a0c4b4a4c0eb97a1aabb8e29f18e917d05abfe1b7a7c07857230879ce7d3d3
will add

images:
- name: postgres
  newName: eu.gcr.io/my-project/postgres
  newTag: latest
- digest: sha256:24a0c4b4a4c0eb97a1aabb8e29f18e917d05abfe1b7a7c07857230879ce7d3d3
  name: my-app
  newName: my-registry/my-app

to the kustomization file if it doesn't exist,
and overwrite the previous ones if the image name exists.

The command
  set image node:8.15.0 mysql=mariadb alpine@sha256:24a0c4b4a4c0eb97a1aabb8e29f18e917d05abfe1b7a7c07857230879ce7d3d3
will add

images:
- name: node
  newTag: 8.15.0
- name: mysql
  newName: mariadb
- digest: sha256:24a0c4b4a4c0eb97a1aabb8e29f18e917d05abfe1b7a7c07857230879ce7d3d3
  name: alpine

to the kustomization file if it doesn't exist,
and overwrite the previous ones if the image name exists.
`,
		RunE: func(cmd *cobra.Command, args []string) error {
			err := o.Validate(args)
			if err != nil {
				return err
			}
			return o.RunSetImage(fsys)
		},
	}
	return cmd
}

type overwrite struct {
	name   string
	digest string
	tag    string
}

// Validate validates setImage command.
func (o *setImageOptions) Validate(args []string) error {
	if len(args) == 0 {
		return errImageNoArgs
	}

	o.imageMap = make(map[string]image.Image)

	for _, arg := range args {

		img, err := parse(arg)
		if err != nil {
			return err
		}
		o.imageMap[img.Name] = img
	}
	return nil
}

// RunSetImage runs setImage command.
func (o *setImageOptions) RunSetImage(fSys fs.FileSystem) error {
	mf, err := kustfile.NewKustomizationFile(fSys)
	if err != nil {
		return err
	}
	m, err := mf.Read()
	if err != nil {
		return err
	}

	// append only new images from ksutomize file
	for _, im := range m.Images {
		if _, ok := o.imageMap[im.Name]; ok {
			continue
		}

		o.imageMap[im.Name] = im
	}

	var images []image.Image
	for _, v := range o.imageMap {
		images = append(images, v)
	}

	sort.Slice(images, func(i, j int) bool {
		return images[i].Name < images[j].Name
	})

	m.Images = images
	return mf.Write(m)
}

func parse(arg string) (image.Image, error) {

	// matches if there is an image name to overwrite
	// <image>=<new-image><:|@><new-tag>
	if s := strings.Split(arg, separator); len(s) == 2 {
		p, err := parseOverwrite(s[1], true)
		return image.Image{
			Name:    s[0],
			NewName: p.name,
			NewTag:  p.tag,
			Digest:  p.digest,
		}, err
	}

	// matches only for <tag|digest> overwrites
	// <image><:|@><new-tag>
	p, err := parseOverwrite(arg, false)
	return image.Image{
		Name:   p.name,
		NewTag: p.tag,
		Digest: p.digest,
	}, err
}

// parseOverwrite parses the overwrite parameters
// from the given arg into a struct
func parseOverwrite(arg string, overwriteImage bool) (overwrite, error) {
	// match <image>@<digest>
	if d := strings.Split(arg, "@"); len(d) > 1 {
		return overwrite{
			name:   d[0],
			digest: d[1],
		}, nil
	}

	// match <image>:<tag>
	if t := pattern.FindStringSubmatch(arg); len(t) == 3 {
		return overwrite{
			name: t[1],
			tag:  t[2],
		}, nil
	}

	// match <image>
	if len(arg) > 0 && overwriteImage {
		return overwrite{
			name: arg,
		}, nil
	}
	return overwrite{}, errImageInvalidArgs
}
