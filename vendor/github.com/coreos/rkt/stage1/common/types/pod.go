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

package types

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"path/filepath"

	"github.com/coreos/rkt/common"
	"github.com/coreos/rkt/pkg/user"

	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"github.com/hashicorp/errwrap"
)

const (
	// The filename where we persist the RuntimePod data
	RuntimeConfigPath = "runtime-config"

	// App-level annotations: streams mode
	AppStdinMode  = "coreos.com/rkt/stage2/stdin"
	AppStdoutMode = "coreos.com/rkt/stage2/stdout"
	AppStderrMode = "coreos.com/rkt/stage2/stderr"
)

// Pod encapsulates a PodManifest and ImageManifests
type Pod struct {
	RuntimePod        // embedded runtime parameters
	Root       string // root directory where the pod will be located
	UUID       types.UUID
	Manifest   *schema.PodManifest
	Images     map[string]*schema.ImageManifest
	UidRange   user.UidRange
}

// RuntimePod stores internal state we'd like access to. There is no interface,
// and noone outside the stage1 should access it. If you find yourself needing
// one of these members outside of the stage1, then it should be set as an
// annotation on the pod.
// This includes things like insecure options and the mds token - they are provided
// during init but needed for `app add`.
type RuntimePod struct {
	MetadataServiceURL string         `json:"MetadataServiceURL"`
	PrivateUsers       string         `json:"PrivateUsers"`
	MDSToken           string         `json:"MDSToken"`
	Hostname           string         `json:"Hostname"`
	Debug              bool           `json:"Debug"`
	Mutable            bool           `json:"Mutable"`
	ResolvConfMode     string         `json:"ResolvConfMode"`
	EtcHostsMode       string         `json:"EtcHostsMode"`
	NetList            common.NetList `json:"NetList"`
	Interactive        bool           `json:"Interactive"`
	InsecureOptions    struct {
		DisablePaths        bool `json:"DisablePaths"`
		DisableCapabilities bool `json:"DisableCapabilities"`
		DisableSeccomp      bool `json:"DisableSeccomp"`
	} `json:"InsecureOptions"`
}

// AppNameToImageName takes the name of an app in the Pod and returns the name
// of the app's image. The mapping between these two is populated when a Pod is
// loaded (using LoadPod).
func (p *Pod) AppNameToImageName(appName types.ACName) types.ACIdentifier {
	image, ok := p.Images[appName.String()]
	if !ok {
		// This should be impossible as we have updated the map in LoadPod().
		panic(fmt.Sprintf("No images for app %q", appName.String()))
	}
	return image.Name
}

// SaveRuntime persists just the runtime state. This should be called when the
// pod is started.
func (p *Pod) SaveRuntime() error {
	path := filepath.Join(p.Root, RuntimeConfigPath)
	buf, err := json.Marshal(p.RuntimePod)
	if err != nil {
		return err
	}

	return ioutil.WriteFile(path, buf, 0644)
}

// LoadPod loads a Pod Manifest (as prepared by stage0), the runtime data, and
// its associated Application Manifests, under $root/stage1/opt/stage1/$apphash
func LoadPod(root string, uuid *types.UUID, rp *RuntimePod) (*Pod, error) {
	p := &Pod{
		Root:     root,
		UUID:     *uuid,
		Images:   make(map[string]*schema.ImageManifest),
		UidRange: *user.NewBlankUidRange(),
	}

	// Unserialize runtime parameters
	if rp != nil {
		p.RuntimePod = *rp
	} else {
		buf, err := ioutil.ReadFile(filepath.Join(p.Root, RuntimeConfigPath))
		if err != nil {
			return nil, errwrap.Wrap(errors.New("failed reading runtime params"), err)
		}
		if err := json.Unmarshal(buf, &p.RuntimePod); err != nil {
			return nil, errwrap.Wrap(errors.New("failed unmarshalling runtime params"), err)
		}
	}

	buf, err := ioutil.ReadFile(common.PodManifestPath(p.Root))
	if err != nil {
		return nil, errwrap.Wrap(errors.New("failed reading pod manifest"), err)
	}

	pm := &schema.PodManifest{}
	if err := json.Unmarshal(buf, pm); err != nil {
		return nil, errwrap.Wrap(errors.New("failed unmarshalling pod manifest"), err)
	}
	p.Manifest = pm

	for i, app := range p.Manifest.Apps {
		impath := common.ImageManifestPath(p.Root, app.Name)
		buf, err := ioutil.ReadFile(impath)
		if err != nil {
			return nil, errwrap.Wrap(fmt.Errorf("failed reading image manifest %q", impath), err)
		}

		im := &schema.ImageManifest{}
		if err = json.Unmarshal(buf, im); err != nil {
			return nil, errwrap.Wrap(fmt.Errorf("failed unmarshalling image manifest %q", impath), err)
		}

		if _, ok := p.Images[app.Name.String()]; ok {
			return nil, fmt.Errorf("got multiple definitions for app: %v", app.Name)
		}
		if app.App == nil {
			p.Manifest.Apps[i].App = im.App
		}
		p.Images[app.Name.String()] = im
	}

	if err := p.UidRange.Deserialize([]byte(p.PrivateUsers)); err != nil {
		return nil, err
	}

	return p, nil
}
