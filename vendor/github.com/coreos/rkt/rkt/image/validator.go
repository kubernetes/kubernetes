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

package image

import (
	"errors"
	"fmt"
	"io"

	"github.com/coreos/rkt/pkg/keystore"
	"github.com/hashicorp/errwrap"

	"github.com/appc/spec/aci"
	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
	"golang.org/x/crypto/openpgp"
	pgperrors "golang.org/x/crypto/openpgp/errors"
)

// validator is a general image checker
type validator struct {
	image    io.ReadSeeker
	manifest *schema.ImageManifest
}

// newValidator returns a validator instance if passed image is indeed
// an ACI.
func newValidator(image io.ReadSeeker) (*validator, error) {
	manifest, err := aci.ManifestFromImage(image)
	if err != nil {
		return nil, err
	}
	v := &validator{
		image:    image,
		manifest: manifest,
	}
	return v, nil
}

// ImageName returns image name as it is in the image manifest.
func (v *validator) ImageName() string {
	return v.manifest.Name.String()
}

// ValidateName checks if desired image name is actually the same as
// the one in the image manifest.
func (v *validator) ValidateName(imageName string) error {
	name := v.ImageName()
	if name != imageName {
		return fmt.Errorf("error when reading the app name: %q expected but %q found",
			imageName, name)
	}
	return nil
}

// ValidateLabels checks if desired image labels are actually the same as
// the ones in the image manifest.
func (v *validator) ValidateLabels(labels map[types.ACIdentifier]string) error {
	for n, rv := range labels {
		if av, ok := v.manifest.GetLabel(n.String()); ok {
			if rv != av {
				return fmt.Errorf("requested value for label %q: %q differs from fetched aci label value: %q", n, rv, av)
			}
		} else {
			return fmt.Errorf("requested label %q not provided by the image manifest", n)
		}
	}
	return nil
}

// ValidateWithSignature verifies the image against a given signature
// file.
func (v *validator) ValidateWithSignature(ks *keystore.Keystore, sig io.ReadSeeker) (*openpgp.Entity, error) {
	if ks == nil {
		return nil, nil
	}
	if _, err := v.image.Seek(0, 0); err != nil {
		return nil, errwrap.Wrap(errors.New("error seeking ACI file"), err)
	}
	if _, err := sig.Seek(0, 0); err != nil {
		return nil, errwrap.Wrap(errors.New("error seeking signature file"), err)
	}
	entity, err := ks.CheckSignature(v.ImageName(), v.image, sig)
	if err == pgperrors.ErrUnknownIssuer {
		log.Print("If you expected the signing key to change, try running:")
		log.Print("    rkt trust --prefix <image>")
	}
	if err != nil {
		return nil, err
	}
	return entity, nil
}
