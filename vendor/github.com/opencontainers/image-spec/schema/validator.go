// Copyright 2016 The Linux Foundation
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

package schema

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"regexp"

	digest "github.com/opencontainers/go-digest"
	"github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
	"github.com/xeipuuv/gojsonschema"
)

// Validator wraps a media type string identifier
// and implements validation against a JSON schema.
type Validator string

type validateFunc func(r io.Reader) error

var mapValidate = map[Validator]validateFunc{
	ValidatorMediaTypeImageConfig: validateConfig,
	ValidatorMediaTypeDescriptor:  validateDescriptor,
	ValidatorMediaTypeImageIndex:  validateIndex,
	ValidatorMediaTypeManifest:    validateManifest,
}

// ValidationError contains all the errors that happened during validation.
type ValidationError struct {
	Errs []error
}

func (e ValidationError) Error() string {
	return fmt.Sprintf("%v", e.Errs)
}

// Validate validates the given reader against the schema of the wrapped media type.
func (v Validator) Validate(src io.Reader) error {
	buf, err := ioutil.ReadAll(src)
	if err != nil {
		return errors.Wrap(err, "unable to read the document file")
	}

	if f, ok := mapValidate[v]; ok {
		if f == nil {
			return fmt.Errorf("internal error: mapValidate[%q] is nil", v)
		}
		err = f(bytes.NewReader(buf))
		if err != nil {
			return err
		}
	}

	sl := gojsonschema.NewReferenceLoaderFileSystem("file:///"+specs[v], fs)
	ml := gojsonschema.NewStringLoader(string(buf))

	result, err := gojsonschema.Validate(sl, ml)
	if err != nil {
		return errors.Wrapf(
			WrapSyntaxError(bytes.NewReader(buf), err),
			"schema %s: unable to validate", v)
	}

	if result.Valid() {
		return nil
	}

	errs := make([]error, 0, len(result.Errors()))
	for _, desc := range result.Errors() {
		errs = append(errs, fmt.Errorf("%s", desc))
	}

	return ValidationError{
		Errs: errs,
	}
}

type unimplemented string

func (v unimplemented) Validate(src io.Reader) error {
	return fmt.Errorf("%s: unimplemented", v)
}

func validateManifest(r io.Reader) error {
	header := v1.Manifest{}

	buf, err := ioutil.ReadAll(r)
	if err != nil {
		return errors.Wrapf(err, "error reading the io stream")
	}

	err = json.Unmarshal(buf, &header)
	if err != nil {
		return errors.Wrap(err, "manifest format mismatch")
	}

	if header.Config.MediaType != string(v1.MediaTypeImageConfig) {
		fmt.Printf("warning: config %s has an unknown media type: %s\n", header.Config.Digest, header.Config.MediaType)
	}

	for _, layer := range header.Layers {
		if layer.MediaType != string(v1.MediaTypeImageLayer) &&
			layer.MediaType != string(v1.MediaTypeImageLayerGzip) &&
			layer.MediaType != string(v1.MediaTypeImageLayerNonDistributable) &&
			layer.MediaType != string(v1.MediaTypeImageLayerNonDistributableGzip) {
			fmt.Printf("warning: layer %s has an unknown media type: %s\n", layer.Digest, layer.MediaType)
		}
	}
	return nil
}

var (
	sha256EncodedRegexp = regexp.MustCompile(`^[a-f0-9]{64}$`)
	sha512EncodedRegexp = regexp.MustCompile(`^[a-f0-9]{128}$`)
)

func validateDescriptor(r io.Reader) error {
	header := v1.Descriptor{}

	buf, err := ioutil.ReadAll(r)
	if err != nil {
		return errors.Wrapf(err, "error reading the io stream")
	}

	err = json.Unmarshal(buf, &header)
	if err != nil {
		return errors.Wrap(err, "descriptor format mismatch")
	}

	if header.Digest.Validate() != nil {
		// we ignore unsupported algorithms
		fmt.Printf("warning: unsupported digest: %q: %v\n", header.Digest, err)
		return nil
	}
	switch header.Digest.Algorithm() {
	case digest.SHA256:
		if !sha256EncodedRegexp.MatchString(header.Digest.Hex()) {
			return errors.Errorf("unexpected sha256 digest: %q", header.Digest)
		}
	case digest.SHA512:
		if !sha512EncodedRegexp.MatchString(header.Digest.Hex()) {
			return errors.Errorf("unexpected sha512 digest: %q", header.Digest)
		}
	}
	return nil
}

func validateIndex(r io.Reader) error {
	header := v1.Index{}

	buf, err := ioutil.ReadAll(r)
	if err != nil {
		return errors.Wrapf(err, "error reading the io stream")
	}

	err = json.Unmarshal(buf, &header)
	if err != nil {
		return errors.Wrap(err, "manifestlist format mismatch")
	}

	for _, manifest := range header.Manifests {
		if manifest.MediaType != string(v1.MediaTypeImageManifest) {
			fmt.Printf("warning: manifest %s has an unknown media type: %s\n", manifest.Digest, manifest.MediaType)
		}

	}

	return nil
}

func validateConfig(r io.Reader) error {
	header := v1.Image{}

	buf, err := ioutil.ReadAll(r)
	if err != nil {
		return errors.Wrapf(err, "error reading the io stream")
	}

	err = json.Unmarshal(buf, &header)
	if err != nil {
		return errors.Wrap(err, "config format mismatch")
	}

	checkPlatform(header.OS, header.Architecture)

	return nil
}

func checkPlatform(OS string, Architecture string) {
	validCombins := map[string][]string{
		"android":   {"arm"},
		"darwin":    {"386", "amd64", "arm", "arm64"},
		"dragonfly": {"amd64"},
		"freebsd":   {"386", "amd64", "arm"},
		"linux":     {"386", "amd64", "arm", "arm64", "ppc64", "ppc64le", "mips64", "mips64le", "s390x"},
		"netbsd":    {"386", "amd64", "arm"},
		"openbsd":   {"386", "amd64", "arm"},
		"plan9":     {"386", "amd64"},
		"solaris":   {"amd64"},
		"windows":   {"386", "amd64"}}
	for os, archs := range validCombins {
		if os == OS {
			for _, arch := range archs {
				if arch == Architecture {
					return
				}
			}
			fmt.Printf("warning: combination of %q and %q is invalid.", OS, Architecture)
		}
	}
	fmt.Printf("warning: operating system %q of the bundle is not supported yet.", OS)
}
