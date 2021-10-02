// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kio

import (
	"bytes"
	"fmt"
	"io"
	"sort"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/kio/kioutil"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

const (
	ResourceListKind       = "ResourceList"
	ResourceListAPIVersion = "config.kubernetes.io/v1alpha1"
)

// ByteReadWriter reads from an input and writes to an output.
type ByteReadWriter struct {
	// Reader is where ResourceNodes are decoded from.
	Reader io.Reader

	// Writer is where ResourceNodes are encoded.
	Writer io.Writer

	// OmitReaderAnnotations will configures Read to skip setting the config.kubernetes.io/index
	// annotation on Resources as they are Read.
	OmitReaderAnnotations bool

	// KeepReaderAnnotations if set will keep the Reader specific annotations when writing
	// the Resources, otherwise they will be cleared.
	KeepReaderAnnotations bool

	// Style is a style that is set on the Resource Node Document.
	Style yaml.Style

	FunctionConfig *yaml.RNode

	Results *yaml.RNode

	NoWrap             bool
	WrappingAPIVersion string
	WrappingKind       string
}

func (rw *ByteReadWriter) Read() ([]*yaml.RNode, error) {
	b := &ByteReader{
		Reader:                rw.Reader,
		OmitReaderAnnotations: rw.OmitReaderAnnotations,
	}
	val, err := b.Read()
	if rw.FunctionConfig == nil {
		rw.FunctionConfig = b.FunctionConfig
	}
	rw.Results = b.Results

	if !rw.NoWrap {
		rw.WrappingAPIVersion = b.WrappingAPIVersion
		rw.WrappingKind = b.WrappingKind
	}
	return val, errors.Wrap(err)
}

func (rw *ByteReadWriter) Write(nodes []*yaml.RNode) error {
	return ByteWriter{
		Writer:                rw.Writer,
		KeepReaderAnnotations: rw.KeepReaderAnnotations,
		Style:                 rw.Style,
		FunctionConfig:        rw.FunctionConfig,
		Results:               rw.Results,
		WrappingAPIVersion:    rw.WrappingAPIVersion,
		WrappingKind:          rw.WrappingKind,
	}.Write(nodes)
}

// ParseAll reads all of the inputs into resources
func ParseAll(inputs ...string) ([]*yaml.RNode, error) {
	return (&ByteReader{
		Reader: bytes.NewBufferString(strings.Join(inputs, "\n---\n")),
	}).Read()
}

// FromBytes reads from a byte slice.
func FromBytes(bs []byte) ([]*yaml.RNode, error) {
	return (&ByteReader{
		OmitReaderAnnotations: true,
		Reader:                bytes.NewBuffer(bs),
	}).Read()
}

// StringAll writes all of the resources to a string
func StringAll(resources []*yaml.RNode) (string, error) {
	var b bytes.Buffer
	err := (&ByteWriter{Writer: &b}).Write(resources)
	return b.String(), err
}

// ByteReader decodes ResourceNodes from bytes.
// By default, Read will set the config.kubernetes.io/index annotation on each RNode as it
// is read so they can be written back in the same order.
type ByteReader struct {
	// Reader is where ResourceNodes are decoded from.
	Reader io.Reader

	// OmitReaderAnnotations will configures Read to skip setting the config.kubernetes.io/index
	// annotation on Resources as they are Read.
	OmitReaderAnnotations bool

	// SetAnnotations is a map of caller specified annotations to set on resources as they are read
	// These are independent of the annotations controlled by OmitReaderAnnotations
	SetAnnotations map[string]string

	FunctionConfig *yaml.RNode

	Results *yaml.RNode

	// DisableUnwrapping prevents Resources in Lists and ResourceLists from being unwrapped
	DisableUnwrapping bool

	// WrappingAPIVersion is set by Read(), and is the apiVersion of the object that
	// the read objects were originally wrapped in.
	WrappingAPIVersion string

	// WrappingKind is set by Read(), and is the kind of the object that
	// the read objects were originally wrapped in.
	WrappingKind string
}

var _ Reader = &ByteReader{}

func (r *ByteReader) Read() ([]*yaml.RNode, error) {
	output := ResourceNodeSlice{}

	// by manually splitting resources -- otherwise the decoder will get the Resource
	// boundaries wrong for header comments.
	input := &bytes.Buffer{}
	_, err := io.Copy(input, r.Reader)
	if err != nil {
		return nil, errors.Wrap(err)
	}

	// replace the ending \r\n (line ending used in windows) with \n and then separate by \n---\n
	values := strings.Split(strings.ReplaceAll(input.String(), "\r\n", "\n"), "\n---\n")

	index := 0
	for i := range values {
		// the Split used above will eat the tail '\n' from each resource. This may affect the
		// literal string value since '\n' is meaningful in it.
		if i != len(values)-1 {
			values[i] += "\n"
		}
		decoder := yaml.NewDecoder(bytes.NewBufferString(values[i]))
		node, err := r.decode(index, decoder)
		if err == io.EOF {
			continue
		}
		if err != nil {
			return nil, errors.Wrap(err)
		}
		if yaml.IsMissingOrNull(node) {
			// empty value
			continue
		}

		// ok if no metadata -- assume not an InputList
		meta, err := node.GetMeta()
		if err != yaml.ErrMissingMetadata && err != nil {
			return nil, errors.WrapPrefixf(err, "[%d]", i)
		}

		// the elements are wrapped in an InputList, unwrap them
		// don't check apiVersion, we haven't standardized on the domain
		if !r.DisableUnwrapping &&
			len(values) == 1 && // Only unwrap if there is only 1 value
			(meta.Kind == ResourceListKind || meta.Kind == "List") &&
			(node.Field("items") != nil || node.Field("functionConfig") != nil) {
			r.WrappingKind = meta.Kind
			r.WrappingAPIVersion = meta.APIVersion

			// unwrap the list
			if fc := node.Field("functionConfig"); fc != nil {
				r.FunctionConfig = fc.Value
			}
			if res := node.Field("results"); res != nil {
				r.Results = res.Value
			}

			items := node.Field("items")
			if items != nil {
				for i := range items.Value.Content() {
					// add items
					output = append(output, yaml.NewRNode(items.Value.Content()[i]))
				}
			}
			continue
		}

		// add the node to the list
		output = append(output, node)

		// increment the index annotation value
		index++
	}
	return output, nil
}

func (r *ByteReader) decode(index int, decoder *yaml.Decoder) (*yaml.RNode, error) {
	node := &yaml.Node{}
	err := decoder.Decode(node)
	if err == io.EOF {
		return nil, io.EOF
	}
	if err != nil {
		return nil, errors.Wrap(err)
	}

	if yaml.IsYNodeEmptyDoc(node) {
		return nil, nil
	}

	// set annotations on the read Resources
	// sort the annotations by key so the output Resources is consistent (otherwise the
	// annotations will be in a random order)
	n := yaml.NewRNode(node)
	if r.SetAnnotations == nil {
		r.SetAnnotations = map[string]string{}
	}
	if !r.OmitReaderAnnotations {
		r.SetAnnotations[kioutil.IndexAnnotation] = fmt.Sprintf("%d", index)
	}
	var keys []string
	for k := range r.SetAnnotations {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		_, err = n.Pipe(yaml.SetAnnotation(k, r.SetAnnotations[k]))
		if err != nil {
			return nil, errors.Wrap(err)
		}
	}
	return yaml.NewRNode(node), nil
}
