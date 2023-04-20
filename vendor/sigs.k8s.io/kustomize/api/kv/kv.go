// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package kv

import (
	"bufio"
	"bytes"
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"

	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/internal/generators"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/errors"
)

var utf8bom = []byte{0xEF, 0xBB, 0xBF}

// loader reads and validates KV pairs.
type loader struct {
	// Used to read the filesystem.
	ldr ifc.Loader

	// Used to validate various k8s data fields.
	validator ifc.Validator
}

func NewLoader(ldr ifc.Loader, v ifc.Validator) ifc.KvLoader {
	return &loader{ldr: ldr, validator: v}
}

func (kvl *loader) Validator() ifc.Validator {
	return kvl.validator
}

func (kvl *loader) Load(
	args types.KvPairSources) (all []types.Pair, err error) {
	pairs, err := kvl.keyValuesFromEnvFiles(args.EnvSources)
	if err != nil {
		return nil, errors.WrapPrefixf(err,
			"env source files: %v",
			args.EnvSources)
	}
	all = append(all, pairs...)

	pairs, err = keyValuesFromLiteralSources(args.LiteralSources)
	if err != nil {
		return nil, errors.WrapPrefixf(err,
			"literal sources %v", args.LiteralSources)
	}
	all = append(all, pairs...)

	pairs, err = kvl.keyValuesFromFileSources(args.FileSources)
	if err != nil {
		return nil, errors.WrapPrefixf(err,
			"file sources: %v", args.FileSources)
	}
	return append(all, pairs...), nil
}

func keyValuesFromLiteralSources(sources []string) ([]types.Pair, error) {
	var kvs []types.Pair
	for _, s := range sources {
		k, v, err := parseLiteralSource(s)
		if err != nil {
			return nil, err
		}
		kvs = append(kvs, types.Pair{Key: k, Value: v})
	}
	return kvs, nil
}

func (kvl *loader) keyValuesFromFileSources(sources []string) ([]types.Pair, error) {
	var kvs []types.Pair
	for _, s := range sources {
		k, fPath, err := generators.ParseFileSource(s)
		if err != nil {
			return nil, err
		}
		content, err := kvl.ldr.Load(fPath)
		if err != nil {
			return nil, err
		}
		kvs = append(kvs, types.Pair{Key: k, Value: string(content)})
	}
	return kvs, nil
}

func (kvl *loader) keyValuesFromEnvFiles(paths []string) ([]types.Pair, error) {
	var kvs []types.Pair
	for _, p := range paths {
		content, err := kvl.ldr.Load(p)
		if err != nil {
			return nil, err
		}
		more, err := kvl.keyValuesFromLines(content)
		if err != nil {
			return nil, err
		}
		kvs = append(kvs, more...)
	}
	return kvs, nil
}

// keyValuesFromLines parses given content in to a list of key-value pairs.
func (kvl *loader) keyValuesFromLines(content []byte) ([]types.Pair, error) {
	var kvs []types.Pair

	scanner := bufio.NewScanner(bytes.NewReader(content))
	currentLine := 0
	for scanner.Scan() {
		// Process the current line, retrieving a key/value pair if
		// possible.
		scannedBytes := scanner.Bytes()
		kv, err := kvl.keyValuesFromLine(scannedBytes, currentLine)
		if err != nil {
			return nil, err
		}
		currentLine++

		if len(kv.Key) == 0 {
			// no key means line was empty or a comment
			continue
		}

		kvs = append(kvs, kv)
	}
	return kvs, nil
}

// KeyValuesFromLine returns a kv with blank key if the line is empty or a comment.
func (kvl *loader) keyValuesFromLine(line []byte, currentLine int) (types.Pair, error) {
	kv := types.Pair{}

	if !utf8.Valid(line) {
		return kv, fmt.Errorf("line %d has invalid utf8 bytes : %v", line, string(line))
	}

	// We trim UTF8 BOM from the first line of the file but no others
	if currentLine == 0 {
		line = bytes.TrimPrefix(line, utf8bom)
	}

	// trim the line from all leading whitespace first
	line = bytes.TrimLeftFunc(line, unicode.IsSpace)

	// If the line is empty or a comment, we return a blank key/value pair.
	if len(line) == 0 || line[0] == '#' {
		return kv, nil
	}

	data := strings.SplitN(string(line), "=", 2)
	key := data[0]
	if err := kvl.validator.IsEnvVarName(key); err != nil {
		return kv, err
	}

	if len(data) == 2 {
		kv.Value = data[1]
	} else {
		// If there is no value (no `=` in the line), we set the value to an empty string
		kv.Value = ""
	}
	kv.Key = key
	return kv, nil
}

// ParseLiteralSource parses the source key=val pair into its component pieces.
// This functionality is distinguished from strings.SplitN(source, "=", 2) since
// it returns an error in the case of empty keys, values, or a missing equals sign.
func parseLiteralSource(source string) (keyName, value string, err error) {
	// leading equal is invalid
	if strings.Index(source, "=") == 0 {
		return "", "", fmt.Errorf("invalid literal source %v, expected key=value", source)
	}
	// split after the first equal (so values can have the = character)
	items := strings.SplitN(source, "=", 2)
	if len(items) != 2 {
		return "", "", fmt.Errorf("invalid literal source %v, expected key=value", source)
	}
	return items[0], removeQuotes(items[1]), nil
}

// removeQuotes removes the surrounding quotes from the provided string only if it is surrounded on both sides
// rather than blindly trimming all quotation marks on either side.
func removeQuotes(str string) string {
	if len(str) == 0 || str[0] != str[len(str)-1] {
		return str
	}
	if str[0] == '"' || str[0] == '\'' {
		return str[1 : len(str)-1]
	}
	return str
}
