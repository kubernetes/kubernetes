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

package configmapandsecret

import (
	"fmt"
	"path"
	"strings"

	"github.com/pkg/errors"
	"k8s.io/cli-runtime/pkg/kustomize/k8sdeps/kv"
	"sigs.k8s.io/kustomize/pkg/ifc"
)

func keyValuesFromLiteralSources(sources []string) ([]kv.Pair, error) {
	var kvs []kv.Pair
	for _, s := range sources {
		k, v, err := parseLiteralSource(s)
		if err != nil {
			return nil, err
		}
		kvs = append(kvs, kv.Pair{Key: k, Value: v})
	}
	return kvs, nil
}

func keyValuesFromFileSources(ldr ifc.Loader, sources []string) ([]kv.Pair, error) {
	var kvs []kv.Pair
	for _, s := range sources {
		k, fPath, err := parseFileSource(s)
		if err != nil {
			return nil, err
		}
		content, err := ldr.Load(fPath)
		if err != nil {
			return nil, err
		}
		kvs = append(kvs, kv.Pair{Key: k, Value: string(content)})
	}
	return kvs, nil
}

func keyValuesFromEnvFile(l ifc.Loader, path string) ([]kv.Pair, error) {
	if path == "" {
		return nil, nil
	}
	content, err := l.Load(path)
	if err != nil {
		return nil, err
	}
	return kv.KeyValuesFromLines(content)
}

// parseFileSource parses the source given.
//
//  Acceptable formats include:
//   1.  source-path: the basename will become the key name
//   2.  source-name=source-path: the source-name will become the key name and
//       source-path is the path to the key file.
//
// Key names cannot include '='.
func parseFileSource(source string) (keyName, filePath string, err error) {
	numSeparators := strings.Count(source, "=")
	switch {
	case numSeparators == 0:
		return path.Base(source), source, nil
	case numSeparators == 1 && strings.HasPrefix(source, "="):
		return "", "", fmt.Errorf("key name for file path %v missing", strings.TrimPrefix(source, "="))
	case numSeparators == 1 && strings.HasSuffix(source, "="):
		return "", "", fmt.Errorf("file path for key name %v missing", strings.TrimSuffix(source, "="))
	case numSeparators > 1:
		return "", "", errors.New("key names or file paths cannot contain '='")
	default:
		components := strings.Split(source, "=")
		return components[0], components[1], nil
	}
}

// parseLiteralSource parses the source key=val pair into its component pieces.
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
	return items[0], strings.Trim(items[1], "\"'"), nil
}
