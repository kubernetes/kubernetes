// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package loading

import (
	"encoding/json"
	"errors"
	"path/filepath"
)

// JSONMatcher matches json for a file loader.
func JSONMatcher(path string) bool {
	ext := filepath.Ext(path)
	return ext == ".json" || ext == ".jsn" || ext == ".jso"
}

// JSONDoc loads a json document from either a file or a remote url.
func JSONDoc(path string, opts ...Option) (json.RawMessage, error) {
	data, err := LoadFromFileOrHTTP(path, opts...)
	if err != nil {
		return nil, errors.Join(err, ErrLoader)
	}
	return json.RawMessage(data), nil
}
