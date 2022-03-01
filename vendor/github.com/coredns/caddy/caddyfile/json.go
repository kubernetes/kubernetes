// Copyright 2015 Light Code Labs, LLC
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

package caddyfile

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
)

const filename = "Caddyfile"

// ToJSON converts caddyfile to its JSON representation.
func ToJSON(caddyfile []byte) ([]byte, error) {
	var j EncodedCaddyfile

	serverBlocks, err := Parse(filename, bytes.NewReader(caddyfile), nil)
	if err != nil {
		return nil, err
	}

	for _, sb := range serverBlocks {
		block := EncodedServerBlock{
			Keys: sb.Keys,
			Body: [][]interface{}{},
		}

		// Extract directives deterministically by sorting them
		var directives = make([]string, len(sb.Tokens))
		for dir := range sb.Tokens {
			directives = append(directives, dir)
		}
		sort.Strings(directives)

		// Convert each directive's tokens into our JSON structure
		for _, dir := range directives {
			disp := NewDispenserTokens(filename, sb.Tokens[dir])
			for disp.Next() {
				block.Body = append(block.Body, constructLine(&disp))
			}
		}

		// tack this block onto the end of the list
		j = append(j, block)
	}

	result, err := json.Marshal(j)
	if err != nil {
		return nil, err
	}

	return result, nil
}

// constructLine transforms tokens into a JSON-encodable structure;
// but only one line at a time, to be used at the top-level of
// a server block only (where the first token on each line is a
// directive) - not to be used at any other nesting level.
func constructLine(d *Dispenser) []interface{} {
	var args []interface{}

	args = append(args, d.Val())

	for d.NextArg() {
		if d.Val() == "{" {
			args = append(args, constructBlock(d))
			continue
		}
		args = append(args, d.Val())
	}

	return args
}

// constructBlock recursively processes tokens into a
// JSON-encodable structure. To be used in a directive's
// block. Goes to end of block.
func constructBlock(d *Dispenser) [][]interface{} {
	block := [][]interface{}{}

	for d.Next() {
		if d.Val() == "}" {
			break
		}
		block = append(block, constructLine(d))
	}

	return block
}

// FromJSON converts JSON-encoded jsonBytes to Caddyfile text
func FromJSON(jsonBytes []byte) ([]byte, error) {
	var j EncodedCaddyfile
	var result string

	err := json.Unmarshal(jsonBytes, &j)
	if err != nil {
		return nil, err
	}

	for sbPos, sb := range j {
		if sbPos > 0 {
			result += "\n\n"
		}
		for i, key := range sb.Keys {
			if i > 0 {
				result += ", "
			}
			//result += standardizeScheme(key)
			result += key
		}
		result += jsonToText(sb.Body, 1)
	}

	return []byte(result), nil
}

// jsonToText recursively transforms a scope of JSON into plain
// Caddyfile text.
func jsonToText(scope interface{}, depth int) string {
	var result string

	switch val := scope.(type) {
	case string:
		if strings.ContainsAny(val, "\" \n\t\r") {
			result += `"` + strings.Replace(val, "\"", "\\\"", -1) + `"`
		} else {
			result += val
		}
	case int:
		result += strconv.Itoa(val)
	case float64:
		result += fmt.Sprintf("%v", val)
	case bool:
		result += fmt.Sprintf("%t", val)
	case [][]interface{}:
		result += " {\n"
		for _, arg := range val {
			result += strings.Repeat("\t", depth) + jsonToText(arg, depth+1) + "\n"
		}
		result += strings.Repeat("\t", depth-1) + "}"
	case []interface{}:
		for i, v := range val {
			if block, ok := v.([]interface{}); ok {
				result += "{\n"
				for _, arg := range block {
					result += strings.Repeat("\t", depth) + jsonToText(arg, depth+1) + "\n"
				}
				result += strings.Repeat("\t", depth-1) + "}"
				continue
			}
			result += jsonToText(v, depth)
			if i < len(val)-1 {
				result += " "
			}
		}
	}

	return result
}

// TODO: Will this function come in handy somewhere else?
/*
// standardizeScheme turns an address like host:https into https://host,
// or "host:" into "host".
func standardizeScheme(addr string) string {
	if hostname, port, err := net.SplitHostPort(addr); err == nil {
		if port == "http" || port == "https" {
			addr = port + "://" + hostname
		}
	}
	return strings.TrimSuffix(addr, ":")
}
*/

// EncodedCaddyfile encapsulates a slice of EncodedServerBlocks.
type EncodedCaddyfile []EncodedServerBlock

// EncodedServerBlock represents a server block ripe for encoding.
type EncodedServerBlock struct {
	Keys []string        `json:"keys"`
	Body [][]interface{} `json:"body"`
}
