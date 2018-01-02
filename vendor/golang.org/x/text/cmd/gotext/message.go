// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// TODO: these definitions should be moved to a package so that the can be used
// by other tools.

// The file contains the structures used to define translations of a certain
// messages.
//
// A translation may have multiple translations strings, or messages, depending
// on the feature values of the various arguments. For instance, consider
// a hypothetical translation from English to English, where the source defines
// the format string "%d file(s) remaining". A completed translation, expressed
// in JS, for this format string could look like:
//
// {
//     "Key": [
//         "\"%d files(s) remaining\""
//     ],
//     "Original": {
//         "Msg": "\"%d files(s) remaining\""
//     },
//     "Translation": {
// 	       "Select": {
// 	           "Feature": "plural",
//             "Arg": 1,
//             "Case": {
//                 "one":   { "Msg": "1 file remaining" },
//                 "other": { "Msg": "%d files remaining" }
//             },
//         },
//     },
//     "Args": [
//         {
//             "ID": 2,
//             "Type": "int",
//             "UnderlyingType": "int",
//             "Expr": "nFiles",
//             "Comment": "number of files remaining",
//             "Position": "golang.org/x/text/cmd/gotext/demo.go:34:3"
//         }
//     ],
//     "Position": "golang.org/x/text/cmd/gotext/demo.go:33:10",
// }
//
// Alternatively, the Translation section could be written as:
//
//     "Translation": {
// 	       "Msg": "%d %[files]s remaining",
//         "Var": {
//             "files" : {
//                 "Select": {
//         	           "Feature": "plural",
//                     "Arg": 1,
//                     "Case": {
//                         "one":   { "Msg": "file" },
//                         "other": { "Msg": "files" }
//                     }
//                 }
//             }
//         }
//     }

// A Translation describes a translation for a single language for a single
// message.
type Translation struct {
	// Key contains a list of identifiers for the message. If this list is empty
	// Original is used as the key.
	Key               []string `json:"key,omitempty"`
	Original          Text     `json:"original"`
	Translation       Text     `json:"translation"`
	ExtractedComment  string   `json:"extractedComment,omitempty"`
	TranslatorComment string   `json:"translatorComment,omitempty"`

	Args []Argument `json:"args,omitempty"`

	// Extraction information.
	Position string `json:"position,omitempty"` // filePosition:line
}

// An Argument contains information about the arguments passed to a message.
type Argument struct {
	ID             interface{} `json:"id"` // An int for printf-style calls, but could be a string.
	Type           string      `json:"type"`
	UnderlyingType string      `json:"underlyingType"`
	Expr           string      `json:"expr"`
	Value          string      `json:"value,omitempty"`
	Comment        string      `json:"comment,omitempty"`
	Position       string      `json:"position,omitempty"`

	// Features contains the features that are available for the implementation
	// of this argument.
	Features []Feature `json:"features,omitempty"`
}

// Feature holds information about a feature that can be implemented by
// an Argument.
type Feature struct {
	Type string `json:"type"` // Right now this is only gender and plural.

	// TODO: possible values and examples for the language under consideration.

}

// Text defines a message to be displayed.
type Text struct {
	// Msg and Select contains the message to be displayed. Within a Text value
	// either Msg or Select is defined.
	Msg    string  `json:"msg,omitempty"`
	Select *Select `json:"select,omitempty"`
	// Var defines a map of variables that may be substituted in the selected
	// message.
	Var map[string]Text `json:"var,omitempty"`
	// Example contains an example message formatted with default values.
	Example string `json:"example,omitempty"`
}

// Type Select selects a Text based on the feature value associated with
// a feature of a certain argument.
type Select struct {
	Feature string          `json:"feature"` // Name of variable or Feature type
	Arg     interface{}     `json:"arg"`     // The argument ID.
	Cases   map[string]Text `json:"cases"`
}
