// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dot implements GraphViz DOT marshaling and unmarshaling of graphs.
//
// See the GraphViz DOT Guide and the DOT grammar for more information
// on using specific aspects of the DOT language:
//
// DOT Guide: https://www.graphviz.org/pdf/dotguide.pdf
//
// DOT grammar: http://www.graphviz.org/doc/info/lang.html
//
// Attribute quoting
//
// Attributes and IDs are quoted if needed during marshalling, to conform with
// valid DOT syntax. Quoted IDs and attributes are unquoted during unmarshaling,
// so the data is kept in raw form. As an exception, quoted text with a leading
// `"<` and a trailing `>"` is not unquoted to ensure preservation of the string
// during a round-trip.
package dot // import "gonum.org/v1/gonum/graph/encoding/dot"
