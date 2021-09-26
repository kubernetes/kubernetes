// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package message implements formatted I/O for localized strings with functions
// analogous to the fmt's print functions. It is a drop-in replacement for fmt.
//
//
// Localized Formatting
//
// A format string can be localized by replacing any of the print functions of
// fmt with an equivalent call to a Printer.
//
//    p := message.NewPrinter(message.MatchLanguage("en"))
//    p.Println(123456.78) // Prints 123,456.78
//
//    p.Printf("%d ducks in a row", 4331) // Prints 4,331 ducks in a row
//
//    p := message.NewPrinter(message.MatchLanguage("nl"))
//    p.Printf("Hoogte: %.1f meter", 1244.9) // Prints Hoogte: 1,244.9 meter
//
//    p := message.NewPrinter(message.MatchLanguage("bn"))
//    p.Println(123456.78) // Prints ১,২৩,৪৫৬.৭৮
//
// Printer currently supports numbers and specialized types for which packages
// exist in x/text. Other builtin types such as time.Time and slices are
// planned.
//
// Format strings largely have the same meaning as with fmt with the following
// notable exceptions:
//   - flag # always resorts to fmt for printing
//   - verb 'f', 'e', 'g', 'd' use localized formatting unless the '#' flag is
//     specified.
//   - verb 'm' inserts a translation of a string argument.
//
// See package fmt for more options.
//
//
// Translation
//
// The format strings that are passed to Printf, Sprintf, Fprintf, or Errorf
// are used as keys to look up translations for the specified languages.
// More on how these need to be specified below.
//
// One can use arbitrary keys to distinguish between otherwise ambiguous
// strings:
//    p := message.NewPrinter(language.English)
//    p.Printf("archive(noun)")  // Prints "archive"
//    p.Printf("archive(verb)")  // Prints "archive"
//
//    p := message.NewPrinter(language.German)
//    p.Printf("archive(noun)")  // Prints "Archiv"
//    p.Printf("archive(verb)")  // Prints "archivieren"
//
// To retain the fallback functionality, use Key:
//    p.Printf(message.Key("archive(noun)", "archive"))
//    p.Printf(message.Key("archive(verb)", "archive"))
//
//
// Translation Pipeline
//
// Format strings that contain text need to be translated to support different
// locales. The first step is to extract strings that need to be translated.
//
// 1. Install gotext
//    go get -u golang.org/x/text/cmd/gotext
//    gotext -help
//
// 2. Mark strings in your source to be translated by using message.Printer,
// instead of the functions of the fmt package.
//
// 3. Extract the strings from your source
//
//    gotext extract
//
// The output will be written to the textdata directory.
//
// 4. Send the files for translation
//
// It is planned to support multiple formats, but for now one will have to
// rewrite the JSON output to the desired format.
//
// 5. Inject translations into program
//
// 6. Repeat from 2
//
// Right now this has to be done programmatically with calls to Set or
// SetString. These functions as well as the methods defined in
// see also package golang.org/x/text/message/catalog can be used to implement
// either dynamic or static loading of messages.
//
//
// Plural and Gender Forms
//
// Translated messages can vary based on the plural and gender forms of
// substitution values. In general, it is up to the translators to provide
// alternative translations for such forms. See the packages in
// golang.org/x/text/feature and golang.org/x/text/message/catalog for more
// information.
//
package message
