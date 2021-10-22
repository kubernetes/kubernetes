// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package number formats numbers according to the customs of different locales.
//
// The number formats of this package allow for greater formatting flexibility
// than passing values to message.Printf calls as is. It currently supports the
// builtin Go types and anything that implements the Convert interface
// (currently internal).
//
//    p := message.NewPrinter(language.English)
//
//    p.Printf("%v bottles of beer on the wall.", number.Decimal(1234))
//    // Prints: 1,234 bottles of beer on the wall.
//
//    p.Printf("%v of gophers lose too much fur", number.Percent(0.12))
//    // Prints: 12% of gophers lose too much fur.
//
//    p := message.NewPrinter(language.Dutch)
//
//    p.Printf("There are %v bikes per household.", number.Decimal(1.2))
//    // Prints: Er zijn 1,2 fietsen per huishouden.
//
//
// The width and scale specified in the formatting directives override the
// configuration of the formatter.
package number
