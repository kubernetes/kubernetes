// Copyright 2017 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package properties provides functions for reading and writing
// ISO-8859-1 and UTF-8 encoded .properties files and has
// support for recursive property expansion.
//
// Java properties files are ISO-8859-1 encoded and use Unicode
// literals for characters outside the ISO character set. Unicode
// literals can be used in UTF-8 encoded properties files but
// aren't necessary.
//
// To load a single properties file use MustLoadFile():
//
//   p := properties.MustLoadFile(filename, properties.UTF8)
//
// To load multiple properties files use MustLoadFiles()
// which loads the files in the given order and merges the
// result. Missing properties files can be ignored if the
// 'ignoreMissing' flag is set to true.
//
// Filenames can contain environment variables which are expanded
// before loading.
//
//   f1 := "/etc/myapp/myapp.conf"
//   f2 := "/home/${USER}/myapp.conf"
//   p := MustLoadFiles([]string{f1, f2}, properties.UTF8, true)
//
// All of the different key/value delimiters ' ', ':' and '=' are
// supported as well as the comment characters '!' and '#' and
// multi-line values.
//
//   ! this is a comment
//   # and so is this
//
//   # the following expressions are equal
//   key value
//   key=value
//   key:value
//   key = value
//   key : value
//   key = val\
//         ue
//
// Properties stores all comments preceding a key and provides
// GetComments() and SetComments() methods to retrieve and
// update them. The convenience functions GetComment() and
// SetComment() allow access to the last comment. The
// WriteComment() method writes properties files including
// the comments and with the keys in the original order.
// This can be used for sanitizing properties files.
//
// Property expansion is recursive and circular references
// and malformed expressions are not allowed and cause an
// error. Expansion of environment variables is supported.
//
//   # standard property
//   key = value
//
//   # property expansion: key2 = value
//   key2 = ${key}
//
//   # recursive expansion: key3 = value
//   key3 = ${key2}
//
//   # circular reference (error)
//   key = ${key}
//
//   # malformed expression (error)
//   key = ${ke
//
//   # refers to the users' home dir
//   home = ${HOME}
//
//   # local key takes precendence over env var: u = foo
//   USER = foo
//   u = ${USER}
//
// The default property expansion format is ${key} but can be
// changed by setting different pre- and postfix values on the
// Properties object.
//
//   p := properties.NewProperties()
//   p.Prefix = "#["
//   p.Postfix = "]#"
//
// Properties provides convenience functions for getting typed
// values with default values if the key does not exist or the
// type conversion failed.
//
//   # Returns true if the value is either "1", "on", "yes" or "true"
//   # Returns false for every other value and the default value if
//   # the key does not exist.
//   v = p.GetBool("key", false)
//
//   # Returns the value if the key exists and the format conversion
//   # was successful. Otherwise, the default value is returned.
//   v = p.GetInt64("key", 999)
//   v = p.GetUint64("key", 999)
//   v = p.GetFloat64("key", 123.0)
//   v = p.GetString("key", "def")
//   v = p.GetDuration("key", 999)
//
// As an alterantive properties may be applied with the standard
// library's flag implementation at any time.
//
//   # Standard configuration
//   v = flag.Int("key", 999, "help message")
//   flag.Parse()
//
//   # Merge p into the flag set
//   p.MustFlag(flag.CommandLine)
//
// Properties provides several MustXXX() convenience functions
// which will terminate the app if an error occurs. The behavior
// of the failure is configurable and the default is to call
// log.Fatal(err). To have the MustXXX() functions panic instead
// of logging the error set a different ErrorHandler before
// you use the Properties package.
//
//   properties.ErrorHandler = properties.PanicHandler
//
//   # Will panic instead of logging an error
//   p := properties.MustLoadFile("config.properties")
//
// You can also provide your own ErrorHandler function. The only requirement
// is that the error handler function must exit after handling the error.
//
//   properties.ErrorHandler = func(err error) {
//	     fmt.Println(err)
//       os.Exit(1)
//   }
//
//   # Will write to stdout and then exit
//   p := properties.MustLoadFile("config.properties")
//
// Properties can also be loaded into a struct via the `Decode`
// method, e.g.
//
//   type S struct {
//       A string        `properties:"a,default=foo"`
//       D time.Duration `properties:"timeout,default=5s"`
//       E time.Time     `properties:"expires,layout=2006-01-02,default=2015-01-01"`
//   }
//
// See `Decode()` method for the full documentation.
//
// The following documents provide a description of the properties
// file format.
//
// http://en.wikipedia.org/wiki/.properties
//
// http://docs.oracle.com/javase/7/docs/api/java/util/Properties.html#load%28java.io.Reader%29
//
package properties
