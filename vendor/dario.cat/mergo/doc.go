// Copyright 2013 Dario Castañé. All rights reserved.
// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
A helper to merge structs and maps in Golang. Useful for configuration default values, avoiding messy if-statements.

Mergo merges same-type structs and maps by setting default values in zero-value fields. Mergo won't merge unexported (private) fields. It will do recursively any exported one. It also won't merge structs inside maps (because they are not addressable using Go reflection).

# Status

It is ready for production use. It is used in several projects by Docker, Google, The Linux Foundation, VMWare, Shopify, etc.

# Important notes

1.0.0

In 1.0.0 Mergo moves to a vanity URL `dario.cat/mergo`.

0.3.9

Please keep in mind that a problematic PR broke 0.3.9. We reverted it in 0.3.10. We consider 0.3.10 as stable but not bug-free. . Also, this version adds suppot for go modules.

Keep in mind that in 0.3.2, Mergo changed Merge() and Map() signatures to support transformers. We added an optional/variadic argument so that it won't break the existing code.

If you were using Mergo before April 6th, 2015, please check your project works as intended after updating your local copy with go get -u dario.cat/mergo. I apologize for any issue caused by its previous behavior and any future bug that Mergo could cause in existing projects after the change (release 0.2.0).

# Install

Do your usual installation procedure:

	go get dario.cat/mergo

	// use in your .go code
	import (
	    "dario.cat/mergo"
	)

# Usage

You can only merge same-type structs with exported fields initialized as zero value of their type and same-types maps. Mergo won't merge unexported (private) fields but will do recursively any exported one. It won't merge empty structs value as they are zero values too. Also, maps will be merged recursively except for structs inside maps (because they are not addressable using Go reflection).

	if err := mergo.Merge(&dst, src); err != nil {
		// ...
	}

Also, you can merge overwriting values using the transformer WithOverride.

	if err := mergo.Merge(&dst, src, mergo.WithOverride); err != nil {
		// ...
	}

Additionally, you can map a map[string]interface{} to a struct (and otherwise, from struct to map), following the same restrictions as in Merge(). Keys are capitalized to find each corresponding exported field.

	if err := mergo.Map(&dst, srcMap); err != nil {
		// ...
	}

Warning: if you map a struct to map, it won't do it recursively. Don't expect Mergo to map struct members of your struct as map[string]interface{}. They will be just assigned as values.

Here is a nice example:

	package main

	import (
		"fmt"
		"dario.cat/mergo"
	)

	type Foo struct {
		A string
		B int64
	}

	func main() {
		src := Foo{
			A: "one",
			B: 2,
		}
		dest := Foo{
			A: "two",
		}
		mergo.Merge(&dest, src)
		fmt.Println(dest)
		// Will print
		// {two 2}
	}

# Transformers

Transformers allow to merge specific types differently than in the default behavior. In other words, now you can customize how some types are merged. For example, time.Time is a struct; it doesn't have zero value but IsZero can return true because it has fields with zero value. How can we merge a non-zero time.Time?

	package main

	import (
		"fmt"
		"dario.cat/mergo"
		"reflect"
		"time"
	)

	type timeTransformer struct {
	}

	func (t timeTransformer) Transformer(typ reflect.Type) func(dst, src reflect.Value) error {
		if typ == reflect.TypeOf(time.Time{}) {
			return func(dst, src reflect.Value) error {
				if dst.CanSet() {
					isZero := dst.MethodByName("IsZero")
					result := isZero.Call([]reflect.Value{})
					if result[0].Bool() {
						dst.Set(src)
					}
				}
				return nil
			}
		}
		return nil
	}

	type Snapshot struct {
		Time time.Time
		// ...
	}

	func main() {
		src := Snapshot{time.Now()}
		dest := Snapshot{}
		mergo.Merge(&dest, src, mergo.WithTransformers(timeTransformer{}))
		fmt.Println(dest)
		// Will print
		// { 2018-01-12 01:15:00 +0000 UTC m=+0.000000001 }
	}

# Contact me

If I can help you, you have an idea or you are using Mergo in your projects, don't hesitate to drop me a line (or a pull request): https://twitter.com/im_dario

# About

Written by Dario Castañé: https://da.rio.hn

# License

BSD 3-Clause license, as Go language.
*/
package mergo
