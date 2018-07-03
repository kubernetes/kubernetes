# Mergo

A helper to merge structs and maps in Golang. Useful for configuration default values, avoiding messy if-statements.

Also a lovely [comune](http://en.wikipedia.org/wiki/Mergo) (municipality) in the Province of Ancona in the Italian region of Marche.

## Status

It is ready for production use. [It is used in several projects by Docker, Google, The Linux Foundation, VMWare, Shopify, etc](https://github.com/imdario/mergo#mergo-in-the-wild).

[![Build Status][1]][2]
[![GoDoc][3]][4]
[![GoCard][5]][6]
[![Coverage Status][7]][8]

[1]: https://travis-ci.org/imdario/mergo.png
[2]: https://travis-ci.org/imdario/mergo
[3]: https://godoc.org/github.com/imdario/mergo?status.svg
[4]: https://godoc.org/github.com/imdario/mergo
[5]: https://goreportcard.com/badge/imdario/mergo
[6]: https://goreportcard.com/report/github.com/imdario/mergo
[7]: https://coveralls.io/repos/github/imdario/mergo/badge.svg?branch=master
[8]: https://coveralls.io/github/imdario/mergo?branch=master

### Latest release

[Release 0.3.1](https://github.com/imdario/mergo/releases/tag/0.3.1) is an important release because it changes `Merge()`and `Map()` signatures to support [transformers](#transformers). An optional/variadic argument has been added, so it won't break existing code.

### Important note

If you were using Mergo **before** April 6th 2015, please check your project works as intended after updating your local copy with ```go get -u github.com/imdario/mergo```. I apologize for any issue caused by its previous behavior and any future bug that Mergo could cause (I hope it won't!) in existing projects after the change (release 0.2.0).

### Mergo in the wild

- [moby/moby](https://github.com/moby/moby)
- [kubernetes/kubernetes](https://github.com/kubernetes/kubernetes)
- [vmware/dispatch](https://github.com/vmware/dispatch)
- [Shopify/themekit](https://github.com/Shopify/themekit)
- [imdario/zas](https://github.com/imdario/zas)
- [matcornic/hermes](https://github.com/matcornic/hermes)
- [OpenBazaar/openbazaar-go](https://github.com/OpenBazaar/openbazaar-go)
- [kataras/iris](https://github.com/kataras/iris)
- [michaelsauter/crane](https://github.com/michaelsauter/crane)
- [go-task/task](https://github.com/go-task/task)
- [sensu/uchiwa](https://github.com/sensu/uchiwa)
- [ory/hydra](https://github.com/ory/hydra)
- [sisatech/vcli](https://github.com/sisatech/vcli)
- [dairycart/dairycart](https://github.com/dairycart/dairycart)
- [projectcalico/felix](https://github.com/projectcalico/felix)
- [resin-os/balena](https://github.com/resin-os/balena)
- [go-kivik/kivik](https://github.com/go-kivik/kivik)
- [Telefonica/govice](https://github.com/Telefonica/govice)
- [supergiant/supergiant](supergiant/supergiant)
- [SergeyTsalkov/brooce](https://github.com/SergeyTsalkov/brooce)
- [soniah/dnsmadeeasy](https://github.com/soniah/dnsmadeeasy)
- [ohsu-comp-bio/funnel](https://github.com/ohsu-comp-bio/funnel)
- [EagerIO/Stout](https://github.com/EagerIO/Stout)
- [lynndylanhurley/defsynth-api](https://github.com/lynndylanhurley/defsynth-api)
- [russross/canvasassignments](https://github.com/russross/canvasassignments)
- [rdegges/cryptly-api](https://github.com/rdegges/cryptly-api)
- [casualjim/exeggutor](https://github.com/casualjim/exeggutor)
- [divshot/gitling](https://github.com/divshot/gitling)
- [RWJMurphy/gorl](https://github.com/RWJMurphy/gorl)
- [andrerocker/deploy42](https://github.com/andrerocker/deploy42)
- [elwinar/rambler](https://github.com/elwinar/rambler)
- [tmaiaroto/gopartman](https://github.com/tmaiaroto/gopartman)
- [jfbus/impressionist](https://github.com/jfbus/impressionist)
- [Jmeyering/zealot](https://github.com/Jmeyering/zealot)
- [godep-migrator/rigger-host](https://github.com/godep-migrator/rigger-host)
- [Dronevery/MultiwaySwitch-Go](https://github.com/Dronevery/MultiwaySwitch-Go)
- [thoas/picfit](https://github.com/thoas/picfit)
- [mantasmatelis/whooplist-server](https://github.com/mantasmatelis/whooplist-server)
- [jnuthong/item_search](https://github.com/jnuthong/item_search)
- [bukalapak/snowboard](https://github.com/bukalapak/snowboard)

## Installation

    go get github.com/imdario/mergo

    // use in your .go code
    import (
        "github.com/imdario/mergo"
    )

## Usage

You can only merge same-type structs with exported fields initialized as zero value of their type and same-types maps. Mergo won't merge unexported (private) fields but will do recursively any exported one. Also maps will be merged recursively except for structs inside maps (because they are not addressable using Go reflection).

```go
if err := mergo.Merge(&dst, src); err != nil {
    // ...
}
```

Also, you can merge overwriting values using the transformer `WithOverride`.

```go
if err := mergo.Merge(&dst, src, WithOverride); err != nil {
    // ...
}
```

Additionally, you can map a `map[string]interface{}` to a struct (and otherwise, from struct to map), following the same restrictions as in `Merge()`. Keys are capitalized to find each corresponding exported field.

```go
if err := mergo.Map(&dst, srcMap); err != nil {
    // ...
}
```

Warning: if you map a struct to map, it won't do it recursively. Don't expect Mergo to map struct members of your struct as `map[string]interface{}`. They will be just assigned as values.

More information and examples in [godoc documentation](http://godoc.org/github.com/imdario/mergo).

### Nice example

```go
package main

import (
	"fmt"
	"github.com/imdario/mergo"
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
```

Note: if test are failing due missing package, please execute:

    go get gopkg.in/yaml.v2

### Transformers

Transformers allow to merge specific types differently than in the default behavior. In other words, now you can customize how some types are merged. For example, `time.Time` is a struct; it doesn't have zero value but IsZero can return true because it has fields with zero value. How can we merge a non-zero `time.Time`?

```go
package main

import (
	"fmt"
        "reflect"
        "time"
)

type timeTransfomer struct {
}

func (t timeTransfomer) Transformer(typ reflect.Type) func(dst, src reflect.Value) error {
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
	mergo.Merge(&dest, src, WithTransformers(timeTransfomer{}))
	fmt.Println(dest)
	// Will print
	// { 2018-01-12 01:15:00 +0000 UTC m=+0.000000001 }
}
```


## Contact me

If I can help you, you have an idea or you are using Mergo in your projects, don't hesitate to drop me a line (or a pull request): [@im_dario](https://twitter.com/im_dario)

## About

Written by [Dario Castañé](http://dario.im).

## License

[BSD 3-Clause](http://opensource.org/licenses/BSD-3-Clause) license, as [Go language](http://golang.org/LICENSE).
