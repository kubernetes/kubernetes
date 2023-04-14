# Mergo

[![GoDoc][3]][4]
[![GitHub release][5]][6]
[![GoCard][7]][8]
[![Build Status][1]][2]
[![Coverage Status][9]][10]
[![Sourcegraph][11]][12]
[![FOSSA Status][13]][14]
[![Become my sponsor][15]][16]
[![Tidelift][17]][18]

[1]: https://travis-ci.org/imdario/mergo.png
[2]: https://travis-ci.org/imdario/mergo
[3]: https://godoc.org/github.com/imdario/mergo?status.svg
[4]: https://godoc.org/github.com/imdario/mergo
[5]: https://img.shields.io/github/release/imdario/mergo.svg
[6]: https://github.com/imdario/mergo/releases
[7]: https://goreportcard.com/badge/imdario/mergo
[8]: https://goreportcard.com/report/github.com/imdario/mergo
[9]: https://coveralls.io/repos/github/imdario/mergo/badge.svg?branch=master
[10]: https://coveralls.io/github/imdario/mergo?branch=master
[11]: https://sourcegraph.com/github.com/imdario/mergo/-/badge.svg
[12]: https://sourcegraph.com/github.com/imdario/mergo?badge
[13]: https://app.fossa.io/api/projects/git%2Bgithub.com%2Fimdario%2Fmergo.svg?type=shield
[14]: https://app.fossa.io/projects/git%2Bgithub.com%2Fimdario%2Fmergo?ref=badge_shield
[15]: https://img.shields.io/github/sponsors/imdario
[16]: https://github.com/sponsors/imdario
[17]: https://tidelift.com/badges/package/go/github.com%2Fimdario%2Fmergo
[18]: https://tidelift.com/subscription/pkg/go-github.com-imdario-mergo

A helper to merge structs and maps in Golang. Useful for configuration default values, avoiding messy if-statements.

Mergo merges same-type structs and maps by setting default values in zero-value fields. Mergo won't merge unexported (private) fields. It will do recursively any exported one. It also won't merge structs inside maps (because they are not addressable using Go reflection).

Also a lovely [comune](http://en.wikipedia.org/wiki/Mergo) (municipality) in the Province of Ancona in the Italian region of Marche.

## Status

It is ready for production use. [It is used in several projects by Docker, Google, The Linux Foundation, VMWare, Shopify, Microsoft, etc](https://github.com/imdario/mergo#mergo-in-the-wild).

### Important note

Please keep in mind that a problematic PR broke [0.3.9](//github.com/imdario/mergo/releases/tag/0.3.9). I reverted it in [0.3.10](//github.com/imdario/mergo/releases/tag/0.3.10), and I consider it stable but not bug-free. Also, this version adds support for go modules.

Keep in mind that in [0.3.2](//github.com/imdario/mergo/releases/tag/0.3.2), Mergo changed `Merge()`and `Map()` signatures to support [transformers](#transformers). I added an optional/variadic argument so that it won't break the existing code.

If you were using Mergo before April 6th, 2015, please check your project works as intended after updating your local copy with ```go get -u github.com/imdario/mergo```. I apologize for any issue caused by its previous behavior and any future bug that Mergo could cause in existing projects after the change (release 0.2.0).

### Donations

If Mergo is useful to you, consider buying me a coffee, a beer, or making a monthly donation to allow me to keep building great free software. :heart_eyes:

<a href='https://ko-fi.com/B0B58839' target='_blank'><img height='36' style='border:0px;height:36px;' src='https://az743702.vo.msecnd.net/cdn/kofi1.png?v=0' border='0' alt='Buy Me a Coffee at ko-fi.com' /></a>
<a href="https://liberapay.com/dario/donate"><img alt="Donate using Liberapay" src="https://liberapay.com/assets/widgets/donate.svg"></a>
<a href='https://github.com/sponsors/imdario' target='_blank'><img alt="Become my sponsor" src="https://img.shields.io/github/sponsors/imdario?style=for-the-badge" /></a>

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
- [containerssh/containerssh](https://github.com/containerssh/containerssh)
- [goreleaser/goreleaser](https://github.com/goreleaser/goreleaser)
- [tjpnz/structbot](https://github.com/tjpnz/structbot)

## Install

    go get github.com/imdario/mergo

    // use in your .go code
    import (
        "github.com/imdario/mergo"
    )

## Usage

You can only merge same-type structs with exported fields initialized as zero value of their type and same-types maps. Mergo won't merge unexported (private) fields but will do recursively any exported one. It won't merge empty structs value as [they are zero values](https://golang.org/ref/spec#The_zero_value) too. Also, maps will be merged recursively except for structs inside maps (because they are not addressable using Go reflection).

```go
if err := mergo.Merge(&dst, src); err != nil {
    // ...
}
```

Also, you can merge overwriting values using the transformer `WithOverride`.

```go
if err := mergo.Merge(&dst, src, mergo.WithOverride); err != nil {
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

Here is a nice example:

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

    go get gopkg.in/yaml.v3

### Transformers

Transformers allow to merge specific types differently than in the default behavior. In other words, now you can customize how some types are merged. For example, `time.Time` is a struct; it doesn't have zero value but IsZero can return true because it has fields with zero value. How can we merge a non-zero `time.Time`?

```go
package main

import (
	"fmt"
	"github.com/imdario/mergo"
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
```

## Contact me

If I can help you, you have an idea or you are using Mergo in your projects, don't hesitate to drop me a line (or a pull request): [@im_dario](https://twitter.com/im_dario)

## About

Written by [Dario Castañé](http://dario.im).

## License

[BSD 3-Clause](http://opensource.org/licenses/BSD-3-Clause) license, as [Go language](http://golang.org/LICENSE).


[![FOSSA Status](https://app.fossa.io/api/projects/git%2Bgithub.com%2Fimdario%2Fmergo.svg?type=large)](https://app.fossa.io/projects/git%2Bgithub.com%2Fimdario%2Fmergo?ref=badge_large)
