# Martini  [![wercker status](https://app.wercker.com/status/9b7dbc6e2654b604cd694d191c3d5487/s/master "wercker status")](https://app.wercker.com/project/bykey/9b7dbc6e2654b604cd694d191c3d5487)[![GoDoc](https://godoc.org/github.com/go-martini/martini?status.png)](http://godoc.org/github.com/go-martini/martini)

MartiniはGolangによる、モジュール形式のウェブアプリケーション/サービスを作成するパワフルなパッケージです。

## はじめに

Goをインストールし、[GOPATH](http://golang.org/doc/code.html#GOPATH)を設定した後、Martiniを始める最初の`.go`ファイルを作りましょう。これを`server.go`とします。

~~~ go
package main

import "github.com/go-martini/martini"

func main() {
  m := martini.Classic()
  m.Get("/", func() string {
    return "Hello world!"
  })
  m.Run()
}
~~~

そのあとで、Martini パッケージをインストールします。(**go 1.1**か、それ以上のバーションが必要です。)

~~~
go get github.com/go-martini/martini
~~~

インストールが完了したら、サーバを起動しましょう。
~~~
go run server.go
~~~

そうすれば`localhost:3000`でMartiniのサーバが起動します。

## 分からないことがあったら？

[メーリングリスト](https://groups.google.com/forum/#!forum/martini-go)に入る

[デモビデオ](http://martini.codegangsta.io/#demo)をみる

Stackoverflowで[martini tag](http://stackoverflow.com/questions/tagged/martini)を使い質問する

GoDoc [documentation](http://godoc.org/github.com/go-martini/martini)


## 特徴
* 非常にシンプルに使用できる
* 押し付けがましくないデザイン
* 他のGolangパッケージとの協調性
* 素晴らしいパスマッチングとルーティング
* モジュラーデザイン - 機能性の付け外しが簡単
* たくさんの良いハンドラ/ミドルウェア
* 優れた 'すぐに使える' 機能たち
* **[http.HandlerFunc](http://godoc.org/net/http#HandlerFunc)との完全な互換性**

## もっとミドルウェアについて知るには？
さらに多くのミドルウェアとその機能について知りたいときは、[martini-contrib](https://github.com/martini-contrib) オーガナイゼーションにあるリポジトリを確認してください。

## 目次(Table of Contents)
* [Classic Martini](#classic-martini)
  * [ハンドラ](#handlers)
  * [ルーティング](#routing)
  * [サービス](#services)
  * [静的ファイル配信](#serving-static-files)
* [ミドルウェアハンドラ](#middleware-handlers)
  * [Next()](#next)
* [Martini Env](#martini-env)
* [FAQ](#faq)

## Classic Martini
立ち上げ、すぐ実行できるように、[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) はほとんどのウェブアプリケーションで機能する、標準的な機能を提供します。
~~~ go
  m := martini.Classic()
  // ... middleware and routing goes here
  m.Run()
~~~

下記が[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic)が自動的に読み込む機能の一覧です。
  * Request/Response Logging - [martini.Logger](http://godoc.org/github.com/go-martini/martini#Logger)
  * Panic Recovery - [martini.Recovery](http://godoc.org/github.com/go-martini/martini#Recovery)
  * Static File serving - [martini.Static](http://godoc.org/github.com/go-martini/martini#Static)
  * Routing - [martini.Router](http://godoc.org/github.com/go-martini/martini#Router)

### ハンドラ
ハンドラはMartiniのコアであり、存在意義でもあります。ハンドラには基本的に、呼び出し可能な全ての関数が適応できます。
~~~ go
m.Get("/", func() {
  println("hello world")
})
~~~

#### Return Values
もしハンドラが何かを返す場合、Martiniはその結果を現在の[http.ResponseWriter](http://godoc.org/net/http#ResponseWriter)にstringとして書き込みます。
~~~ go
m.Get("/", func() string {
  return "hello world" // HTTP 200 : "hello world"
})
~~~

任意でステータスコードを返すこともできます。
~~~ go
m.Get("/", func() (int, string) {
  return 418, "i'm a teapot" // HTTP 418 : "i'm a teapot"
})
~~~

#### Service Injection
ハンドラはリフレクションによって実行されます。Martiniはハンドラの引数内の依存関係を**依存性の注入(Dependency Injection)**を使って解決しています。**これによって、Martiniはgolangの`http.HandlerFunc`と完全な互換性を備えています。**

ハンドラに引数を追加すると、Martiniは内部のサービスを検索し、依存性をtype assertionによって解決しようと試みます。
~~~ go
m.Get("/", func(res http.ResponseWriter, req *http.Request) { // res and req are injected by Martini
  res.WriteHeader(200) // HTTP 200
})
~~~

[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic)にはこれらのサービスが内包されています:
  * [*log.Logger](http://godoc.org/log#Logger) - Martiniのためのグローバルなlogger.
  * [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) - http request context.
  * [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) - `map[string]string`型の、ルートマッチングによって検出されたパラメータ
  * [martini.Routes](http://godoc.org/github.com/go-martini/martini#Routes) - Route helper service.
  * [http.ResponseWriter](http://godoc.org/net/http/#ResponseWriter) - http Response writer interface.
  * [*http.Request](http://godoc.org/net/http/#Request) - http Request.

### ルーティング
Martiniでは、ルーティングはHTTPメソッドとURL-matching patternによって対になっており、それぞれが一つ以上のハンドラメソッドを持つことができます。
~~~ go
m.Get("/", func() {
  // show something
})

m.Patch("/", func() {
  // update something
})

m.Post("/", func() {
  // create something
})

m.Put("/", func() {
  // replace something
})

m.Delete("/", func() {
  // destroy something
})

m.Options("/", func() {
  // http options
})

m.NotFound(func() {
  // handle 404
})
~~~

ルーティングはそれらの定義された順番に検索され、最初にマッチしたルーティングが呼ばれます。

名前付きパラメータを定義することもできます。これらのパラメータは[martini.Params](http://godoc.org/github.com/go-martini/martini#Params)サービスを通じてアクセスすることができます:
~~~ go
m.Get("/hello/:name", func(params martini.Params) string {
  return "Hello " + params["name"]
})
~~~

ワイルドカードを使用することができます:
~~~ go
m.Get("/hello/**", func(params martini.Params) string {
  return "Hello " + params["_1"]
})
~~~

正規表現も、このように使うことができます:
~~~go
m.Get("/hello/(?P<name>[a-zA-Z]+)", func(params martini.Params) string {
  return fmt.Sprintf ("Hello %s", params["name"])
})
~~~

もっと正規表現の構文をしりたい場合は、[Go documentation](http://golang.org/pkg/regexp/syntax/) を見てください。


ハンドラは互いの上に積み重ねてることができます。これは、認証や承認処理の際に便利です:
~~~ go
m.Get("/secret", authorize, func() {
  // this will execute as long as authorize doesn't write a response
})
~~~

ルーティンググループも、Groupメソッドを使用することで追加できます。
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
})
~~~

ハンドラにミドルウェアを渡せるのと同じように、グループにもミドルウェアを渡すことができます:
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
}, MyMiddleware1, MyMiddleware2)
~~~

### サービス
サービスはハンドラの引数として注入されることで利用可能になるオブジェクトです。これらは*グローバル*、または*リクエスト*のレベルでマッピングすることができます。

#### Global Mapping
Martiniのインスタンスはinject.Injectorのインターフェースを実装しています。なので、サービスをマッピングすることは簡単です:
~~~ go
db := &MyDatabase{}
m := martini.Classic()
m.Map(db) // the service will be available to all handlers as *MyDatabase
// ...
m.Run()
~~~

#### Request-Level Mapping
リクエストレベルでのマッピングは[martini.Context](http://godoc.org/github.com/go-martini/martini#Context)を使い、ハンドラ内で行うことができます:
~~~ go
func MyCustomLoggerHandler(c martini.Context, req *http.Request) {
  logger := &MyCustomLogger{req}
  c.Map(logger) // mapped as *MyCustomLogger
}
~~~

#### Mapping values to Interfaces
サービスの最も強力なことの一つは、インターフェースにサービスをマッピングできる機能です。例えば、[http.ResponseWriter](http://godoc.org/net/http#ResponseWriter)を機能を追加して上書きしたい場合、このようにハンドラを書くことができます:
~~~ go
func WrapResponseWriter(res http.ResponseWriter, c martini.Context) {
  rw := NewSpecialResponseWriter(res)
  c.MapTo(rw, (*http.ResponseWriter)(nil)) // override ResponseWriter with our wrapper ResponseWriter
}
~~~

### 静的ファイル配信

[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) インスタンスは、自動的にルート直下の "public" ディレクトリ以下の静的ファイルを配信します。[martini.Static](http://godoc.org/github.com/go-martini/martini#Static)を追加することで、もっと多くのディレクトリを配信することもできます:
~~~ go
m.Use(martini.Static("assets")) // serve from the "assets" directory as well
~~~

## ミドルウェア　ハンドラ
ミドルウェア ハンドラは次に来るhttpリクエストとルーターの間に位置します。本質的には、その他のハンドラとの違いはありません。ミドルウェア　ハンドラの追加はこのように行います:
~~~ go
m.Use(func() {
  // do some middleware stuff
})
~~~

`Handlers`関数を使えば、ミドルウェアスタックを完全に制御できます。これは以前に設定されている全てのハンドラを置き換えます:

~~~ go
m.Handlers(
  Middleware1,
  Middleware2,
  Middleware3,
)
~~~

ミドルウェア ハンドラはロギング、認証、承認プロセス、セッション、gzipping、エラーページの表示、その他httpリクエストの前後で怒らなければならないような場合に素晴らしく効果を発揮します。
~~~ go
// validate an api key
m.Use(func(res http.ResponseWriter, req *http.Request) {
  if req.Header.Get("X-API-KEY") != "secret123" {
    res.WriteHeader(http.StatusUnauthorized)
  }
})
~~~

### Next()

[Context.Next()](http://godoc.org/github.com/go-martini/martini#Context) は他のハンドラが実行されたことを取得するために使用する機能です。これはhttpリクエストのあとに実行したい任意の関数があるときに素晴らしく機能します:
~~~ go
// log before and after a request
m.Use(func(c martini.Context, log *log.Logger){
  log.Println("before a request")

  c.Next()

  log.Println("after a request")
})
~~~

## Martini Env

いくつかのMartiniのハンドラはdevelopment環境とproduction環境で別々の動作を提供するために`martini.Env`グローバル変数を使用しています。Martiniサーバを本番環境にデプロイする際には、`MARTINI_ENV=production`環境変数をセットすることをおすすめします。

## FAQ

### Middlewareを見つけるには?

[martini-contrib](https://github.com/martini-contrib)プロジェクトをみることから始めてください。もし望みのものがなければ、新しいリポジトリをオーガナイゼーションに追加するために、martini-contribチームのメンバーにコンタクトを取ってみてください。

* [auth](https://github.com/martini-contrib/auth) - Handlers for authentication.
* [binding](https://github.com/martini-contrib/binding) - Handler for mapping/validating a raw request into a structure.
* [gzip](https://github.com/martini-contrib/gzip) - Handler for adding gzip compress to requests
* [render](https://github.com/martini-contrib/render) - Handler that provides a service for easily rendering JSON and HTML templates.
* [acceptlang](https://github.com/martini-contrib/acceptlang) - Handler for parsing the `Accept-Language` HTTP header.
* [sessions](https://github.com/martini-contrib/sessions) - Handler that provides a Session service.
* [strip](https://github.com/martini-contrib/strip) - URL Prefix stripping.
* [method](https://github.com/martini-contrib/method) - HTTP method overriding via Header or form fields.
* [secure](https://github.com/martini-contrib/secure) - Implements a few quick security wins.
* [encoder](https://github.com/martini-contrib/encoder) - Encoder service for rendering data in several formats and content negotiation.
* [cors](https://github.com/martini-contrib/cors) - Handler that enables CORS support.
* [oauth2](https://github.com/martini-contrib/oauth2) - Handler that provides OAuth 2.0 login for Martini apps. Google Sign-in, Facebook Connect and Github login is supported.

### 既存のサーバに組み込むには?

Martiniのインスタンスは`http.Handler`を実装しているので、既存のGoサーバ上でサブツリーを提供するのは簡単です。例えばこれは、Google App Engine上で動くMartiniアプリです:

~~~ go
package hello

import (
  "net/http"
  "github.com/go-martini/martini"
)

func init() {
  m := martini.Classic()
  m.Get("/", func() string {
    return "Hello world!"
  })
  http.Handle("/", m)
}
~~~

### どうやってポート/ホストをかえるの?

Martiniの`Run`関数はPORTとHOSTという環境変数を探し、その値を使用します。見つからない場合はlocalhost:3000がデフォルトで使用されます。もっと柔軟性をもとめるなら、`martini.RunOnAddr`関数が役に立ちます:

~~~ go
  m := martini.Classic()
  // ...
  log.Fatal(m.RunOnAddr(":8080"))
~~~

### Live code reload?

[gin](https://github.com/codegangsta/gin) と [fresh](https://github.com/pilu/fresh) 両方がMartiniアプリケーションを自動リロードできます。

## Contributing
Martini本体は小さく、クリーンであるべきであり、ほとんどのコントリビューションは[martini-contrib](https://github.com/martini-contrib) オーガナイゼーション内で完結すべきですが、もしMartiniのコアにコントリビュートすることがあるなら、自由に行ってください。

## About

Inspired by [express](https://github.com/visionmedia/express) and [sinatra](https://github.com/sinatra/sinatra)

Martini is obsessively designed by none other than the [Code Gangsta](http://codegangsta.io/)
