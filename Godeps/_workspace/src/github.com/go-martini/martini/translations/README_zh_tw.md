# Martini  [![wercker status](https://app.wercker.com/status/9b7dbc6e2654b604cd694d191c3d5487/s/master "wercker status")](https://app.wercker.com/project/bykey/9b7dbc6e2654b604cd694d191c3d5487)[![GoDoc](https://godoc.org/github.com/go-martini/martini?status.png)](http://godoc.org/github.com/go-martini/martini)

Martini 是一個使用 Go 語言來快速開發模組化 Web 應用程式或服務的強大套件

## 開始

在您安裝Go語言以及設定好
[GOPATH](http://golang.org/doc/code.html#GOPATH)環境變數後,
開始寫您第一支`.go`檔, 我們將稱它為`server.go`

~~~ go
package main

import "github.com/go-martini/martini"

func main() {
  m := martini.Classic()
  m.Get("/", func() string {
    return "Hello 世界!"
  })
  m.Run()
}
~~~

然後安裝Martini套件 (**go 1.1**以上的版本是必要的)
~~~
go get github.com/go-martini/martini
~~~

然後利用以下指令執行你的程式:
~~~
go run server.go
~~~

此時, 您將會看到一個 Martini Web 伺服器在`localhost:3000`上執行

## 尋求幫助

可以加入 [Mailing list](https://groups.google.com/forum/#!forum/martini-go)

觀看 [Demo Video](http://martini.codegangsta.io/#demo)

## 功能

* 超容易使用
* 非侵入式設計
* 很容易跟其他Go套件同時使用
* 很棒的路徑matching和routing方式
* 模組化設計 - 容易增加或移除功能
* 有很多handlers或middlewares可以直接使用
* 已經提供很多內建功能
* **跟[http.HandlerFunc](http://godoc.org/net/http#HandlerFunc) 介面**完全相容
* 預設document服務 (例如, 提供AngularJS在HTML5模式的服務)

## 其他Middleware
尋找更多的middleware或功能, 請到  [martini-contrib](https://github.com/martini-contrib)程式集搜尋

## 目錄
* [Classic Martini](#classic-martini)
* [Handlers](#handlers)
* [Routing](#routing)
* [Services (服務)](#services)
* [Serving Static Files (伺服靜態檔案)](#serving-static-files)
* [Middleware Handlers](#middleware-handlers)
* [Next()](#next)
* [Martini Env](#martini-env)
* [FAQ (常見問題與答案)](#faq)

## Classic Martini

[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic)
提供大部份web應用程式所需要的基本預設功能:

~~~ go
  m := martini.Classic()
  // ... middleware 或 routing 寫在這裡
  m.Run()
~~~
[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic)
 會自動提供以下功能
* Request/Response Logging - [martini.Logger](http://godoc.org/github.com/go-martini/martini#Logger)
* Panic Recovery - [martini.Recovery](http://godoc.org/github.com/go-martini/martini#Recovery)
* Static File serving - [martini.Static](http://godoc.org/github.com/go-martini/martini#Static)
* Routing - [martini.Router](http://godoc.org/github.com/go-martini/martini#Router)


### Handlers
Handlers 是 Martini 的核心, 每個 handler 就是一個基本的呼叫函式, 例如:
~~~ go
m.Get("/", func() {
  println("hello 世界")
})
~~~

#### 回傳值
如果一個 handler 有回傳值, Martini就會用字串的方式將結果寫回現在的
[http.ResponseWriter](http://godoc.org/net/http#ResponseWriter), 例如:
~~~ go
m.Get("/", func() string {
  return "hello 世界" // HTTP 200 : "hello 世界"
})
~~~

你也可以選擇回傳狀態碼, 例如:
~~~ go
m.Get("/", func() (int, string) {
  return 418, "我是一個茶壺" // HTTP 418 : "我是一個茶壺"
})
~~~

#### 注入服務 (Service Injection)
Handlers 是透過 reflection 方式被喚起, Martini 使用 *Dependency Injection* 的方法
載入 Handler 變數所需要的相關物件 **這也是 Martini 跟 Go 語言`http.HandlerFunc`介面
完全相容的原因**

如果你在 Handler 裡加入一個變數, Martini 會嘗試著從它的服務清單裡透過 type assertion
方式將相關物件載入
~~~ go
m.Get("/", func(res http.ResponseWriter, req *http.Request) { // res 和 req 是由 Martini 注入
  res.WriteHeader(200) // HTTP 200
})
~~~

[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) 包含以下物件:
  * [*log.Logger](http://godoc.org/log#Logger) - Martini 的全區域 Logger.
  * [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) - http request 內文.
  * [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) - `map[string]string` of named params found by route matching.
  * [martini.Routes](http://godoc.org/github.com/go-martini/martini#Routes) - Route helper 服務.
  * [http.ResponseWriter](http://godoc.org/net/http/#ResponseWriter) - http 回應 writer 介面.
  * [*http.Request](http://godoc.org/net/http/#Request) - http 請求.

### Routing
在 Martini 裡, 一個 route 就是一個 HTTP 方法與其 URL 的比對模式.
每個 route 可以有ㄧ或多個 handler 方法:
~~~ go
m.Get("/", func() {
  // 顯示（值）
})

m.Patch("/", func() {
  // 更新
})

m.Post("/", func() {
  // 產生
})

m.Put("/", func() {
  // 取代
})

m.Delete("/", func() {
  // 刪除
})

m.Options("/", func() {
  // http 選項
})

m.NotFound(func() {
  // handle 404
})
~~~

Routes 依照它們被定義時的順序做比對. 第一個跟請求 (request) 相同的 route 就被執行.

Route 比對模式可以包含變數部分, 可以透過 [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) 物件來取值:
~~~ go
m.Get("/hello/:name", func(params martini.Params) string {
  return "Hello " + params["name"]
})
~~~

Routes 也可以用 "**" 來配對, 例如:
~~~ go
m.Get("/hello/**", func(params martini.Params) string {
  return "Hello " + params["_1"]
})
~~~

也可以用正規表示法 (regular expressions) 來做比對, 例如:
~~~go
m.Get("/hello/(?P<name>[a-zA-Z]+)", func(params martini.Params) string {
  return fmt.Sprintf ("Hello %s", params["name"])
})
~~~
更多有關正規表示法文法的資訊, 請參考 [Go 文件](http://golang.org/pkg/regexp/syntax/).

Route handlers 也可以相互堆疊, 尤其是認證與授權相當好用:
~~~ go
m.Get("/secret", authorize, func() {
  // 這裏開始處理授權問題, 而非寫出回應
})
~~~

也可以用 Group 方法, 將 route 編成一組.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
})
~~~

跟對 handler 增加 middleware 方法一樣, 你也可以為一組 routes 增加 middleware.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
}, MyMiddleware1, MyMiddleware2)
~~~

### Services
服務是一些物件可以被注入 Handler 變數裡的東西, 可以分對應到 *Global* 或 *Request* 兩種等級.

#### Global Mapping (全域級對應)
一個 Martini 實體 (instance) 實現了 inject.Injector 介面, 所以非常容易對應到所需要的服務, 例如:
~~~ go
db := &MyDatabase{}
m := martini.Classic()
m.Map(db) // 所以 *MyDatabase 就可以被所有的 handlers 使用
// ...
m.Run()
~~~

#### Request-Level Mapping (請求級對應)
如果只在一個 handler 裡定義, 透由  [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) 獲得一個請求 (request) 級的對應:
~~~ go
func MyCustomLoggerHandler(c martini.Context, req *http.Request) {
  logger := &MyCustomLogger{req}
  c.Map(logger) // 對應到 *MyCustomLogger
}
~~~

#### 透由介面對應
有關服務, 最強的部分是它還能對應到一個介面 (interface), 例如,
如果你想要包裹並增加一個變數而改寫 (override) 原有的 [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter), 你的 handler 可以寫成:
~~~ go
func WrapResponseWriter(res http.ResponseWriter, c martini.Context) {
  rw := NewSpecialResponseWriter(res)
  c.MapTo(rw, (*http.ResponseWriter)(nil)) // 我們包裹的 ResponseWriter 蓋掉原始的 ResponseWrite
}
~~~

### Serving Static Files
一個[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) 實體會將伺服器根目錄下 public 子目錄裡的檔案自動當成靜態檔案處理. 你也可以手動用 [martini.Static](http://godoc.org/github.com/go-martini/martini#Static) 增加其他目錄, 例如.
~~~ go
m.Use(martini.Static("assets")) // "assets" 子目錄裡, 也視為靜態檔案
~~~

#### Serving a Default Document
當某些 URL 找不到時, 你也可以指定本地檔案的 URL 來顯示.
你也可以用開頭除外 (exclusion prefix) 的方式, 來忽略某些 URLs,
它尤其在某些伺服器同時伺服靜態檔案, 而且還有額外 handlers 處理 (例如 REST API) 時, 特別好用.
比如說, 在比對找不到之後, 想要用靜態檔來處理特別好用.

以下範例, 就是在 URL 開頭不是`/api/v`而且也不是本地檔案的情況下, 顯示`/index.html`檔:
~~~ go
static := martini.Static("assets", martini.StaticOptions{Fallback: "/index.html", Exclude: "/api/v"})
m.NotFound(static, http.NotFound)
~~~

## Middleware Handlers
Middleware Handlers 位於進來的 http 請求與 router 之間, 在 Martini 裡, 本質上它跟其他
 Handler 沒有什麼不同, 例如, 你可加入一個 middleware 方法如下
~~~ go
m.Use(func() {
  // 做 middleware 的事
})
~~~

你也可以用`Handlers`完全控制 middelware 層, 把先前設定的 handlers 都替換掉, 例如:
~~~ go
m.Handlers(
  Middleware1,
  Middleware2,
  Middleware3,
)
~~~

Middleware Handlers 成被拿來處理 http 請求之前和之後的事, 尤其是用來紀錄logs, 授權, 認證,
sessions, 壓縮 （gzipping), 顯示錯誤頁面等等, 都非常好用, 例如:
~~~ go
// validate an api key
m.Use(func(res http.ResponseWriter, req *http.Request) {
  if req.Header.Get("X-API-KEY") != "secret123" {
    res.WriteHeader(http.StatusUnauthorized)
  }
})
~~~

### Next()
[Context.Next()](http://godoc.org/github.com/go-martini/martini#Context) 是 Middleware Handlers 可以呼叫的選項功能, 用來等到其他 handlers 處理完再開始執行.
它常常被用來處理那些必須在 http 請求之後才能發生的事件, 例如:
~~~ go
// 在請求前後加 logs
m.Use(func(c martini.Context, log *log.Logger){
  log.Println("before a request")

  c.Next()

  log.Println("after a request")
})
~~~

## Martini Env

有些 Martini handlers 使用 `martini.Env` 全區域變數, 來當成開發環境或是上架 (production)
環境的設定判斷. 建議用 `MARTINI_ENV=production` 環境變數來設定 Martini 伺服器是上架與否.

## FAQ

### 我去哪可以找到 middleware X?

可以從 [martini-contrib](https://github.com/martini-contrib) 裡的專案找起.
如果那裡沒有, 請與 martini-contrib 團隊聯絡, 將它加入.

* [auth](https://github.com/martini-contrib/auth) - 處理認證的 Handler.
* [binding](https://github.com/martini-contrib/binding) -
處理一個單純的請求對應到一個結構體與確認內容正確與否的 Handler.
* [gzip](https://github.com/martini-contrib/gzip) - 對請求加 gzip 壓縮的 Handler.
* [render](https://github.com/martini-contrib/render) - 提供簡單處理 JSON 和
HTML 樣板成形 (rendering) 的 Handler.
* [acceptlang](https://github.com/martini-contrib/acceptlang) - 解析 `Accept-Language` HTTP 檔頭的 Handler.
* [sessions](https://github.com/martini-contrib/sessions) - 提供 Session 服務的 Handler.
* [strip](https://github.com/martini-contrib/strip) - URL 字頭處理 (Prefix stripping).
* [method](https://github.com/martini-contrib/method) - 透過 Header 或表格 (form) 欄位蓋過 HTTP 方法 (method).
* [secure](https://github.com/martini-contrib/secure) - 提供一些簡單的安全機制.
* [encoder](https://github.com/martini-contrib/encoder) - 轉換資料格式之 Encoder 服務.
* [cors](https://github.com/martini-contrib/cors) - 啟動支援 CORS 之 Handler.
* [oauth2](https://github.com/martini-contrib/oauth2) - 讓 Martini 應用程式能提供 OAuth 2.0 登入的 Handler. 其中支援 Google 登錄, Facebook Connect 與 Github 的登入等.
* [vauth](https://github.com/rafecolton/vauth) - 處理 vender webhook 認證的 Handler (目前支援 GitHub 以及 TravisCI)

### 我如何整合到現有的伺服器?

Martini 實作 `http.Handler`,所以可以非常容易整合到現有的 Go 伺服器裡.
以下寫法, 是一個能在 Google App Engine 上運行的 Martini 應用程式:

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

### 我要如何改變 port/host?

Martini 的 `Run` 功能會看 PORT 及 HOST 當時的環境變數, 否則 Martini 會用 localhost:3000
當預設值. 讓 port 及 host 更有彈性, 可以用 `martini.RunOnAddr` 取代.

~~~ go
  m := martini.Classic()
  // ...
  log.Fatal(m.RunOnAddr(":8080"))
~~~

### 可以線上更新 (live reload) 嗎?

[gin](https://github.com/codegangsta/gin) 和 [fresh](https://github.com/pilu/fresh) 可以幫 Martini 程式做到線上更新.

## 貢獻
Martini 盡量保持小而美的精神, 大多數的程式貢獻者可以在 [martini-contrib](https://github.com/martini-contrib) 組織提供代碼. 如果你想要對 Martini 核心提出貢獻, 請丟出 Pull Request.

## 關於

靈感來自與 [express](https://github.com/visionmedia/express) 以及 [sinatra](https://github.com/sinatra/sinatra)

Martini 由 [Code Gangsta](http://codegangsta.io/) 公司設計出品 (著魔地)
