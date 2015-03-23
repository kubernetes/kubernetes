# Martini  [![wercker status](https://app.wercker.com/status/9b7dbc6e2654b604cd694d191c3d5487/s/master "wercker status")](https://app.wercker.com/project/bykey/9b7dbc6e2654b604cd694d191c3d5487)[![GoDoc](https://godoc.org/github.com/go-martini/martini?status.png)](http://godoc.org/github.com/go-martini/martini)

Martini Go dilinde hızlı ve modüler web uygulamaları ve servisleri için güçlü bir pakettir.


## Başlangıç

Go kurulumu ve [GOPATH](http://golang.org/doc/code.html#GOPATH) ayarını yaptıktan sonra, ilk `.go` uzantılı dosyamızı oluşturuyoruz. Bu oluşturduğumuz dosyayı `server.go` olarak adlandıracağız.

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

Martini paketini kurduktan sonra (**go 1.1** ve daha üst go sürümü gerekmektedir.):

~~~
go get github.com/go-martini/martini
~~~

Daha sonra server'ımızı çalıştırıyoruz:

~~~
go run server.go
~~~

Şimdi elimizde çalışan bir adet Martini webserver `localhost:3000`  adresinde bulunmaktadır.


## Yardım Almak İçin 

[Mail Listesi](https://groups.google.com/forum/#!forum/martini-go)

[Örnek Video](http://martini.codegangsta.io/#demo)

Stackoverflow üzerinde [martini etiketine](http://stackoverflow.com/questions/tagged/martini) sahip sorular

[GO Diline ait Dökümantasyonlar](http://godoc.org/github.com/go-martini/martini)


## Özellikler 
* Oldukça basit bir kullanıma sahip.
* Kısıtlama yok.
* Golang paketleri ile rahat bir şekilde kullanılıyor.
* Müthiş bir şekilde path eşleştirme ve yönlendirme.
* Modüler dizayn - Kolay eklenen fonksiyonellik.
* handlers/middlewares kullanımı çok iyi.
* Büyük 'kutu dışarı' özellik seti.
* **[http.HandlerFunc](http://godoc.org/net/http#HandlerFunc) arayüzü ile tam uyumludur.**
* Varsayılan belgelendirme işlemleri (örnek olarak, AngularJS uygulamalarının HTML5 modunda servis edilmesi).

## Daha Fazla Middleware(Ara Katman)

Daha fazla ara katman ve fonksiyonellik için, şu repoları inceleyin [martini-contrib](https://github.com/martini-contrib).

## Tablo İçerikleri
* [Classic Martini](#classic-martini)
  * [İşleyiciler / Handlers](#handlers)
  * [Yönlendirmeler / Routing](#routing)
  * [Servisler](#services)
  * [Statik Dosyaların Sunumu](#serving-static-files)
* [Katman İşleyiciler / Middleware Handlers](#middleware-handlers)
  * [Next()](#next)
* [Martini Env](#martini-env)
* [FAQ](#faq)

## Classic Martini
[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) hızlıca projeyi çalıştırır ve çoğu web uygulaması için iyi çalışan bazı makul varsayılanlar sağlar:

~~~ go
  m := martini.Classic()
  // ... middleware and routing goes here
  m.Run()
~~~

[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) aşağıdaki bazı fonsiyonelleri  otomatik olarak çeker:

  * İstek/Yanıt Kayıtları (Request/Response Logging) - [martini.Logger](http://godoc.org/github.com/go-martini/martini#Logger)
  * Hataların Düzeltilmesi (Panic Recovery) - [martini.Recovery](http://godoc.org/github.com/go-martini/martini#Recovery)
  * Statik Dosyaların Sunumu (Static File serving) - [martini.Static](http://godoc.org/github.com/go-martini/martini#Static)
  * Yönlendirmeler (Routing) - [martini.Router](http://godoc.org/github.com/go-martini/martini#Router)

### İşleyiciler (Handlers)
İşleyiciler Martini'nin ruhu ve kalbidir. Bir işleyici temel olarak her türlü fonksiyonu çağırabilir:

~~~ go
m.Get("/", func() {
  println("hello world")
})
~~~

#### Geriye Dönen Değerler

Eğer bir işleyici geriye bir şey dönderiyorsa, Martini string olarak sonucu [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter) ile yazacaktır:

~~~ go
m.Get("/", func() string {
  return "hello world" // HTTP 200 : "hello world"
})
~~~

Ayrıca isteğe bağlı bir durum kodu dönderebilir:
~~~ go
m.Get("/", func() (int, string) {
  return 418, "i'm a teapot" // HTTP 418 : "i'm a teapot"
})
~~~

#### Service Injection
İşlemciler yansıma yoluyla çağrılır. Martini *Dependency Injection* kullanarak arguman listesindeki bağımlıkları giderir.**Bu sayede Martini go programlama dilinin `http.HandlerFunc` arayüzü ile tamamen uyumlu hale getirilir.**

Eğer işleyiciye bir arguman eklersek, Martini "type assertion" ile servis listesinde arayacak ve bağımlılıkları çözmek için girişimde bulunacaktır:

~~~ go
m.Get("/", func(res http.ResponseWriter, req *http.Request) { // res and req are injected by Martini
  res.WriteHeader(200) // HTTP 200
})
~~~

Aşağıdaki servislerin içerikleri

[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic):
  * [*log.Logger](http://godoc.org/log#Logger) - Martini için Global loglayıcı.
  * [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) - http request içereği.
  * [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) - `map[string]string` ile yol eşleme tarafından params olarak isimlendirilen yapılar bulundu.
  * [martini.Routes](http://godoc.org/github.com/go-martini/martini#Routes) - Yönledirilme için yardımcı olan yapıdır.
  * [http.ResponseWriter](http://godoc.org/net/http/#ResponseWriter) - http yanıtlarını yazacak olan yapıdır.
  * [*http.Request](http://godoc.org/net/http/#Request) - http Request(http isteği yapar).

### Yönlendirme - Routing
Martini'de bir yol HTTP metodu URL-matching pattern'i ile eşleştirilir.
Her bir yol bir veya daha fazla işleyici metod alabilir:
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

Yollar sırayla tanımlandıkları şekilde eşleştirilir.Request ile eşleşen ilk rota çağrılır.

Yol patternleri [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) servisi tarafından adlandırılan parametreleri içerebilir:
~~~ go
m.Get("/hello/:name", func(params martini.Params) string {
  return "Hello " + params["name"]
})
~~~

Yollar globaller ile eşleşebilir:
~~~ go
m.Get("/hello/**", func(params martini.Params) string {
  return "Hello " + params["_1"]
})
~~~

Düzenli ifadeler kullanılabilir:
~~~go
m.Get("/hello/(?P<name>[a-zA-Z]+)", func(params martini.Params) string {
  return fmt.Sprintf ("Hello %s", params["name"])
})
~~~
Düzenli ifadeler hakkında daha fazla bilgiyi [Go dökümanlarından](http://golang.org/pkg/regexp/syntax/) elde edebilirsiniz.

Yol işleyicileri birbirlerinin üstüne istiflenebilir. Bu durum doğrulama ve yetkilendirme(authentication and authorization) işlemleri için iyi bir yöntemdir: 
~~~ go
m.Get("/secret", authorize, func() {
  // this will execute as long as authorize doesn't write a response
})
~~~

Yol grupları Grup metodlar kullanılarak eklenebilir.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
})
~~~

Tıpkı ara katmanların işleyiciler için bazı ara katman işlemlerini atlayabileceği gibi gruplar içinde atlayabilir.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
}, MyMiddleware1, MyMiddleware2)
~~~

### Servisler

Servisler işleyicilerin arguman listesine enjekte edilecek kullanılabilir nesnelerdir. İstenildiği taktirde bir servis *Global* ve *Request* seviyesinde eşlenebilir.

#### Global Eşleme - Global Mapping

Bir martini örneği(instance) projeye enjekte edilir. 
A Martini instance implements the inject.Enjekte arayüzü, çok kolay bir şekilde servis eşlemesi yapar:
~~~ go
db := &MyDatabase{}
m := martini.Classic()
m.Map(db) // the service will be available to all handlers as *MyDatabase
// ...
m.Run()
~~~

#### Request-Level Mapping
Request düzeyinde eşleme yapmak üzere işleyici [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) ile oluşturulabilir:
~~~ go
func MyCustomLoggerHandler(c martini.Context, req *http.Request) {
  logger := &MyCustomLogger{req}
  c.Map(logger) // mapped as *MyCustomLogger
}
~~~

#### Arayüz Eşleme Değerleri
Servisler hakkındaki en güçlü şeylerden birisi bir arabirim ile bir servis eşleşmektedir. Örneğin, istenirse [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter) yapısı paketlenmiş ve ekstra işlemleri gerçekleştirilen bir nesne ile  override edilebilir. Şu işleyici yazılabilir:

~~~ go
func WrapResponseWriter(res http.ResponseWriter, c martini.Context) {
  rw := NewSpecialResponseWriter(res)
  c.MapTo(rw, (*http.ResponseWriter)(nil)) // override ResponseWriter with our wrapper ResponseWriter
}
~~~

### Statik Dosyaların Sunumu

[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) örneği otomatik olarak statik dosyaları serverda root içinde yer alan "public" dizininden servis edilir.

Eğer istenirse daha fazla [martini.Static](http://godoc.org/github.com/go-martini/martini#Static) işleyicisi eklenerek daha fazla dizin servis edilebilir.
~~~ go
m.Use(martini.Static("assets")) // serve from the "assets" directory as well
~~~

#### Standart Dökümanların Sunulması - Serving a Default Document

Eğer istenilen URL bulunamaz ise özel bir URL dönderilebilir. Ayrıca bir dışlama(exclusion) ön eki ile bazı URL'ler göz ardı edilir. Bu durum statik dosyaların ve ilave işleyiciler için kullanışlıdır(Örneğin, REST API). Bunu yaparken, bu işlem ile NotFound zincirinin bir parçası olan statik işleyiciyi tanımlamak kolaydır.

Herhangi bir URL isteği bir local dosya ile eşleşmediği ve `/api/v` ile başlamadığı zaman aşağıdaki örnek `/index.html` dosyasını sonuç olarak geriye döndürecektir.
~~~ go
static := martini.Static("assets", martini.StaticOptions{Fallback: "/index.html", Exclude: "/api/v"})
m.NotFound(static, http.NotFound)
~~~

## Ara Katman İşleyicileri
Ara katmana ait işleyiciler http isteği ve yönlendirici arasında bulunmaktadır. Özünde onlar diğer Martini işleyicilerinden farklı değildirler. İstenildiği taktirde bir yığına ara katman işleyicisi şu şekilde eklenebilir:
~~~ go
m.Use(func() {
  // do some middleware stuff
})
~~~

`Handlers` fonksiyonu ile ara katman yığını üzerinde tüm kontrole sahip olunabilir. Bu daha önceden ayarlanmış herhangi bir işleyicinin yerini alacaktır:
~~~ go
m.Handlers(
  Middleware1,
  Middleware2,
  Middleware3,
)
~~~

Orta katman işleyicileri loglama, giriş , yetkilendirme , sessionlar, sıkıştırma(gzipping) , hata sayfaları ve HTTP isteklerinden önce ve sonra herhangi bir olay sonucu oluşan durumlar için gerçekten iyi bir yapıya sahiptir:

~~~ go
// validate an api key
m.Use(func(res http.ResponseWriter, req *http.Request) {
  if req.Header.Get("X-API-KEY") != "secret123" {
    res.WriteHeader(http.StatusUnauthorized)
  }
})
~~~

### Next()
[Context.Next()](http://godoc.org/github.com/go-martini/martini#Context) orta katman işleyicilerinin diğer işleyiciler yok edilmeden çağrılmasını sağlayan opsiyonel bir fonksiyondur.Bu iş http işlemlerinden sonra gerçekleşecek işlemler için gerçekten iyidir:
~~~ go
// log before and after a request
m.Use(func(c martini.Context, log *log.Logger){
  log.Println("before a request")

  c.Next()

  log.Println("after a request")
})
~~~

## Martini Env

Bazı Martini işleyicileri `martini.Env` yapısının özel fonksiyonlarını kullanmak için geliştirici ortamları, üretici ortamları vs. kullanır.Bu üretim ortamına Martini sunucu kurulurken `MARTINI_ENV=production` şeklinde ortam değişkeninin ayarlanması gerekir.

## FAQ

### Ara Katmanda X'i Nerede Bulurum?

[martini-contrib](https://github.com/martini-contrib) projelerine bakarak başlayın. Eğer aradığınız şey orada mevcut değil ise yeni bir repo eklemek için martini-contrib takım üyeleri ile iletişime geçin.

* [auth](https://github.com/martini-contrib/auth) - Kimlik doğrulama için işleyiciler.
* [binding](https://github.com/martini-contrib/binding) - Mapping/Validating yapısı içinde ham request'i doğrulamak için kullanılan işleyici(handler)
* [gzip](https://github.com/martini-contrib/gzip) - İstekleri gzip sıkışıtırıp eklemek için kullanılan işleyici
* [render](https://github.com/martini-contrib/render) - Kolay bir şekilde JSON ve HTML şablonları oluşturmak için kullanılan işleyici.
* [acceptlang](https://github.com/martini-contrib/acceptlang) - `Kabul edilen dile` göre HTTP başlığını oluşturmak için kullanılan işleyici.
* [sessions](https://github.com/martini-contrib/sessions) - Oturum hizmeti vermek için kullanılır.
* [strip](https://github.com/martini-contrib/strip) - İşleyicilere gitmeden önce URL'ye ait ön şeriti değiştirme işlemini yapar.
* [method](https://github.com/martini-contrib/method) - Formlar ve başlık için http metodunu override eder.
* [secure](https://github.com/martini-contrib/secure) - Birkaç hızlı güvenlik uygulaması ile kazanımda bulundurur.
* [encoder](https://github.com/martini-contrib/encoder) - Encoder servis veri işlemleri için çeşitli format ve içerik sağlar.
* [cors](https://github.com/martini-contrib/cors) - İşleyicilerin CORS desteği bulunur.
* [oauth2](https://github.com/martini-contrib/oauth2) - İşleyiciler OAuth 2.0 için Martini uygulamalarına giriş sağlar. Google , Facebook ve Github için desteği mevcuttur.
* [vauth](https://github.com/rafecolton/vauth) - Webhook için giriş izni sağlar. (şimdilik sadece GitHub ve TravisCI ile)

### Mevcut Sunucular ile Nasıl Entegre Edilir?

Bir martini örneği `http.Handler`'ı projeye dahil eder, bu sayde kolay bir şekilde mevcut olan Go sunucularında  bulunan alt ağaçlarda kullanabilir. Örnek olarak, bu olay Google App Engine için hazırlanmış Martini uygulamalarında kullanılmaktadır:

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

### port/hostu nasıl değiştiririm?

Martini'ye ait `Run` fonksiyounu PORT ve HOST'a ait ortam değişkenlerini arar ve bunları kullanır. Aksi taktirde standart olarak localhost:3000 adresini port ve host olarak kullanacaktır.

Port ve host için daha fazla esneklik isteniyorsa `martini.RunOnAddr` fonksiyonunu kullanın.

~~~ go
  m := martini.Classic()
  // ...
  log.Fatal(m.RunOnAddr(":8080"))
~~~

### Anlık Kod Yüklemesi?

[gin](https://github.com/codegangsta/gin) ve [fresh](https://github.com/pilu/fresh) anlık kod yüklemeleri yapan martini uygulamalarıdır.

## Katkıda Bulunmak
Martini'nin temiz ve düzenli olaması gerekiyordu. 
Martini is meant to be kept tiny and clean. Tüm kullanıcılar katkı yapmak için [martini-contrib](https://github.com/martini-contrib) organizasyonunda yer alan repoları bitirmelidirler. Eğer martini core için katkıda bulunacaksanız fork işlemini yaparak başlayabilirsiniz.

## Hakkında 

[express](https://github.com/visionmedia/express) ve [sinatra](https://github.com/sinatra/sinatra) projelerinden esinlenmiştir.

Martini [Code Gangsta](http://codegangsta.io/) tarafından tasarlanılmıştır.
