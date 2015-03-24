# Martini  [![wercker status](https://app.wercker.com/status/9b7dbc6e2654b604cd694d191c3d5487/s/master "wercker status")](https://app.wercker.com/project/bykey/9b7dbc6e2654b604cd694d191c3d5487)[![GoDoc](https://godoc.org/github.com/go-martini/martini?status.png)](http://godoc.org/github.com/go-martini/martini)

마티니(Martini)는 강력하고 손쉬운 웹애플리케이션 / 웹서비스개발을 위한 Golang 패키지입니다.

## 시작하기

Go 인스톨 및 [GOPATH](http://golang.org/doc/code.html#GOPATH) 환경변수 설정 이후에, `.go` 파일 하나를 만들어 보죠..흠... 일단 `server.go`라고 부르겠습니다.
~~~go
package main

import "github.com/go-martini/martini"

func main() {
  m := martini.Classic()
  m.Get("/", func() string {
    return "Hello, 세계!"
  })
  m.Run()
}
~~~

마티니 패키지를 인스톨 합니다. (**go 1.1** 혹은 그 이상 버젼 필요):
~~~
go get github.com/go-martini/martini
~~~

이제 서버를 돌려 봅시다:
~~~
go run server.go
~~~

마티니 웹서버가 `localhost:3000`에서 돌아가고 있는 것을 확인하실 수 있을 겁니다.

## 도움이 필요하다면?

[메일링 리스트](https://groups.google.com/forum/#!forum/martini-go)에 가입해 주세요

[데모 비디오](http://martini.codegangsta.io/#demo)도 있어요.

혹은 Stackoverflow에 [마티니 태크](http://stackoverflow.com/questions/tagged/martini)를 이용해서 물어봐 주세요

GoDoc [문서(documentation)](http://godoc.org/github.com/go-martini/martini)

문제는 전부다 영어로 되어 있다는 건데요 -_-;;;
나는 한글 아니면 보기 싫다! 하는 분들은 아래 링크를 참조하세요
- [golang-korea](https://code.google.com/p/golang-korea/)
- 혹은 ([RexK](http://github.com/RexK))의 이메일로 연락주세요.

## 주요기능
* 사용하기 엄청 쉽습니다.
* 비간섭(Non-intrusive) 디자인
* 다른 Golang 패키지들과 잘 어울립니다.
* 끝내주는 경로 매칭과 라우팅.
* 모듈 형 디자인 - 기능추가 쉽고, 코드 꺼내오기도 쉬움.
* 쓸모있는 핸들러와 미들웨어가 많음.
* 훌률한 패키지화(out of the box) 기능들
* **[http.HandlerFunc](http://godoc.org/net/http#HandlerFunc) 인터페이스와 호환율 100%**

## 미들웨어(Middleware)
미들웨어들과 추가기능들은 [martini-contrib](https://github.com/martini-contrib)에서 확인해 주세요.

## 목차
* [Classic Martini](#classic-martini)
  * [핸들러](#핸들러handlers)
  * [라우팅](#라우팅routing)
  * [서비스](#서비스services)
  * [정적파일 서빙](#정적파일-서빙serving-static-files)
* [미들웨어 핸들러](#미들웨어-핸들러middleware-handlers)
  * [Next()](#next)
* [Martini Env](#martini-env)
* [FAQ](#faq)

## Classic Martini
마티니를 쉽고 빠르게 이용하시려면, [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic)를 이용해 보세요. 보통 웹애플리케이션에서 사용하는 설정들이 이미 포함되어 있습니다.
~~~ go
  m := martini.Classic()
  // ... 미들웨어와 라우팅 설정은 이곳에 오면 작성하면 됩니다.
  m.Run()
~~~

아래는 [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic)의 자동으로 장착하는 기본 기능들입니다.

  * Request/Response 로그 기능 - [martini.Logger](http://godoc.org/github.com/go-martini/martini#Logger)
  * 패닉 리커버리 (Panic Recovery) - [martini.Recovery](http://godoc.org/github.com/go-martini/martini#Recovery)
  * 정적 파일 서빙 - [martini.Static](http://godoc.org/github.com/go-martini/martini#Static)
  * 라우팅(Routing) - [martini.Router](http://godoc.org/github.com/go-martini/martini#Router)

### 핸들러(Handlers)

핸들러(Handlers)는 마티니의 핵심입니다. 핸들러는 기본적으로 실행 가능한 모든형태의 함수들입니다.
~~~ go
m.Get("/", func() {
  println("hello 세계")
})
~~~

#### 반환 값 (Return Values)
핸들러가 반환을 하는 함수라면, 마티니는 반환 값을 [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter)에 스트링으로 입력 할 것입니다.
~~~ go
m.Get("/", func() string {
  return "hello 세계" // HTTP 200 : "hello 세계"
})
~~~

원하신다면, 선택적으로 상태코드도 함께 반화 할 수 있습니다.
~~~ go
m.Get("/", func() (int, string) {
  return 418, "난 주전자야!" // HTTP 418 : "난 주전자야!"
})
~~~

#### 서비스 주입(Service Injection)
핸들러들은 리플렉션을 통해 호출됩니다. 마티니는 *의존성 주입*을 이용해서 핸들러의 인수들을 주입합니다. **이것이 마티니를 `http.HandlerFunc` 인터페이스와 100% 호환할 수 있게 해줍니다.**

핸들러의 인수를 입력했다면, 마티니가 서비스 목록들을 살펴본 후 타입확인(type assertion)을 통해 의존성을 해결을 시도 할 것입니다.
~~~ go
m.Get("/", func(res http.ResponseWriter, req *http.Request) { // res와 req는 마티니에 의해 주입되었다.
  res.WriteHeader(200) // HTTP 200
})
~~~

아래 서비스들은 [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic):에 포함되어 있습니다.
  * [*log.Logger](http://godoc.org/log#Logger) - 마티니의 글러벌(전역) 로그.
  * [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) - http 요청 컨텍스트.
  * [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) - 루트 매칭으로 찾은 인자를 `map[string]string`으로 변형.
  * [martini.Routes](http://godoc.org/github.com/go-martini/martini#Routes) - 루트 도우미 서미스.
  * [http.ResponseWriter](http://godoc.org/net/http/#ResponseWriter) - http Response writer 인터페이스.
  * [*http.Request](http://godoc.org/net/http/#Request) - http 리퀘스트.

### 라우팅(Routing)
마티니에서 루트는 HTTP 메소드와 URL매칭 패턴의 페어입니다.
각 루트는 하나 혹은 그 이상의 핸들러 메소드를 가질 수 있습니다.
~~~ go
m.Get("/", func() {
  // 보여줘 봐
})

m.Patch("/", func() {
  // 업데이트 좀 해
})

m.Post("/", func() {
  // 만들어봐
})

m.Put("/", func() {
  // 교환해봐
})

m.Delete("/", func() {
  // 없애버려!
})

m.Options("/", func() {
  // http 옵션 메소드
})

m.NotFound(func() {
  // 404 해결하기
})
~~~

루트들은 정의된 순서대로 매칭된다. 들어온 요그에 첫번째 매칭된 루트가 호출된다.

루트 패턴은 [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) service로 액세스 가능한 인자들을 포함하기도 한다:
~~~ go
m.Get("/hello/:name", func(params martini.Params) string {
  return "Hello " + params["name"]			// :name을 Params인자에서 추출
})
~~~

루트는 별표식(\*)으로 매칭 될 수도 있습니다:
~~~ go
m.Get("/hello/**", func(params martini.Params) string {
  return "Hello " + params["_1"]
})
~~~

Regular expressions can be used as well:
정규식도 사용가능합니다:
~~~go
m.Get("/hello/(?P<name>[a-zA-Z]+)", func(params martini.Params) string {
  return fmt.Sprintf ("Hello %s", params["name"])
})
~~~
정규식에 관하여 더 자세히 알고 싶다면 [Go documentation](http://golang.org/pkg/regexp/syntax/)을 참조해 주세요.

루트 핸들러는 스택을 쌓아 올릴 수 있습니다. 특히 유저 인증작업이나, 허가작업에 유용히 쓰일 수 있죠.
~~~ go
m.Get("/secret", authorize, func() {
  // 이 함수는 authorize 함수가 resopnse에 결과를 쓰지 않는이상 실행 될 거에요.
})
~~~

루트그룹은 루트들을 한 곳에 모아 정리하는데 유용합니다.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
})
~~~

핸들러에 미들웨어를 집어넣을 수 있었듯이, 그룹에도 미들웨어 집어넣는게 가능합니다.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
}, MyMiddleware1, MyMiddleware2)
~~~

### 서비스(Services)
서비스는 핸들러의 인수목록에 주입될수 있는 오브젝트들을 말합니다. 서비스는 *글로벌* 혹은 *리퀘스트* 레벨단위로 주입이 가능합니다.

#### 글로벌 맵핑(Global Mapping)
마타니 인스턴스는 서비스 맵핑을 쉽게 하기 위해서 inject.Injector 인터페이스를 반형합니다:
~~~ go
db := &MyDatabase{}
m := martini.Classic()
m.Map(db) // 서비스가 모든 핸들러에서 *MyDatabase로서 사용될 수 있습니다.
// ...
m.Run()
~~~

#### 리퀘스트 레벨 맵핑(Request-Level Mapping)
리퀘스트 레벨 맵핑은 핸들러안에서 [martini.Context](http://godoc.org/github.com/go-martini/martini#Context)를 사용하면 됩니다:
~~~ go
func MyCustomLoggerHandler(c martini.Context, req *http.Request) {
  logger := &MyCustomLogger{req}
  c.Map(logger) // *MyCustomLogger로서 맵핑 됨
}
~~~

#### 인터페이스로 값들 맵핑(Mapping values to Interfaces)
서비스의 강력한 기능중 하나는 서비스를 인터페이스로 맵핑이 가능하다는 것입니다. 예를들어, [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter)를 치환(override)해서 부가 기능들을 수행하게 하고 싶으시다면, 아래와 같은 핸들러를 작성 하시면 됩니다.

~~~ go
func WrapResponseWriter(res http.ResponseWriter, c martini.Context) {
  rw := NewSpecialResponseWriter(res)
  c.MapTo(rw, (*http.ResponseWriter)(nil)) // ResponseWriter를 NewResponseWriter로 치환(override)
}
~~~

### 정적파일 서빙(Serving Static Files)
[martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) 인스턴스는 "public" 폴더안에 있는 파일들은 정적파일로서 자동으로 서빙합니다. 더 많은 폴더들은 정적파일 폴더에 포함시키시려면 [martini.Static](http://godoc.org/github.com/go-martini/martini#Static) 핸들러를 이용하시면 됩니다.

~~~ go
m.Use(martini.Static("assets")) // "assets" 폴더에서도 정적파일 서빙.
~~~

## 미들웨어 핸들러(Middleware Handlers)
미들웨어 핸들러는 http request와 라우팅 사이에서 작동합니다. 미들웨어 핸들러는 근본적으로 다른 핸들러들과는 다릅니다. 사용방법은 아래와 같습니다:
~~~ go
m.Use(func() {
  // 미들웨어 임무 수행!
})
~~~

`Handlers`를 이용하여 미들웨어 스택들의 완전 컨트롤이 가능합니다. 다만, 이렇게 설정하시면 이전에 `Handlers`를 이용하여 설정한 핸들러들은 사라집니다:
~~~ go
m.Handlers(
  Middleware1,
  Middleware2,
  Middleware3,
)
~~~

미들웨어 핸들러는 로깅(logging), 허가(authorization), 인가(authentication), 세션, 압축(gzipping), 에러 페이지 등 등, http request의 전후로 실행되어야 할 일들을 처리하기 아주 좋습니다:
~~~ go
// API 키 확인작업
m.Use(func(res http.ResponseWriter, req *http.Request) {
  if req.Header.Get("X-API-KEY") != "비밀암호!!!" {
    res.WriteHeader(http.StatusUnauthorized)	// HTTP 401
  }
})
~~~

### Next()
[Context.Next()](http://godoc.org/github.com/go-martini/martini#Context)는  선택적 함수입니다. 이 함수는 http request가 다 작동 될때까지 기다립니다.따라서 http request 이후에 실행 되어야 할 업무들을 수행하기 좋은 함수입니다.
~~~ go
// log before and after a request
m.Use(func(c martini.Context, log *log.Logger){
  log.Println("request전입니다.")

  c.Next()

  log.Println("request후 입니다.")
})
~~~

## Martini Env
마티니 핸들러들은 `martini.Env` 글로벌 변수를 사용하여 개발환경에서는 프로덕션 환경과는 다르게 작동하기도 합니다. 따라서, 프로덕션 서버로 마티니 서보를 배포하시게 된다면 꼭 환경변수 `MARTINI_ENV=production`를 세팅해주시기 바랍니다.

## FAQ

### 미들웨어들을 어디서 찾아야 하나요?

깃헙에서 [martini-contrib](https://github.com/martini-contrib) 프로젝트들을 탖아보세요. 만약에 못 찾으시겠으면, martini-contrib 팀원들에게 연락해서 하나 만들어 달라고 해보세요.
* [auth](https:	//github.com/martini-contrib/auth) - 인증작업을 도와주는 핸들러.
* [binding](https://github.com/martini-contrib/binding) - request를 맵핑하고 검사하는 핸들러.
* [gzip](https://github.com/martini-contrib/gzip) - gzip 핸들러.
* [render](https://github.com/martini-contrib/render) - HTML 템플레이트들과 JSON를 사용하기 편하게 해주는 핸들러.
* [acceptlang](https://github.com/martini-contrib/acceptlang) - `Accept-Language` HTTP 해더를 파싱할때 유용한 핸들러.
* [sessions](https://github.com/martini-contrib/sessions) - 세션 서비스를 제공하는 핸들러.
* [strip](https://github.com/martini-contrib/strip) - URL 프리틱스를 없애주는 핸들러.
* [method](https://github.com/martini-contrib/method) - 해더나 폼필드를 이용한 HTTP 메소드 치환.
* [secure](https://github.com/martini-contrib/secure) - 몇몇 보안설정을 위한 핸들러.
* [encoder](https://github.com/martini-contrib/encoder) - 데이터 렌더링과 컨텐트 타엽을위한 인코딩 서비스.
* [cors](https://github.com/martini-contrib/cors) - CORS 서포트를 위한 핸들러.
* [oauth2](https://github.com/martini-contrib/oauth2) - OAuth2.0 로그인 핸들러. 페이스북, 구글, 깃헙 지원.

### 현재 작동중인 서버에 마티니를 적용하려면?

마티니 인스턴스는 `http.Handler` 인터페이스를 차용합니다. 따라서 Go 서버 서브트리로 쉽게 사용될 수 있습니다. 아래 코드는 구글 앱 엔진에서 작동하는 마티니 앱입니다:

~~~ go
package hello

import (
  "net/http"
  "github.com/go-martini/martini"
)

func init() {
  m := martini.Classic()
  m.Get("/", func() string {
    return "Hello 세계!"
  })
  http.Handle("/", m)
}
~~~

### 포트와 호스트는 어떻게 바꾸나요?

마티니의 `Run` 함수는 PORT와 HOST 환경변수를 이용하는데, 설정이 안되어 있다면 localhost:3000으로 설정 되어 집니다.
좀더 유연하게 설정을 하고 싶다면, `martini.RunOnAddr`를 활용해 주세요.

~~~ go
  m := martini.Classic()
  // ...
  log.Fatal(m.RunOnAddr(":8080"))
~~~

### 라이브 포드 리로드?

[gin](https://github.com/codegangsta/gin) and [fresh](https://github.com/pilu/fresh) 마티니 앱의 라이브 리로드를 도와줍니다.

## 공헌하기(Contributing)

마티니는 간단하고 가벼운 패키지로 남을 것입니다. 따라서 보통 대부분의 공헌들은 [martini-contrib](https://github.com/martini-contrib) 그룹의 저장소로 가게 됩니다. 만약 마티니 코어에 기여하고 싶으시다면 주저없이 Pull Request를 해주세요.

## About

[express](https://github.com/visionmedia/express) 와 [sinatra](https://github.com/sinatra/sinatra)의 영향을 받았습니다.

마티니는 [Code Gangsta](http://codegangsta.io/)가 디자인 했습니다.
