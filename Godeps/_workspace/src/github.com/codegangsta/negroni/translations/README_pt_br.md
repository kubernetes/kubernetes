# Negroni [![GoDoc](https://godoc.org/github.com/codegangsta/negroni?status.svg)](http://godoc.org/github.com/codegangsta/negroni) [![wercker status](https://app.wercker.com/status/13688a4a94b82d84a0b8d038c4965b61/s "wercker status")](https://app.wercker.com/project/bykey/13688a4a94b82d84a0b8d038c4965b61)

Negroni é uma abordagem idiomática para middleware web em Go. É pequeno, não intrusivo, e incentiva uso da biblioteca `net/http`.

Se gosta da idéia do [Martini](http://github.com/go-martini/martini), mas acha que contém muita mágica, então Negroni é ideal.

## Começando

Depois de instalar Go e definir seu [GOPATH](http://golang.org/doc/code.html#GOPATH), criar seu primeirto arquivo `.go`. Iremos chamá-lo `server.go`.

~~~ go
package main

import (
  "github.com/codegangsta/negroni"
  "net/http"
  "fmt"
)

func main() {
  mux := http.NewServeMux()
  mux.HandleFunc("/", func(w http.ResponseWriter, req *http.Request) {
    fmt.Fprintf(w, "Welcome to the home page!")
  })

  n := negroni.Classic()
  n.UseHandler(mux)
  n.Run(":3000")
}
~~~

Depois instale o pacote Negroni (**go 1.1** ou superior)
~~~
go get github.com/codegangsta/negroni
~~~

Depois execute seu servidor:
~~~
go run server.go
~~~

Agora terá um servidor web Go net/http rodando em `localhost:3000`.

## Precisa de Ajuda?
Se você tem uma pergunta ou pedido de recurso,[go ask the mailing list](https://groups.google.com/forum/#!forum/negroni-users). O Github issuses para o Negroni será usado exclusivamente para Reportar bugs e pull requests.

## Negroni é um Framework?
Negroni **não** é a framework. É uma biblioteca que é desenhada para trabalhar diretamente com net/http.

## Roteamento?
Negroni é TSPR(Traga seu próprio Roteamento). A comunidade Go já tem um grande número de roteadores http disponíveis, Negroni tenta rodar bem com todos eles pelo suporte total `net/http`/ Por exemplo, a integração com [Gorilla Mux](http://github.com/gorilla/mux) se parece com isso:

~~~ go
router := mux.NewRouter()
router.HandleFunc("/", HomeHandler)

n := negroni.New(Middleware1, Middleware2)
// Or use a middleware with the Use() function
n.Use(Middleware3)
// router goes last
n.UseHandler(router)

n.Run(":3000")
~~~

## `negroni.Classic()`
`negroni.Classic()`  fornece alguns middlewares padrão que são útil para maioria das aplicações:

* `negroni.Recovery` - Panic Recovery Middleware.
* `negroni.Logging` - Request/Response Logging Middleware.
* `negroni.Static` - Static File serving under the "public" directory.

Isso torna muito fácil começar com alguns recursos úteis do Negroni.

## Handlers
Negroni fornece um middleware de fluxo bidirecional. Isso é feito através da interface `negroni.Handler`:

~~~ go
type Handler interface {
  ServeHTTP(rw http.ResponseWriter, r *http.Request, next http.HandlerFunc)
}
~~~

Se um middleware não tenha escrito o ResponseWriter, ele deve chamar a próxima `http.HandlerFunc` na cadeia para produzir o próximo handler middleware. Isso pode ser usado muito bem:

~~~ go
func MyMiddleware(rw http.ResponseWriter, r *http.Request, next http.HandlerFunc) {
  // do some stuff before
  next(rw, r)
  // do some stuff after
}
~~~

E pode mapear isso para a cadeia de handler com a função `Use`:

~~~ go
n := negroni.New()
n.Use(negroni.HandlerFunc(MyMiddleware))
~~~

Você também pode mapear `http.Handler` antigos:

~~~ go
n := negroni.New()

mux := http.NewServeMux()
// map your routes

n.UseHandler(mux)

n.Run(":3000")
~~~

## `Run()`
Negroni tem uma função de conveniência chamada `Run`. `Run` pega um endereço de string idêntico para [http.ListenAndServe](http://golang.org/pkg/net/http#ListenAndServe).

~~~ go
n := negroni.Classic()
// ...
log.Fatal(http.ListenAndServe(":8080", n))
~~~

## Middleware para Rotas Específicas
Se você tem um grupo de rota com rotas que precisam ser executadas por um middleware específico, pode simplesmente criar uma nova instância de Negroni e usar no seu Manipular de rota.

~~~ go
router := mux.NewRouter()
adminRoutes := mux.NewRouter()
// add admin routes here

// Criar um middleware negroni para admin
router.Handle("/admin", negroni.New(
  Middleware1,
  Middleware2,
  negroni.Wrap(adminRoutes),
))
~~~

## Middleware de Terceiros

Aqui está uma lista atual de Middleware Compatíveis com Negroni. Sinta se livre para mandar um PR vinculando seu middleware if construiu um:


| Middleware | Autor | Descrição |
| -----------|--------|-------------|
| [Graceful](https://github.com/stretchr/graceful) | [Tyler Bunnell](https://github.com/tylerb) | Graceful HTTP Shutdown |
| [secure](https://github.com/unrolled/secure) | [Cory Jacobsen](https://github.com/unrolled) |  Implementa rapidamente itens de segurança.|
| [binding](https://github.com/mholt/binding) | [Matt Holt](https://github.com/mholt) | Handler para mapeamento/validação de um request a estrutura. |
| [logrus](https://github.com/meatballhat/negroni-logrus) | [Dan Buch](https://github.com/meatballhat) | Logrus-based logger |
| [render](https://github.com/unrolled/render) | [Cory Jacobsen](https://github.com/unrolled) | Pacote para renderizar JSON, XML, e templates HTML. |
| [gorelic](https://github.com/jingweno/negroni-gorelic) | [Jingwen Owen Ou](https://github.com/jingweno) | New Relic agent for Go runtime |
| [gzip](https://github.com/phyber/negroni-gzip) | [phyber](https://github.com/phyber) | Handler para adicionar compreção gzip para as requisições |
| [oauth2](https://github.com/goincremental/negroni-oauth2) | [David Bochenski](https://github.com/bochenski) | Handler que prove sistema de login OAuth 2.0 para aplicações Martini. Google Sign-in, Facebook Connect e Github login são suportados. |
| [sessions](https://github.com/goincremental/negroni-sessions) | [David Bochenski](https://github.com/bochenski) | Handler que provê o serviço de sessão. |
| [permissions](https://github.com/xyproto/permissions) | [Alexander Rødseth](https://github.com/xyproto) | Cookies, usuários e permissões. |
| [onthefly](https://github.com/xyproto/onthefly) | [Alexander Rødseth](https://github.com/xyproto) | Pacote para gerar TinySVG, HTML e CSS em tempo real. |

## Exemplos
[Alexander Rødseth](https://github.com/xyproto) criou [mooseware](https://github.com/xyproto/mooseware), uma estrutura para escrever um handler middleware Negroni.

## Servidor com autoreload?
[gin](https://github.com/codegangsta/gin) e [fresh](https://github.com/pilu/fresh) são aplicativos para autoreload do Negroni.

## Sobre
Negroni é obsessivamente desenhado por ninguém menos que  [Code Gangsta](http://codegangsta.io/)
