# Martini  [![wercker status](https://app.wercker.com/status/9b7dbc6e2654b604cd694d191c3d5487/s/master "wercker status")](https://app.wercker.com/project/bykey/9b7dbc6e2654b604cd694d191c3d5487)[![GoDoc](https://godoc.org/github.com/go-martini/martini?status.png)](http://godoc.org/github.com/go-martini/martini)

Martini é um poderoso pacote para escrever aplicações/serviços modulares em Golang..


## Vamos começar

Após a instalação do Go e de configurar o [GOPATH](http://golang.org/doc/code.html#GOPATH), crie seu primeiro arquivo `.go`. Vamos chamá-lo de `server.go`.

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

Então instale o pacote do Martini (É necessário **go 1.1** ou superior):
~~~
go get github.com/go-martini/martini
~~~

Então rode o servidor:
~~~
go run server.go
~~~

Agora você tem um webserver Martini rodando na porta `localhost:3000`.

## Obtenha ajuda

Assine a [Lista de email](https://groups.google.com/forum/#!forum/martini-go)

Veja o [Vídeo demonstrativo](http://martini.codegangsta.io/#demo)

Use a tag [martini](http://stackoverflow.com/questions/tagged/martini) para perguntas no Stackoverflow



## Caracteríticas
* Extrema simplicidade de uso.
* Design não intrusivo.
* Boa integração com outros pacotes Golang.
* Router impressionante.
* Design modular - Fácil para adicionar e remover funcionalidades.
* Muito bom no uso handlers/middlewares.
* Grandes caracteríticas inovadoras.
* **Completa compatibilidade com a interface [http.HandlerFunc](http://godoc.org/net/http#HandlerFunc).**

## Mais Middleware
Para mais middleware e funcionalidades, veja os repositórios em [martini-contrib](https://github.com/martini-contrib).

## Tabela de Conteudos
* [Classic Martini](#classic-martini)
  * [Handlers](#handlers)
  * [Routing](#routing)
  * [Services](#services)
  * [Serving Static Files](#serving-static-files)
* [Middleware Handlers](#middleware-handlers)
  * [Next()](#next)
* [Martini Env](#martini-env)
* [FAQ](#faq)

## Classic Martini
Para iniciar rapidamente, [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) provê algumas ferramentas razoáveis para maioria das aplicações web:
~~~ go
  m := martini.Classic()
  // ... middleware e rota aqui
  m.Run()
~~~

Algumas das funcionalidade que o [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) oferece automaticamente são:
  * Request/Response Logging - [martini.Logger](http://godoc.org/github.com/go-martini/martini#Logger)
  * Panic Recovery - [martini.Recovery](http://godoc.org/github.com/go-martini/martini#Recovery)
  * Servidor de arquivos státicos - [martini.Static](http://godoc.org/github.com/go-martini/martini#Static)
  * Rotas - [martini.Router](http://godoc.org/github.com/go-martini/martini#Router)

### Handlers
Handlers são o coração e a alma do Martini. Um handler é basicamente qualquer função que pode ser chamada:
~~~ go
m.Get("/", func() {
  println("hello world")
})
~~~

#### Retorno de Valores
Se um handler retornar alguma coisa, Martini irá escrever o valor retornado como uma string ao [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter):
~~~ go
m.Get("/", func() string {
  return "hello world" // HTTP 200 : "hello world"
})
~~~

Você também pode retornar o código de status:
~~~ go
m.Get("/", func() (int, string) {
  return 418, "Eu sou um bule" // HTTP 418 : "Eu sou um bule"
})
~~~

#### Injeção de Serviços
Handlers são chamados via reflexão. Martini utiliza *Injeção de Dependencia* para resolver as dependencias nas listas de argumentos dos Handlers . **Isso faz Martini ser completamente compatível com a interface `http.HandlerFunc` do golang.**

Se você adicionar um argumento ao seu Handler, Martini ira procurar na sua lista de serviços e tentar resolver sua dependencia pelo seu tipo:
~~~ go
m.Get("/", func(res http.ResponseWriter, req *http.Request) { // res e req são injetados pelo Martini
  res.WriteHeader(200) // HTTP 200
})
~~~

Os seguintes serviços são incluídos com [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic):
  * [*log.Logger](http://godoc.org/log#Logger) - Log Global para Martini.
  * [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) - http request context.
  * [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) - `map[string]string` de nomes dos parâmetros buscados pela rota.
  * [martini.Routes](http://godoc.org/github.com/go-martini/martini#Routes) - Serviço de auxílio as rotas.
  * [http.ResponseWriter](http://godoc.org/net/http/#ResponseWriter) - http Response escreve a interface.
  * [*http.Request](http://godoc.org/net/http/#Request) - http Request.

### Rotas
No Martini, uma rota é um método HTTP emparelhado com um padrão de URL de correspondência.
Cada rota pode ter um ou mais métodos handler:
~~~ go
m.Get("/", func() {
  // mostra alguma coisa
})

m.Patch("/", func() {
  // altera alguma coisa
})

m.Post("/", func() {
  // cria alguma coisa
})

m.Put("/", func() {
  // sobrescreve alguma coisa
})

m.Delete("/", func() {
  // destrói alguma coisa
})

m.Options("/", func() {
  // opções do HTTP
})

m.NotFound(func() {
  // manipula 404
})
~~~

As rotas são combinadas na ordem em que são definidas. A primeira rota que corresponde a solicitação é chamada.

O padrão de rotas pode incluir parâmetros que podem ser acessados via [martini.Params](http://godoc.org/github.com/go-martini/martini#Params):
~~~ go
m.Get("/hello/:name", func(params martini.Params) string {
  return "Hello " + params["name"]
})
~~~

As rotas podem ser combinados com expressões regulares e globs:
~~~ go
m.Get("/hello/**", func(params martini.Params) string {
  return "Hello " + params["_1"]
})
~~~

Expressões regulares podem ser bem usadas:
~~~go
m.Get("/hello/(?P<name>[a-zA-Z]+)", func(params martini.Params) string {
  return fmt.Sprintf ("Hello %s", params["name"])
})
~~~
Dê uma olhada na [documentação](http://golang.org/pkg/regexp/syntax/) para mais informações sobre expressões regulares.


Handlers de rota podem ser empilhados em cima uns dos outros, o que é útil para coisas como autenticação e autorização:
~~~ go
m.Get("/secret", authorize, func() {
  // Será executado quando authorize não escrever uma resposta
})
~~~

Grupos de rota podem ser adicionados usando o método Group.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
})
~~~

Assim como você pode passar middlewares para um manipulador você pode passar middlewares para grupos.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
}, MyMiddleware1, MyMiddleware2)
~~~

### Serviços
Serviços são objetos que estão disponíveis para ser injetado em uma lista de argumentos de Handler. Você pode mapear um serviço num nível *Global* ou *Request*.

#### Mapeamento Global
Um exemplo onde o Martini implementa a interface inject.Injector, então o mapeamento de um serviço é fácil:
~~~ go
db := &MyDatabase{}
m := martini.Classic()
m.Map(db) // o serviço estará disponível para todos os handlers *MyDatabase.
// ...
m.Run()
~~~

#### Mapeamento por requisição
Mapeamento do nível de request pode ser feito via handler através [martini.Context](http://godoc.org/github.com/go-martini/martini#Context):
~~~ go
func MyCustomLoggerHandler(c martini.Context, req *http.Request) {
  logger := &MyCustomLogger{req}
  c.Map(logger) // mapeamento é *MyCustomLogger
}
~~~

#### Valores de Mapeamento para Interfaces
Uma das partes mais poderosas sobre os serviços é a capacidade para mapear um serviço de uma interface. Por exemplo, se você quiser substituir o [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter) com um objeto que envolveu-o e realizou operações extras, você pode escrever o seguinte handler:
~~~ go
func WrapResponseWriter(res http.ResponseWriter, c martini.Context) {
  rw := NewSpecialResponseWriter(res)
  c.MapTo(rw, (*http.ResponseWriter)(nil)) // substituir ResponseWriter com nosso ResponseWriter invólucro
}
~~~

### Servindo Arquivos Estáticos
Uma instância de [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) serve automaticamente arquivos estáticos do diretório "public" na raiz do seu servidor.
Você pode servir de mais diretórios, adicionando mais [martini.Static](http://godoc.org/github.com/go-martini/martini#Static) handlers.
~~~ go
m.Use(martini.Static("assets")) // servindo os arquivos do diretório "assets"
~~~

## Middleware Handlers
Middleware Handlers ficam entre a solicitação HTTP e o roteador. Em essência, eles não são diferentes de qualquer outro Handler no Martini. Você pode adicionar um handler de middleware para a pilha assim:
~~~ go
m.Use(func() {
  // faz algo com middleware
})
~~~

Você pode ter o controle total sobre a pilha de middleware com a função `Handlers`. Isso irá substituir quaisquer manipuladores que foram previamente definidos:
~~~ go
m.Handlers(
  Middleware1,
  Middleware2,
  Middleware3,
)
~~~

Middleware Handlers trabalham muito bem com princípios com logging, autorização, autenticação, sessão, gzipping, páginas de erros e uma série de outras operações que devem acontecer antes ou depois de uma solicitação HTTP:
~~~ go
// Valida uma chave de API
m.Use(func(res http.ResponseWriter, req *http.Request) {
  if req.Header.Get("X-API-KEY") != "secret123" {
    res.WriteHeader(http.StatusUnauthorized)
  }
})
~~~

### Next()
[Context.Next()](http://godoc.org/github.com/go-martini/martini#Context) é uma função opcional que Middleware Handlers podem chamar para aguardar a execução de outros Handlers. Isso funciona muito bem para operações que devem acontecer após uma requisição:
~~~ go
// log antes e depois do request
m.Use(func(c martini.Context, log *log.Logger){
  log.Println("antes do request")

  c.Next()

  log.Println("depois do request")
})
~~~

## Martini Env

Martini handlers fazem uso do `martini.Env`, uma variável global para fornecer funcionalidade especial para ambientes de desenvolvimento e ambientes de produção. É recomendado que a variável `MARTINI_ENV=production` seja definida quando a implementação estiver em um ambiente de produção.

## FAQ

### Onde posso encontrar o middleware X?

Inicie sua busca nos projetos [martini-contrib](https://github.com/martini-contrib). Se ele não estiver lá não hesite em contactar um membro da equipe martini-contrib sobre como adicionar um novo repo para a organização.

* [auth](https://github.com/martini-contrib/auth) - Handlers para autenticação.
* [binding](https://github.com/martini-contrib/binding) - Handler para mapeamento/validação de um request a estrutura.
* [gzip](https://github.com/martini-contrib/gzip) - Handler para adicionar compreção gzip para o requests
* [render](https://github.com/martini-contrib/render) - Handler que providencia uma rederização simples para JSON e templates HTML.
* [acceptlang](https://github.com/martini-contrib/acceptlang) - Handler para parsing do `Accept-Language` no header HTTP.
* [sessions](https://github.com/martini-contrib/sessions) - Handler que prove o serviço de sessão.
* [strip](https://github.com/martini-contrib/strip) - URL Prefix stripping.
* [method](https://github.com/martini-contrib/method) - HTTP método de substituição via cabeçalho ou campos do formulário.
* [secure](https://github.com/martini-contrib/secure) - Implementa rapidamente itens de segurança.
* [encoder](https://github.com/martini-contrib/encoder) - Serviço Encoder para renderização de dados em vários formatos e negociação de conteúdo.
* [cors](https://github.com/martini-contrib/cors) - Handler que habilita suporte a CORS.
* [oauth2](https://github.com/martini-contrib/oauth2) - Handler que prove sistema de login OAuth 2.0 para aplicações Martini. Google Sign-in, Facebook Connect e Github login são suportados.

### Como faço para integrar com os servidores existentes?

Uma instância do Martini implementa `http.Handler`, de modo que pode ser facilmente utilizado para servir sub-rotas e diretórios
em servidores Go existentes. Por exemplo, este é um aplicativo Martini trabalhando para Google App Engine:

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

### Como faço para alterar a porta/host?

A função `Run` do Martini olha para as variáveis PORT e HOST para utilizá-las. Caso contrário o Martini assume como padrão localhost:3000.
Para ter mais flexibilidade sobre a porta e host use a função `martini.RunOnAddr`.

~~~ go
  m := martini.Classic()
  // ...
  log.Fatal(m.RunOnAddr(":8080"))
~~~

### Servidor com autoreload?

[gin](https://github.com/codegangsta/gin) e [fresh](https://github.com/pilu/fresh) são aplicativos para autoreload do Martini.

## Contribuindo
Martini é feito para ser mantido pequeno e limpo. A maioria das contribuições devem ser feitas no repositório [martini-contrib](https://github.com/martini-contrib). Se quiser contribuir com o core do Martini fique livre para fazer um Pull Request.

## Sobre

Inspirado por [express](https://github.com/visionmedia/express) e [sinatra](https://github.com/sinatra/sinatra)

Martini is obsessively designed by none other than the [Code Gangsta](http://codegangsta.io/)
