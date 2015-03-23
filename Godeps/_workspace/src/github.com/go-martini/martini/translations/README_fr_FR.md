# Martini  [![wercker status](https://app.wercker.com/status/9b7dbc6e2654b604cd694d191c3d5487/s/master "wercker status")](https://app.wercker.com/project/bykey/9b7dbc6e2654b604cd694d191c3d5487)[![GoDoc](https://godoc.org/github.com/go-martini/martini?status.png)](http://godoc.org/github.com/go-martini/martini)

Martini est une puissante bibliothèque pour développer rapidement des applications et services web en Golang.


## Pour commencer

Après avoir installé Go et configuré le chemin d'accès pour [GOPATH](http://golang.org/doc/code.html#GOPATH), créez votre premier fichier '.go'. Nous l'appellerons 'server.go'.

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

Installez ensuite le paquet Martini (**go 1.1** ou supérieur est requis) :

~~~
go get github.com/go-martini/martini
~~~

La commande suivante vous permettra de lancer votre serveur :
~~~
go run server.go
~~~

Vous avez à présent un serveur web Martini à l'écoute sur l'adresse et le port suivants : `localhost:3000`.

## Besoin d'aide
- Souscrivez à la [Liste d'emails](https://groups.google.com/forum/#!forum/martini-go)
- Regardez les vidéos [Demo en vidéo](http://martini.codegangsta.io/#demo)
- Posez vos questions sur StackOverflow.com en utilisant le tag [martini](http://stackoverflow.com/questions/tagged/martini)
- La documentation GoDoc [documentation](http://godoc.org/github.com/go-martini/martini)


## Caractéristiques
* Simple d'utilisation
* Design non-intrusif
* Compatible avec les autres paquets Golang
* Gestionnaire d'URL et routeur disponibles
* Modulable, permettant l'ajout et le retrait de fonctionnalités
* Un grand nombre de handlers/middlewares disponibles
* Prêt pour une utilisation immédiate
* **Entièrement compatible avec l'interface [http.HandlerFunc](http://godoc.org/net/http#HandlerFunc).**

## Plus de Middleware
Pour plus de middlewares et de fonctionnalités, consultez le dépôt [martini-contrib](https://github.com/martini-contrib).

## Table des matières
* [Classic Martini](#classic-martini)
  * [Handlers](#handlers)
  * [Routage](#routing)
  * [Services](#services)
  * [Serveur de fichiers statiques](#serving-static-files)
* [Middleware Handlers](#middleware-handlers)
  * [Next()](#next)
* [Martini Env](#martini-env)
* [FAQ](#faq)

## Classic Martini
Pour vous faire gagner un temps précieux, [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) est configuré avec des paramètres qui devraient couvrir les besoins de la plupart des applications web :

~~~ go
  m := martini.Classic()
  // ... les middlewares and le routage sont insérés ici...
  m.Run()
~~~

Voici quelques handlers/middlewares que [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) intègre par défault :
  * Logging des requêtes/réponses - [martini.Logger](http://godoc.org/github.com/go-martini/martini#Logger)
  * Récupération sur erreur - [martini.Recovery](http://godoc.org/github.com/go-martini/martini#Recovery)
  * Serveur de fichiers statiques - [martini.Static](http://godoc.org/github.com/go-martini/martini#Static)
  * Routage - [martini.Router](http://godoc.org/github.com/go-martini/martini#Router)

### Handlers
Les Handlers sont le coeur et l'âme de Martini. N'importe quelle fonction peut être utilisée comme un handler.
~~~ go
m.Get("/", func() {
  println("hello world")
})
~~~

#### Valeurs retournées
Si un handler retourne une valeur, Martini écrira le résultat dans l'instance [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter) courante sous forme de ```string```:
~~~ go
m.Get("/", func() string {
  return "hello world" // HTTP 200 : "hello world"
})
~~~
Vous pouvez aussi optionnellement renvoyer un code de statut HTTP :
~~~ go
m.Get("/", func() (int, string) {
  return 418, "i'm a teapot" // HTTP 418 : "i'm a teapot"
})
~~~

#### Injection de services
Les handlers sont appelés via réflexion. Martini utilise "l'injection par dépendance" pour résoudre les dépendances des handlers dans la liste d'arguments. **Cela permet à Martini d'être parfaitement compatible avec l'interface golang ```http.HandlerFunc```.**

Si vous ajoutez un argument à votre Handler, Martini parcourera la liste des services et essayera de déterminer ses dépendances selon son type :
~~~ go
m.Get("/", func(res http.ResponseWriter, req *http.Request) { // res and req are injected by Martini
  res.WriteHeader(200) // HTTP 200
})
~~~
Les services suivants sont inclus avec [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic):
  * [log.Logger](http://godoc.org/log#Logger) - Global logger for Martini.
  * [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) - Contexte d'une requête HTTP.
  * [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) - `map[string]string` contenant les paramètres retrouvés par correspondance des routes.
  * [martini.Routes](http://godoc.org/github.com/go-martini/martini#Routes) - Service d'aide au routage.
  * [http.ResponseWriter](http://godoc.org/net/http/#ResponseWriter) - Interface d'écriture de réponses HTTP.
  * [*http.Request](http://godoc.org/net/http/#Request) - Requête HTTP.

### Routeur
Dans Martini, un chemin est une méthode HTTP liée à un modèle d'adresse URL.
Chaque chemin peut avoir un seul ou plusieurs méthodes *handler* :
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
Les chemins seront traités dans l'ordre dans lequel ils auront été définis. Le *handler* du premier chemin trouvé qui correspondra à la requête sera invoqué.


Les chemins peuvent inclure des paramètres nommés, accessibles avec le service [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) :
~~~ go
m.Get("/hello/:name", func(params martini.Params) string {
  return "Hello " + params["name"]
})
~~~

Les chemins peuvent correspondre à des globs :
~~~ go
m.Get("/hello/**", func(params martini.Params) string {
  return "Hello " + params["_1"]
})
~~~
Les expressions régulières peuvent aussi être utilisées :
~~~go
m.Get("/hello/(?P<name>[a-zA-Z]+)", func(params martini.Params) string {
  return fmt.Sprintf ("Hello %s", params["name"])
})
~~~
Jetez un oeil à la documentation [Go documentation](http://golang.org/pkg/regexp/syntax/) pour plus d'informations sur la syntaxe des expressions régulières.

Les handlers d'un chemins peuvent être superposés, ce qui s'avère particulièrement pratique pour des tâches comme la gestion de l'authentification et des autorisations :
~~~ go
m.Get("/secret", authorize, func() {
  // this will execute as long as authorize doesn't write a response
})
~~~

Un groupe de chemins peut aussi être ajouté en utilisant la méthode ```Group``` :
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
})
~~~

Comme vous pouvez passer des middlewares à un handler, vous pouvez également passer des middlewares à des groupes :
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
}, MyMiddleware1, MyMiddleware2)
~~~

### Services
Les services sont des objets injectés dans la liste d'arguments d'un handler. Un service peut être défini pour une *requête*, ou de manière *globale*.


#### Global Mapping
Les instances Martini implémentent l'interace inject.Injector, ce qui facilite grandement le mapping de services :
~~~ go
db := &MyDatabase{}
m := martini.Classic()
m.Map(db) // the service will be available to all handlers as *MyDatabase
// ...
m.Run()
~~~

#### Requête-Level Mapping
Pour une déclaration au niveau d'une requête, il suffit d'utiliser un handler via [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) :
~~~ go
func MyCustomLoggerHandler(c martini.Context, req *http.Request) {
  logger := &MyCustomLogger{req}
  c.Map(logger) // mapped as *MyCustomLogger
}
~~~

#### Mapping de valeurs à des interfaces
L'un des aspects les plus intéressants des services réside dans le fait qu'ils peuvent être liés à des interfaces. Par exemple, pour surcharger [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter) avec un objet qui l'enveloppe et étend ses fonctionnalités, vous pouvez utiliser le handler suivant :
~~~ go
func WrapResponseWriter(res http.ResponseWriter, c martini.Context) {
  rw := NewSpecialResponseWriter(res)
  c.MapTo(rw, (*http.ResponseWriter)(nil)) // override ResponseWriter with our wrapper ResponseWriter
}
~~~

### Serveur de fichiers statiques
Une instance [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) est déjà capable de servir les fichiers statiques qu'elle trouvera dans le dossier *public* à la racine de votre serveur.
Vous pouvez indiquer d'autres dossiers sources à l'aide du handler  [martini.Static](http://godoc.org/github.com/go-martini/martini#Static).
~~~ go
m.Use(martini.Static("assets")) // serve from the "assets" directory as well
~~~

## Les middleware Handlers
Les *middleware handlers* sont placés entre la requête HTTP entrante et le routeur. Ils ne sont aucunement différents des autres handlers présents dans Martini. Vous pouvez ajouter un middleware handler comme ceci :
~~~ go
m.Use(func() {
  // do some middleware stuff
})
~~~
Vous avez un contrôle total sur la structure middleware avec la fonction ```Handlers```. Son exécution écrasera tous les handlers configurés précédemment :
~~~ go
m.Handlers(
  Middleware1,
  Middleware2,
  Middleware3,
)
~~~
Middleware Handlers est très pratique pour automatiser des fonctions comme le logging, l'autorisation, l'authentification, sessions, gzipping, pages d'erreur, et toutes les opérations qui se font avant ou après chaque requête HTTP :
~~~ go
// validate an api key
m.Use(func(res http.ResponseWriter, req *http.Request) {
  if req.Header.Get("X-API-KEY") != "secret123" {
    res.WriteHeader(http.StatusUnauthorized)
  }
})
~~~

### Next() (Suivant)
[Context.Next()](http://godoc.org/github.com/go-martini/martini#Context) est une fonction optionnelle que les Middleware Handlers peuvent appeler pour patienter jusqu'à ce que tous les autres handlers aient été exécutés. Cela fonctionne très bien pour toutes opérations qui interviennent après une requête HTTP :
~~~ go
// log before and after a request
m.Use(func(c martini.Context, log *log.Logger){
  log.Println("avant la requête")

  c.Next()

  log.Println("après la requête")
})
~~~

## Martini Env
Plusieurs Martini handlers utilisent 'martini.Env' comme variable globale pour fournir des fonctionnalités particulières qui diffèrent entre l'environnement de développement et l'environnement de production. Il est recommandé que la variable 'MARTINI_ENV=production' soit définie pour déployer un serveur Martini en environnement de production.

## FAQ (Foire aux questions)

### Où puis-je trouver des middleware ?
Commencer par regarder dans le [martini-contrib](https://github.com/martini-contrib) projet. S'il n'y est pas, n'hésitez pas à contacter un membre de l'équipe martini-contrib pour ajouter un nouveau dépôt à l'organisation.

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
* [vauth](https://github.com/rafecolton/vauth) - Handlers for vender webhook authentication (currently GitHub and TravisCI)

### Comment puis-je m'intègrer avec des serveurs existants ?
Une instance Martini implémente ```http.Handler```. Elle peut donc utilisée pour alimenter des sous-arbres sur des serveurs Go existants. Voici l'exemple d'une application Martini pour Google App Engine :

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

### Comment changer le port/adresse ?

La fonction ```Run``` de Martini utilise le port et l'adresse spécifiés dans les variables d'environnement. Si elles ne peuvent y être trouvées, Martini utilisera *localhost:3000* par default.
Pour avoir plus de flexibilité sur le port et l'adresse, utilisez la fonction `martini.RunOnAddr` à la place.

~~~ go
  m := martini.Classic()
  // ...
  log.Fatal(m.RunOnAddr(":8080"))
~~~

### Rechargement du code en direct ?

[gin](https://github.com/codegangsta/gin) et [fresh](https://github.com/pilu/fresh) sont tous les capables de recharger le code des applications martini chaque fois qu'il est modifié.

## Contribuer
Martini est destiné à rester restreint et épuré. Toutes les contributions doivent finir dans un dépot dans l'organisation [martini-contrib](https://github.com/martini-contrib). Si vous avez une contribution pour le noyau de Martini, n'hésitez pas à envoyer une Pull Request.

## A propos de Martini

Inspiré par [express](https://github.com/visionmedia/express) et [Sinatra](https://github.com/sinatra/sinatra), Martini est l'oeuvre de nul d'autre que [Code Gangsta](http://codegangsta.io/), votre serviteur.
