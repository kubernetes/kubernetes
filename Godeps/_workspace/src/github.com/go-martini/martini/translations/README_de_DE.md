# Martini  [![wercker status](https://app.wercker.com/status/9b7dbc6e2654b604cd694d191c3d5487/s/master "wercker status")](https://app.wercker.com/project/bykey/9b7dbc6e2654b604cd694d191c3d5487)[![GoDoc](https://godoc.org/github.com/go-martini/martini?status.png)](http://godoc.org/github.com/go-martini/martini)

Martini ist eine mächtiges Package zur schnellen Entwicklung von modularen Webanwendungen und -services in Golang. 

## Ein Projekt starten

Nach der Installation von Go und dem Einrichten des [GOPATH](http://golang.org/doc/code.html#GOPATH), erstelle Deine erste `.go`-Datei. Speichere sie unter `server.go`.

~~~ go
package main

import "github.com/go-martini/martini"

func main() {
  m := martini.Classic()
  m.Get("/", func() string {
    return "Hallo Welt!"
  })
  m.Run()
}
~~~

Installiere anschließend das Martini Package (**Go 1.1** oder höher wird vorausgesetzt):
~~~
go get github.com/go-martini/martini
~~~

Starte den Server:
~~~
go run server.go
~~~

Der Martini-Webserver ist nun unter `localhost:3000` erreichbar.

## Hilfe

Aboniere den [Emailverteiler](https://groups.google.com/forum/#!forum/martini-go)

Schaue das [Demovideo](http://martini.codegangsta.io/#demo)

Stelle Fragen auf Stackoverflow mit dem [Martini-Tag](http://stackoverflow.com/questions/tagged/martini)

GoDoc [Dokumentation](http://godoc.org/github.com/go-martini/martini)


## Eigenschaften
* Sehr einfach nutzbar
* Nicht-intrusives Design
* Leicht kombinierbar mit anderen Golang Packages
* Ausgezeichnetes Path Matching und Routing
* Modulares Design - einfaches Hinzufügen und Entfernen von Funktionen
* Eine Vielzahl von guten Handlern/Middlewares nutzbar
* Großer Funktionsumfang mitgeliefert
* **Voll kompatibel mit dem [http.HandlerFunc](http://godoc.org/net/http#HandlerFunc) Interface.**
* Standardmäßges ausliefern von Dateien (z.B. von AngularJS-Apps im HTML5-Modus)

## Mehr Middleware
Mehr Informationen zur Middleware und Funktionalität findest Du in den Repositories der [martini-contrib](https://github.com/martini-contrib) Gruppe.

## Inhaltsverzeichnis
* [Classic Martini](#classic-martini)
  * [Handler](#handler)
  * [Routing](#routing)
  * [Services](#services)
  * [Statische Dateien bereitstellen](#statische-dateien-bereitstellen)
* [Middleware Handler](#middleware-handler)
  * [Next()](#next)
* [Martini Env](#martini-env)
* [FAQ](#faq)

## Classic Martini
Einen schnellen Start in ein Projekt ermöglicht [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic), dessen Voreinstellungen sich für die meisten Webanwendungen eignen:
~~~ go
  m := martini.Classic()
  // ... Middleware und Routing hier einfügen
  m.Run()
~~~

Aufgelistet findest Du einige Aspekte, die [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) automatich berücksichtigt:

  * Request/Response Logging - [martini.Logger](http://godoc.org/github.com/go-martini/martini#Logger)
  * Panic Recovery - [martini.Recovery](http://godoc.org/github.com/go-martini/martini#Recovery)
  * Static File serving - [martini.Static](http://godoc.org/github.com/go-martini/martini#Static)
  * Routing - [martini.Router](http://godoc.org/github.com/go-martini/martini#Router)

### Handler
Handler sind das Herz und die Seele von Martini. Ein Handler ist grundsätzlich jede Art von aufrufbaren Funktionen:
~~~ go
m.Get("/", func() {
  println("Hallo Welt")
})
~~~

#### Rückgabewerte
Wenn ein Handler Rückgabewerte beinhaltet, übergibt Martini diese an den aktuellen [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter) in Form eines String:
~~~ go
m.Get("/", func() string {
  return "Hallo Welt" // HTTP 200 : "Hallo Welt"
})
~~~

Die Rückgabe eines Statuscode ist optional:
~~~ go
m.Get("/", func() (int, string) {
  return 418, "Ich bin eine Teekanne" // HTTP 418 : "Ich bin eine Teekanne"
})
~~~

#### Service Injection
Handler werden per Reflection aufgerufen. Martini macht Gebrauch von *Dependency Injection*, um Abhängigkeiten in der Argumentliste von Handlern aufzulösen. **Dies macht Martini komplett inkompatibel mit Golangs `http.HandlerFunc` Interface.**

Fügst Du einem Handler ein Argument hinzu, sucht Martini in seiner Liste von Services und versucht, die Abhängigkeiten via Type Assertion aufzulösen. 
~~~ go
m.Get("/", func(res http.ResponseWriter, req *http.Request) { // res und req wurden von Martini injiziert
  res.WriteHeader(200) // HTTP 200
})
~~~

Die Folgenden Services sind Bestandteil von [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic):

  * [*log.Logger](http://godoc.org/log#Logger) - Globaler Logger für Martini.
  * [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) - http request context.
  * [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) - `map[string]string` von benannten Parametern, welche durch Route Matching gefunden wurden.
  * [martini.Routes](http://godoc.org/github.com/go-martini/martini#Routes) - Routen Hilfeservice.
  * [http.ResponseWriter](http://godoc.org/net/http/#ResponseWriter) - http Response writer interface.
  * [*http.Request](http://godoc.org/net/http/#Request) - http Request.

### Routing
Eine Route ist in Martini eine HTTP-Methode gepaart mit einem URL-Matching-Pattern. Jede Route kann eine oder mehrere Handler-Methoden übernehmen:
~~~ go
m.Get("/", func() {
  // zeige etwas an
})

m.Patch("/", func() {
  // aktualisiere etwas
})

m.Post("/", func() {
  // erstelle etwas
})

m.Put("/", func() {
  // ersetzte etwas
})

m.Delete("/", func() {
  // lösche etwas
})

m.Options("/", func() {
  // HTTP Optionen
})

m.NotFound(func() {
  // bearbeite 404-Fehler
})
~~~

Routen werden in der Reihenfolge, in welcher sie definiert wurden, zugeordnet. Die bei einer Anfrage zuerst zugeordnete Route wird daraufhin aufgerufen.  

Routenmuster enhalten ggf. benannte Parameter, die über den [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) Service abrufbar sind:
~~~ go
m.Get("/hello/:name", func(params martini.Params) string {
  return "Hallo " + params["name"]
})
~~~

Routen können mit Globs versehen werden:
~~~ go
m.Get("/hello/**", func(params martini.Params) string {
  return "Hallo " + params["_1"]
})
~~~

Reguläre Ausdrücke sind ebenfalls möglich:
~~~go
m.Get("/hello/(?P<name>[a-zA-Z]+)", func(params martini.Params) string {
  return fmt.Sprintf ("Hallo %s", params["name"])
})
~~~
Weitere Informationen zum Syntax regulärer Ausdrücke findest Du in der [Go Dokumentation](http://golang.org/pkg/regexp/syntax/).

Routen-Handler können auch in einander verschachtelt werden. Dies ist bei der Authentifizierung und Berechtigungen nützlich.
~~~ go
m.Get("/secret", authorize, func() {
  // wird ausgeführt, solange authorize nichts zurückgibt
})
~~~

Routengruppen können durch die Group-Methode hinzugefügt werden.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
})
~~~

Sowohl Handlern als auch Middlewares können Gruppen übergeben werden.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
}, MyMiddleware1, MyMiddleware2)
~~~

### Services
Services sind Okjekte, welche der Argumentliste von Handlern beigefügt werden können.
Du kannst einem Service der *Global* oder *Request* Ebene zuordnen.

#### Global Mapping
Eine Martini-Instanz implementiert das inject.Injector Interface, sodass ein Service leicht zugeordnet werden kann:
~~~ go
db := &MyDatabase{}
m := martini.Classic()
m.Map(db) // der Service ist allen Handlern unter *MyDatabase verfügbar
// ...
m.Run()
~~~

#### Request-Level Mapping
Das Zuordnen auf der Request-Ebene kann in einem Handler via  [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) realisiert werden:
~~~ go
func MyCustomLoggerHandler(c martini.Context, req *http.Request) {
  logger := &MyCustomLogger{req}
  c.Map(logger) // zugeordnet als *MyCustomLogger
}
~~~

#### Werten einem Interface zuordnen
Einer der mächtigsten Aspekte von Services ist dessen Fähigkeit, einen Service einem Interface zuzuordnen. Möchtest Du den [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter) mit einem Decorator (Objekt) und dessen Zusatzfunktionen überschreiben, definiere den Handler wie folgt:
~~~ go
func WrapResponseWriter(res http.ResponseWriter, c martini.Context) {
  rw := NewSpecialResponseWriter(res)
  c.MapTo(rw, (*http.ResponseWriter)(nil)) // überschribe ResponseWriter mit dem  ResponseWriter Decorator
}
~~~

### Statische Dateien bereitstellen
Eine [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) Instanz übertragt automatisch statische Dateien aus dem "public"-Ordner im Stammverzeichnis Deines Servers. Dieses Verhalten lässt sich durch weitere [martini.Static](http://godoc.org/github.com/go-martini/martini#Static) Handler auf andere Verzeichnisse übertragen.
~~~ go
m.Use(martini.Static("assets")) // überträgt auch vom "assets"-Verzeichnis
~~~

#### Eine voreingestelle Datei übertragen
Du kannst die URL zu einer lokalen Datei angeben, sollte die URL einer Anfrage nicht gefunden werden. Durch einen Präfix können bestimmte URLs ignoriert werden.
Dies ist für Server nützlich, welche statische Dateien übertragen und ggf. zusätzliche Handler defineren (z.B. eine REST-API). Ist dies der Fall, so ist das Anlegen eines Handlers in der NotFound-Reihe nützlich.

Das gezeigte Beispiel zeigt die `/index.html` immer an, wenn die angefrage URL keiner lokalen Datei zugeordnet werden kann bzw. wenn sie nicht mit `/api/v` beginnt:
~~~ go
static := martini.Static("assets", martini.StaticOptions{Fallback: "/index.html", Exclude: "/api/v"})
m.NotFound(static, http.NotFound)
~~~

## Middleware Handler
Middleware-Handler befinden sich logisch zwischen einer Anfrage via HTTP und dem Router. Im wesentlichen unterscheiden sie sich nicht von anderen Handlern in Martini.
Du kannst einen Middleware-Handler dem Stack folgendermaßen anfügen:
~~~ go
m.Use(func() {
  // durchlaufe die Middleware
})
~~~

Volle Kontrolle über den Middleware Stack erlangst Du mit der `Handlers`-Funktion.
Sie ersetzt jeden zuvor definierten Handler:
~~~ go
m.Handlers(
  Middleware1,
  Middleware2,
  Middleware3,
)
~~~

Middleware Handler arbeiten gut mit Aspekten wie Logging, Berechtigungen, Authentifizierung, Sessions, Komprimierung durch gzip, Fehlerseiten und anderen Operationen zusammen, die vor oder nach einer Anfrage passieren.
~~~ go
// überprüfe einen API-Schlüssel
m.Use(func(res http.ResponseWriter, req *http.Request) {
  if req.Header.Get("X-API-KEY") != "secret123" {
    res.WriteHeader(http.StatusUnauthorized)
  }
})
~~~

### Next()
[Context.Next()](http://godoc.org/github.com/go-martini/martini#Context) ist eine optionale Funktion, die Middleware-Handler aufrufen können, um sie nach dem Beenden der anderen Handler auszuführen. Dies funktioniert besonders gut, wenn Operationen nach einer HTTP-Anfrage ausgeführt werden müssen.
~~~ go
// protokolliere vor und nach einer Anfrage
m.Use(func(c martini.Context, log *log.Logger){
  log.Println("vor einer Anfrage")

  c.Next()

  log.Println("nach einer Anfrage")
})
~~~

## Martini Env

Einige Martini-Handler machen von der globalen `martini.Env` Variable gebrauch, die der Entwicklungsumgebung erweiterte Funktionen bietet, welche die Produktivumgebung nicht enthält. Es wird empfohlen, die `MARTINI_ENV=production` Umgebungsvariable zu setzen, sobald der Martini-Server in den Live-Betrieb übergeht.

## FAQ

### Wo finde ich eine bestimmte Middleware?

Starte die Suche mit einem Blick in die Projekte von [martini-contrib](https://github.com/martini-contrib). Solltest Du nicht fündig werden, kontaktiere ein Mitglied des martini-contrib Teams, um eine neue Repository anzulegen.

* [auth](https://github.com/martini-contrib/auth) - Handler zur Authentifizierung.
* [binding](https://github.com/martini-contrib/binding) - Handler zum Zuordnen/Validieren einer Anfrage zu einem Struct.
* [gzip](https://github.com/martini-contrib/gzip) - Handler zum Ermöglichen von  gzip-Kompression bei Anfragen.
* [render](https://github.com/martini-contrib/render) - Handler der einen einfachen Service zum  Rendern von JSON und HTML Templates bereitstellt.
* [acceptlang](https://github.com/martini-contrib/acceptlang) - Handler zum Parsen des `Accept-Language` HTTP-Header.
* [sessions](https://github.com/martini-contrib/sessions) - Handler mit einem Session service.
* [strip](https://github.com/martini-contrib/strip) - URL Prefix stripping.
* [method](https://github.com/martini-contrib/method) - Überschreibe eine HTTP-Method via Header oder Formularfelder.
* [secure](https://github.com/martini-contrib/secure) - Implementation von Sicherheitsfunktionen
* [encoder](https://github.com/martini-contrib/encoder) - Encoderservice zum Datenrendering in den verschiedensten Formaten.
* [cors](https://github.com/martini-contrib/cors) - Handler der CORS ermöglicht.
* [oauth2](https://github.com/martini-contrib/oauth2) - Handler der den Login mit OAuth 2.0 in Martinianwendungen ermöglicht. Google Sign-in, Facebook Connect und Github werden ebenfalls unterstützt.
* [vauth](https://github.com/rafecolton/vauth) - Handlers zur Webhook Authentifizierung (momentan nur GitHub und TravisCI)

### Wie integriere ich in bestehende Systeme?

Eine Martiniinstanz implementiert `http.Handler`, sodass Subrouten in bestehenden Servern einfach genutzt werden können. Hier ist eine funktionierende Martinianwendungen für die Google App Engine:

~~~ go
package hello

import (
  "net/http"
  "github.com/go-martini/martini"
)

func init() {
  m := martini.Classic()
  m.Get("/", func() string {
    return "Hallo Welt!"
  })
  http.Handle("/", m)
}
~~~

### Wie ändere ich den Port/Host?

Martinis `Run` Funktion sucht automatisch nach den PORT und HOST Umgebungsvariablen, um diese zu nutzen. Andernfalls ist localhost:3000 voreingestellt.
Für mehr Flexibilität über den Port und den Host nutze stattdessen die `martini.RunOnAddr` Funktion.

~~~ go
  m := martini.Classic()
  // ...
  log.Fatal(m.RunOnAddr(":8080"))
~~~

### Automatisches Aktualisieren?

[Gin](https://github.com/codegangsta/gin) und [Fresh](https://github.com/pilu/fresh) aktualisieren Martini-Apps live.

## Bei Martini mitwirken

Martinis Maxime ist Minimalismus und sauberer Code. Die meisten Beiträge sollten sich in den Repositories der [martini-contrib](https://github.com/martini-contrib) Gruppe wiederfinden. Beinhaltet Dein Beitrag Veränderungen am Kern von Martini, zögere nicht, einen Pull Request zu machen.

## Über das Projekt

Inspiriert von [Express](https://github.com/visionmedia/express) und [Sinatra](https://github.com/sinatra/sinatra)

Martini wird leidenschaftlich von Niemand gerigeren als dem [Code Gangsta](http://codegangsta.io/) entwickelt
