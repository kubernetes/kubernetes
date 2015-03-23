# Martini  [![wercker status](https://app.wercker.com/status/9b7dbc6e2654b604cd694d191c3d5487/s/master "wercker status")](https://app.wercker.com/project/bykey/9b7dbc6e2654b604cd694d191c3d5487)[![GoDoc](https://godoc.org/github.com/go-martini/martini?status.png)](http://godoc.org/github.com/go-martini/martini)

Martini - мощный пакет для быстрой разработки веб приложений и сервисов на Golang.

## Начало работы

После установки Golang и настройки вашего [GOPATH](http://golang.org/doc/code.html#GOPATH), создайте ваш первый `.go` файл. Назовем его `server.go`.

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

Потом установите пакет Martini (требуется **go 1.1** или выше):
~~~
go get github.com/go-martini/martini
~~~

Потом запустите ваш сервер:
~~~
go run server.go
~~~

И вы получите запущенный Martini сервер на `localhost:3000`.

## Помощь

Присоединяйтесь к [рассылке](https://groups.google.com/forum/#!forum/martini-go)

Смотрите [демо видео](http://martini.codegangsta.io/#demo)

Задавайте вопросы на Stackoverflow используя [тэг martini](http://stackoverflow.com/questions/tagged/martini)

GoDoc [документация](http://godoc.org/github.com/go-martini/martini)


## Возможности
* Очень прост в использовании.
* Ненавязчивый дизайн.
* Хорошо сочетается с другими пакетами.
* Потрясающий роутинг и маршрутизация.
* Модульный дизайн - легко добавлять и исключать функциональность.
* Большое количество хороших обработчиков/middlewares, готовых к использованию.
* Отличный набор 'из коробки'.
* **Полностью совместим с интерфейсом [http.HandlerFunc](http://godoc.org/net/http#HandlerFunc).**

## Больше Middleware
Смотрите репозитории организации [martini-contrib](https://github.com/martini-contrib), для большей информации о функциональности и middleware.

## Содержание
* [Classic Martini](#classic-martini)
  * [Обработчики](#%D0%9E%D0%B1%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%87%D0%B8%D0%BA%D0%B8)
  * [Роутинг](#%D0%A0%D0%BE%D1%83%D1%82%D0%B8%D0%BD%D0%B3)
  * [Сервисы](#%D0%A1%D0%B5%D1%80%D0%B2%D0%B8%D1%81%D1%8B)
  * [Отдача статических файлов](#%D0%9E%D1%82%D0%B4%D0%B0%D1%87%D0%B0-%D1%81%D1%82%D0%B0%D1%82%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D1%85-%D1%84%D0%B0%D0%B9%D0%BB%D0%BE%D0%B2)
* [Middleware обработчики](#middleware-%D0%9E%D0%B1%D1%80%D0%B0%D0%B1%D0%BE%D1%82%D1%87%D0%B8%D0%BA%D0%B8)
  * [Next()](#next)
* [Окружение](#%D0%9E%D0%BA%D1%80%D1%83%D0%B6%D0%B5%D0%BD%D0%B8%D0%B5)
* [FAQ](#faq)

## Classic Martini
Для быстрого старта [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) предлагает несколько предустановок, это используется для большинства веб приложений:
~~~ go
  m := martini.Classic()
  // ... middleware и роутинг здесь
  m.Run()
~~~

Ниже представлена уже подключенная [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) функциональность:  

  * Request/Response логгирование - [martini.Logger](http://godoc.org/github.com/go-martini/martini#Logger)
  * Panic Recovery - [martini.Recovery](http://godoc.org/github.com/go-martini/martini#Recovery)
  * Отдача статики - [martini.Static](http://godoc.org/github.com/go-martini/martini#Static)
  * Роутинг - [martini.Router](http://godoc.org/github.com/go-martini/martini#Router)

### Обработчики
Обработчики - это сердце и душа Martini. Обработчик - любая функция, которая может быть вызвана:
~~~ go
m.Get("/", func() {
  println("hello world")
})
~~~

#### Возвращаемые значения
Если обработчик возвращает что либо, Martini запишет это как результат в текущий [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter), в виде строки:
~~~ go
m.Get("/", func() string {
  return "hello world" // HTTP 200 : "hello world"
})
~~~

Так же вы можете возвращать код статуса, опционально:
~~~ go
m.Get("/", func() (int, string) {
  return 418, "i'm a teapot" // HTTP 418 : "i'm a teapot"
})
~~~

#### Внедрение сервисов
Обработчики вызываются посредством рефлексии. Martini использует **Внедрение зависимости** для разрешения зависимостей в списке аргумента обработчика. **Это делает Martini полностью совместимым с интерфейсом `http.HandlerFunc`.**

Если вы добавите аргументы в ваш обработчик, Martini будет пытаться найти этот список сервисов за счет проверки типов(type assertion):
~~~ go
m.Get("/", func(res http.ResponseWriter, req *http.Request) { // res и req будут внедрены  Martini
  res.WriteHeader(200) // HTTP 200
})
~~~

Следующие сервисы включены в [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic):

  * [*log.Logger](http://godoc.org/log#Logger) - Глобальный логгер для Martini.
  * [martini.Context](http://godoc.org/github.com/go-martini/martini#Context) - http request контекст.
  * [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) - `map[string]string` именованых аргументов из роутера.
  * [martini.Routes](http://godoc.org/github.com/go-martini/martini#Routes) - Хэлпер роутеров.
  * [http.ResponseWriter](http://godoc.org/net/http/#ResponseWriter) - http Response writer интерфейс.
  * [*http.Request](http://godoc.org/net/http/#Request) - http Request.

### Роутинг
В Martini, роут - это объединенные паттерн и HTTP метод.
Каждый роут может принимать один или несколько обработчиков:
~~~ go
m.Get("/", func() {
  // показать что-то
})

m.Patch("/", func() {
  // обновить что-то
})

m.Post("/", func() {
  // создать что-то
})

m.Put("/", func() {
  // изменить что-то
})

m.Delete("/", func() {
  // удалить что-то
})

m.Options("/", func() {
  // http опции
})

m.NotFound(func() {
  // обработчик 404
})
~~~

Роуты могут сопоставляться с http запросами только в порядке объявления. Вызывается первый роут, который соответствует запросу.

Паттерны роутов могут включать именованные параметры, доступные через [martini.Params](http://godoc.org/github.com/go-martini/martini#Params) сервис:
~~~ go
m.Get("/hello/:name", func(params martini.Params) string {
  return "Hello " + params["name"]
})
~~~

Роуты можно объявлять как glob'ы:
~~~ go
m.Get("/hello/**", func(params martini.Params) string {
  return "Hello " + params["_1"]
})
~~~

Так же могут использоваться регулярные выражения:
~~~go
m.Get("/hello/(?P<name>[a-zA-Z]+)", func(params martini.Params) string {
  return fmt.Sprintf ("Hello %s", params["name"])
})
~~~
Синтаксис регулярных выражений смотрите [Go documentation](http://golang.org/pkg/regexp/syntax/).

Обработчики роутов так же могут быть выстроены в стек, друг перед другом. Это очень удобно для таких задач как авторизация и аутентификация:
~~~ go
m.Get("/secret", authorize, func() {
  // будет вызываться, в случае если authorize ничего не записал в ответ
})
~~~

Роуты так же могут быть объединены в группы, посредством метода Group:
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
})
~~~

Так же как вы можете добавить middleware для обычного обработчика, вы можете добавить middleware и для группы.
~~~ go
m.Group("/books", func(r martini.Router) {
    r.Get("/:id", GetBooks)
    r.Post("/new", NewBook)
    r.Put("/update/:id", UpdateBook)
    r.Delete("/delete/:id", DeleteBook)
}, MyMiddleware1, MyMiddleware2)
~~~

### Сервисы
Сервисы - это объекты, которые доступны для внедрения в аргументы обработчиков. Вы можете замапить сервисы на уровне всего приложения либо на уровне запроса.

#### Глобальный маппинг
Экземпляр Martini реализует интерфейс inject.Injector, поэтому замаппить сервис легко:
~~~ go
db := &MyDatabase{}
m := martini.Classic()
m.Map(db) // сервис будет доступен для всех обработчиков как *MyDatabase
// ...
m.Run()
~~~

#### Маппинг уровня запроса
Маппинг на уровне запроса можно сделать при помощи [martini.Context](http://godoc.org/github.com/go-martini/martini#Context):
~~~ go
func MyCustomLoggerHandler(c martini.Context, req *http.Request) {
  logger := &MyCustomLogger{req}
  c.Map(logger) // как *MyCustomLogger
}
~~~

#### Маппинг на определенный интерфейс
Одна из мощных частей, того что касается сервисов - маппинг сервиса на определенный интерфейс. Например, если вы хотите переопределить [http.ResponseWriter](http://godoc.org/net/http#ResponseWriter) объектом, который оборачивает и добавляет новые операции, вы можете написать следующее:
~~~ go
func WrapResponseWriter(res http.ResponseWriter, c martini.Context) {
  rw := NewSpecialResponseWriter(res)
  c.MapTo(rw, (*http.ResponseWriter)(nil)) // переопределить ResponseWriter нашей оберткой
}
~~~

### Отдача статических файлов
Экземпляр [martini.Classic()](http://godoc.org/github.com/go-martini/martini#Classic) автоматически отдает статические файлы из директории "public" в корне, рядом с вашим файлом `server.go`.
Вы можете добавить еще директорий, добавляя [martini.Static](http://godoc.org/github.com/go-martini/martini#Static) обработчики.  
~~~ go
m.Use(martini.Static("assets")) // отдача файлов из "assets" директории
~~~

## Middleware Обработчики
Middleware обработчики находятся между входящим http запросом и роутом. По сути, они ничем не отличаются от любого другого обработчика Martini. Вы можете добавить middleware обработчик в стек следующим образом:
~~~ go
m.Use(func() {
  // делать какую то middleware работу
})
~~~

Для полного контроля над стеком middleware существует метод `Handlers`. В этом примере будут заменены все обработчики, которые были до этого:
~~~ go
m.Handlers(
  Middleware1,
  Middleware2,
  Middleware3,
)
~~~

Middleware обработчики очень хорошо работают для таких вещей как логгирование, авторизация, аутентификация, сессии, сжатие, страницы ошибок и любые другие операции, которые должны быть выполнены до или после http запроса:
~~~ go
// валидация api ключа
m.Use(func(res http.ResponseWriter, req *http.Request) {
  if req.Header.Get("X-API-KEY") != "secret123" {
    res.WriteHeader(http.StatusUnauthorized)
  }
})
~~~

### Next()
[Context.Next()](http://godoc.org/github.com/go-martini/martini#Context) опциональная функция, которая может быть вызвана в Middleware обработчике, для выхода из контекста, и возврата в него, после вызова всего стека обработчиков. Это можно использовать для операций, которые должны быть выполнены после http запроса:
~~~ go
// логгирование до и после http запроса
m.Use(func(c martini.Context, log *log.Logger){
  log.Println("до запроса")

  c.Next()

  log.Println("после запроса")
})
~~~

## Окружение
Некоторые Martini обработчики используют глобальную переменную `martini.Env` для того, чтоб предоставить специальную функциональность для девелопмент и продакшн окружения. Рекомендуется устанавливать `MARTINI_ENV=production`, когда вы деплоите приложение на продакшн.

## FAQ

### Где найти готовые middleware?

Начните поиск с [martini-contrib](https://github.com/martini-contrib) проектов. Если нет ничего подходящего, без колебаний пишите члену команды martini-contrib о добавлении нового репозитория в организацию.

* [auth](https://github.com/martini-contrib/auth) - Обработчики для аутентификации.
* [binding](https://github.com/martini-contrib/binding) - Обработчик для маппинга/валидации сырого запроса в определенную структуру(struct).
* [gzip](https://github.com/martini-contrib/gzip) - Обработчик, добавляющий gzip сжатие для запросов.
* [render](https://github.com/martini-contrib/render) - Обработчик, которые предоставляет сервис для легкого рендеринга JSON и HTML шаблонов.
* [acceptlang](https://github.com/martini-contrib/acceptlang) - Обработчик для парсинга `Accept-Language` HTTP заголовка.
* [sessions](https://github.com/martini-contrib/sessions) - Сервис сессий.
* [strip](https://github.com/martini-contrib/strip) - Удаление префиксов из URL.
* [method](https://github.com/martini-contrib/method) - Подмена HTTP метода через заголовок.
* [secure](https://github.com/martini-contrib/secure) - Набор для безопасности.
* [encoder](https://github.com/martini-contrib/encoder) - Сервис для представления данных в нескольких форматах и взаимодействия с контентом.
* [cors](https://github.com/martini-contrib/cors) - Поддержка CORS.
* [oauth2](https://github.com/martini-contrib/oauth2) - Обработчик, предоставляющий OAuth 2.0 логин для Martini приложений. Вход через Google, Facebook и через Github поддерживаются.

### Как интегрироваться с существуюшими серверами?

Экземпляр Martini реализует интерфейс `http.Handler`, потому - это очень просто использовать вместе с существующим Go проектом. Например, это работает для платформы Google App Engine:
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

### Как изменить порт и/или хост?
Функция `Run` смотрит переменные окружиения PORT и HOST, и использует их.
В противном случае Martini по умолчанию будет использовать `localhost:3000`.
Для большей гибкости используйте вместо этого функцию `martini.RunOnAddr`.

~~~ go
  m := martini.Classic()
  // ...
  log.Fatal(m.RunOnAddr(":8080"))
~~~

### Живая перезагрузка кода?

[gin](https://github.com/codegangsta/gin) и [fresh](https://github.com/pilu/fresh) могут работать вместе с Martini.

## Вклад в обшее дело

Подразумевается что Martini чистый и маленький. Большинство улучшений должны быть в организации [martini-contrib](https://github.com/martini-contrib). Но если вы хотите улучшить ядро Martini, отправляйте пулл реквесты.

## О проекте

Вдохновлен [express](https://github.com/visionmedia/express) и [sinatra](https://github.com/sinatra/sinatra)

Martini создан [Code Gangsta](http://codegangsta.io/)
