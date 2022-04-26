# Tagliatelle

[![Sponsor](https://img.shields.io/badge/Sponsor%20me-%E2%9D%A4%EF%B8%8F-pink)](https://github.com/sponsors/ldez)
[![Build Status](https://github.com/ldez/tagliatelle/workflows/Main/badge.svg?branch=master)](https://github.com/ldez/tagliatelle/actions)

A linter that handles struct tags.

Supported string casing:

- `camel`
- `pascal`
- `kebab`
- `snake`
- `goCamel` Respects [Go's common initialisms](https://github.com/golang/lint/blob/83fdc39ff7b56453e3793356bcff3070b9b96445/lint.go#L770-L809) (e.g. HttpResponse -> HTTPResponse).
- `goPascal` Respects [Go's common initialisms](https://github.com/golang/lint/blob/83fdc39ff7b56453e3793356bcff3070b9b96445/lint.go#L770-L809) (e.g. HttpResponse -> HTTPResponse).
- `goKebab` Respects [Go's common initialisms](https://github.com/golang/lint/blob/83fdc39ff7b56453e3793356bcff3070b9b96445/lint.go#L770-L809) (e.g. HttpResponse -> HTTPResponse).
- `goSnake` Respects [Go's common initialisms](https://github.com/golang/lint/blob/83fdc39ff7b56453e3793356bcff3070b9b96445/lint.go#L770-L809) (e.g. HttpResponse -> HTTPResponse).
- `upper`
- `lower`

| Source         | Camel Case     | Go Camel Case  |
|----------------|----------------|----------------|
| GooID          | gooId          | gooID          |
| HTTPStatusCode | httpStatusCode | httpStatusCode |
| FooBAR         | fooBar         | fooBar         |
| URL            | url            | url            |
| ID             | id             | id             |
| hostIP         | hostIp         | hostIP         |
| JSON           | json           | json           |
| JSONName       | jsonName       | jsonName       |
| NameJSON       | nameJson       | nameJSON       |
| UneTête        | uneTête        | uneTête        |

| Source         | Pascal Case    | Go Pascal Case |
|----------------|----------------|----------------|
| GooID          | GooId          | GooID          |
| HTTPStatusCode | HttpStatusCode | HTTPStatusCode |
| FooBAR         | FooBar         | FooBar         |
| URL            | Url            | URL            |
| ID             | Id             | ID             |
| hostIP         | HostIp         | HostIP         |
| JSON           | Json           | JSON           |
| JSONName       | JsonName       | JSONName       |
| NameJSON       | NameJson       | NameJSON       |
| UneTête        | UneTête        | UneTête        |

| Source         | Snake Case       | Go Snake Case    |
|----------------|------------------|------------------|
| GooID          | goo_id           | goo_ID           |
| HTTPStatusCode | http_status_code | HTTP_status_code |
| FooBAR         | foo_bar          | foo_bar          |
| URL            | url              | URL              |
| ID             | id               | ID               |
| hostIP         | host_ip          | host_IP          |
| JSON           | json             | JSON             |
| JSONName       | json_name        | JSON_name        |
| NameJSON       | name_json        | name_JSON        |
| UneTête        | une_tête         | une_tête         |

| Source         | Kebab Case       | Go KebabCase     |
|----------------|------------------|------------------|
| GooID          | goo-id           | goo-ID           |
| HTTPStatusCode | http-status-code | HTTP-status-code |
| FooBAR         | foo-bar          | foo-bar          |
| URL            | url              | URL              |
| ID             | id               | ID               |
| hostIP         | host-ip          | host-IP          |
| JSON           | json             | JSON             |
| JSONName       | json-name        | JSON-name        |
| NameJSON       | name-json        | name-JSON        |
| UneTête        | une-tête         | une-tête         |


## Examples

```go
// json and camel case
type Foo struct {
    ID     string `json:"ID"` // must be "id"
    UserID string `json:"UserID"`// must be "userId"
    Name   string `json:"name"`
    Value  string `json:"val,omitempty"`// must be "value"
}
```
