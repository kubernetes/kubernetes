# dnspod-go

A Go client for the [DNSPod API](https://www.dnspod.cn/docs/index.html).

Borrowed from : [dnsimple](https://github.com/weppos/dnsimple-go/dnsimple)

## Installation

```
$ go get github.com/decker502/dnspod-go
```


## Getting Started

This library is a Go client you can use to interact with the [DNSPod API](https://www.dnspod.cn/docs/index.html). Here are some examples.


```go
package main

import (
  "fmt"
  "github.com/decker502/dnspod-go"
)

func main() {
  apiToken := "xxxxxxx"

  params := dnspod.CommonParams{LoginToken: "dnspod login token"}
  client := dnspod.NewClient(apiToken)

  // Get a list of your domains
  domains, _, _ := client.Domains.List()
  for _, domain := range domains {
      fmt.Printf("Domain: %s (id: %d)\n", domain.Name, domain.Id)
  }

  // Get a list of your domains (with error management)
  domains, _, error := client.Domains.List()
  if error != nil {
      log.Fatalln(error)
  }
  for _, domain := range domains {
      fmt.Printf("Domain: %s (id: %d)\n", domain.Name, domain.Id)
  }

  // Create a new Domain
  newDomain := Domain{Name: "example.com"}
  domain, _, _ := client.Domains.Create(newDomain)
  fmt.Printf("Domain: %s\n (id: %d)", domain.Name, domain.Id)
}
```
## License

This is Free Software distributed under the MIT license.
