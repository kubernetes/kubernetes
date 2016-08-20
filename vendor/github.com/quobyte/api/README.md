# Quobyte API Clients

Get the quoybte api client

```bash
go get github.com/quobyte/api
```

## Usage

```go
package main

import (
  "log"
  quobyte_api "github.com/quobyte/api"
)

func main() {
    client := quobyte_api.NewQuobyteClient("http://apiserver:7860", "user", "password")
    volume_uuid, err := client.CreateVolume("MyVolume", "root", "root")
    if err != nil {
        log.Fatalf("Error:", err)
    }

    log.Printf("%s", volume_uuid)
}
```
