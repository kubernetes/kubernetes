# Quobyte API Clients

Get the quobyte api client

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
    client.SetAPIRetryPolicy(quobyte_api.RetryInfinitely) // Default quobyte_api.RetryInteractive
    req := &quobyte_api.CreateVolumeRequest{
        Name:              "MyVolume",
        RootUserID:        "root",
        RootGroupID:       "root",
        ConfigurationName: "BASE",
    }

    volumeUUID, err := client.CreateVolume(req)
    if err != nil {
        log.Fatalf("Error:", err)
    }

    log.Printf("%s", volumeUUID)
}
```