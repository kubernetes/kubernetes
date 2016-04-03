# cAdvisor API Clients

There is an official Go client implementation in the [client](../client/) directory. You can use it on your own Go project by including it like this:

```go
import "github.com/google/cadvisor/client"

client, err = client.NewClient("http://localhost:8080/")
mInfo, err := client.MachineInfo()
```

Do you know of another cAdvisor client? Maybe in another language? Please let us know! We'd be happy to add a note on this page.
