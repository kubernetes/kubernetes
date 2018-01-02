# Photon-controller-go-SDK

# Getting Started

1. If you haven't already, set up a Go workspace according to the
   [Go docs](http://golang.org/doc).
2. Install the Go SDK. Normally this is done with "go get".
3. Setup GOPATH environment variable

	Then:
	```
	mkdir -p $GOPATH/src/github.com/vmware
	cd $GOPATH/src/github.com/vmware
	git clone (github.com/vmware or gerrit)/photon-controller-go-sdk
	```

## Sample App

Here's a quick sample app that will retrieve Photon Controller status from a
[local devbox].
In this example, it's under $GOPATH/src/sdkexample/main.go:

```golang
package main

import (
	"fmt"
	"github.com/vmware/photon-controller-go-sdk/photon"
	"log"
)

func main() {
	client := photon.NewClient("http://localhost:9080", nil, nil)
	status, err := client.Status.Get()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Print(status)
}
```

Then build it and run it:

```
cd $GOPATH/src/sdkexample
go build ./...
./sdkexample
```

And the output should look something like this:
`&{READY [{PHOTON_CONTROLLER  READY}]}`

## Using APIs that return tasks

Most Photon Controller APIs use a task model. The API will return a task object,
which will indicate the state of the task (such as queued, completed, error, etc).
These tasks return immediately and the caller must poll to find out when the task
has been completed.The Go SDK provides a tasks API to do this for you,
with built-in retry and error handling.

Let's expand the sample app to create a new tenant:

```
package main

import (
	"fmt"
	"github.com/vmware/photon-controller-go-sdk/photon"
	"log"
)

func main() {
	client := photon.NewClient("http://localhost:9080", nil, nil)
	status, err := client.Status.Get()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(status)

	// Let's create a new tenant
	tenantSpec := &photon.TenantCreateSpec{Name: "new-tenant"}

	task, err := client.Tenants.Create(tenantSpec)
	if err != nil {
		log.Fatal(err)
	}

	// Wait for task completion
	task, err = client.Tasks.Wait(task.ID)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("ID of new tenant is: %s\n", task.Entity.ID)
}

```

It should now output this:

```
&{READY [{PHOTON_CONTROLLER  READY}]}
ID of new tenant is: c8989a40-0fa4-4d9a-8e73-2fe4d28d0065
```
