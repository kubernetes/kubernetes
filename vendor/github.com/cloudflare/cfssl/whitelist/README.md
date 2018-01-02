## whitelist

This is a simple whitelisting package that encompasses several common
patterns into a reusable package.

The basic type of a whitelist is the `ACL` type, which provides
a single method on a `net.IP` value:

* `Permitted` determines whether the IP address is whitelisted and
  therefore should be permitted access. It should return true if the
  address is whitelisted.

Additionally, there are two other types that are built on the `ACL`
type; the `HostACL` stores individual hosts and the `NetACL` stores
networks. Each of these provides two functions that differ in the
types of their arguments.

* `Add` whitelists the IP address.
* `Remove` drops the IP address from the whitelist.

The `HostACL` operates on `net.IP` values, while the `NetACL` operates
on `*net.IPNet`s.

There are currently four implementations of `ACL` provided in this
package; a basic implementation of the two types of ACLs and a stub
type for each:

* `Basic` is a simple host-based whitelister that converts the IP
  addresses to strings; the whitelist is implemented as a set of
  string addresses. The set is implemented as a `map[string]bool`, and
  uses a `sync.Mutex` to coordinate updates to the whitelist.
* `BasicNet` is a simple network-based whitelister that similarly uses
  a mutex and an array to store networks. This has a number of
  limitations: operations are /O(n)/, and subsets/supersets of
  existing networks isn't detected. That is, if 192.168.3.0/24 is
  removed from a whitelist that has 192.168.0.0/16 permitted, **that
  subnet will not actually be removed**. Exact networks are required
  for `Add` and `Remove` at this time.
* `HostStub` and `NetStub` are stand-in whitelists that always permits
  addresses. They are vocal about logging warning messages noting that
  whitelisting is stubbed. They are designed to be used in cases where
  whitelisting is desired, but the mechanics of whitelisting
  (i.e. administration of the whitelist) is not yet implemented,
  perhaps to keep whitelists in the system's flow.

Two convenience functions are provided here for extracting IP addresses:

* `NetConnLookup` accepts a `net.Conn` value, and returns the `net.IP`
  value from the connection.
* `HTTPRequestLookup` accepts a `*http.Request` and returns the
  `net.IP` value from the request.

There are also two functions for whitelisting HTTP endpoints:

* `NewHandler` returns an `http.Handler`
* `NewHandlerFunc` returns an `http.HandlerFunc`

These endpoints will work with both `HostACL` and `NetACL`.

### Example `http.Handler`

This is a file server that uses a pair of whitelists. The admin
whitelist permits modifications to the user whitelist only by the
localhost. The user whitelist controls which hosts have access to
the file server.

```
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"

	"github.com/cloudflare/cfssl/whitelist"
)

var wl = whitelist.NewBasic()

func addIP(w http.ResponseWriter, r *http.Request) {
	addr := r.FormValue("ip")

	ip := net.ParseIP(addr)
	wl.Add(ip)
	log.Printf("request to add %s to the whitelist", addr)
	w.Write([]byte(fmt.Sprintf("Added %s to whitelist.\n", addr)))
}

func delIP(w http.ResponseWriter, r *http.Request) {
	addr := r.FormValue("ip")

	ip := net.ParseIP(addr)
	wl.Remove(ip)
	log.Printf("request to remove %s from the whitelist", addr)
	w.Write([]byte(fmt.Sprintf("Removed %s from whitelist.\n", ip)))
}

func dumpWhitelist(w http.ResponseWriter, r *http.Request) {
	out, err := json.Marshal(wl)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
	} else {
		w.Write(out)
	}
}

type handler struct {
	h func(http.ResponseWriter, *http.Request)
}

func newHandler(h func(w http.ResponseWriter, r *http.Request)) http.Handler {
	return &handler{h: h}
}

func (h *handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.h(w, r)
}

func main() {
	root := flag.String("root", "files/", "file server root")
	flag.Parse()

	fileServer := http.StripPrefix("/files/",
		http.FileServer(http.Dir(*root)))
	wl.Add(net.IP{127, 0, 0, 1})

	adminWL := whitelist.NewBasic()
	adminWL.Add(net.IP{127, 0, 0, 1})
	adminWL.Add(net.ParseIP("::1"))

	protFiles, err := whitelist.NewHandler(fileServer, nil, wl)
	if err != nil {
		log.Fatalf("%v", err)
	}

	addHandler, err := whitelist.NewHandlerFunc(addIP, nil, adminWL)
	if err != nil {
		log.Fatalf("%v", err)
	}

	delHandler, err := whitelist.NewHandlerFunc(delIP, nil, adminWL)
	if err != nil {
		log.Fatalf("%v", err)
	}

	dumpHandler, err := whitelist.NewHandlerFunc(dumpWhitelist, nil, adminWL)
	if err != nil {
		log.Fatalf("%v", err)
	}

	http.Handle("/files/", protFiles)
	http.Handle("/add", addHandler)
	http.Handle("/del", delHandler)
	http.Handle("/dump", dumpHandler)

	log.Println("Serving files on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
```


