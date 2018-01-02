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
