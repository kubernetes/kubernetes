package main

import (
	"fmt"
	"net/http"
)

func helloFromKubernetes(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Hello World! -- Kubernetes")
}

func main() {
	http.HandleFunc("/", helloFromKubernetes)
	err := http.ListenAndServe(":8080", nil)
	if err != nil {
		panic("ListenAndServe: " + err.Error())
	}
}
