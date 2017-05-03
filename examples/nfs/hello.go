package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
)

func file(w http.ResponseWriter, r *http.Request) {
	b, err := ioutil.ReadFile("/data/hello.txt")
	if err != nil {
		panic(err)
	}
	fmt.Println(string(b))
	fmt.Fprintln(w, string(b))
}

func hello(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Hello world!  This is the working hello handler!")
}

func any(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Any URL?!  This is the working any handler!")
}

func main() {

	http.HandleFunc("/hello", hello)
	http.HandleFunc("/file", file)
	http.HandleFunc("/", any)
	fmt.Println("Started, serving at 12345")
	err := http.ListenAndServe(":12345", nil)
	if err != nil {
		panic("ListenAndServe: " + err.Error())
	}
}
