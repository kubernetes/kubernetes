package main

import (
	"fmt"
	"net/http"
	"io/ioutil"
)

func main() {
	http.HandleFunc("/browse", browseHandler)
	http.HandleFunc("/", helloHandler)

	fmt.Println("Started, serving at 8888")
	err := http.ListenAndServe(":8888", nil)
	if err != nil {
		panic("ListenAndServe: " + err.Error())
	}
}


func helloHandler(w http.ResponseWriter, r *http.Request) {
	fmt.Fprintln(w, "Hello World!")
}

func browseHandler(w http.ResponseWriter, r *http.Request) {

	data, err := ioutil.ReadFile("/test/mydata.txt")
	if err != nil {
		fmt.Fprintf(w, "Error reading /test/mydata.txt : %+v\n", err)
	}

	fmt.Fprintln(w, string(data))
}
