package main

import (
	"fmt"
	"github.com/googleapis/gnostic/plugins/gnostic-go-generator/examples/v2.0/xkcd/xkcd"
)

func main() {
	c := xkcd.NewClient("http://xkcd.com")

	comic, err := c.Get_info_0_json()
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", comic)

	comic, err = c.Get_comicId_info_0_json(1800)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%+v\n", comic)
}
