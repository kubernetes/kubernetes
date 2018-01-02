package main

import (
	"fmt"
	"github.com/petar/GoLLRB/llrb"
)

func lessInt(a, b interface{}) bool { return a.(int) < b.(int) }

func main() {
	tree := llrb.New(lessInt)
	tree.ReplaceOrInsert(1)
	tree.ReplaceOrInsert(2)
	tree.ReplaceOrInsert(3)
	tree.ReplaceOrInsert(4)
	tree.DeleteMin()
	tree.Delete(4)
	c := tree.IterAscend()
	for {
		u := <-c
		if u == nil {
			break
		}
		fmt.Printf("%d\n", int(u.(int)))
	}
}
