// Copyright (c) 2013 Phillip Bond
// Licensed under the MIT License
// see file LICENSE

package systemstat_test

import (
	"bitbucket.org/bertimus9/systemstat"
	"fmt"
)

var sample systemstat.MemSample

// This example shows how easy it is to get memory information
func Example_simple() {
	sample = systemstat.GetMemSample()
	fmt.Println("Total available RAM in kb:", sample.MemTotal, "k total")
	fmt.Println("Used RAM in kb:", sample.MemUsed, "k used")
	fmt.Println("Free RAM in kb:", sample.MemFree, "k free")
	fmt.Printf("The output is similar to, but somewhat different than:\n\ttop -n1 | grep Mem:\n")
}
