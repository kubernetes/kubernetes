package main

import (
	"fmt"
	"math/rand"
	"os"
	"strconv"
	"time"
)

var outQuote = "We've done the impossible, and that makes us mighty."
var errQuote = "Ah, curse your sudden but inevitable betrayal!"

var randomQuotes = []string{
	"Can we maybe vote on the whole murdering people issue?",
	"I swear by my pretty floral bonnet, I will end you.",
	"My work's illegal, but at least it's honest.",
}

func main() {
	fmt.Fprintln(os.Stdout, outQuote)
	fmt.Fprintln(os.Stderr, errQuote)

	randomIndex := rand.New(rand.NewSource(time.Now().UnixNano())).Intn(len(randomQuotes))

	time.Sleep(100 * time.Millisecond)

	fmt.Fprintln(os.Stdout, randomQuotes[randomIndex])

	if len(os.Args) == 2 {
		exitCode, _ := strconv.Atoi(os.Args[1])
		os.Exit(exitCode)
	} else {
		os.Exit(randomIndex)
	}
}
