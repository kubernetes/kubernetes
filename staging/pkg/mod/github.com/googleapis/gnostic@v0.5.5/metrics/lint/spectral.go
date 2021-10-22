package linter

import (
	"bufio"
	"log"
	"os"
	"regexp"
	"strconv"
)

// parseOutput creates a new Message struct given a message type, message, path
// and line. The new struct is then returned.
func parseOutput(output []string) []*Message {
	messages := make([]*Message, 0)
	for _, line := range output {
		array := regexp.MustCompile("[]: *]").Split(line, 6)
		line, _ := strconv.ParseInt(array[1], 0, 64)
		temp := &Message{
			Type:    array[3],
			Message: array[5],
			Line:    int32(line),
		}
		messages = append(messages, temp)
	}
	return messages
}

// Text takes the name of the filename that contains the linter results
// from the spectral linter and parses it into a string slice
func openAndReadText(filename string) []string {
	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	output := make([]string, 0)

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		output = append(output, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		log.Fatal(err)
	}
	return output
}

// LintSpectral functions serves as a linter results translater. The function takes the filename
// which contains the text results of Stoplights's spectral and creates a new instance of
// the linter struct using the text data.
func LintSpectral(filename string) {
	output := openAndReadText(filename)
	messages := parseOutput(output)
	linterResult := &Linter{
		Messages: messages,
	}
	writePb(linterResult)
}
