/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"

	"github.com/golang/glog"
	"github.com/spf13/pflag"
)

type Config struct {
	inputFile    string
	outputFile   string
	glossaryFile string
}

type GlossaryEntry struct {
	Words []string
	Link  string
}

type Glossary struct {
	Items []GlossaryEntry
}

type WordLocation struct {
	word string
	link string
	ix   int
}

func loadGlossary(file string) (*Glossary, error) {
	glossary := Glossary{}
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(data, &glossary); err != nil {
		return nil, err
	}
	return &glossary, nil
}

func buildWordMap(glossary *Glossary) (map[string]*GlossaryEntry, error) {
	result := map[string]*GlossaryEntry{}
	for ix := range glossary.Items {
		entry := &glossary.Items[ix]
		for _, word := range entry.Words {
			_, exists := result[word]
			if exists {
				return nil, fmt.Errorf("Duplicate word: %s", word)
			}
			result[word] = entry
		}
	}
	return result, nil
}

func indexOf(source, substr string, startingIndex int) int {
	ix := strings.Index(source[startingIndex:], substr)
	if ix == -1 {
		return ix
	}
	return ix + startingIndex
}

func main() {
	c := Config{}
	pflag.StringVar(&c.inputFile, "input-file", "", "The input file to process, '-' for stdin")
	pflag.StringVar(&c.outputFile, "output-file", "-", "The output file to process, '-' for stdout")
	pflag.StringVar(&c.glossaryFile, "glossary", "", "The glossary file to use.")

	pflag.Parse()

	if len(c.inputFile) == 0 {
		glog.Fatalf("--input-file is required.")
	}
	if len(c.outputFile) == 0 {
		glog.Fatal("--output-file is required.")
	}
	if len(c.glossaryFile) == 0 {
		glog.Fatal("--glossary is required.")
	}

	glossary, err := loadGlossary(c.glossaryFile)
	if err != nil {
		glog.Fatalf("Couldn't load glossary: %v", err)
	}
	wordMap, err := buildWordMap(glossary)
	if err != nil {
		glog.Fatalf("Couldn't build word map: %v", err)
	}
	var reader io.Reader
	if c.inputFile == "-" {
		reader = os.Stdin
	} else {
		var err error
		reader, err = os.Open(c.inputFile)
		if err != nil {
			glog.Fatalf("Can't open %s: %v", c.inputFile, err)
		}
	}

	var writer *os.File
	if c.outputFile == "-" {
		writer = os.Stdout
	} else {
		var err error
		writer, err = os.Open(c.outputFile)
		if err != nil {
			glog.Fatalf("Can't open %s for writing: %v", c.outputFile, err)
		}
		writer.Truncate(0)
	}

	scanner := bufio.NewScanner(reader)
	lines := []string{}
	inCode := false
	locs := map[string]WordLocation{}
	for lineIx := 0; scanner.Scan(); lineIx++ {
		line := scanner.Text()
		lines = append(lines, line)
		if strings.HasPrefix(line, "```") {
			inCode = !inCode
		}
		if inCode {
			continue
		}
		if len(line) > 0 && line[0] == '#' {
			continue
		}
		words := strings.Split(line, " ")
		charIx := 0
		for wordIx, word := range words {
			word := strings.TrimSpace(word)
			if entry, exists := wordMap[word]; exists {
				loc, found := locs[entry.Words[0]]
				kubernetesPrefix := wordIx != 0 && words[wordIx-1] == "Kubernetes"
				if found {
					if !kubernetesPrefix || strings.HasPrefix(loc.word, "Kubernetes") {
						break
					}
				}
				if kubernetesPrefix {
					word = "Kubernetes " + word
				}
				charIx = indexOf(line, word, charIx)
				// Skip already linkified or code blocks
				if charIx != -1 && charIx == 0 || (line[charIx-1] != '[' && line[charIx-1] != '`') {
					locs[entry.Words[0]] = WordLocation{word: word, ix: lineIx, link: entry.Link}
				}
			}
		}
	}
	for _, loc := range locs {
		replace := fmt.Sprintf("[%s](%s)", loc.word, loc.link)
		line := strings.Replace(lines[loc.ix], loc.word, replace, 1)
		lines[loc.ix] = line
	}
	for _, line := range lines {
		writer.Write([]byte(fmt.Sprintf("%s\n", line)))
	}
}
