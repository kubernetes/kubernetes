/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// Add k8s style header for api-reference html docs; And fix the alignment.

package main

import (
	"flag"
	"golang.org/x/net/html"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
)

var (
	templateHTML = flag.String("templateHTML", "http://kubernetes.io/v1.0/index.html", "URL of the html file that has the k8s header")
)

func detachNode(n *html.Node) {
	n.Parent = nil
	n.NextSibling = nil
	n.PrevSibling = nil
}

func addHeader(filename string) error {
	//extract header from template URL
	response, err := http.Get(*templateHTML)
	if err != nil {
		return err
	}
	defer response.Body.Close()
	doc, err := html.Parse(response.Body)
	if err != nil {
		return err
	}
	//<head>
	headNode := doc.FirstChild.NextSibling.FirstChild

	bodyNode := headNode.NextSibling.NextSibling
	//<header id="nav" class="mobile-menu-slide">
	headerNode := bodyNode.FirstChild.NextSibling
	//<div id="mobile-nav-container" class="visible-sm-block visible-xs-block mobile-menu-slide">
	navContainerDivNode := headerNode.NextSibling.NextSibling

	targetHTML, err := os.OpenFile(filename, os.O_RDWR, 0666)
	if err != nil {
		return err
	}
	defer targetHTML.Close()
	target, err := html.Parse(targetHTML)
	if err != nil {
		return err
	}
	//<head>
	htmlNode2 := target.FirstChild.NextSibling

	headNode2 := htmlNode2.FirstChild

	//append source <head> after target <head>
	detachNode(headNode)
	htmlNode2.InsertBefore(headNode, headNode2)
	bodyNode2 := headNode2.NextSibling.NextSibling
	//insert nav header
	detachNode(headerNode)
	bodyNode2.InsertBefore(headerNode, bodyNode2.FirstChild)
	//insert navContainerDivNode as target bodyNode's first child. Need to clean navContainerDivNode's Parent and Siblings first.
	detachNode(navContainerDivNode)
	bodyNode2.InsertBefore(navContainerDivNode, bodyNode2.FirstChild)

	if err := targetHTML.Truncate(0); err != nil {
		return err
	}
	if _, err := targetHTML.Seek(0, 0); err != nil {
		return err
	}
	err = html.Render(targetHTML, target)
	return err
}

//TODO: These really should be fixed when generating the original html files
func fixHeadAlign(filename string) error {
	file, err := ioutil.ReadFile(filename)
	if err != nil {
		return err
	}
	newContents := strings.Replace(string(file), "h2 { font-size: 2.3125em; }", "h2 { font-size: 2.3125em; text-align: left;}", 1)
	newContents = strings.Replace(newContents, "h4 { font-size: 1.4375em; } }", "h4 { font-size: 1.4375em; text-align: left;} }", 1)
	newContents = strings.Replace(newContents, "<h2 id=\"_paths\">Paths</h2>", "<h2 id=\"_paths\">Operations</h2>", 1)

	if err = ioutil.WriteFile(filename, []byte(newContents), 0666); err != nil {
		return err
	}
	return nil
}
