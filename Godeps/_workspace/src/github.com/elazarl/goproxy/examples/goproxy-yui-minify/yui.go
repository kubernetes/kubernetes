// This example would minify standalone Javascript files (identified by their content type)
// using the command line utility YUI compressor http://yui.github.io/yuicompressor/
// Example usage:
//
//    ./yui -java /usr/local/bin/java -yuicompressor ~/Downloads/yuicompressor-2.4.8.jar
//    $ curl -vx localhost:8080  http://golang.org/lib/godoc/godocs.js
//    (function(){function g(){var u=$("#search");if(u.length===0){return}function t(){if(....
//    $ curl http://golang.org/lib/godoc/godocs.js | head -n 3
//    // Copyright 2012 The Go Authors. All rights reserved.
//    // Use of this source code is governed by a BSD-style
//    // license that can be found in the LICENSE file.
package main

import (
	"flag"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"os"
	"os/exec"
	"path"
	"strings"

	"github.com/elazarl/goproxy"
)

func main() {
	verbose := flag.Bool("v", false, "should every proxy request be logged to stdout")
	addr := flag.String("addr", ":8080", "proxy listen address")
	java := flag.String("javapath", "java", "where the Java executable is located")
	yuicompressor := flag.String("yuicompressor", "", "where the yuicompressor is located, assumed to be in CWD")
	yuicompressordir := flag.String("yuicompressordir", ".", "a folder to search yuicompressor in, will be ignored if yuicompressor is set")
	flag.Parse()
	if *yuicompressor == "" {
		files, err := ioutil.ReadDir(*yuicompressordir)
		if err != nil {
			log.Fatal("Cannot find yuicompressor jar")
		}
		for _, file := range files {
			if strings.HasPrefix(file.Name(), "yuicompressor") && strings.HasSuffix(file.Name(), ".jar") {
				c := path.Join(*yuicompressordir, file.Name())
				yuicompressor = &c
				break
			}
		}
	}
	if *yuicompressor == "" {
		log.Fatal("Can't find yuicompressor jar, searched yuicompressor*.jar in dir ", *yuicompressordir)
	}
	if _, err := os.Stat(*yuicompressor); os.IsNotExist(err) {
		log.Fatal("Can't find yuicompressor jar specified ", *yuicompressor)
	}
	proxy := goproxy.NewProxyHttpServer()
	proxy.Verbose = *verbose
	proxy.OnResponse().DoFunc(func(resp *http.Response, ctx *goproxy.ProxyCtx) *http.Response {
		contentType := resp.Header.Get("Content-Type")
		if contentType == "application/javascript" || contentType == "application/x-javascript" {
			// in real code, response should be streamed as well
			var err error
			cmd := exec.Command(*java, "-jar", *yuicompressor, "--type", "js")
			cmd.Stdin = resp.Body
			resp.Body, err = cmd.StdoutPipe()
			if err != nil {
				ctx.Warnf("Cannot minify content in %v: %v", ctx.Req.URL, err)
				return goproxy.TextResponse(ctx.Req, "Error getting stdout pipe")
			}
			stderr, err := cmd.StderrPipe()
			if err != nil {
				ctx.Logf("Error obtaining stderr from yuicompress: %s", err)
				return goproxy.TextResponse(ctx.Req, "Error getting stderr pipe")
			}
			if err := cmd.Start(); err != nil {
				ctx.Warnf("Cannot minify content in %v: %v", ctx.Req.URL, err)
			}
			go func() {
				defer stderr.Close()
				const kb = 1024
				msg, err := ioutil.ReadAll(&io.LimitedReader{stderr, 50 * kb})
				if len(msg) != 0 {
					ctx.Logf("Error executing yuicompress: %s", string(msg))
				}
				if err != nil {
					ctx.Logf("Error reading stderr from yuicompress: %s", string(msg))
				}
			}()
		}
		return resp
	})
	log.Fatal(http.ListenAndServe(*addr, proxy))
}
