// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"strings"

	gmail "google.golang.org/api/gmail/v1"
)

func init() {
	registerDemo("gmail", gmail.MailGoogleComScope, gmailMain)
}

type message struct {
	size    int64
	gmailID string
	date    string // retrieved from message header
	snippet string
}

// gmailMain is an example that demonstrates calling the Gmail API.
// It iterates over all messages of a user that are larger
// than 5MB, sorts them by size, and then interactively asks the user to
// choose either to Delete, Skip, or Quit for each message.
//
// Example usage:
//   go build -o go-api-demo *.go
//   go-api-demo -clientid="my-clientid" -secret="my-secret" gmail
func gmailMain(client *http.Client, argv []string) {
	if len(argv) != 0 {
		fmt.Fprintln(os.Stderr, "Usage: gmail")
		return
	}

	svc, err := gmail.New(client)
	if err != nil {
		log.Fatalf("Unable to create Gmail service: %v", err)
	}

	var total int64
	msgs := []message{}
	pageToken := ""
	for {
		req := svc.Users.Messages.List("me").Q("larger:5M")
		if pageToken != "" {
			req.PageToken(pageToken)
		}
		r, err := req.Do()
		if err != nil {
			log.Fatalf("Unable to retrieve messages: %v", err)
		}

		log.Printf("Processing %v messages...\n", len(r.Messages))
		for _, m := range r.Messages {
			msg, err := svc.Users.Messages.Get("me", m.Id).Do()
			if err != nil {
				log.Fatalf("Unable to retrieve message %v: %v", m.Id, err)
			}
			total += msg.SizeEstimate
			date := ""
			for _, h := range msg.Payload.Headers {
				if h.Name == "Date" {
					date = h.Value
					break
				}
			}
			msgs = append(msgs, message{
				size:    msg.SizeEstimate,
				gmailID: msg.Id,
				date:    date,
				snippet: msg.Snippet,
			})
		}

		if r.NextPageToken == "" {
			break
		}
		pageToken = r.NextPageToken
	}
	log.Printf("total: %v\n", total)

	sortBySize(msgs)
	reader := bufio.NewReader(os.Stdin)
	count, deleted := 0, 0
	for _, m := range msgs {
		count++
		fmt.Printf("\nMessage URL: https://mail.google.com/mail/u/0/#all/%v\n", m.gmailID)
		fmt.Printf("Size: %v, Date: %v, Snippet: %q\n", m.size, m.date, m.snippet)
		fmt.Printf("Options: (d)elete, (s)kip, (q)uit: [s] ")
		val := ""
		if val, err = reader.ReadString('\n'); err != nil {
			log.Fatalf("unable to scan input: %v", err)
		}
		val = strings.TrimSpace(val)
		switch val {
		case "d": // delete message
			if err := svc.Users.Messages.Delete("me", m.gmailID).Do(); err != nil {
				log.Fatalf("unable to delete message %v: %v", m.gmailID, err)
			}
			log.Printf("Deleted message %v.\n", m.gmailID)
			deleted++
		case "q": // quit
			log.Printf("Done.  %v messages processed, %v deleted\n", count, deleted)
			os.Exit(0)
		default:
		}
	}
}

type messageSorter struct {
	msg  []message
	less func(i, j message) bool
}

func sortBySize(msg []message) {
	sort.Sort(messageSorter{msg, func(i, j message) bool {
		return i.size > j.size
	}})
}

func (s messageSorter) Len() int {
	return len(s.msg)
}

func (s messageSorter) Swap(i, j int) {
	s.msg[i], s.msg[j] = s.msg[j], s.msg[i]
}

func (s messageSorter) Less(i, j int) bool {
	return s.less(s.msg[i], s.msg[j])
}
