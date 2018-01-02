// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"bufio"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/textproto"
	"os"
	"strings"

	pubsub "google.golang.org/api/pubsub/v1beta2"
)

const USAGE = `Available arguments are:
    <project_id> list_topics
    <project_id> create_topic <topic>
    <project_id> delete_topic <topic>
    <project_id> list_subscriptions
    <project_id> create_subscription <subscription> <linked topic>
    <project_id> delete_subscription <subscription>
    <project_id> connect_irc <topic> <server> <channel>
    <project_id> pull_messages <subscription>
`

type IRCBot struct {
	server   string
	port     string
	nick     string
	user     string
	channel  string
	conn     net.Conn
	tpReader *textproto.Reader
}

func NewIRCBot(server, channel, nick string) *IRCBot {
	return &IRCBot{
		server:  server,
		port:    "6667",
		nick:    nick,
		channel: channel,
		conn:    nil,
		user:    nick,
	}
}

func (bot *IRCBot) Connect() {
	conn, err := net.Dial("tcp", bot.server+":"+bot.port)
	if err != nil {
		log.Fatal("unable to connect to IRC server ", err)
	}
	bot.conn = conn
	log.Printf("Connected to IRC server %s (%s)\n",
		bot.server, bot.conn.RemoteAddr())
	bot.tpReader = textproto.NewReader(bufio.NewReader(bot.conn))
	bot.Sendf("USER %s 8 * :%s\r\n", bot.nick, bot.nick)
	bot.Sendf("NICK %s\r\n", bot.nick)
	bot.Sendf("JOIN %s\r\n", bot.channel)
}

func (bot *IRCBot) CheckConnection() {
	for {
		line, err := bot.ReadLine()
		if err != nil {
			log.Fatal("Unable to read a line during checking the connection.")
		}
		if parts := strings.Split(line, " "); len(parts) > 1 {
			if parts[1] == "004" {
				log.Println("The nick accepted.")
			} else if parts[1] == "433" {
				log.Fatalf("The nick is already in use: %s", line)
			} else if parts[1] == "366" {
				log.Println("Starting to publish messages.")
				return
			}
		}
	}
}

func (bot *IRCBot) Sendf(format string, args ...interface{}) {
	fmt.Fprintf(bot.conn, format, args...)
}

func (bot *IRCBot) Close() {
	bot.conn.Close()
}

func (bot *IRCBot) ReadLine() (line string, err error) {
	return bot.tpReader.ReadLine()
}

func init() {
	registerDemo("pubsub", pubsub.PubsubScope, pubsubMain)
}

func pubsubUsage() {
	fmt.Fprint(os.Stderr, USAGE)
}

// Returns a fully qualified resource name for Cloud Pub/Sub.
func fqrn(res, proj, name string) string {
	return fmt.Sprintf("projects/%s/%s/%s", proj, res, name)
}

func fullTopicName(proj, topic string) string {
	return fqrn("topics", proj, topic)
}

func fullSubName(proj, topic string) string {
	return fqrn("subscriptions", proj, topic)
}

// Check the length of the arguments.
func checkArgs(argv []string, min int) {
	if len(argv) < min {
		pubsubUsage()
		os.Exit(2)
	}
}

func listTopics(service *pubsub.Service, argv []string) {
	next := ""
	for {
		topicsList, err := service.Projects.Topics.List(fmt.Sprintf("projects/%s", argv[0])).PageToken(next).Do()
		if err != nil {
			log.Fatalf("listTopics query.Do() failed: %v", err)
		}
		for _, topic := range topicsList.Topics {
			fmt.Println(topic.Name)
		}
		next = topicsList.NextPageToken
		if next == "" {
			break
		}
	}
}

func createTopic(service *pubsub.Service, argv []string) {
	checkArgs(argv, 3)
	topic, err := service.Projects.Topics.Create(fullTopicName(argv[0], argv[2]), &pubsub.Topic{}).Do()
	if err != nil {
		log.Fatalf("createTopic Create().Do() failed: %v", err)
	}
	fmt.Printf("Topic %s was created.\n", topic.Name)
}

func deleteTopic(service *pubsub.Service, argv []string) {
	checkArgs(argv, 3)
	topicName := fullTopicName(argv[0], argv[2])
	if _, err := service.Projects.Topics.Delete(topicName).Do(); err != nil {
		log.Fatalf("deleteTopic Delete().Do() failed: %v", err)
	}
	fmt.Printf("Topic %s was deleted.\n", topicName)
}

func listSubscriptions(service *pubsub.Service, argv []string) {
	next := ""
	for {
		subscriptionsList, err := service.Projects.Subscriptions.List(fmt.Sprintf("projects/%s", argv[0])).PageToken(next).Do()
		if err != nil {
			log.Fatalf("listSubscriptions query.Do() failed: %v", err)
		}
		for _, subscription := range subscriptionsList.Subscriptions {
			sub_text, _ := json.MarshalIndent(subscription, "", "  ")
			fmt.Printf("%s\n", sub_text)
		}
		next = subscriptionsList.NextPageToken
		if next == "" {
			break
		}
	}
}

func createSubscription(service *pubsub.Service, argv []string) {
	checkArgs(argv, 4)
	name := fullSubName(argv[0], argv[2])
	sub := &pubsub.Subscription{Topic: fullTopicName(argv[0], argv[3])}
	subscription, err := service.Projects.Subscriptions.Create(name, sub).Do()
	if err != nil {
		log.Fatalf("createSubscription Create().Do() failed: %v", err)
	}
	fmt.Printf("Subscription %s was created.\n", subscription.Name)
}

func deleteSubscription(service *pubsub.Service, argv []string) {
	checkArgs(argv, 3)
	name := fullSubName(argv[0], argv[2])
	if _, err := service.Projects.Subscriptions.Delete(name).Do(); err != nil {
		log.Fatalf("deleteSubscription Delete().Do() failed: %v", err)
	}
	fmt.Printf("Subscription %s was deleted.\n", name)
}

func connectIRC(service *pubsub.Service, argv []string) {
	checkArgs(argv, 5)
	topicName := fullTopicName(argv[0], argv[2])
	server := argv[3]
	channel := argv[4]
	nick := fmt.Sprintf("bot-%s", argv[2])
	ircbot := NewIRCBot(server, channel, nick)
	ircbot.Connect()
	defer ircbot.Close()
	ircbot.CheckConnection()
	privMark := fmt.Sprintf("PRIVMSG %s :", ircbot.channel)
	for {
		line, err := ircbot.ReadLine()
		if err != nil {
			log.Fatal("Unable to read a line from the connection.")
		}
		parts := strings.Split(line, " ")
		if len(parts) > 0 && parts[0] == "PING" {
			ircbot.Sendf("PONG %s\r\n", parts[1])
		} else {
			pos := strings.Index(line, privMark)
			if pos == -1 {
				continue
			}
			privMsg := line[pos+len(privMark) : len(line)]
			pubsubMessage := &pubsub.PubsubMessage{
				Data: base64.StdEncoding.EncodeToString([]byte(privMsg)),
			}
			publishRequest := &pubsub.PublishRequest{
				Messages: []*pubsub.PubsubMessage{pubsubMessage},
			}
			if _, err := service.Projects.Topics.Publish(topicName, publishRequest).Do(); err != nil {
				log.Fatalf("connectIRC Publish().Do() failed: %v", err)
			}
			log.Println("Published a message to the topic.")
		}
	}
}

func pullMessages(service *pubsub.Service, argv []string) {
	checkArgs(argv, 3)
	subName := fullSubName(argv[0], argv[2])
	pullRequest := &pubsub.PullRequest{
		ReturnImmediately: false,
		MaxMessages:       1,
	}
	for {
		pullResponse, err := service.Projects.Subscriptions.Pull(subName, pullRequest).Do()
		if err != nil {
			log.Fatalf("pullMessages Pull().Do() failed: %v", err)
		}
		for _, receivedMessage := range pullResponse.ReceivedMessages {
			data, err := base64.StdEncoding.DecodeString(receivedMessage.Message.Data)
			if err != nil {
				log.Fatalf("pullMessages DecodeString() failed: %v", err)
			}
			fmt.Printf("%s\n", data)
			ackRequest := &pubsub.AcknowledgeRequest{
				AckIds: []string{receivedMessage.AckId},
			}
			if _, err = service.Projects.Subscriptions.Acknowledge(subName, ackRequest).Do(); err != nil {
				log.Printf("pullMessages Acknowledge().Do() failed: %v", err)
			}
		}
	}
}

// This example demonstrates calling the Cloud Pub/Sub API. As of 20
// Aug 2014, the Cloud Pub/Sub API is only available if you're
// whitelisted. If you're interested in using it, please apply for the
// Limited Preview program at the following form:
// http://goo.gl/Wql9HL
//
// Also, before running this example, be sure to enable Cloud Pub/Sub
// service on your project in Developer Console at:
// https://console.developers.google.com/
//
// It has 8 subcommands as follows:
//
//  <project_id> list_topics
//  <project_id> create_topic <topic>
//  <project_id> delete_topic <topic>
//  <project_id> list_subscriptions
//  <project_id> create_subscription <subscription> <linked topic>
//  <project_id> delete_subscription <subscription>
//  <project_id> connect_irc <topic> <server> <channel>
//  <project_id> pull_messages <subscription>
//
// You can use either of your alphanumerical or numerial Cloud Project
// ID for project_id. You can choose any names for topic and
// subscription as long as they follow the naming rule described at:
// https://developers.google.com/pubsub/overview#names
//
// You can list/create/delete topics/subscriptions by self-explanatory
// subcommands, as well as connect to an IRC channel and publish
// messages from the IRC channel to a specified Cloud Pub/Sub topic by
// the "connect_irc" subcommand, or continuously pull messages from a
// specified Cloud Pub/Sub subscription and display the data by the
// "pull_messages" subcommand.
func pubsubMain(client *http.Client, argv []string) {
	checkArgs(argv, 2)
	service, err := pubsub.New(client)
	if err != nil {
		log.Fatalf("Unable to create PubSub service: %v", err)
	}

	m := map[string]func(service *pubsub.Service, argv []string){
		"list_topics":         listTopics,
		"create_topic":        createTopic,
		"delete_topic":        deleteTopic,
		"list_subscriptions":  listSubscriptions,
		"create_subscription": createSubscription,
		"delete_subscription": deleteSubscription,
		"connect_irc":         connectIRC,
		"pull_messages":       pullMessages,
	}
	f, ok := m[argv[1]]
	if !ok {
		pubsubUsage()
		os.Exit(2)
	}
	f(service, argv)
}
