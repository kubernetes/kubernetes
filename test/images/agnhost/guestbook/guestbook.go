/*
Copyright 2019 The Kubernetes Authors.

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

package guestbook

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/spf13/cobra"

	utilnet "k8s.io/apimachinery/pkg/util/net"
)

// CmdGuestbook is used by agnhost Cobra.
var CmdGuestbook = &cobra.Command{
	Use:   "guestbook",
	Short: "Creates a HTTP server with various endpoints representing a guestbook app",
	Long: `Starts a HTTP server on the given --http-port (default: 80), serving various endpoints representing a guestbook app. The endpoints and their purpose are:

- /register: A guestbook slave will subscribe to a master, to its given --slaveof endpoint. The master will then push any updates it receives to its registered slaves through the --backend-port.
- /get: Returns '{"data": value}', where the value is the stored value for the given key if non-empty, or the entire store.
- /set: Will set the given key-value pair in its own store and propagate it to its slaves, if any. Will return '{"data": "Updated"}' to the caller on success.
- /guestbook: Will proxy the request to agnhost-master if the given cmd is 'set', or agnhost-slave if the given cmd is 'get'.`,
	Args: cobra.MaximumNArgs(0),
	Run:  main,
}

var (
	httpPort    string
	backendPort string
	slaveOf     string
	slaves      []string
	store       map[string]interface{}
)

const (
	timeout = time.Duration(15) * time.Second
	sleep   = time.Duration(1) * time.Second
)

func init() {
	CmdGuestbook.Flags().StringVar(&httpPort, "http-port", "80", "HTTP Listen Port")
	CmdGuestbook.Flags().StringVar(&backendPort, "backend-port", "6379", "Backend's HTTP Listen Port")
	CmdGuestbook.Flags().StringVar(&slaveOf, "slaveof", "", "The host's name to register to")
	store = make(map[string]interface{})
}

func main(cmd *cobra.Command, args []string) {
	go registerNode(slaveOf, backendPort)
	startHTTPServer(httpPort)
}

func registerNode(registerTo, port string) {
	if registerTo == "" {
		return
	}

	hostPort := net.JoinHostPort(registerTo, backendPort)
	_, err := net.ResolveTCPAddr("tcp", hostPort)
	if err != nil {
		log.Fatalf("--slaveof param and/or --backend-port param are invalid. %v", err)
		return
	}

	start := time.Now()
	for time.Since(start) < timeout {
		response, err := dialHTTP("register", hostPort)
		if err != nil {
			log.Printf("encountered error while registering to master: %v. Retrying in 1 second.", err)
			time.Sleep(sleep)
			continue
		}

		responseJSON := make(map[string]interface{})
		err = json.Unmarshal([]byte(response), &responseJSON)
		if err != nil {
			log.Fatalf("Error while unmarshaling master's response: %v", err)
		}

		var ok bool
		store, ok = responseJSON["data"].(map[string]interface{})
		if !ok {
			log.Fatalf("Could not cast responseJSON: %s", responseJSON["data"])
		}
		log.Printf("Registered to node: %s", registerTo)
		return
	}

	log.Fatal("Timed out while registering to master.")
}

func startHTTPServer(port string) {
	http.HandleFunc("/register", registerHandler)
	http.HandleFunc("/get", getHandler)
	http.HandleFunc("/set", setHandler)
	http.HandleFunc("/guestbook", guestbookHandler)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%s", port), nil))
}

// registerHandler will register the caller in this server's list of slaves.
// /set requests will be propagated to slaves, if any.
func registerHandler(w http.ResponseWriter, r *http.Request) {
	ip, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		fmt.Fprintf(w, "userip: %q is not IP:port", r.RemoteAddr)
		return
	}
	log.Printf("GET /register, IP: %s", ip)

	// send all the store to the slave as well.
	output := make(map[string]interface{})
	output["data"] = store
	bytes, err := json.Marshal(output)
	if err != nil {
		http.Error(w, fmt.Sprintf("response could not be serialized. %v", err), http.StatusExpectationFailed)
		return
	}
	fmt.Fprint(w, string(bytes))
	slaves = append(slaves, ip)
	log.Printf("Node '%s' registered.", ip)
}

// getHandler will return '{"data": value}', where value is the stored value for
// the given key if non-empty, or entire store.
func getHandler(w http.ResponseWriter, r *http.Request) {
	values, err := url.Parse(r.URL.RequestURI())
	if err != nil {
		http.Error(w, fmt.Sprintf("%v", err), http.StatusBadRequest)
		return
	}

	key := values.Query().Get("key")

	log.Printf("GET /get?key=%s", key)

	output := make(map[string]interface{})
	if key == "" {
		output["data"] = store
	} else {
		value, found := store[key]
		if !found {
			value = ""
		}
		output["data"] = value
	}

	bytes, err := json.Marshal(output)
	if err == nil {
		fmt.Fprint(w, string(bytes))
	} else {
		http.Error(w, fmt.Sprintf("response could not be serialized. %v", err), http.StatusExpectationFailed)
	}
}

// setHandler will set the given key-value pair in its own store and propagate
// it to its slaves, if any. Will return '{"message": "Updated"}' to the caller on success.
func setHandler(w http.ResponseWriter, r *http.Request) {
	values, err := url.Parse(r.URL.RequestURI())
	if err != nil {
		http.Error(w, fmt.Sprintf("%v", err), http.StatusBadRequest)
		return
	}

	key := values.Query().Get("key")
	value := values.Query().Get("value")

	log.Printf("GET /set?key=%s&value=%s", key, value)

	if key == "" {
		http.Error(w, "cannot set with empty key.", http.StatusBadRequest)
		return
	}

	store[key] = value
	request := fmt.Sprintf("set?key=%s&value=%s", key, value)
	for _, slave := range slaves {
		hostPort := net.JoinHostPort(slave, backendPort)
		_, err = dialHTTP(request, hostPort)
		if err != nil {
			http.Error(w, fmt.Sprintf("encountered error while propagating to slave '%s': %v", slave, err), http.StatusExpectationFailed)
			return
		}
	}

	output := map[string]string{}
	output["message"] = "Updated"
	bytes, err := json.Marshal(output)
	if err == nil {
		fmt.Fprint(w, string(bytes))
	} else {
		http.Error(w, fmt.Sprintf("response could not be serialized. %v", err), http.StatusExpectationFailed)
	}
}

// guestbookHandler will proxy the request to agnhost-master if the given cmd is
// 'set' or agnhost-slave if the given cmd is 'get'.
func guestbookHandler(w http.ResponseWriter, r *http.Request) {
	values, err := url.Parse(r.URL.RequestURI())
	if err != nil {
		http.Error(w, fmt.Sprintf("%v", err), http.StatusBadRequest)
		return
	}

	cmd := strings.ToLower(values.Query().Get("cmd"))
	key := values.Query().Get("key")
	value := values.Query().Get("value")

	log.Printf("GET /guestbook?cmd=%s&key=%s&value=%s", cmd, key, value)

	if cmd != "get" && cmd != "set" {
		http.Error(w, fmt.Sprintf("unsupported cmd: '%s'", cmd), http.StatusBadRequest)
		return
	}
	if cmd == "set" && key == "" {
		http.Error(w, "cannot set with empty key.", http.StatusBadRequest)
		return
	}

	host := "agnhost-master"
	if cmd == "get" {
		host = "agnhost-slave"
	}

	hostPort := net.JoinHostPort(host, backendPort)
	_, err = net.ResolveTCPAddr("tcp", hostPort)
	if err != nil {
		http.Error(w, fmt.Sprintf("host and/or port param are invalid. %v", err), http.StatusBadRequest)
		return
	}

	request := fmt.Sprintf("%s?key=%s&value=%s", cmd, key, value)
	response, err := dialHTTP(request, hostPort)
	if err == nil {
		fmt.Fprint(w, response)
	} else {
		http.Error(w, fmt.Sprintf("encountered error: %v", err), http.StatusExpectationFailed)
	}
}

func dialHTTP(request, hostPort string) (string, error) {
	transport := utilnet.SetTransportDefaults(&http.Transport{})
	httpClient := createHTTPClient(transport)
	resp, err := httpClient.Get(fmt.Sprintf("http://%s/%s", hostPort, request))
	defer transport.CloseIdleConnections()
	if err == nil {
		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err == nil {
			return string(body), nil
		}
	}
	return "", err
}

func createHTTPClient(transport *http.Transport) *http.Client {
	client := &http.Client{
		Transport: transport,
		Timeout:   5 * time.Second,
	}
	return client
}
