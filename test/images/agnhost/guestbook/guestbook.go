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
	"io"
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

- /register: A guestbook replica will subscribe to a primary, to its given --replicaof endpoint. The primary will then push any updates it receives to its registered replicas through the --backend-port.
- /get: Returns '{"data": value}', where the value is the stored value for the given key if non-empty, or the entire store.
- /set: Will set the given key-value pair in its own store and propagate it to its replicas, if any. Will return '{"data": "Updated"}' to the caller on success.
- /guestbook: Will proxy the request to agnhost-primary if the given cmd is 'set', or agnhost-replica if the given cmd is 'get'.`,
	Args: cobra.MaximumNArgs(0),
	Run:  main,
}

var (
	httpPort    string
	backendPort string
	replicaOf   string
	replicas    []string
	store       map[string]interface{}
)

const (
	timeout = time.Duration(15) * time.Second
	sleep   = time.Duration(1) * time.Second
)

func init() {
	CmdGuestbook.Flags().StringVar(&httpPort, "http-port", "80", "HTTP Listen Port")
	CmdGuestbook.Flags().StringVar(&backendPort, "backend-port", "6379", "Backend's HTTP Listen Port")
	CmdGuestbook.Flags().StringVar(&replicaOf, "replicaof", "", "The host's name to register to")
	store = make(map[string]interface{})
}

func main(cmd *cobra.Command, args []string) {
	go registerNode(replicaOf, backendPort)
	startHTTPServer(httpPort)
}

func registerNode(registerTo, port string) {
	if registerTo == "" {
		return
	}

	hostPort := net.JoinHostPort(registerTo, backendPort)

	start := time.Now()
	for time.Since(start) < timeout {
		host, err := getIP(hostPort)
		if err != nil {
			log.Printf("unable to get IP %s: %v. Retrying in %s.", hostPort, err, sleep)
			time.Sleep(sleep)
			continue
		}

		request := fmt.Sprintf("register?host=%s", host.String())
		log.Printf("Registering to primary: %s/%s", hostPort, request)
		_, err = net.ResolveTCPAddr("tcp", hostPort)
		if err != nil {
			log.Printf("unable to resolve %s, --replicaof param and/or --backend-port param are invalid: %v. Retrying in %s.", hostPort, err, sleep)
			time.Sleep(sleep)
			continue
		}

		response, err := dialHTTP(request, hostPort)
		if err != nil {
			log.Printf("encountered error while registering to primary: %v. Retrying in %s.", err, sleep)
			time.Sleep(sleep)
			continue
		}

		responseJSON := make(map[string]interface{})
		err = json.Unmarshal([]byte(response), &responseJSON)
		if err != nil {
			log.Fatalf("Error while unmarshaling primary's response: %v", err)
		}

		var ok bool
		store, ok = responseJSON["data"].(map[string]interface{})
		if !ok {
			log.Fatalf("Could not cast responseJSON: %s", responseJSON["data"])
		}
		log.Printf("Registered to node: %s", registerTo)
		return
	}

	log.Fatal("Timed out while registering to primary.")
}

func startHTTPServer(port string) {
	http.HandleFunc("/register", registerHandler)
	http.HandleFunc("/get", getHandler)
	http.HandleFunc("/set", setHandler)
	http.HandleFunc("/guestbook", guestbookHandler)
	log.Fatal(http.ListenAndServe(fmt.Sprintf(":%s", port), nil))
}

// registerHandler will register the caller in this server's list of replicas.
// /set requests will be propagated to replicas, if any.
func registerHandler(w http.ResponseWriter, r *http.Request) {
	values, err := url.Parse(r.URL.RequestURI())
	if err != nil {
		http.Error(w, fmt.Sprintf("%v", err), http.StatusBadRequest)
		return
	}

	ip := values.Query().Get("host")
	log.Printf("GET /register?host=%s", ip)

	// send all the store to the replica as well.
	output := make(map[string]interface{})
	output["data"] = store
	bytes, err := json.Marshal(output)
	if err != nil {
		http.Error(w, fmt.Sprintf("response could not be serialized. %v", err), http.StatusExpectationFailed)
		return
	}
	fmt.Fprint(w, string(bytes))
	replicas = append(replicas, ip)
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
// it to its replicas, if any. Will return '{"message": "Updated"}' to the caller on success.
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
	for _, replica := range replicas {
		hostPort := net.JoinHostPort(replica, backendPort)
		_, err = dialHTTP(request, hostPort)
		if err != nil {
			http.Error(w, fmt.Sprintf("encountered error while propagating to replica '%s': %v", replica, err), http.StatusExpectationFailed)
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

// guestbookHandler will proxy the request to agnhost-primary if the given cmd is
// 'set' or agnhost-replica if the given cmd is 'get'.
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

	host := "agnhost-primary"
	if cmd == "get" {
		host = "agnhost-replica"
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
		body, err := io.ReadAll(resp.Body)
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

func getIP(hostPort string) (net.IP, error) {
	conn, err := net.Dial("udp", hostPort)
	if err != nil {
		return []byte{}, err
	}
	defer conn.Close()

	localAddr := conn.LocalAddr().(*net.UDPAddr)
	return localAddr.IP, nil
}
