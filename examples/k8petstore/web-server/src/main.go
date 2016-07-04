/*
Copyright 2014 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"

	"github.com/codegangsta/negroni"
	"github.com/gorilla/mux"
	"github.com/xyproto/simpleredis"
)

//return the path to static assets (i.e. index.html)
func pathToStaticContents() string {
	var static_content = os.Getenv("STATIC_FILES")
	// Take a wild guess.  This will work in dev environment.
	if static_content == "" {
		println("*********** WARNING: DIDNT FIND ENV VAR 'STATIC_FILES', guessing your running in dev.")
		static_content = "../../static/"
	} else {
		println("=========== Read ENV 'STATIC_FILES', path to assets : " + static_content)
	}

	//Die if no the static files are missing.
	_, err := os.Stat(static_content)
	if err != nil {
		println("*********** os.Stat failed on " + static_content + " This means no static files are available.  Dying...")
		os.Exit(2)
	}
	return static_content
}

func main() {

	var connection = os.Getenv("REDISMASTER_SERVICE_HOST") + ":" + os.Getenv("REDISMASTER_SERVICE_PORT")

	if connection == ":" {
		print("WARNING ::: If in kube, this is a failure: Missing env variable REDISMASTER_SERVICE_HOST")
		print("WARNING ::: Attempting to connect redis localhost.")
		connection = "127.0.0.1:6379"
	} else {
		print("Found redis master host " + os.Getenv("REDISMASTER_SERVICE_PORT"))
		connection = os.Getenv("REDISMASTER_SERVICE_HOST") + ":" + os.Getenv("REDISMASTER_SERVICE_PORT")
	}

	println("Now connecting to : " + connection)
	/**
	 *  Create a connection pool.  ?The pool pointer will otherwise
	 *  not be of any use.?https://gist.github.com/jayunit100/1d00e6d343056401ef00
	 */
	pool = simpleredis.NewConnectionPoolHost(connection)

	println("Connection pool established : " + connection)

	defer pool.Close()

	r := mux.NewRouter()

	println("Router created ")

	/**
	 * Define a REST path.
	 *  - The parameters (key) can be accessed via mux.Vars.
	 *  - The Methods (GET) will be bound to a handler function.
	 */
	r.Path("/info").Methods("GET").HandlerFunc(InfoHandler)
	r.Path("/lrange/{key}").Methods("GET").HandlerFunc(ListRangeHandler)
	r.Path("/rpush/{key}/{value}").Methods("GET").HandlerFunc(ListPushHandler)
	r.Path("/llen").Methods("GET").HandlerFunc(LLENHandler)

	//for dev environment, the site is one level up...

	r.PathPrefix("/").Handler(http.FileServer(http.Dir(pathToStaticContents())))

	r.Path("/env").Methods("GET").HandlerFunc(EnvHandler)

	list := simpleredis.NewList(pool, "k8petstore")
	HandleError(nil, list.Add("jayunit100"))
	HandleError(nil, list.Add("tstclaire"))
	HandleError(nil, list.Add("rsquared"))

	// Verify that this is 3 on startup.
	infoL := HandleError(pool.Get(0).Do("LLEN", "k8petstore")).(int64)
	fmt.Printf("\n=========== Starting DB has %d elements \n", infoL)
	if infoL < 3 {
		print("Not enough entries in DB.  something is wrong w/ redis querying")
		print(infoL)
		panic("Failed ... ")
	}

	println("===========  Now launching negroni...this might take a second...")
	n := negroni.Classic()
	n.UseHandler(r)
	n.Run(":3000")
	println("Done ! Web app is now running.")

}

/**
* the Pool will be populated on startup,
* it will be an instance of a connection pool.
* Hence, we reference its address rather than copying.
 */
var pool *simpleredis.ConnectionPool

/**
*  REST
*  input: key
*
*  Writes  all members to JSON.
 */
func ListRangeHandler(rw http.ResponseWriter, req *http.Request) {
	println("ListRangeHandler")

	key := mux.Vars(req)["key"]

	list := simpleredis.NewList(pool, key)

	//members := HandleError(list.GetAll()).([]string)
	members := HandleError(list.GetLastN(4)).([]string)

	print(members)
	membersJSON := HandleError(json.MarshalIndent(members, "", "  ")).([]byte)

	print("RETURN MEMBERS = " + string(membersJSON))
	rw.Write(membersJSON)
}

func LLENHandler(rw http.ResponseWriter, req *http.Request) {
	println("=========== LLEN HANDLER")

	infoL := HandleError(pool.Get(0).Do("LLEN", "k8petstore")).(int64)
	fmt.Printf("=========== LLEN is %d ", infoL)
	lengthJSON := HandleError(json.MarshalIndent(infoL, "", "  ")).([]byte)
	fmt.Printf("================ LLEN json is %d", infoL)

	print("RETURN LEN = " + string(lengthJSON))
	rw.Write(lengthJSON)

}

func ListPushHandler(rw http.ResponseWriter, req *http.Request) {
	println("ListPushHandler")

	/**
	 *  Expect a key and value as input.
	 *
	 */
	key := mux.Vars(req)["key"]
	value := mux.Vars(req)["value"]

	println("New list " + key + " " + value)
	list := simpleredis.NewList(pool, key)
	HandleError(nil, list.Add(value))
	ListRangeHandler(rw, req)
}

func InfoHandler(rw http.ResponseWriter, req *http.Request) {
	println("InfoHandler")

	info := HandleError(pool.Get(0).Do("INFO")).([]byte)
	rw.Write(info)
}

func EnvHandler(rw http.ResponseWriter, req *http.Request) {
	println("EnvHandler")

	environment := make(map[string]string)
	for _, item := range os.Environ() {
		splits := strings.Split(item, "=")
		key := splits[0]
		val := strings.Join(splits[1:], "=")
		environment[key] = val
	}

	envJSON := HandleError(json.MarshalIndent(environment, "", "  ")).([]byte)
	rw.Write(envJSON)
}

func HandleError(result interface{}, err error) (r interface{}) {
	if err != nil {
		print("ERROR :  " + err.Error())
		//panic(err)
	}
	return result
}
