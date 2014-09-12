package main

import (
	"encoding/json"
	"net/http"
	"os"
	"strings"

	"github.com/codegangsta/negroni"
	"github.com/gorilla/mux"
	"github.com/xyproto/simpleredis"
)

var pool *simpleredis.ConnectionPool

func ListRangeHandler(rw http.ResponseWriter, req *http.Request) {
	members, err := simpleredis.NewList(pool, mux.Vars(req)["key"]).GetAll()
	if err != nil {
		panic(err)
	}

	membersJSON, err := json.MarshalIndent(members, "", "  ")
	if err != nil {
		panic(err)
	}

	rw.WriteHeader(200)
	rw.Header().Set("Content-Type", "application/json")
	rw.Write([]byte(membersJSON))
}

func ListPushHandler(rw http.ResponseWriter, req *http.Request) {
	set := simpleredis.NewList(pool, mux.Vars(req)["key"])
	err := set.Add(mux.Vars(req)["value"])
	if err != nil {
		panic(err)
	}

	members, err := set.GetAll()
	if err != nil {
		panic(err)
	}

	membersJSON, err := json.MarshalIndent(members, "", "  ")
	if err != nil {
		panic(err)
	}

	rw.WriteHeader(200)
	rw.Header().Set("Content-Type", "application/json")
	rw.Write([]byte(membersJSON))
}

func InfoHandler(rw http.ResponseWriter, req *http.Request) {
	info, err := pool.Get(0).Do("INFO")
	if err != nil {
		panic(err)
	}

	infoString := string(info.([]uint8))

	rw.WriteHeader(200)
	rw.Write([]byte(infoString))
}

func EnvHandler(rw http.ResponseWriter, req *http.Request) {
	getenvironment := func(data []string, getkeyval func(item string) (key, val string)) map[string]string {
		items := make(map[string]string)
		for _, item := range data {
			key, val := getkeyval(item)
			items[key] = val
		}
		return items
	}
	environment := getenvironment(os.Environ(), func(item string) (key, val string) {
		splits := strings.Split(item, "=")
		key = splits[0]
		val = strings.Join(splits[1:], "=")
		return
	})

	envJSON, err := json.MarshalIndent(environment, "", "  ")
	if err != nil {
		panic(err)
	}

	rw.WriteHeader(200)
	rw.Write([]byte(envJSON))
}

func main() {
	pool = simpleredis.NewConnectionPoolHost(os.Getenv("SERVICE_HOST") + ":" + os.Getenv("REDIS_MASTER_SERVICE_SERVICE_PORT"))
	defer pool.Close()

	r := mux.NewRouter()
	r.Path("/lrange/{key}").Methods("GET").HandlerFunc(ListRangeHandler)
	r.Path("/rpush/{key}/{value}").Methods("GET").HandlerFunc(ListPushHandler)
	r.Path("/info").Methods("GET").HandlerFunc(InfoHandler)
	r.Path("/env").Methods("GET").HandlerFunc(EnvHandler)

	n := negroni.Classic()
	n.UseHandler(r)
	n.Run(":3000")
}
