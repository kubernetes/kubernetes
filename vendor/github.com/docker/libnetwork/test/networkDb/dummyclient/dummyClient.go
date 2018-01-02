package dummyclient

import (
	"fmt"
	"log"
	"net/http"

	events "github.com/docker/go-events"
	"github.com/docker/libnetwork/diagnose"
	"github.com/docker/libnetwork/networkdb"
	"github.com/sirupsen/logrus"
)

// DummyClientPaths2Func exported paths for the client
var DummyClientPaths2Func = map[string]diagnose.HTTPHandlerFunc{
	"/watchtable":          watchTable,
	"/watchedtableentries": watchTableEntries,
}

const (
	missingParameter = "missing parameter"
)

type tableHandler struct {
	cancelWatch func()
	entries     map[string]string
}

var clientWatchTable = map[string]tableHandler{}

func watchTable(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	diagnose.DebugHTTPForm(r)
	if len(r.Form["tname"]) < 1 {
		diagnose.HTTPReplyError(w, missingParameter, fmt.Sprintf("%s?tname=table_name", r.URL.Path))
		return
	}

	tableName := r.Form["tname"][0]
	if _, ok := clientWatchTable[tableName]; ok {
		fmt.Fprintf(w, "OK\n")
		return
	}

	nDB, ok := ctx.(*networkdb.NetworkDB)
	if ok {
		ch, cancel := nDB.Watch(tableName, "", "")
		clientWatchTable[tableName] = tableHandler{cancelWatch: cancel, entries: make(map[string]string)}
		go handleTableEvents(tableName, ch)

		fmt.Fprintf(w, "OK\n")
	}
}

func watchTableEntries(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	diagnose.DebugHTTPForm(r)
	if len(r.Form["tname"]) < 1 {
		diagnose.HTTPReplyError(w, missingParameter, fmt.Sprintf("%s?tname=table_name", r.URL.Path))
		return
	}

	tableName := r.Form["tname"][0]
	table, ok := clientWatchTable[tableName]
	if !ok {
		fmt.Fprintf(w, "Table %s not watched\n", tableName)
		return
	}

	fmt.Fprintf(w, "total elements: %d\n", len(table.entries))
	i := 0
	for k, v := range table.entries {
		fmt.Fprintf(w, "%d) k:`%s` -> v:`%s`\n", i, k, v)
		i++
	}
}

func handleTableEvents(tableName string, ch *events.Channel) {
	var (
		// nid   string
		eid   string
		value []byte
		isAdd bool
	)

	logrus.Infof("Started watching table:%s", tableName)
	for {
		select {
		case <-ch.Done():
			logrus.Infof("End watching %s", tableName)
			return

		case evt := <-ch.C:
			logrus.Infof("Recevied new event on:%s", tableName)
			switch event := evt.(type) {
			case networkdb.CreateEvent:
				// nid = event.NetworkID
				eid = event.Key
				value = event.Value
				isAdd = true
			case networkdb.DeleteEvent:
				// nid = event.NetworkID
				eid = event.Key
				value = event.Value
				isAdd = false
			default:
				log.Fatalf("Unexpected table event = %#v", event)
			}
			if isAdd {
				// logrus.Infof("Add %s %s", tableName, eid)
				clientWatchTable[tableName].entries[eid] = string(value)
			} else {
				// logrus.Infof("Del %s %s", tableName, eid)
				delete(clientWatchTable[tableName].entries, eid)
			}
		}
	}
}
