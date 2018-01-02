package networkdb

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/docker/libnetwork/diagnose"
)

const (
	missingParameter = "missing parameter"
)

// NetDbPaths2Func TODO
var NetDbPaths2Func = map[string]diagnose.HTTPHandlerFunc{
	"/join":         dbJoin,
	"/networkpeers": dbPeers,
	"/clusterpeers": dbClusterPeers,
	"/joinnetwork":  dbJoinNetwork,
	"/leavenetwork": dbLeaveNetwork,
	"/createentry":  dbCreateEntry,
	"/updateentry":  dbUpdateEntry,
	"/deleteentry":  dbDeleteEntry,
	"/getentry":     dbGetEntry,
	"/gettable":     dbGetTable,
}

func dbJoin(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	diagnose.DebugHTTPForm(r)
	if len(r.Form["members"]) < 1 {
		diagnose.HTTPReplyError(w, missingParameter, fmt.Sprintf("%s?members=ip1,ip2,...", r.URL.Path))
		return
	}

	nDB, ok := ctx.(*NetworkDB)
	if ok {
		err := nDB.Join(strings.Split(r.Form["members"][0], ","))
		if err != nil {
			fmt.Fprintf(w, "%s error in the DB join %s\n", r.URL.Path, err)
			return
		}

		fmt.Fprintf(w, "OK\n")
	}
}

func dbPeers(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	diagnose.DebugHTTPForm(r)
	if len(r.Form["nid"]) < 1 {
		diagnose.HTTPReplyError(w, missingParameter, fmt.Sprintf("%s?nid=test", r.URL.Path))
		return
	}

	nDB, ok := ctx.(*NetworkDB)
	if ok {
		peers := nDB.Peers(r.Form["nid"][0])
		fmt.Fprintf(w, "Network:%s Total peers: %d\n", r.Form["nid"], len(peers))
		for i, peerInfo := range peers {
			fmt.Fprintf(w, "%d) %s -> %s\n", i, peerInfo.Name, peerInfo.IP)
		}
	}
}

func dbClusterPeers(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	nDB, ok := ctx.(*NetworkDB)
	if ok {
		peers := nDB.ClusterPeers()
		fmt.Fprintf(w, "Total peers: %d\n", len(peers))
		for i, peerInfo := range peers {
			fmt.Fprintf(w, "%d) %s -> %s\n", i, peerInfo.Name, peerInfo.IP)
		}
	}
}

func dbCreateEntry(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	diagnose.DebugHTTPForm(r)
	if len(r.Form["tname"]) < 1 ||
		len(r.Form["nid"]) < 1 ||
		len(r.Form["key"]) < 1 ||
		len(r.Form["value"]) < 1 {
		diagnose.HTTPReplyError(w, missingParameter, fmt.Sprintf("%s?tname=table_name&nid=network_id&key=k&value=v", r.URL.Path))
		return
	}

	tname := r.Form["tname"][0]
	nid := r.Form["nid"][0]
	key := r.Form["key"][0]
	value := r.Form["value"][0]

	nDB, ok := ctx.(*NetworkDB)
	if ok {
		if err := nDB.CreateEntry(tname, nid, key, []byte(value)); err != nil {
			diagnose.HTTPReplyError(w, err.Error(), "")
			return
		}
		fmt.Fprintf(w, "OK\n")
	}
}

func dbUpdateEntry(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	diagnose.DebugHTTPForm(r)
	if len(r.Form["tname"]) < 1 ||
		len(r.Form["nid"]) < 1 ||
		len(r.Form["key"]) < 1 ||
		len(r.Form["value"]) < 1 {
		diagnose.HTTPReplyError(w, missingParameter, fmt.Sprintf("%s?tname=table_name&nid=network_id&key=k&value=v", r.URL.Path))
		return
	}

	tname := r.Form["tname"][0]
	nid := r.Form["nid"][0]
	key := r.Form["key"][0]
	value := r.Form["value"][0]

	nDB, ok := ctx.(*NetworkDB)
	if ok {
		if err := nDB.UpdateEntry(tname, nid, key, []byte(value)); err != nil {
			diagnose.HTTPReplyError(w, err.Error(), "")
			return
		}
		fmt.Fprintf(w, "OK\n")
	}
}

func dbDeleteEntry(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	diagnose.DebugHTTPForm(r)
	if len(r.Form["tname"]) < 1 ||
		len(r.Form["nid"]) < 1 ||
		len(r.Form["key"]) < 1 {
		diagnose.HTTPReplyError(w, missingParameter, fmt.Sprintf("%s?tname=table_name&nid=network_id&key=k", r.URL.Path))
		return
	}

	tname := r.Form["tname"][0]
	nid := r.Form["nid"][0]
	key := r.Form["key"][0]

	nDB, ok := ctx.(*NetworkDB)
	if ok {
		err := nDB.DeleteEntry(tname, nid, key)
		if err != nil {
			diagnose.HTTPReplyError(w, err.Error(), "")
			return
		}
		fmt.Fprintf(w, "OK\n")
	}
}

func dbGetEntry(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	diagnose.DebugHTTPForm(r)
	if len(r.Form["tname"]) < 1 ||
		len(r.Form["nid"]) < 1 ||
		len(r.Form["key"]) < 1 {
		diagnose.HTTPReplyError(w, missingParameter, fmt.Sprintf("%s?tname=table_name&nid=network_id&key=k", r.URL.Path))
		return
	}

	tname := r.Form["tname"][0]
	nid := r.Form["nid"][0]
	key := r.Form["key"][0]

	nDB, ok := ctx.(*NetworkDB)
	if ok {
		value, err := nDB.GetEntry(tname, nid, key)
		if err != nil {
			diagnose.HTTPReplyError(w, err.Error(), "")
			return
		}
		fmt.Fprintf(w, "key:`%s` value:`%s`\n", key, string(value))
	}
}

func dbJoinNetwork(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	diagnose.DebugHTTPForm(r)
	if len(r.Form["nid"]) < 1 {
		diagnose.HTTPReplyError(w, missingParameter, fmt.Sprintf("%s?nid=network_id", r.URL.Path))
		return
	}

	nid := r.Form["nid"][0]

	nDB, ok := ctx.(*NetworkDB)
	if ok {
		if err := nDB.JoinNetwork(nid); err != nil {
			diagnose.HTTPReplyError(w, err.Error(), "")
			return
		}
		fmt.Fprintf(w, "OK\n")
	}
}

func dbLeaveNetwork(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	diagnose.DebugHTTPForm(r)
	if len(r.Form["nid"]) < 1 {
		diagnose.HTTPReplyError(w, missingParameter, fmt.Sprintf("%s?nid=network_id", r.URL.Path))
		return
	}

	nid := r.Form["nid"][0]

	nDB, ok := ctx.(*NetworkDB)
	if ok {
		if err := nDB.LeaveNetwork(nid); err != nil {
			diagnose.HTTPReplyError(w, err.Error(), "")
			return
		}
		fmt.Fprintf(w, "OK\n")
	}
}

func dbGetTable(ctx interface{}, w http.ResponseWriter, r *http.Request) {
	r.ParseForm()
	diagnose.DebugHTTPForm(r)
	if len(r.Form["tname"]) < 1 ||
		len(r.Form["nid"]) < 1 {
		diagnose.HTTPReplyError(w, missingParameter, fmt.Sprintf("%s?tname=table_name&nid=network_id", r.URL.Path))
		return
	}

	tname := r.Form["tname"][0]
	nid := r.Form["nid"][0]

	nDB, ok := ctx.(*NetworkDB)
	if ok {
		table := nDB.GetTableByNetwork(tname, nid)
		fmt.Fprintf(w, "total elements: %d\n", len(table))
		i := 0
		for k, v := range table {
			fmt.Fprintf(w, "%d) k:`%s` -> v:`%s`\n", i, k, string(v.([]byte)))
			i++
		}
	}
}
