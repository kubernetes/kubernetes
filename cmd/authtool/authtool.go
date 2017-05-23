/*
Copyright 2014 Google Inc. All rights reserved.

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

// The authinit binary generates the initial authorization and
// authentication files needed to setup a Kubernetes cluster.
//
// In addition to generating auth(n|z) for essential system users, it also
// can create accounts for any number of user names specified on the command-line
// with several built-in policy types.
//
// Generation is done in a go program instead of e.g. a python program
// since not all distributions have python or the like (coreOS doesn't).
//
// TODO: will need auto-edit mode, to update existing policies if, upon
// a kubernetes upgrade:
//   - new object types were added that users need access too.
//   - new privileges are required for a system user, such as kubelet.

package main

import (
	"flag"
	"fmt"
	"math/rand"
	"os"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/golang/glog"
)

var (
	tokenAuthFile = flag.String("token_auth_file", "",
		"Filename to create with list of users and tokens.  Will not overwrite.")
	clientAuthDir = flag.String("client_auth_dir", "",
		"Directory in which to make $USERNAME.kubernetes_auth files. "+
			"Admin will need to distribute to users.")
	policyFile = flag.String("policy_file", "",
		"Filename to create holding generated policies.  Will not overwrite.")
	extraReadUsers  util.StringList
	extraWriteUsers util.StringList
	extraAdminUsers util.StringList
)

// TODO: move extra users generation to using kubectl calls once users and
// policy can be created via REST API.
func init() {
	flag.Var(&extraReadUsers, "extra_read_users",
		"List of usernames who should be generated credentials and "+
			"given authorization to read non-admin objects.")
	flag.Var(&extraWriteUsers, "extra_write_users",
		"List of usernames who should be generated credentials and "+
			"given authorization to read and write non-admin objects.")
	flag.Var(&extraAdminUsers, "extra_admin_users",
		"List of usernames who should be generated credentials and "+
			"given authorization to read and write all objects.")
	rand.Seed(time.Now().UTC().UnixNano())

}

var alnums = []rune("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

const tokenLen = 32

func newToken() string {
	t := make([]rune, tokenLen)
	for i := range t {
		t[i] = alnums[rand.Intn(len(alnums))]
	}
	return string(t)
}

type role int

const (
	adminRole = iota
	writeRole
	readRole
	kubeletRole
)

type user struct {
	name string
	id   int
	role role
}

func main() {
	flag.Parse()

	// Build a list of users.
	l := make([]user, 0)

	// Essential users:
	id := 1
	l = append(l, user{"admin", id, adminRole})
	id += 1
	l = append(l, user{"kubelet", id, kubeletRole})
	// TODO: add users and policies for scheduler, etc.

	for _, u := range extraAdminUsers {
		l = append(l, user{u, id, adminRole})
		id += 1
	}
	for _, u := range extraWriteUsers {
		l = append(l, user{u, id, writeRole})
		id += 1
	}
	for _, u := range extraReadUsers {
		l = append(l, user{u, id, readRole})
		id += 1
	}

	// Check for duplicates.
	seenNames := make(map[string]bool)
	seenIds := make(map[int]bool)
	for _, u := range l {
		if _, ok := seenNames[u.name]; ok {
			glog.Fatalf("Duplicated user: %s", u.name)
		}
		if _, ok := seenIds[u.id]; ok {
			glog.Fatalf("Duplicated id %s", u.id)
		}
	}

	// Generate token file and auth files.
	tf, err := os.Create(*tokenAuthFile)
	if err != nil {
		glog.Fatalf("Unable to create tokenfile: %s", *tokenAuthFile)
	}
	defer tf.Close()
	for _, u := range l {
		token := newToken()
		// Append user and token to apiservers file with all users and tokens.
		line := fmt.Sprintf("%s,%s,%d\n", token, u.name, u.id)
		_, err := tf.WriteString(line)
		if err != nil {
			glog.Fatalf("Unable to write to open tokenfile")
		}
		// Make a kubernetes_auth file to be distributed to that user.
		filename := fmt.Sprintf("%s/%s.kubernetes_auth", *clientAuthDir, u.name)
		kf, err := os.Create(filename)
		defer kf.Close() // TODO: close sooner if lots of users
		if err != nil {
			glog.Fatalf("Unable to create file: %s", filename)
			kf.Close()
		}
		cfg := fmt.Sprintf(`{
  "BearerToken": "%s",
  "Insecure": true,
}
`,
			token)
		_, err = kf.WriteString(cfg)
		if err != nil {
			glog.Fatalf("Unable to write to %s", filename)
		}
	}

	// Generate policy file.
	pf, err := os.Create(*policyFile)
	if err != nil {
		glog.Fatalf("Unable to create policy file: %s", *policyFile)
	}
	defer pf.Close()
	for _, u := range l {
		// TODO: tighten down these policies
		switch u.role {
		case adminRole:
			pf.WriteString(fmt.Sprintf(`{"user":"%s"}`+"\n", u.name))
		case readRole:
			pf.WriteString(fmt.Sprintf(`{"user":"%s", "readonly": true}`+"\n", u.name))
		case writeRole:
			pf.WriteString(fmt.Sprintf(`{"user":"%s", "readonly": true}`+"\n", u.name))
			pf.WriteString(fmt.Sprintf(`{"user":"%s", "kind": "pods"}`+"\n", u.name))
			pf.WriteString(fmt.Sprintf(`{"user":"%s", "kind": "replicationControllers"}`+"\n", u.name))
			pf.WriteString(fmt.Sprintf(`{"user":"%s", "kind": "services"}`+"\n", u.name))
		case kubeletRole:
			pf.WriteString(fmt.Sprintf(`{"user":"%s", "readonly": true, "kind": "pods"}`+"\n", u.name))
			pf.WriteString(fmt.Sprintf(`{"user":"%s", "readonly": true, "kind": "services"}`+"\n", u.name))
			pf.WriteString(fmt.Sprintf(`{"user":"%s", "kind": "events"}`+"\n", u.name))
		}
	}
}
