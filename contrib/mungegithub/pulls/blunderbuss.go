/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package pulls

import (
	"flag"
	"math/rand"
	"os"
	"strings"

	"github.com/golang/glog"
	"github.com/google/go-github/github"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/yaml"
)

// A BlunderbussConfig maps a set of file prefixes to a set of owner names (github users)
type BlunderbussConfig struct {
	PrefixMap map[string][]string `json:"prefixMap,omitempty" yaml:"prefixMap,omitempty"`
}

func (b *BlunderbussConfig) FindOwners(filename string) []string {
	owners := util.StringSet{}
	for prefix, ownersList := range b.PrefixMap {
		if strings.HasPrefix(filename, prefix) {
			owners.Insert(ownersList...)
		}
	}
	return owners.List()
}

type BlunderbussMunger struct {
	config *BlunderbussConfig
}

var blunderbussConfig = flag.String("blunderbuss-config", "", "Path to the blunderbuss config file")

func init() {
	blunderbuss := &BlunderbussMunger{}
	RegisterMungerOrDie(blunderbuss)
}

func (b *BlunderbussMunger) Name() string { return "blunderbuss" }

func (b *BlunderbussMunger) loadConfig() {
	if len(*blunderbussConfig) == 0 {
		glog.Fatalf("--blunderbuss-config is required with the blunderbuss munger")
	}
	file, err := os.Open(*blunderbussConfig)
	if err != nil {
		glog.Fatalf("Failed to load blunderbuss config: %v", err)
	}
	defer file.Close()

	b.config = &BlunderbussConfig{}
	if err := yaml.NewYAMLToJSONDecoder(file).Decode(b.config); err != nil {
		glog.Fatalf("Failed to load blunderbuss config: %v", err)
	}
	glog.V(4).Infof("Loaded config from %s", *blunderbussConfig)
}

func (b *BlunderbussMunger) MungePullRequest(client *github.Client, org, project string, pr *github.PullRequest, issue *github.Issue, commits []github.RepositoryCommit, events []github.IssueEvent, dryrun bool) {
	if b.config == nil {
		b.loadConfig()
	}
	if issue.Assignee != nil {
		return
	}
	potentialOwners := util.StringSet{}
	for _, commit := range commits {
		commit, _, err := client.Repositories.GetCommit(*commit.Author.Login, project, *commit.SHA)
		if err != nil {
			glog.Errorf("Can't load commit %s %s %s", *commit.Author.Login, project, *commit.SHA)
			continue
		}
		for _, file := range commit.Files {
			fileOwners := b.config.FindOwners(*file.Filename)
			if len(fileOwners) == 0 {
				glog.Warningf("Couldn't find an owner for: %s", *file.Filename)
			}
			potentialOwners.Insert(fileOwners...)
		}
	}
	if potentialOwners.Len() == 0 {
		glog.Errorf("No owners found for PR %d", *pr.Number)
		return
	}
	ix := rand.Int() % potentialOwners.Len()
	owner := potentialOwners.List()[ix]
	if dryrun {
		glog.Infof("would have assigned %s to PR %d", owner, *pr.Number)
		return
	}
	if _, _, err := client.Issues.Edit(org, project, *pr.Number, &github.IssueRequest{Assignee: &owner}); err != nil {
		glog.Errorf("Error updating issue: %v", err)
	}
}
