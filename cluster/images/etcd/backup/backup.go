/*
Copyright 2020 The Kubernetes Authors.

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
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/spf13/cobra"
)

const (
	// maxFilesAllowed determines maximum number of backup files created under initial backup folder
	// the gke master backup pod running on the same VM uploads and deletes these backup files
	maxFilesAllowed = 3

	// minFileInterval (in seconds) determines minimum time interval between consecutively created
	// backup files.
	minFileInterval = 300

	dbFilePath          = "/member/snap/db"
	initialBackupFolder = "/initial-data-backups/"
)

var (
	backupCmd = &cobra.Command{
		Short: "Create backup files for etcd",
		Long: `Create backup files for etcd. The tools is intended to run before etcd server starts.
		If db file is found, the tool creates a 'initial-data-backups' folder under the same data diretory,
		under which it creates a copy of that db file.`,
		Run: func(cmd *cobra.Command, args []string) {
			createBackup()
		},
	}
	opts = backupOpts{}
)

func main() {
	flags := backupCmd.Flags()
	flags.StringVar(&opts.dataDir, "data-dir", "",
		"etcd data directory of etcd server to backup. If unset fallbacks to DATA_DIRECTORY env.")
	err := backupCmd.Execute()
	if err != nil {
		log.Printf("Failed to execute backupCmd: %v", err)
	}
}

func createBackup() {
	if err := opts.validateAndDefault(); err != nil {
		log.Fatalf("%v", err)
	}

	dbFile := filepath.Join(opts.dataDir, dbFilePath)
	initialBackupDir := filepath.Join(opts.dataDir, initialBackupFolder)
	if !fileExists(dbFile) {
		log.Printf("no etcd db file found under %v", opts.dataDir)
		return
	}
	log.Printf("found etcd db file %v, attempt to create a copy", dbFile)

	if _, err := os.Stat(initialBackupDir); os.IsNotExist(err) {
		if err := os.MkdirAll(initialBackupDir, 0755); err != nil {
			log.Fatalf("unable to create directory %v: %v", initialBackupDir, err)
		}
	}

	files, err := ioutil.ReadDir(initialBackupDir)
	if err != nil {
		log.Fatalf("unable to read files in directory %v: %v", initialBackupDir, err)
	}
	if len(files) >= maxFilesAllowed {
		log.Printf("number of backup files already reached max value [%d]. Skip the current backup.", maxFilesAllowed)
		return
	}

	sf, err := os.Open(dbFile)
	if err != nil {
		log.Fatalf("unable to open file %v", dbFile)
	}
	defer sf.Close()

	// use filename to rate limit the file creation in case of server crashlooping
	dfFilename := filepath.Join(initialBackupDir, strconv.Itoa(int(time.Now().Unix()/minFileInterval))+"_snapshot.db")
	df, err := os.Create(dfFilename)
	if err != nil {
		log.Fatalf("unable to create file %v: %v", dfFilename, err)
	}
	defer df.Close()

	_, err = io.Copy(df, sf)
	if err != nil {
		log.Fatalf("unable to copy %v to %v: %v", sf, df, err)
	}

	fi, _ := os.Stat(dbFile) // we have already checked that the file exists
	if err := os.Chmod(dfFilename, fi.Mode()); err != nil {
		log.Fatalf("unable to chmod file %v: %v", dfFilename, err)
	}
	log.Printf("successfully created initial backup file %v", dfFilename)
}

func fileExists(filename string) bool {
	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}
