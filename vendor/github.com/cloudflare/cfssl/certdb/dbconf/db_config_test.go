package dbconf

import (
	"testing"

	_ "github.com/mattn/go-sqlite3" // import just to initialize SQLite testing
)

func TestLoadFile(t *testing.T) {
	config, err := LoadFile("testdata/db-config.json")
	if err != nil || config == nil {
		t.Fatal("Failed to load test db-config file ", err)
	}

	config, err = LoadFile("nonexistent")
	if err == nil || config != nil {
		t.Fatal("Expected failure loading nonexistent configuration file")
	}
}

func TestDBFromConfig(t *testing.T) {
	db, err := DBFromConfig("testdata/db-config.json")
	if err != nil || db == nil {
		t.Fatal("Failed to open db from test db-config file")
	}

	db, err = DBFromConfig("testdata/bad-db-config.json")
	if err == nil || db != nil {
		t.Fatal("Expected failure opening invalid db")
	}
}
