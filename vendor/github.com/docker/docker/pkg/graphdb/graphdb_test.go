package graphdb

import (
	"database/sql"
	"fmt"
	"os"
	"path"
	"strconv"
	"testing"

	_ "code.google.com/p/gosqlite/sqlite3"
)

func newTestDb(t *testing.T) (*Database, string) {
	p := path.Join(os.TempDir(), "sqlite.db")
	conn, err := sql.Open("sqlite3", p)
	db, err := NewDatabase(conn)
	if err != nil {
		t.Fatal(err)
	}
	return db, p
}

func destroyTestDb(dbPath string) {
	os.Remove(dbPath)
}

func TestNewDatabase(t *testing.T) {
	db, dbpath := newTestDb(t)
	if db == nil {
		t.Fatal("Database should not be nil")
	}
	db.Close()
	defer destroyTestDb(dbpath)
}

func TestCreateRootEntity(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)
	root := db.RootEntity()
	if root == nil {
		t.Fatal("Root entity should not be nil")
	}
}

func TestGetRootEntity(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	e := db.Get("/")
	if e == nil {
		t.Fatal("Entity should not be nil")
	}
	if e.ID() != "0" {
		t.Fatalf("Entity id should be 0, got %s", e.ID())
	}
}

func TestSetEntityWithDifferentName(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	db.Set("/test", "1")
	if _, err := db.Set("/other", "1"); err != nil {
		t.Fatal(err)
	}
}

func TestSetDuplicateEntity(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	if _, err := db.Set("/foo", "42"); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/foo", "43"); err == nil {
		t.Fatalf("Creating an entry with a duplicate path did not cause an error")
	}
}

func TestCreateChild(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	child, err := db.Set("/db", "1")
	if err != nil {
		t.Fatal(err)
	}
	if child == nil {
		t.Fatal("Child should not be nil")
	}
	if child.ID() != "1" {
		t.Fail()
	}
}

func TestParents(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	for i := 1; i < 6; i++ {
		a := strconv.Itoa(i)
		if _, err := db.Set("/"+a, a); err != nil {
			t.Fatal(err)
		}
	}

	for i := 6; i < 11; i++ {
		a := strconv.Itoa(i)
		p := strconv.Itoa(i - 5)

		key := fmt.Sprintf("/%s/%s", p, a)

		if _, err := db.Set(key, a); err != nil {
			t.Fatal(err)
		}

		parents, err := db.Parents(key)
		if err != nil {
			t.Fatal(err)
		}

		if len(parents) != 1 {
			t.Fatalf("Expected 1 entry for %s got %d", key, len(parents))
		}

		if parents[0] != p {
			t.Fatalf("ID %s received, %s expected", parents[0], p)
		}
	}
}

func TestChildren(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	str := "/"
	for i := 1; i < 6; i++ {
		a := strconv.Itoa(i)
		if _, err := db.Set(str+a, a); err != nil {
			t.Fatal(err)
		}

		str = str + a + "/"
	}

	str = "/"
	for i := 10; i < 30; i++ { // 20 entities
		a := strconv.Itoa(i)
		if _, err := db.Set(str+a, a); err != nil {
			t.Fatal(err)
		}

		str = str + a + "/"
	}
	entries, err := db.Children("/", 5)
	if err != nil {
		t.Fatal(err)
	}

	if len(entries) != 11 {
		t.Fatalf("Expect 11 entries for / got %d", len(entries))
	}

	entries, err = db.Children("/", 20)
	if err != nil {
		t.Fatal(err)
	}

	if len(entries) != 25 {
		t.Fatalf("Expect 25 entries for / got %d", len(entries))
	}
}

func TestListAllRootChildren(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	for i := 1; i < 6; i++ {
		a := strconv.Itoa(i)
		if _, err := db.Set("/"+a, a); err != nil {
			t.Fatal(err)
		}
	}
	entries := db.List("/", -1)
	if len(entries) != 5 {
		t.Fatalf("Expect 5 entries for / got %d", len(entries))
	}
}

func TestListAllSubChildren(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	_, err := db.Set("/webapp", "1")
	if err != nil {
		t.Fatal(err)
	}
	child2, err := db.Set("/db", "2")
	if err != nil {
		t.Fatal(err)
	}
	child4, err := db.Set("/logs", "4")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/db/logs", child4.ID()); err != nil {
		t.Fatal(err)
	}

	child3, err := db.Set("/sentry", "3")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/sentry", child3.ID()); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/db", child2.ID()); err != nil {
		t.Fatal(err)
	}

	entries := db.List("/webapp", 1)
	if len(entries) != 3 {
		t.Fatalf("Expect 3 entries for / got %d", len(entries))
	}

	entries = db.List("/webapp", 0)
	if len(entries) != 2 {
		t.Fatalf("Expect 2 entries for / got %d", len(entries))
	}
}

func TestAddSelfAsChild(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	child, err := db.Set("/test", "1")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/test/other", child.ID()); err == nil {
		t.Fatal("Error should not be nil")
	}
}

func TestAddChildToNonExistantRoot(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	if _, err := db.Set("/myapp", "1"); err != nil {
		t.Fatal(err)
	}

	if _, err := db.Set("/myapp/proxy/db", "2"); err == nil {
		t.Fatal("Error should not be nil")
	}
}

func TestWalkAll(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)
	_, err := db.Set("/webapp", "1")
	if err != nil {
		t.Fatal(err)
	}
	child2, err := db.Set("/db", "2")
	if err != nil {
		t.Fatal(err)
	}
	child4, err := db.Set("/db/logs", "4")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/logs", child4.ID()); err != nil {
		t.Fatal(err)
	}

	child3, err := db.Set("/sentry", "3")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/sentry", child3.ID()); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/db", child2.ID()); err != nil {
		t.Fatal(err)
	}

	child5, err := db.Set("/gograph", "5")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/same-ref-diff-name", child5.ID()); err != nil {
		t.Fatal(err)
	}

	if err := db.Walk("/", func(p string, e *Entity) error {
		t.Logf("Path: %s Entity: %s", p, e.ID())
		return nil
	}, -1); err != nil {
		t.Fatal(err)
	}
}

func TestGetEntityByPath(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)
	_, err := db.Set("/webapp", "1")
	if err != nil {
		t.Fatal(err)
	}
	child2, err := db.Set("/db", "2")
	if err != nil {
		t.Fatal(err)
	}
	child4, err := db.Set("/logs", "4")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/db/logs", child4.ID()); err != nil {
		t.Fatal(err)
	}

	child3, err := db.Set("/sentry", "3")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/sentry", child3.ID()); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/db", child2.ID()); err != nil {
		t.Fatal(err)
	}

	child5, err := db.Set("/gograph", "5")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/same-ref-diff-name", child5.ID()); err != nil {
		t.Fatal(err)
	}

	entity := db.Get("/webapp/db/logs")
	if entity == nil {
		t.Fatal("Entity should not be nil")
	}
	if entity.ID() != "4" {
		t.Fatalf("Expected to get entity with id 4, got %s", entity.ID())
	}
}

func TestEnitiesPaths(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)
	_, err := db.Set("/webapp", "1")
	if err != nil {
		t.Fatal(err)
	}
	child2, err := db.Set("/db", "2")
	if err != nil {
		t.Fatal(err)
	}
	child4, err := db.Set("/logs", "4")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/db/logs", child4.ID()); err != nil {
		t.Fatal(err)
	}

	child3, err := db.Set("/sentry", "3")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/sentry", child3.ID()); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/db", child2.ID()); err != nil {
		t.Fatal(err)
	}

	child5, err := db.Set("/gograph", "5")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/same-ref-diff-name", child5.ID()); err != nil {
		t.Fatal(err)
	}

	out := db.List("/", -1)
	for _, p := range out.Paths() {
		t.Log(p)
	}
}

func TestDeleteRootEntity(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	if err := db.Delete("/"); err == nil {
		t.Fatal("Error should not be nil")
	}
}

func TestDeleteEntity(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)
	_, err := db.Set("/webapp", "1")
	if err != nil {
		t.Fatal(err)
	}
	child2, err := db.Set("/db", "2")
	if err != nil {
		t.Fatal(err)
	}
	child4, err := db.Set("/logs", "4")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/db/logs", child4.ID()); err != nil {
		t.Fatal(err)
	}

	child3, err := db.Set("/sentry", "3")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/sentry", child3.ID()); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/db", child2.ID()); err != nil {
		t.Fatal(err)
	}

	child5, err := db.Set("/gograph", "5")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Set("/webapp/same-ref-diff-name", child5.ID()); err != nil {
		t.Fatal(err)
	}

	if err := db.Delete("/webapp/sentry"); err != nil {
		t.Fatal(err)
	}
	entity := db.Get("/webapp/sentry")
	if entity != nil {
		t.Fatal("Entity /webapp/sentry should be nil")
	}
}

func TestCountRefs(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	db.Set("/webapp", "1")

	if db.Refs("1") != 1 {
		t.Fatal("Expect reference count to be 1")
	}

	db.Set("/db", "2")
	db.Set("/webapp/db", "2")
	if db.Refs("2") != 2 {
		t.Fatal("Expect reference count to be 2")
	}
}

func TestPurgeId(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	db.Set("/webapp", "1")

	if c := db.Refs("1"); c != 1 {
		t.Fatalf("Expect reference count to be 1, got %d", c)
	}

	db.Set("/db", "2")
	db.Set("/webapp/db", "2")

	count, err := db.Purge("2")
	if err != nil {
		t.Fatal(err)
	}
	if count != 2 {
		t.Fatalf("Expected 2 references to be removed, got %d", count)
	}
}

// Regression test https://github.com/docker/docker/issues/12334
func TestPurgeIdRefPaths(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	db.Set("/webapp", "1")
	db.Set("/db", "2")

	db.Set("/db/webapp", "1")

	if c := db.Refs("1"); c != 2 {
		t.Fatalf("Expected 2 reference for webapp, got %d", c)
	}
	if c := db.Refs("2"); c != 1 {
		t.Fatalf("Expected 1 reference for db, got %d", c)
	}

	if rp := db.RefPaths("2"); len(rp) != 1 {
		t.Fatalf("Expected 1 reference path for db, got %d", len(rp))
	}

	count, err := db.Purge("2")
	if err != nil {
		t.Fatal(err)
	}

	if count != 2 {
		t.Fatalf("Expected 2 rows to be removed, got %d", count)
	}

	if c := db.Refs("2"); c != 0 {
		t.Fatalf("Expected 0 reference for db, got %d", c)
	}
	if c := db.Refs("1"); c != 1 {
		t.Fatalf("Expected 1 reference for webapp, got %d", c)
	}
}

func TestRename(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	db.Set("/webapp", "1")

	if db.Refs("1") != 1 {
		t.Fatal("Expect reference count to be 1")
	}

	db.Set("/db", "2")
	db.Set("/webapp/db", "2")

	if db.Get("/webapp/db") == nil {
		t.Fatal("Cannot find entity at path /webapp/db")
	}

	if err := db.Rename("/webapp/db", "/webapp/newdb"); err != nil {
		t.Fatal(err)
	}
	if db.Get("/webapp/db") != nil {
		t.Fatal("Entity should not exist at /webapp/db")
	}
	if db.Get("/webapp/newdb") == nil {
		t.Fatal("Cannot find entity at path /webapp/newdb")
	}

}

func TestCreateMultipleNames(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	db.Set("/db", "1")
	if _, err := db.Set("/myapp", "1"); err != nil {
		t.Fatal(err)
	}

	db.Walk("/", func(p string, e *Entity) error {
		t.Logf("%s\n", p)
		return nil
	}, -1)
}

func TestRefPaths(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	db.Set("/webapp", "1")

	db.Set("/db", "2")
	db.Set("/webapp/db", "2")

	refs := db.RefPaths("2")
	if len(refs) != 2 {
		t.Fatalf("Expected reference count to be 2, got %d", len(refs))
	}
}

func TestExistsTrue(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	db.Set("/testing", "1")

	if !db.Exists("/testing") {
		t.Fatalf("/tesing should exist")
	}
}

func TestExistsFalse(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	db.Set("/toerhe", "1")

	if db.Exists("/testing") {
		t.Fatalf("/tesing should not exist")
	}

}

func TestGetNameWithTrailingSlash(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	db.Set("/todo", "1")

	e := db.Get("/todo/")
	if e == nil {
		t.Fatalf("Entity should not be nil")
	}
}

func TestConcurrentWrites(t *testing.T) {
	db, dbpath := newTestDb(t)
	defer destroyTestDb(dbpath)

	errs := make(chan error, 2)

	save := func(name string, id string) {
		if _, err := db.Set(fmt.Sprintf("/%s", name), id); err != nil {
			errs <- err
		}
		errs <- nil
	}
	purge := func(id string) {
		if _, err := db.Purge(id); err != nil {
			errs <- err
		}
		errs <- nil
	}

	save("/1", "1")

	go purge("1")
	go save("/2", "2")

	any := false
	for i := 0; i < 2; i++ {
		if err := <-errs; err != nil {
			any = true
			t.Log(err)
		}
	}
	if any {
		t.Fail()
	}
}
