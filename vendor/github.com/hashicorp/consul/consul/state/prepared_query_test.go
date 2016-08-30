package state

import (
	"reflect"
	"strings"
	"testing"

	"github.com/hashicorp/consul/consul/structs"
)

func TestStateStore_PreparedQuery_isUUID(t *testing.T) {
	cases := map[string]bool{
		"":     false,
		"nope": false,
		"f004177f-2c28-83b7-4229-eacc25fe55d1":  true,
		"F004177F-2C28-83B7-4229-EACC25FE55D1":  true,
		"x004177f-2c28-83b7-4229-eacc25fe55d1":  false, // Bad hex
		"f004177f-xc28-83b7-4229-eacc25fe55d1":  false, // Bad hex
		"f004177f-2c28-x3b7-4229-eacc25fe55d1":  false, // Bad hex
		"f004177f-2c28-83b7-x229-eacc25fe55d1":  false, // Bad hex
		"f004177f-2c28-83b7-4229-xacc25fe55d1":  false, // Bad hex
		" f004177f-2c28-83b7-4229-eacc25fe55d1": false, // Leading whitespace
		"f004177f-2c28-83b7-4229-eacc25fe55d1 ": false, // Trailing whitespace
	}
	for i := 0; i < 100; i++ {
		cases[testUUID()] = true
	}

	for str, expected := range cases {
		if actual := isUUID(str); actual != expected {
			t.Fatalf("bad: '%s'", str)
		}
	}
}

func TestStateStore_PreparedQuerySet_PreparedQueryGet(t *testing.T) {
	s := testStateStore(t)

	// Querying with no results returns nil.
	idx, res, err := s.PreparedQueryGet(testUUID())
	if idx != 0 || res != nil || err != nil {
		t.Fatalf("expected (0, nil, nil), got: (%d, %#v, %#v)", idx, res, err)
	}

	// Inserting a query with empty ID is disallowed.
	if err := s.PreparedQuerySet(1, &structs.PreparedQuery{}); err == nil {
		t.Fatalf("expected %#v, got: %#v", ErrMissingQueryID, err)
	}

	// Index is not updated if nothing is saved.
	if idx := s.maxIndex("prepared-queries"); idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}

	// Build a legit-looking query with the most basic options.
	query := &structs.PreparedQuery{
		ID:      testUUID(),
		Session: "nope",
		Service: structs.ServiceQuery{
			Service: "redis",
		},
	}

	// The set will still fail because the session is bogus.
	err = s.PreparedQuerySet(1, query)
	if err == nil || !strings.Contains(err.Error(), "failed session lookup") {
		t.Fatalf("bad: %v", err)
	}

	// Index is not updated if nothing is saved.
	if idx := s.maxIndex("prepared-queries"); idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}

	// Now register the service and remove the bogus session.
	testRegisterNode(t, s, 1, "foo")
	testRegisterService(t, s, 2, "foo", "redis")
	query.Session = ""

	// This should go through.
	if err := s.PreparedQuerySet(3, query); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make sure the index got updated.
	if idx := s.maxIndex("prepared-queries"); idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}

	// Read it back out and verify it.
	expected := &structs.PreparedQuery{
		ID: query.ID,
		Service: structs.ServiceQuery{
			Service: "redis",
		},
		RaftIndex: structs.RaftIndex{
			CreateIndex: 3,
			ModifyIndex: 3,
		},
	}
	idx, actual, err := s.PreparedQueryGet(query.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("bad: %v", actual)
	}

	// Give it a name and set it again.
	query.Name = "test-query"
	if err := s.PreparedQuerySet(4, query); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make sure the index got updated.
	if idx := s.maxIndex("prepared-queries"); idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}

	// Read it back and verify the data was updated as well as the index.
	expected.Name = "test-query"
	expected.ModifyIndex = 4
	idx, actual, err = s.PreparedQueryGet(query.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("bad: %v", actual)
	}

	// Try to tie it to a bogus session.
	query.Session = testUUID()
	err = s.PreparedQuerySet(5, query)
	if err == nil || !strings.Contains(err.Error(), "invalid session") {
		t.Fatalf("bad: %v", err)
	}

	// Index is not updated if nothing is saved.
	if idx := s.maxIndex("prepared-queries"); idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}

	// Now make a session and try again.
	session := &structs.Session{
		ID:   query.Session,
		Node: "foo",
	}
	if err := s.SessionCreate(5, session); err != nil {
		t.Fatalf("err: %s", err)
	}
	if err := s.PreparedQuerySet(6, query); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make sure the index got updated.
	if idx := s.maxIndex("prepared-queries"); idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}

	// Read it back and verify the data was updated as well as the index.
	expected.Session = query.Session
	expected.ModifyIndex = 6
	idx, actual, err = s.PreparedQueryGet(query.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("bad: %v", actual)
	}

	// Try to register a query with the same name and make sure it fails.
	{
		evil := &structs.PreparedQuery{
			ID:   testUUID(),
			Name: query.Name,
			Service: structs.ServiceQuery{
				Service: "redis",
			},
		}
		err := s.PreparedQuerySet(7, evil)
		if err == nil || !strings.Contains(err.Error(), "aliases an existing query name") {
			t.Fatalf("bad: %v", err)
		}

		// Sanity check to make sure it's not there.
		idx, actual, err := s.PreparedQueryGet(evil.ID)
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 6 {
			t.Fatalf("bad index: %d", idx)
		}
		if actual != nil {
			t.Fatalf("bad: %v", actual)
		}
	}

	// Try to abuse the system by trying to register a query whose name
	// aliases a real query ID.
	{
		evil := &structs.PreparedQuery{
			ID:   testUUID(),
			Name: query.ID,
			Service: structs.ServiceQuery{
				Service: "redis",
			},
		}
		err := s.PreparedQuerySet(8, evil)
		if err == nil || !strings.Contains(err.Error(), "aliases an existing query ID") {
			t.Fatalf("bad: %v", err)
		}

		// Sanity check to make sure it's not there.
		idx, actual, err := s.PreparedQueryGet(evil.ID)
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 6 {
			t.Fatalf("bad index: %d", idx)
		}
		if actual != nil {
			t.Fatalf("bad: %v", actual)
		}
	}

	// Try to register a template that squats on the existing query's name.
	{
		evil := &structs.PreparedQuery{
			ID:   testUUID(),
			Name: query.Name,
			Template: structs.QueryTemplateOptions{
				Type: structs.QueryTemplateTypeNamePrefixMatch,
			},
			Service: structs.ServiceQuery{
				Service: "redis",
			},
		}
		err := s.PreparedQuerySet(8, evil)
		if err == nil || !strings.Contains(err.Error(), "aliases an existing query name") {
			t.Fatalf("bad: %v", err)
		}

		// Sanity check to make sure it's not there.
		idx, actual, err := s.PreparedQueryGet(evil.ID)
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 6 {
			t.Fatalf("bad index: %d", idx)
		}
		if actual != nil {
			t.Fatalf("bad: %v", actual)
		}
	}

	// Index is not updated if nothing is saved.
	if idx := s.maxIndex("prepared-queries"); idx != 6 {
		t.Fatalf("bad index: %d", idx)
	}

	// Turn the query into a template with an empty name.
	query.Name = ""
	query.Template = structs.QueryTemplateOptions{
		Type: structs.QueryTemplateTypeNamePrefixMatch,
	}
	if err := s.PreparedQuerySet(9, query); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make sure the index got updated.
	if idx := s.maxIndex("prepared-queries"); idx != 9 {
		t.Fatalf("bad index: %d", idx)
	}

	// Read it back and verify the data was updated as well as the index.
	expected.Name = ""
	expected.Template = structs.QueryTemplateOptions{
		Type: structs.QueryTemplateTypeNamePrefixMatch,
	}
	expected.ModifyIndex = 9
	idx, actual, err = s.PreparedQueryGet(query.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 9 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("bad: %v", actual)
	}

	// Try to register a template that squats on the empty prefix.
	{
		evil := &structs.PreparedQuery{
			ID:   testUUID(),
			Name: "",
			Template: structs.QueryTemplateOptions{
				Type: structs.QueryTemplateTypeNamePrefixMatch,
			},
			Service: structs.ServiceQuery{
				Service: "redis",
			},
		}
		err := s.PreparedQuerySet(10, evil)
		if err == nil || !strings.Contains(err.Error(), "query template with an empty name already exists") {
			t.Fatalf("bad: %v", err)
		}

		// Sanity check to make sure it's not there.
		idx, actual, err := s.PreparedQueryGet(evil.ID)
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 9 {
			t.Fatalf("bad index: %d", idx)
		}
		if actual != nil {
			t.Fatalf("bad: %v", actual)
		}
	}

	// Give the query template a name.
	query.Name = "prefix"
	if err := s.PreparedQuerySet(11, query); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make sure the index got updated.
	if idx := s.maxIndex("prepared-queries"); idx != 11 {
		t.Fatalf("bad index: %d", idx)
	}

	// Read it back and verify the data was updated as well as the index.
	expected.Name = "prefix"
	expected.ModifyIndex = 11
	idx, actual, err = s.PreparedQueryGet(query.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 11 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("bad: %v", actual)
	}

	// Try to register a template that squats on the prefix.
	{
		evil := &structs.PreparedQuery{
			ID:   testUUID(),
			Name: "prefix",
			Template: structs.QueryTemplateOptions{
				Type: structs.QueryTemplateTypeNamePrefixMatch,
			},
			Service: structs.ServiceQuery{
				Service: "redis",
			},
		}
		err := s.PreparedQuerySet(12, evil)
		if err == nil || !strings.Contains(err.Error(), "aliases an existing query name") {
			t.Fatalf("bad: %v", err)
		}

		// Sanity check to make sure it's not there.
		idx, actual, err := s.PreparedQueryGet(evil.ID)
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 11 {
			t.Fatalf("bad index: %d", idx)
		}
		if actual != nil {
			t.Fatalf("bad: %v", actual)
		}
	}

	// Try to register a template that doesn't compile.
	{
		evil := &structs.PreparedQuery{
			ID:   testUUID(),
			Name: "legit-prefix",
			Template: structs.QueryTemplateOptions{
				Type: structs.QueryTemplateTypeNamePrefixMatch,
			},
			Service: structs.ServiceQuery{
				Service: "${nope",
			},
		}
		err := s.PreparedQuerySet(13, evil)
		if err == nil || !strings.Contains(err.Error(), "failed compiling template") {
			t.Fatalf("bad: %v", err)
		}

		// Sanity check to make sure it's not there.
		idx, actual, err := s.PreparedQueryGet(evil.ID)
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 11 {
			t.Fatalf("bad index: %d", idx)
		}
		if actual != nil {
			t.Fatalf("bad: %v", actual)
		}
	}
}

func TestStateStore_PreparedQueryDelete(t *testing.T) {
	s := testStateStore(t)

	// Set up our test environment.
	testRegisterNode(t, s, 1, "foo")
	testRegisterService(t, s, 2, "foo", "redis")

	// Create a new query.
	query := &structs.PreparedQuery{
		ID: testUUID(),
		Service: structs.ServiceQuery{
			Service: "redis",
		},
	}

	// Deleting a query that doesn't exist should be a no-op.
	if err := s.PreparedQueryDelete(3, query.ID); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Index is not updated if nothing is saved.
	if idx := s.maxIndex("prepared-queries"); idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}

	// Now add the query to the data store.
	if err := s.PreparedQuerySet(3, query); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make sure the index got updated.
	if idx := s.maxIndex("prepared-queries"); idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}

	// Read it back out and verify it.
	expected := &structs.PreparedQuery{
		ID: query.ID,
		Service: structs.ServiceQuery{
			Service: "redis",
		},
		RaftIndex: structs.RaftIndex{
			CreateIndex: 3,
			ModifyIndex: 3,
		},
	}
	idx, actual, err := s.PreparedQueryGet(query.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("bad: %v", actual)
	}

	// Now delete it.
	if err := s.PreparedQueryDelete(4, query.ID); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make sure the index got updated.
	if idx := s.maxIndex("prepared-queries"); idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}

	// Sanity check to make sure it's not there.
	idx, actual, err = s.PreparedQueryGet(query.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 4 {
		t.Fatalf("bad index: %d", idx)
	}
	if actual != nil {
		t.Fatalf("bad: %v", actual)
	}
}

func TestStateStore_PreparedQueryResolve(t *testing.T) {
	s := testStateStore(t)

	// Set up our test environment.
	testRegisterNode(t, s, 1, "foo")
	testRegisterService(t, s, 2, "foo", "redis")

	// Create a new query.
	query := &structs.PreparedQuery{
		ID:   testUUID(),
		Name: "my-test-query",
		Service: structs.ServiceQuery{
			Service: "redis",
		},
	}

	// Try to lookup a query that's not there using something that looks
	// like a real ID.
	idx, actual, err := s.PreparedQueryResolve(query.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}
	if actual != nil {
		t.Fatalf("bad: %v", actual)
	}

	// Try to lookup a query that's not there using something that looks
	// like a name
	idx, actual, err = s.PreparedQueryResolve(query.Name)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}
	if actual != nil {
		t.Fatalf("bad: %v", actual)
	}

	// Now actually insert the query.
	if err := s.PreparedQuerySet(3, query); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Make sure the index got updated.
	if idx := s.maxIndex("prepared-queries"); idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}

	// Read it back out using the ID and verify it.
	expected := &structs.PreparedQuery{
		ID:   query.ID,
		Name: "my-test-query",
		Service: structs.ServiceQuery{
			Service: "redis",
		},
		RaftIndex: structs.RaftIndex{
			CreateIndex: 3,
			ModifyIndex: 3,
		},
	}
	idx, actual, err = s.PreparedQueryResolve(query.ID)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("bad: %v", actual)
	}

	// Read it back using the name and verify it again.
	idx, actual, err = s.PreparedQueryResolve(query.Name)
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 3 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("bad: %v", actual)
	}

	// Make sure an empty lookup is well-behaved if there are actual queries
	// in the state store.
	idx, actual, err = s.PreparedQueryResolve("")
	if err != ErrMissingQueryID {
		t.Fatalf("bad: %v ", err)
	}
	if idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}
	if actual != nil {
		t.Fatalf("bad: %v", actual)
	}

	// Create two prepared query templates, one a longer prefix of the
	// other.
	tmpl1 := &structs.PreparedQuery{
		ID:   testUUID(),
		Name: "prod-",
		Template: structs.QueryTemplateOptions{
			Type: structs.QueryTemplateTypeNamePrefixMatch,
		},
		Service: structs.ServiceQuery{
			Service: "${name.suffix}",
		},
	}
	if err := s.PreparedQuerySet(4, tmpl1); err != nil {
		t.Fatalf("err: %s", err)
	}
	tmpl2 := &structs.PreparedQuery{
		ID:   testUUID(),
		Name: "prod-redis",
		Template: structs.QueryTemplateOptions{
			Type:   structs.QueryTemplateTypeNamePrefixMatch,
			Regexp: "^prod-(.*)$",
		},
		Service: structs.ServiceQuery{
			Service: "${match(1)}-master",
		},
	}
	if err := s.PreparedQuerySet(5, tmpl2); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Resolve the less-specific prefix.
	expected = &structs.PreparedQuery{
		ID:   tmpl1.ID,
		Name: "prod-",
		Template: structs.QueryTemplateOptions{
			Type: structs.QueryTemplateTypeNamePrefixMatch,
		},
		Service: structs.ServiceQuery{
			Service: "mongodb",
		},
		RaftIndex: structs.RaftIndex{
			CreateIndex: 4,
			ModifyIndex: 4,
		},
	}
	idx, actual, err = s.PreparedQueryResolve("prod-mongodb")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("bad: %v", actual)
	}

	// Now resolve the more specific prefix.
	expected = &structs.PreparedQuery{
		ID:   tmpl2.ID,
		Name: "prod-redis",
		Template: structs.QueryTemplateOptions{
			Type:   structs.QueryTemplateTypeNamePrefixMatch,
			Regexp: "^prod-(.*)$",
		},
		Service: structs.ServiceQuery{
			Service: "redis-foobar-master",
		},
		RaftIndex: structs.RaftIndex{
			CreateIndex: 5,
			ModifyIndex: 5,
		},
	}
	idx, actual, err = s.PreparedQueryResolve("prod-redis-foobar")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("bad: %v", actual)
	}

	// Resolve an exact-match prefix. The output of this one doesn't match a
	// sensical service name, but it still renders.
	expected = &structs.PreparedQuery{
		ID:   tmpl1.ID,
		Name: "prod-",
		Template: structs.QueryTemplateOptions{
			Type: structs.QueryTemplateTypeNamePrefixMatch,
		},
		Service: structs.ServiceQuery{
			Service: "",
		},
		RaftIndex: structs.RaftIndex{
			CreateIndex: 4,
			ModifyIndex: 4,
		},
	}
	idx, actual, err = s.PreparedQueryResolve("prod-")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("bad: %v", actual)
	}

	// Make sure you can't run a prepared query template by ID, since that
	// makes no sense.
	_, _, err = s.PreparedQueryResolve(tmpl1.ID)
	if err == nil || !strings.Contains(err.Error(), "prepared query templates can only be resolved up by name") {
		t.Fatalf("bad: %v", err)
	}
}

func TestStateStore_PreparedQueryList(t *testing.T) {
	s := testStateStore(t)

	// Make sure nothing is returned for an empty query
	idx, actual, err := s.PreparedQueryList()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 0 {
		t.Fatalf("bad index: %d", idx)
	}
	if len(actual) != 0 {
		t.Fatalf("bad: %v", actual)
	}

	// Set up our test environment.
	testRegisterNode(t, s, 1, "foo")
	testRegisterService(t, s, 2, "foo", "redis")
	testRegisterService(t, s, 3, "foo", "mongodb")

	// Create some queries.
	queries := structs.PreparedQueries{
		&structs.PreparedQuery{
			ID:   testUUID(),
			Name: "alice",
			Service: structs.ServiceQuery{
				Service: "redis",
			},
		},
		&structs.PreparedQuery{
			ID:   testUUID(),
			Name: "bob",
			Service: structs.ServiceQuery{
				Service: "mongodb",
			},
		},
	}

	// Force the sort order of the UUIDs before we create them so the
	// order is deterministic.
	queries[0].ID = "a" + queries[0].ID[1:]
	queries[1].ID = "b" + queries[1].ID[1:]

	// Now create the queries.
	for i, query := range queries {
		if err := s.PreparedQuerySet(uint64(4+i), query); err != nil {
			t.Fatalf("err: %s", err)
		}
	}

	// Read it back and verify.
	expected := structs.PreparedQueries{
		&structs.PreparedQuery{
			ID:   queries[0].ID,
			Name: "alice",
			Service: structs.ServiceQuery{
				Service: "redis",
			},
			RaftIndex: structs.RaftIndex{
				CreateIndex: 4,
				ModifyIndex: 4,
			},
		},
		&structs.PreparedQuery{
			ID:   queries[1].ID,
			Name: "bob",
			Service: structs.ServiceQuery{
				Service: "mongodb",
			},
			RaftIndex: structs.RaftIndex{
				CreateIndex: 5,
				ModifyIndex: 5,
			},
		},
	}
	idx, actual, err = s.PreparedQueryList()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("bad: %v", actual)
	}
}

func TestStateStore_PreparedQuery_Snapshot_Restore(t *testing.T) {
	s := testStateStore(t)

	// Set up our test environment.
	testRegisterNode(t, s, 1, "foo")
	testRegisterService(t, s, 2, "foo", "redis")
	testRegisterService(t, s, 3, "foo", "mongodb")

	// Create some queries.
	queries := structs.PreparedQueries{
		&structs.PreparedQuery{
			ID:   testUUID(),
			Name: "alice",
			Service: structs.ServiceQuery{
				Service: "redis",
			},
		},
		&structs.PreparedQuery{
			ID:   testUUID(),
			Name: "bob-",
			Template: structs.QueryTemplateOptions{
				Type: structs.QueryTemplateTypeNamePrefixMatch,
			},
			Service: structs.ServiceQuery{
				Service: "${name.suffix}",
			},
		},
	}

	// Force the sort order of the UUIDs before we create them so the
	// order is deterministic.
	queries[0].ID = "a" + queries[0].ID[1:]
	queries[1].ID = "b" + queries[1].ID[1:]

	// Now create the queries.
	for i, query := range queries {
		if err := s.PreparedQuerySet(uint64(4+i), query); err != nil {
			t.Fatalf("err: %s", err)
		}
	}

	// Snapshot the queries.
	snap := s.Snapshot()
	defer snap.Close()

	// Alter the real state store.
	if err := s.PreparedQueryDelete(6, queries[0].ID); err != nil {
		t.Fatalf("err: %s", err)
	}

	// Verify the snapshot.
	if idx := snap.LastIndex(); idx != 5 {
		t.Fatalf("bad index: %d", idx)
	}
	expected := structs.PreparedQueries{
		&structs.PreparedQuery{
			ID:   queries[0].ID,
			Name: "alice",
			Service: structs.ServiceQuery{
				Service: "redis",
			},
			RaftIndex: structs.RaftIndex{
				CreateIndex: 4,
				ModifyIndex: 4,
			},
		},
		&structs.PreparedQuery{
			ID:   queries[1].ID,
			Name: "bob-",
			Template: structs.QueryTemplateOptions{
				Type: structs.QueryTemplateTypeNamePrefixMatch,
			},
			Service: structs.ServiceQuery{
				Service: "${name.suffix}",
			},
			RaftIndex: structs.RaftIndex{
				CreateIndex: 5,
				ModifyIndex: 5,
			},
		},
	}
	dump, err := snap.PreparedQueries()
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if !reflect.DeepEqual(dump, expected) {
		t.Fatalf("bad: %v", dump)
	}

	// Restore the values into a new state store.
	func() {
		s := testStateStore(t)
		restore := s.Restore()
		for _, query := range dump {
			if err := restore.PreparedQuery(query); err != nil {
				t.Fatalf("err: %s", err)
			}
		}
		restore.Commit()

		// Read the restored queries back out and verify that they
		// match.
		idx, actual, err := s.PreparedQueryList()
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if idx != 5 {
			t.Fatalf("bad index: %d", idx)
		}
		if !reflect.DeepEqual(actual, expected) {
			t.Fatalf("bad: %v", actual)
		}

		// Make sure the second query, which is a template, was compiled
		// and can be resolved.
		_, query, err := s.PreparedQueryResolve("bob-backwards-is-bob")
		if err != nil {
			t.Fatalf("err: %s", err)
		}
		if query == nil {
			t.Fatalf("should have resolved the query")
		}
		if query.Service.Service != "backwards-is-bob" {
			t.Fatalf("bad: %s", query.Service.Service)
		}
	}()
}

func TestStateStore_PreparedQuery_Watches(t *testing.T) {
	s := testStateStore(t)

	// Set up our test environment.
	testRegisterNode(t, s, 1, "foo")
	testRegisterService(t, s, 2, "foo", "redis")

	query := &structs.PreparedQuery{
		ID: testUUID(),
		Service: structs.ServiceQuery{
			Service: "redis",
		},
	}

	// Call functions that update the queries table and make sure a watch
	// fires each time.
	verifyWatch(t, s.getTableWatch("prepared-queries"), func() {
		if err := s.PreparedQuerySet(3, query); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	verifyWatch(t, s.getTableWatch("prepared-queries"), func() {
		if err := s.PreparedQueryDelete(4, query.ID); err != nil {
			t.Fatalf("err: %s", err)
		}
	})
	verifyWatch(t, s.getTableWatch("prepared-queries"), func() {
		restore := s.Restore()
		if err := restore.PreparedQuery(query); err != nil {
			t.Fatalf("err: %s", err)
		}
		restore.Commit()
	})
}
