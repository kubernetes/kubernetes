package state

import (
	"fmt"
	"regexp"

	"github.com/hashicorp/consul/consul/prepared_query"
	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/go-memdb"
)

// validUUID is used to check if a given string looks like a UUID
var validUUID = regexp.MustCompile(`(?i)^[\da-f]{8}-[\da-f]{4}-[\da-f]{4}-[\da-f]{4}-[\da-f]{12}$`)

// isUUID returns true if the given string is a valid UUID.
func isUUID(str string) bool {
	return validUUID.MatchString(str)
}

// queryWrapper is an internal structure that is used to store a query alongside
// its compiled template, which can be nil.
type queryWrapper struct {
	// We embed the PreparedQuery structure so that the UUID field indexer
	// can see the ID directly.
	*structs.PreparedQuery

	// ct is the compiled template, or nil if the query isn't a template. The
	// state store manages this and keeps it up to date every time the query
	// changes.
	ct *prepared_query.CompiledTemplate
}

// toPreparedQuery unwraps the internal form of a prepared query and returns
// the regular struct.
func toPreparedQuery(wrapped interface{}) *structs.PreparedQuery {
	if wrapped == nil {
		return nil
	}
	return wrapped.(*queryWrapper).PreparedQuery
}

// PreparedQueries is used to pull all the prepared queries from the snapshot.
func (s *StateSnapshot) PreparedQueries() (structs.PreparedQueries, error) {
	queries, err := s.tx.Get("prepared-queries", "id")
	if err != nil {
		return nil, err
	}

	var ret structs.PreparedQueries
	for wrapped := queries.Next(); wrapped != nil; wrapped = queries.Next() {
		ret = append(ret, toPreparedQuery(wrapped))
	}
	return ret, nil
}

// PrepparedQuery is used when restoring from a snapshot. For general inserts,
// use PreparedQuerySet.
func (s *StateRestore) PreparedQuery(query *structs.PreparedQuery) error {
	// If this is a template, compile it, otherwise leave the compiled
	// template field nil.
	var ct *prepared_query.CompiledTemplate
	if prepared_query.IsTemplate(query) {
		var err error
		ct, err = prepared_query.Compile(query)
		if err != nil {
			return fmt.Errorf("failed compiling template: %s", err)
		}
	}

	// Insert the wrapped query.
	if err := s.tx.Insert("prepared-queries", &queryWrapper{query, ct}); err != nil {
		return fmt.Errorf("failed restoring prepared query: %s", err)
	}
	if err := indexUpdateMaxTxn(s.tx, query.ModifyIndex, "prepared-queries"); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	s.watches.Arm("prepared-queries")
	return nil
}

// PreparedQuerySet is used to create or update a prepared query.
func (s *StateStore) PreparedQuerySet(idx uint64, query *structs.PreparedQuery) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	if err := s.preparedQuerySetTxn(tx, idx, query); err != nil {
		return err
	}

	tx.Commit()
	return nil
}

// preparedQuerySetTxn is the inner method used to insert a prepared query with
// the proper indexes into the state store.
func (s *StateStore) preparedQuerySetTxn(tx *memdb.Txn, idx uint64, query *structs.PreparedQuery) error {
	// Check that the ID is set.
	if query.ID == "" {
		return ErrMissingQueryID
	}

	// Check for an existing query.
	wrapped, err := tx.First("prepared-queries", "id", query.ID)
	if err != nil {
		return fmt.Errorf("failed prepared query lookup: %s", err)
	}
	existing := toPreparedQuery(wrapped)

	// Set the indexes.
	if existing != nil {
		query.CreateIndex = existing.CreateIndex
		query.ModifyIndex = idx
	} else {
		query.CreateIndex = idx
		query.ModifyIndex = idx
	}

	// Verify that the query name doesn't already exist, or that we are
	// updating the same instance that has this name. If this is a template
	// and the name is empty then we make sure there's not an empty template
	// already registered.
	if query.Name != "" {
		wrapped, err := tx.First("prepared-queries", "name", query.Name)
		if err != nil {
			return fmt.Errorf("failed prepared query lookup: %s", err)
		}
		other := toPreparedQuery(wrapped)
		if other != nil && (existing == nil || existing.ID != other.ID) {
			return fmt.Errorf("name '%s' aliases an existing query name", query.Name)
		}
	} else if prepared_query.IsTemplate(query) {
		wrapped, err := tx.First("prepared-queries", "template", query.Name)
		if err != nil {
			return fmt.Errorf("failed prepared query lookup: %s", err)
		}
		other := toPreparedQuery(wrapped)
		if other != nil && (existing == nil || existing.ID != other.ID) {
			return fmt.Errorf("a query template with an empty name already exists")
		}
	}

	// Verify that the name doesn't alias any existing ID. We allow queries
	// to be looked up by ID *or* name so we don't want anyone to try to
	// register a query with a name equal to some other query's ID in an
	// attempt to hijack it. We also look up by ID *then* name in order to
	// prevent this, but it seems prudent to prevent these types of rogue
	// queries from ever making it into the state store. Note that we have
	// to see if the name looks like a UUID before checking since the UUID
	// index will complain if we look up something that's not formatted
	// like one.
	if isUUID(query.Name) {
		wrapped, err := tx.First("prepared-queries", "id", query.Name)
		if err != nil {
			return fmt.Errorf("failed prepared query lookup: %s", err)
		}
		if wrapped != nil {
			return fmt.Errorf("name '%s' aliases an existing query ID", query.Name)
		}
	}

	// Verify that the session exists.
	if query.Session != "" {
		sess, err := tx.First("sessions", "id", query.Session)
		if err != nil {
			return fmt.Errorf("failed session lookup: %s", err)
		}
		if sess == nil {
			return fmt.Errorf("invalid session %#v", query.Session)
		}
	}

	// We do not verify the service here, nor the token, if any. These are
	// checked at execute time and not doing integrity checking on them
	// helps avoid bootstrapping chicken and egg problems.

	// If this is a template, compile it, otherwise leave the compiled
	// template field nil.
	var ct *prepared_query.CompiledTemplate
	if prepared_query.IsTemplate(query) {
		var err error
		ct, err = prepared_query.Compile(query)
		if err != nil {
			return fmt.Errorf("failed compiling template: %s", err)
		}
	}

	// Insert the wrapped query.
	if err := tx.Insert("prepared-queries", &queryWrapper{query, ct}); err != nil {
		return fmt.Errorf("failed inserting prepared query: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"prepared-queries", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	tx.Defer(func() { s.tableWatches["prepared-queries"].Notify() })
	return nil
}

// PreparedQueryDelete deletes the given query by ID.
func (s *StateStore) PreparedQueryDelete(idx uint64, queryID string) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	watches := NewDumbWatchManager(s.tableWatches)
	if err := s.preparedQueryDeleteTxn(tx, idx, watches, queryID); err != nil {
		return fmt.Errorf("failed prepared query delete: %s", err)
	}

	tx.Defer(func() { watches.Notify() })
	tx.Commit()
	return nil
}

// preparedQueryDeleteTxn is the inner method used to delete a prepared query
// with the proper indexes into the state store.
func (s *StateStore) preparedQueryDeleteTxn(tx *memdb.Txn, idx uint64, watches *DumbWatchManager,
	queryID string) error {
	// Pull the query.
	wrapped, err := tx.First("prepared-queries", "id", queryID)
	if err != nil {
		return fmt.Errorf("failed prepared query lookup: %s", err)
	}
	if wrapped == nil {
		return nil
	}

	// Delete the query and update the index.
	if err := tx.Delete("prepared-queries", wrapped); err != nil {
		return fmt.Errorf("failed prepared query delete: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"prepared-queries", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	watches.Arm("prepared-queries")
	return nil
}

// PreparedQueryGet returns the given prepared query by ID.
func (s *StateStore) PreparedQueryGet(queryID string) (uint64, *structs.PreparedQuery, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("PreparedQueryGet")...)

	// Look up the query by its ID.
	wrapped, err := tx.First("prepared-queries", "id", queryID)
	if err != nil {
		return 0, nil, fmt.Errorf("failed prepared query lookup: %s", err)
	}
	return idx, toPreparedQuery(wrapped), nil
}

// PreparedQueryResolve returns the given prepared query by looking up an ID or
// Name. If the query was looked up by name and it's a template, then the
// template will be rendered before it is returned.
func (s *StateStore) PreparedQueryResolve(queryIDOrName string) (uint64, *structs.PreparedQuery, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("PreparedQueryResolve")...)

	// Explicitly ban an empty query. This will never match an ID and the
	// schema is set up so it will never match a query with an empty name,
	// but we check it here to be explicit about it (we'd never want to
	// return the results from the first query w/o a name).
	if queryIDOrName == "" {
		return 0, nil, ErrMissingQueryID
	}

	// Try first by ID if it looks like they gave us an ID. We check the
	// format before trying this because the UUID index will complain if
	// we look up something that's not formatted like one.
	if isUUID(queryIDOrName) {
		wrapped, err := tx.First("prepared-queries", "id", queryIDOrName)
		if err != nil {
			return 0, nil, fmt.Errorf("failed prepared query lookup: %s", err)
		}
		if wrapped != nil {
			query := toPreparedQuery(wrapped)
			if prepared_query.IsTemplate(query) {
				return idx, nil, fmt.Errorf("prepared query templates can only be resolved up by name, not by ID")
			}
			return idx, query, nil
		}
	}

	// prep will check to see if the query is a template and render it
	// first, otherwise it will just return a regular query.
	prep := func(wrapped interface{}) (uint64, *structs.PreparedQuery, error) {
		wrapper := wrapped.(*queryWrapper)
		if prepared_query.IsTemplate(wrapper.PreparedQuery) {
			render, err := wrapper.ct.Render(queryIDOrName)
			if err != nil {
				return idx, nil, err
			}
			return idx, render, nil
		} else {
			return idx, wrapper.PreparedQuery, nil
		}
	}

	// Next, look for an exact name match. This is the common case for static
	// prepared queries, and could also apply to templates.
	{
		wrapped, err := tx.First("prepared-queries", "name", queryIDOrName)
		if err != nil {
			return 0, nil, fmt.Errorf("failed prepared query lookup: %s", err)
		}
		if wrapped != nil {
			return prep(wrapped)
		}
	}

	// Next, look for the longest prefix match among the prepared query
	// templates.
	{
		wrapped, err := tx.LongestPrefix("prepared-queries", "template_prefix", queryIDOrName)
		if err != nil {
			return 0, nil, fmt.Errorf("failed prepared query lookup: %s", err)
		}
		if wrapped != nil {
			return prep(wrapped)
		}
	}

	return idx, nil, nil
}

// PreparedQueryList returns all the prepared queries.
func (s *StateStore) PreparedQueryList() (uint64, structs.PreparedQueries, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("PreparedQueryList")...)

	// Query all of the prepared queries in the state store.
	queries, err := tx.Get("prepared-queries", "id")
	if err != nil {
		return 0, nil, fmt.Errorf("failed prepared query lookup: %s", err)
	}

	// Go over all of the queries and build the response.
	var result structs.PreparedQueries
	for wrapped := queries.Next(); wrapped != nil; wrapped = queries.Next() {
		result = append(result, toPreparedQuery(wrapped))
	}
	return idx, result, nil
}
