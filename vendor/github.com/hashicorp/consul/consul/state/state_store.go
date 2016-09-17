package state

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/types"
	"github.com/hashicorp/go-memdb"
	"github.com/hashicorp/serf/coordinate"
)

var (
	// ErrMissingNode is the error returned when trying an operation
	// which requires a node registration but none exists.
	ErrMissingNode = errors.New("Missing node registration")

	// ErrMissingService is the error we return if trying an
	// operation which requires a service but none exists.
	ErrMissingService = errors.New("Missing service registration")

	// ErrMissingSessionID is returned when a session registration
	// is attempted with an empty session ID.
	ErrMissingSessionID = errors.New("Missing session ID")

	// ErrMissingACLID is returned when an ACL set is called on
	// an ACL with an empty ID.
	ErrMissingACLID = errors.New("Missing ACL ID")

	// ErrMissingQueryID is returned when a Query set is called on
	// a Query with an empty ID.
	ErrMissingQueryID = errors.New("Missing Query ID")
)

// StateStore is where we store all of Consul's state, including
// records of node registrations, services, checks, key/value
// pairs and more. The DB is entirely in-memory and is constructed
// from the Raft log through the FSM.
type StateStore struct {
	schema *memdb.DBSchema
	db     *memdb.MemDB

	// tableWatches holds all the full table watches, indexed by table name.
	tableWatches map[string]*FullTableWatch

	// kvsWatch holds the special prefix watch for the key value store.
	kvsWatch *PrefixWatchManager

	// kvsGraveyard manages tombstones for the key value store.
	kvsGraveyard *Graveyard

	// lockDelay holds expiration times for locks associated with keys.
	lockDelay *Delay
}

// StateSnapshot is used to provide a point-in-time snapshot. It
// works by starting a read transaction against the whole state store.
type StateSnapshot struct {
	store     *StateStore
	tx        *memdb.Txn
	lastIndex uint64
}

// StateRestore is used to efficiently manage restoring a large amount of
// data to a state store.
type StateRestore struct {
	store   *StateStore
	tx      *memdb.Txn
	watches *DumbWatchManager
}

// IndexEntry keeps a record of the last index per-table.
type IndexEntry struct {
	Key   string
	Value uint64
}

// sessionCheck is used to create a many-to-one table such that
// each check registered by a session can be mapped back to the
// session table. This is only used internally in the state
// store and thus it is not exported.
type sessionCheck struct {
	Node    string
	CheckID types.CheckID
	Session string
}

// NewStateStore creates a new in-memory state storage layer.
func NewStateStore(gc *TombstoneGC) (*StateStore, error) {
	// Create the in-memory DB.
	schema := stateStoreSchema()
	db, err := memdb.NewMemDB(schema)
	if err != nil {
		return nil, fmt.Errorf("Failed setting up state store: %s", err)
	}

	// Build up the all-table watches.
	tableWatches := make(map[string]*FullTableWatch)
	for table, _ := range schema.Tables {
		if table == "kvs" || table == "tombstones" {
			continue
		}

		tableWatches[table] = NewFullTableWatch()
	}

	// Create and return the state store.
	s := &StateStore{
		schema:       schema,
		db:           db,
		tableWatches: tableWatches,
		kvsWatch:     NewPrefixWatchManager(),
		kvsGraveyard: NewGraveyard(gc),
		lockDelay:    NewDelay(),
	}
	return s, nil
}

// Snapshot is used to create a point-in-time snapshot of the entire db.
func (s *StateStore) Snapshot() *StateSnapshot {
	tx := s.db.Txn(false)

	var tables []string
	for table, _ := range s.schema.Tables {
		tables = append(tables, table)
	}
	idx := maxIndexTxn(tx, tables...)

	return &StateSnapshot{s, tx, idx}
}

// LastIndex returns that last index that affects the snapshotted data.
func (s *StateSnapshot) LastIndex() uint64 {
	return s.lastIndex
}

// Close performs cleanup of a state snapshot.
func (s *StateSnapshot) Close() {
	s.tx.Abort()
}

// Nodes is used to pull the full list of nodes for use during snapshots.
func (s *StateSnapshot) Nodes() (memdb.ResultIterator, error) {
	iter, err := s.tx.Get("nodes", "id")
	if err != nil {
		return nil, err
	}
	return iter, nil
}

// Services is used to pull the full list of services for a given node for use
// during snapshots.
func (s *StateSnapshot) Services(node string) (memdb.ResultIterator, error) {
	iter, err := s.tx.Get("services", "node", node)
	if err != nil {
		return nil, err
	}
	return iter, nil
}

// Checks is used to pull the full list of checks for a given node for use
// during snapshots.
func (s *StateSnapshot) Checks(node string) (memdb.ResultIterator, error) {
	iter, err := s.tx.Get("checks", "node", node)
	if err != nil {
		return nil, err
	}
	return iter, nil
}

// Sessions is used to pull the full list of sessions for use during snapshots.
func (s *StateSnapshot) Sessions() (memdb.ResultIterator, error) {
	iter, err := s.tx.Get("sessions", "id")
	if err != nil {
		return nil, err
	}
	return iter, nil
}

// ACLs is used to pull all the ACLs from the snapshot.
func (s *StateSnapshot) ACLs() (memdb.ResultIterator, error) {
	iter, err := s.tx.Get("acls", "id")
	if err != nil {
		return nil, err
	}
	return iter, nil
}

// Coordinates is used to pull all the coordinates from the snapshot.
func (s *StateSnapshot) Coordinates() (memdb.ResultIterator, error) {
	iter, err := s.tx.Get("coordinates", "id")
	if err != nil {
		return nil, err
	}
	return iter, nil
}

// Restore is used to efficiently manage restoring a large amount of data into
// the state store. It works by doing all the restores inside of a single
// transaction.
func (s *StateStore) Restore() *StateRestore {
	tx := s.db.Txn(true)
	watches := NewDumbWatchManager(s.tableWatches)
	return &StateRestore{s, tx, watches}
}

// Abort abandons the changes made by a restore. This or Commit should always be
// called.
func (s *StateRestore) Abort() {
	s.tx.Abort()
}

// Commit commits the changes made by a restore. This or Abort should always be
// called.
func (s *StateRestore) Commit() {
	// Fire off a single KVS watch instead of a zillion prefix ones, and use
	// a dumb watch manager to single-fire all the full table watches.
	s.tx.Defer(func() { s.store.kvsWatch.Notify("", true) })
	s.tx.Defer(func() { s.watches.Notify() })

	s.tx.Commit()
}

// Registration is used to make sure a node, service, and check registration is
// performed within a single transaction to avoid race conditions on state
// updates.
func (s *StateRestore) Registration(idx uint64, req *structs.RegisterRequest) error {
	if err := s.store.ensureRegistrationTxn(s.tx, idx, s.watches, req); err != nil {
		return err
	}
	return nil
}

// Session is used when restoring from a snapshot. For general inserts, use
// SessionCreate.
func (s *StateRestore) Session(sess *structs.Session) error {
	// Insert the session.
	if err := s.tx.Insert("sessions", sess); err != nil {
		return fmt.Errorf("failed inserting session: %s", err)
	}

	// Insert the check mappings.
	for _, checkID := range sess.Checks {
		mapping := &sessionCheck{
			Node:    sess.Node,
			CheckID: checkID,
			Session: sess.ID,
		}
		if err := s.tx.Insert("session_checks", mapping); err != nil {
			return fmt.Errorf("failed inserting session check mapping: %s", err)
		}
	}

	// Update the index.
	if err := indexUpdateMaxTxn(s.tx, sess.ModifyIndex, "sessions"); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	s.watches.Arm("sessions")
	return nil
}

// ACL is used when restoring from a snapshot. For general inserts, use ACLSet.
func (s *StateRestore) ACL(acl *structs.ACL) error {
	if err := s.tx.Insert("acls", acl); err != nil {
		return fmt.Errorf("failed restoring acl: %s", err)
	}

	if err := indexUpdateMaxTxn(s.tx, acl.ModifyIndex, "acls"); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	s.watches.Arm("acls")
	return nil
}

// Coordinates is used when restoring from a snapshot. For general inserts, use
// CoordinateBatchUpdate. We do less vetting of the updates here because they
// already got checked on the way in during a batch update.
func (s *StateRestore) Coordinates(idx uint64, updates structs.Coordinates) error {
	for _, update := range updates {
		if err := s.tx.Insert("coordinates", update); err != nil {
			return fmt.Errorf("failed restoring coordinate: %s", err)
		}
	}

	if err := indexUpdateMaxTxn(s.tx, idx, "coordinates"); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	s.watches.Arm("coordinates")
	return nil
}

// maxIndex is a helper used to retrieve the highest known index
// amongst a set of tables in the db.
func (s *StateStore) maxIndex(tables ...string) uint64 {
	tx := s.db.Txn(false)
	defer tx.Abort()
	return maxIndexTxn(tx, tables...)
}

// maxIndexTxn is a helper used to retrieve the highest known index
// amongst a set of tables in the db.
func maxIndexTxn(tx *memdb.Txn, tables ...string) uint64 {
	var lindex uint64
	for _, table := range tables {
		ti, err := tx.First("index", "id", table)
		if err != nil {
			panic(fmt.Sprintf("unknown index: %s err: %s", table, err))
		}
		if idx, ok := ti.(*IndexEntry); ok && idx.Value > lindex {
			lindex = idx.Value
		}
	}
	return lindex
}

// indexUpdateMaxTxn is used when restoring entries and sets the table's index to
// the given idx only if it's greater than the current index.
func indexUpdateMaxTxn(tx *memdb.Txn, idx uint64, table string) error {
	ti, err := tx.First("index", "id", table)
	if err != nil {
		return fmt.Errorf("failed to retrieve existing index: %s", err)
	}

	// Always take the first update, otherwise do the > check.
	if ti == nil {
		if err := tx.Insert("index", &IndexEntry{table, idx}); err != nil {
			return fmt.Errorf("failed updating index %s", err)
		}
	} else if cur, ok := ti.(*IndexEntry); ok && idx > cur.Value {
		if err := tx.Insert("index", &IndexEntry{table, idx}); err != nil {
			return fmt.Errorf("failed updating index %s", err)
		}
	}

	return nil
}

// getWatchTables returns the list of tables that should be watched and used for
// max index calculations for the given query method. This is used for all
// methods except for KVS. This will panic if the method is unknown.
func (s *StateStore) getWatchTables(method string) []string {
	switch method {
	case "GetNode", "Nodes":
		return []string{"nodes"}
	case "Services":
		return []string{"services"}
	case "ServiceNodes", "NodeServices":
		return []string{"nodes", "services"}
	case "NodeChecks", "ServiceChecks", "ChecksInState":
		return []string{"checks"}
	case "CheckServiceNodes", "NodeInfo", "NodeDump":
		return []string{"nodes", "services", "checks"}
	case "SessionGet", "SessionList", "NodeSessions":
		return []string{"sessions"}
	case "ACLGet", "ACLList":
		return []string{"acls"}
	case "Coordinates":
		return []string{"coordinates"}
	case "PreparedQueryGet", "PreparedQueryResolve", "PreparedQueryList":
		return []string{"prepared-queries"}
	}

	panic(fmt.Sprintf("Unknown method %s", method))
}

// getTableWatch returns a full table watch for the given table. This will panic
// if the table doesn't have a full table watch.
func (s *StateStore) getTableWatch(table string) Watch {
	if watch, ok := s.tableWatches[table]; ok {
		return watch
	}

	panic(fmt.Sprintf("Unknown watch for table %s", table))
}

// GetQueryWatch returns a watch for the given query method. This is
// used for all methods except for KV; you should call GetKVSWatch instead.
// This will panic if the method is unknown.
func (s *StateStore) GetQueryWatch(method string) Watch {
	tables := s.getWatchTables(method)
	if len(tables) == 1 {
		return s.getTableWatch(tables[0])
	}

	var watches []Watch
	for _, table := range tables {
		watches = append(watches, s.getTableWatch(table))
	}
	return NewMultiWatch(watches...)
}

// GetKVSWatch returns a watch for the given prefix in the key value store.
func (s *StateStore) GetKVSWatch(prefix string) Watch {
	return s.kvsWatch.NewPrefixWatch(prefix)
}

// EnsureRegistration is used to make sure a node, service, and check
// registration is performed within a single transaction to avoid race
// conditions on state updates.
func (s *StateStore) EnsureRegistration(idx uint64, req *structs.RegisterRequest) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	watches := NewDumbWatchManager(s.tableWatches)
	if err := s.ensureRegistrationTxn(tx, idx, watches, req); err != nil {
		return err
	}

	tx.Defer(func() { watches.Notify() })
	tx.Commit()
	return nil
}

// ensureRegistrationTxn is used to make sure a node, service, and check
// registration is performed within a single transaction to avoid race
// conditions on state updates.
func (s *StateStore) ensureRegistrationTxn(tx *memdb.Txn, idx uint64, watches *DumbWatchManager,
	req *structs.RegisterRequest) error {
	// Add the node.
	node := &structs.Node{
		Node:            req.Node,
		Address:         req.Address,
		TaggedAddresses: req.TaggedAddresses,
	}
	if err := s.ensureNodeTxn(tx, idx, watches, node); err != nil {
		return fmt.Errorf("failed inserting node: %s", err)
	}

	// Add the service, if any.
	if req.Service != nil {
		if err := s.ensureServiceTxn(tx, idx, watches, req.Node, req.Service); err != nil {
			return fmt.Errorf("failed inserting service: %s", err)
		}
	}

	// Add the checks, if any.
	if req.Check != nil {
		if err := s.ensureCheckTxn(tx, idx, watches, req.Check); err != nil {
			return fmt.Errorf("failed inserting check: %s", err)
		}
	}
	for _, check := range req.Checks {
		if err := s.ensureCheckTxn(tx, idx, watches, check); err != nil {
			return fmt.Errorf("failed inserting check: %s", err)
		}
	}

	return nil
}

// EnsureNode is used to upsert node registration or modification.
func (s *StateStore) EnsureNode(idx uint64, node *structs.Node) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// Call the node upsert
	watches := NewDumbWatchManager(s.tableWatches)
	if err := s.ensureNodeTxn(tx, idx, watches, node); err != nil {
		return err
	}

	tx.Defer(func() { watches.Notify() })
	tx.Commit()
	return nil
}

// ensureNodeTxn is the inner function called to actually create a node
// registration or modify an existing one in the state store. It allows
// passing in a memdb transaction so it may be part of a larger txn.
func (s *StateStore) ensureNodeTxn(tx *memdb.Txn, idx uint64, watches *DumbWatchManager,
	node *structs.Node) error {
	// Check for an existing node
	existing, err := tx.First("nodes", "id", node.Node)
	if err != nil {
		return fmt.Errorf("node lookup failed: %s", err)
	}

	// Get the indexes
	if existing != nil {
		node.CreateIndex = existing.(*structs.Node).CreateIndex
		node.ModifyIndex = idx
	} else {
		node.CreateIndex = idx
		node.ModifyIndex = idx
	}

	// Insert the node and update the index
	if err := tx.Insert("nodes", node); err != nil {
		return fmt.Errorf("failed inserting node: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"nodes", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	watches.Arm("nodes")
	return nil
}

// GetNode is used to retrieve a node registration by node ID.
func (s *StateStore) GetNode(id string) (uint64, *structs.Node, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("GetNode")...)

	// Retrieve the node from the state store
	node, err := tx.First("nodes", "id", id)
	if err != nil {
		return 0, nil, fmt.Errorf("node lookup failed: %s", err)
	}
	if node != nil {
		return idx, node.(*structs.Node), nil
	}
	return idx, nil, nil
}

// Nodes is used to return all of the known nodes.
func (s *StateStore) Nodes() (uint64, structs.Nodes, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("Nodes")...)

	// Retrieve all of the nodes
	nodes, err := tx.Get("nodes", "id")
	if err != nil {
		return 0, nil, fmt.Errorf("failed nodes lookup: %s", err)
	}

	// Create and return the nodes list.
	var results structs.Nodes
	for node := nodes.Next(); node != nil; node = nodes.Next() {
		results = append(results, node.(*structs.Node))
	}
	return idx, results, nil
}

// DeleteNode is used to delete a given node by its ID.
func (s *StateStore) DeleteNode(idx uint64, nodeID string) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// Call the node deletion.
	if err := s.deleteNodeTxn(tx, idx, nodeID); err != nil {
		return err
	}

	tx.Commit()
	return nil
}

// deleteNodeTxn is the inner method used for removing a node from
// the store within a given transaction.
func (s *StateStore) deleteNodeTxn(tx *memdb.Txn, idx uint64, nodeID string) error {
	// Look up the node.
	node, err := tx.First("nodes", "id", nodeID)
	if err != nil {
		return fmt.Errorf("node lookup failed: %s", err)
	}
	if node == nil {
		return nil
	}

	// Use a watch manager since the inner functions can perform multiple
	// ops per table.
	watches := NewDumbWatchManager(s.tableWatches)

	// Delete all services associated with the node and update the service index.
	services, err := tx.Get("services", "node", nodeID)
	if err != nil {
		return fmt.Errorf("failed service lookup: %s", err)
	}
	var sids []string
	for service := services.Next(); service != nil; service = services.Next() {
		sids = append(sids, service.(*structs.ServiceNode).ServiceID)
	}

	// Do the delete in a separate loop so we don't trash the iterator.
	for _, sid := range sids {
		if err := s.deleteServiceTxn(tx, idx, watches, nodeID, sid); err != nil {
			return err
		}
	}

	// Delete all checks associated with the node. This will invalidate
	// sessions as necessary.
	checks, err := tx.Get("checks", "node", nodeID)
	if err != nil {
		return fmt.Errorf("failed check lookup: %s", err)
	}
	var cids []types.CheckID
	for check := checks.Next(); check != nil; check = checks.Next() {
		cids = append(cids, check.(*structs.HealthCheck).CheckID)
	}

	// Do the delete in a separate loop so we don't trash the iterator.
	for _, cid := range cids {
		if err := s.deleteCheckTxn(tx, idx, watches, nodeID, cid); err != nil {
			return err
		}
	}

	// Delete any coordinate associated with this node.
	coord, err := tx.First("coordinates", "id", nodeID)
	if err != nil {
		return fmt.Errorf("failed coordinate lookup: %s", err)
	}
	if coord != nil {
		if err := tx.Delete("coordinates", coord); err != nil {
			return fmt.Errorf("failed deleting coordinate: %s", err)
		}
		if err := tx.Insert("index", &IndexEntry{"coordinates", idx}); err != nil {
			return fmt.Errorf("failed updating index: %s", err)
		}
		watches.Arm("coordinates")
	}

	// Delete the node and update the index.
	if err := tx.Delete("nodes", node); err != nil {
		return fmt.Errorf("failed deleting node: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"nodes", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	// Invalidate any sessions for this node.
	sessions, err := tx.Get("sessions", "node", nodeID)
	if err != nil {
		return fmt.Errorf("failed session lookup: %s", err)
	}
	var ids []string
	for sess := sessions.Next(); sess != nil; sess = sessions.Next() {
		ids = append(ids, sess.(*structs.Session).ID)
	}

	// Do the delete in a separate loop so we don't trash the iterator.
	for _, id := range ids {
		if err := s.deleteSessionTxn(tx, idx, watches, id); err != nil {
			return fmt.Errorf("failed session delete: %s", err)
		}
	}

	watches.Arm("nodes")
	tx.Defer(func() { watches.Notify() })
	return nil
}

// EnsureService is called to upsert creation of a given NodeService.
func (s *StateStore) EnsureService(idx uint64, node string, svc *structs.NodeService) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// Call the service registration upsert
	watches := NewDumbWatchManager(s.tableWatches)
	if err := s.ensureServiceTxn(tx, idx, watches, node, svc); err != nil {
		return err
	}

	tx.Defer(func() { watches.Notify() })
	tx.Commit()
	return nil
}

// ensureServiceTxn is used to upsert a service registration within an
// existing memdb transaction.
func (s *StateStore) ensureServiceTxn(tx *memdb.Txn, idx uint64, watches *DumbWatchManager,
	node string, svc *structs.NodeService) error {
	// Check for existing service
	existing, err := tx.First("services", "id", node, svc.ID)
	if err != nil {
		return fmt.Errorf("failed service lookup: %s", err)
	}

	// Create the service node entry and populate the indexes. Note that
	// conversion doesn't populate any of the node-specific information
	// (Address and TaggedAddresses). That's always populated when we read
	// from the state store.
	entry := svc.ToServiceNode(node)
	if existing != nil {
		entry.CreateIndex = existing.(*structs.ServiceNode).CreateIndex
		entry.ModifyIndex = idx
	} else {
		entry.CreateIndex = idx
		entry.ModifyIndex = idx
	}

	// Get the node
	n, err := tx.First("nodes", "id", node)
	if err != nil {
		return fmt.Errorf("failed node lookup: %s", err)
	}
	if n == nil {
		return ErrMissingNode
	}

	// Insert the service and update the index
	if err := tx.Insert("services", entry); err != nil {
		return fmt.Errorf("failed inserting service: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"services", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	watches.Arm("services")
	return nil
}

// Services returns all services along with a list of associated tags.
func (s *StateStore) Services() (uint64, structs.Services, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("Services")...)

	// List all the services.
	services, err := tx.Get("services", "id")
	if err != nil {
		return 0, nil, fmt.Errorf("failed querying services: %s", err)
	}

	// Rip through the services and enumerate them and their unique set of
	// tags.
	unique := make(map[string]map[string]struct{})
	for service := services.Next(); service != nil; service = services.Next() {
		svc := service.(*structs.ServiceNode)
		tags, ok := unique[svc.ServiceName]
		if !ok {
			unique[svc.ServiceName] = make(map[string]struct{})
			tags = unique[svc.ServiceName]
		}
		for _, tag := range svc.ServiceTags {
			tags[tag] = struct{}{}
		}
	}

	// Generate the output structure.
	var results = make(structs.Services)
	for service, tags := range unique {
		results[service] = make([]string, 0)
		for tag, _ := range tags {
			results[service] = append(results[service], tag)
		}
	}
	return idx, results, nil
}

// ServiceNodes returns the nodes associated with a given service name.
func (s *StateStore) ServiceNodes(serviceName string) (uint64, structs.ServiceNodes, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("ServiceNodes")...)

	// List all the services.
	services, err := tx.Get("services", "service", serviceName)
	if err != nil {
		return 0, nil, fmt.Errorf("failed service lookup: %s", err)
	}
	var results structs.ServiceNodes
	for service := services.Next(); service != nil; service = services.Next() {
		results = append(results, service.(*structs.ServiceNode))
	}

	// Fill in the address details.
	results, err = s.parseServiceNodes(tx, results)
	if err != nil {
		return 0, nil, fmt.Errorf("failed parsing service nodes: %s", err)
	}
	return idx, results, nil
}

// ServiceTagNodes returns the nodes associated with a given service, filtering
// out services that don't contain the given tag.
func (s *StateStore) ServiceTagNodes(service, tag string) (uint64, structs.ServiceNodes, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("ServiceNodes")...)

	// List all the services.
	services, err := tx.Get("services", "service", service)
	if err != nil {
		return 0, nil, fmt.Errorf("failed service lookup: %s", err)
	}

	// Gather all the services and apply the tag filter.
	var results structs.ServiceNodes
	for service := services.Next(); service != nil; service = services.Next() {
		svc := service.(*structs.ServiceNode)
		if !serviceTagFilter(svc, tag) {
			results = append(results, svc)
		}
	}

	// Fill in the address details.
	results, err = s.parseServiceNodes(tx, results)
	if err != nil {
		return 0, nil, fmt.Errorf("failed parsing service nodes: %s", err)
	}
	return idx, results, nil
}

// serviceTagFilter returns true (should filter) if the given service node
// doesn't contain the given tag.
func serviceTagFilter(sn *structs.ServiceNode, tag string) bool {
	tag = strings.ToLower(tag)

	// Look for the lower cased version of the tag.
	for _, t := range sn.ServiceTags {
		if strings.ToLower(t) == tag {
			return false
		}
	}

	// If we didn't hit the tag above then we should filter.
	return true
}

// parseServiceNodes iterates over a services query and fills in the node details,
// returning a ServiceNodes slice.
func (s *StateStore) parseServiceNodes(tx *memdb.Txn, services structs.ServiceNodes) (structs.ServiceNodes, error) {
	var results structs.ServiceNodes
	for _, sn := range services {
		// Note that we have to clone here because we don't want to
		// modify the node-related fields on the object in the database,
		// which is what we are referencing.
		s := sn.PartialClone()

		// Grab the corresponding node record.
		n, err := tx.First("nodes", "id", sn.Node)
		if err != nil {
			return nil, fmt.Errorf("failed node lookup: %s", err)
		}

		// Populate the node-related fields. The tagged addresses may be
		// used by agents to perform address translation if they are
		// configured to do that.
		node := n.(*structs.Node)
		s.Address = node.Address
		s.TaggedAddresses = node.TaggedAddresses

		results = append(results, s)
	}
	return results, nil
}

// NodeServices is used to query service registrations by node ID.
func (s *StateStore) NodeServices(nodeID string) (uint64, *structs.NodeServices, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("NodeServices")...)

	// Query the node
	n, err := tx.First("nodes", "id", nodeID)
	if err != nil {
		return 0, nil, fmt.Errorf("node lookup failed: %s", err)
	}
	if n == nil {
		return 0, nil, nil
	}
	node := n.(*structs.Node)

	// Read all of the services
	services, err := tx.Get("services", "node", nodeID)
	if err != nil {
		return 0, nil, fmt.Errorf("failed querying services for node %q: %s", nodeID, err)
	}

	// Initialize the node services struct
	ns := &structs.NodeServices{
		Node:     node,
		Services: make(map[string]*structs.NodeService),
	}

	// Add all of the services to the map.
	for service := services.Next(); service != nil; service = services.Next() {
		svc := service.(*structs.ServiceNode).ToNodeService()
		ns.Services[svc.ID] = svc
	}

	return idx, ns, nil
}

// DeleteService is used to delete a given service associated with a node.
func (s *StateStore) DeleteService(idx uint64, nodeID, serviceID string) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// Call the service deletion
	watches := NewDumbWatchManager(s.tableWatches)
	if err := s.deleteServiceTxn(tx, idx, watches, nodeID, serviceID); err != nil {
		return err
	}

	tx.Defer(func() { watches.Notify() })
	tx.Commit()
	return nil
}

// deleteServiceTxn is the inner method called to remove a service
// registration within an existing transaction.
func (s *StateStore) deleteServiceTxn(tx *memdb.Txn, idx uint64, watches *DumbWatchManager, nodeID, serviceID string) error {
	// Look up the service.
	service, err := tx.First("services", "id", nodeID, serviceID)
	if err != nil {
		return fmt.Errorf("failed service lookup: %s", err)
	}
	if service == nil {
		return nil
	}

	// Delete any checks associated with the service. This will invalidate
	// sessions as necessary.
	checks, err := tx.Get("checks", "node_service", nodeID, serviceID)
	if err != nil {
		return fmt.Errorf("failed service check lookup: %s", err)
	}
	var cids []types.CheckID
	for check := checks.Next(); check != nil; check = checks.Next() {
		cids = append(cids, check.(*structs.HealthCheck).CheckID)
	}

	// Do the delete in a separate loop so we don't trash the iterator.
	for _, cid := range cids {
		if err := s.deleteCheckTxn(tx, idx, watches, nodeID, cid); err != nil {
			return err
		}
	}

	// Update the index.
	if err := tx.Insert("index", &IndexEntry{"checks", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	// Delete the service and update the index
	if err := tx.Delete("services", service); err != nil {
		return fmt.Errorf("failed deleting service: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"services", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	watches.Arm("services")
	return nil
}

// EnsureCheck is used to store a check registration in the db.
func (s *StateStore) EnsureCheck(idx uint64, hc *structs.HealthCheck) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// Call the check registration
	watches := NewDumbWatchManager(s.tableWatches)
	if err := s.ensureCheckTxn(tx, idx, watches, hc); err != nil {
		return err
	}

	tx.Defer(func() { watches.Notify() })
	tx.Commit()
	return nil
}

// ensureCheckTransaction is used as the inner method to handle inserting
// a health check into the state store. It ensures safety against inserting
// checks with no matching node or service.
func (s *StateStore) ensureCheckTxn(tx *memdb.Txn, idx uint64, watches *DumbWatchManager,
	hc *structs.HealthCheck) error {
	// Check if we have an existing health check
	existing, err := tx.First("checks", "id", hc.Node, string(hc.CheckID))
	if err != nil {
		return fmt.Errorf("failed health check lookup: %s", err)
	}

	// Set the indexes
	if existing != nil {
		hc.CreateIndex = existing.(*structs.HealthCheck).CreateIndex
		hc.ModifyIndex = idx
	} else {
		hc.CreateIndex = idx
		hc.ModifyIndex = idx
	}

	// Use the default check status if none was provided
	if hc.Status == "" {
		hc.Status = structs.HealthCritical
	}

	// Get the node
	node, err := tx.First("nodes", "id", hc.Node)
	if err != nil {
		return fmt.Errorf("failed node lookup: %s", err)
	}
	if node == nil {
		return ErrMissingNode
	}

	// If the check is associated with a service, check that we have
	// a registration for the service.
	if hc.ServiceID != "" {
		service, err := tx.First("services", "id", hc.Node, hc.ServiceID)
		if err != nil {
			return fmt.Errorf("failed service lookup: %s", err)
		}
		if service == nil {
			return ErrMissingService
		}

		// Copy in the service name
		hc.ServiceName = service.(*structs.ServiceNode).ServiceName
	}

	// Delete any sessions for this check if the health is critical.
	if hc.Status == structs.HealthCritical {
		mappings, err := tx.Get("session_checks", "node_check", hc.Node, string(hc.CheckID))
		if err != nil {
			return fmt.Errorf("failed session checks lookup: %s", err)
		}

		var ids []string
		for mapping := mappings.Next(); mapping != nil; mapping = mappings.Next() {
			ids = append(ids, mapping.(*sessionCheck).Session)
		}

		// Delete the session in a separate loop so we don't trash the
		// iterator.
		watches := NewDumbWatchManager(s.tableWatches)
		for _, id := range ids {
			if err := s.deleteSessionTxn(tx, idx, watches, id); err != nil {
				return fmt.Errorf("failed deleting session: %s", err)
			}
		}
		tx.Defer(func() { watches.Notify() })
	}

	// Persist the check registration in the db.
	if err := tx.Insert("checks", hc); err != nil {
		return fmt.Errorf("failed inserting check: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"checks", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	watches.Arm("checks")
	return nil
}

// NodeChecks is used to retrieve checks associated with the
// given node from the state store.
func (s *StateStore) NodeChecks(nodeID string) (uint64, structs.HealthChecks, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("NodeChecks")...)

	// Return the checks.
	checks, err := tx.Get("checks", "node", nodeID)
	if err != nil {
		return 0, nil, fmt.Errorf("failed check lookup: %s", err)
	}
	return s.parseChecks(idx, checks)
}

// ServiceChecks is used to get all checks associated with a
// given service ID. The query is performed against a service
// _name_ instead of a service ID.
func (s *StateStore) ServiceChecks(serviceName string) (uint64, structs.HealthChecks, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("ServiceChecks")...)

	// Return the checks.
	checks, err := tx.Get("checks", "service", serviceName)
	if err != nil {
		return 0, nil, fmt.Errorf("failed check lookup: %s", err)
	}
	return s.parseChecks(idx, checks)
}

// ChecksInState is used to query the state store for all checks
// which are in the provided state.
func (s *StateStore) ChecksInState(state string) (uint64, structs.HealthChecks, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("ChecksInState")...)

	// Query all checks if HealthAny is passed
	if state == structs.HealthAny {
		checks, err := tx.Get("checks", "status")
		if err != nil {
			return 0, nil, fmt.Errorf("failed check lookup: %s", err)
		}
		return s.parseChecks(idx, checks)
	}

	// Any other state we need to query for explicitly
	checks, err := tx.Get("checks", "status", state)
	if err != nil {
		return 0, nil, fmt.Errorf("failed check lookup: %s", err)
	}
	return s.parseChecks(idx, checks)
}

// parseChecks is a helper function used to deduplicate some
// repetitive code for returning health checks.
func (s *StateStore) parseChecks(idx uint64, iter memdb.ResultIterator) (uint64, structs.HealthChecks, error) {
	// Gather the health checks and return them properly type casted.
	var results structs.HealthChecks
	for check := iter.Next(); check != nil; check = iter.Next() {
		results = append(results, check.(*structs.HealthCheck))
	}
	return idx, results, nil
}

// DeleteCheck is used to delete a health check registration.
func (s *StateStore) DeleteCheck(idx uint64, node string, checkID types.CheckID) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// Call the check deletion
	watches := NewDumbWatchManager(s.tableWatches)
	if err := s.deleteCheckTxn(tx, idx, watches, node, checkID); err != nil {
		return err
	}

	tx.Defer(func() { watches.Notify() })
	tx.Commit()
	return nil
}

// deleteCheckTxn is the inner method used to call a health
// check deletion within an existing transaction.
func (s *StateStore) deleteCheckTxn(tx *memdb.Txn, idx uint64, watches *DumbWatchManager, node string, checkID types.CheckID) error {
	// Try to retrieve the existing health check.
	hc, err := tx.First("checks", "id", node, string(checkID))
	if err != nil {
		return fmt.Errorf("check lookup failed: %s", err)
	}
	if hc == nil {
		return nil
	}

	// Delete the check from the DB and update the index.
	if err := tx.Delete("checks", hc); err != nil {
		return fmt.Errorf("failed removing check: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"checks", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	// Delete any sessions for this check.
	mappings, err := tx.Get("session_checks", "node_check", node, string(checkID))
	if err != nil {
		return fmt.Errorf("failed session checks lookup: %s", err)
	}
	var ids []string
	for mapping := mappings.Next(); mapping != nil; mapping = mappings.Next() {
		ids = append(ids, mapping.(*sessionCheck).Session)
	}

	// Do the delete in a separate loop so we don't trash the iterator.
	for _, id := range ids {
		if err := s.deleteSessionTxn(tx, idx, watches, id); err != nil {
			return fmt.Errorf("failed deleting session: %s", err)
		}
	}

	watches.Arm("checks")
	return nil
}

// CheckServiceNodes is used to query all nodes and checks for a given service
// The results are compounded into a CheckServiceNodes, and the index returned
// is the maximum index observed over any node, check, or service in the result
// set.
func (s *StateStore) CheckServiceNodes(serviceName string) (uint64, structs.CheckServiceNodes, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("CheckServiceNodes")...)

	// Query the state store for the service.
	services, err := tx.Get("services", "service", serviceName)
	if err != nil {
		return 0, nil, fmt.Errorf("failed service lookup: %s", err)
	}

	// Return the results.
	var results structs.ServiceNodes
	for service := services.Next(); service != nil; service = services.Next() {
		results = append(results, service.(*structs.ServiceNode))
	}
	return s.parseCheckServiceNodes(tx, idx, results, err)
}

// CheckServiceTagNodes is used to query all nodes and checks for a given
// service, filtering out services that don't contain the given tag. The results
// are compounded into a CheckServiceNodes, and the index returned is the maximum
// index observed over any node, check, or service in the result set.
func (s *StateStore) CheckServiceTagNodes(serviceName, tag string) (uint64, structs.CheckServiceNodes, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("CheckServiceNodes")...)

	// Query the state store for the service.
	services, err := tx.Get("services", "service", serviceName)
	if err != nil {
		return 0, nil, fmt.Errorf("failed service lookup: %s", err)
	}

	// Return the results, filtering by tag.
	var results structs.ServiceNodes
	for service := services.Next(); service != nil; service = services.Next() {
		svc := service.(*structs.ServiceNode)
		if !serviceTagFilter(svc, tag) {
			results = append(results, svc)
		}
	}
	return s.parseCheckServiceNodes(tx, idx, results, err)
}

// parseCheckServiceNodes is used to parse through a given set of services,
// and query for an associated node and a set of checks. This is the inner
// method used to return a rich set of results from a more simple query.
func (s *StateStore) parseCheckServiceNodes(
	tx *memdb.Txn, idx uint64, services structs.ServiceNodes,
	err error) (uint64, structs.CheckServiceNodes, error) {
	if err != nil {
		return 0, nil, err
	}

	// Special-case the zero return value to nil, since this ends up in
	// external APIs.
	if len(services) == 0 {
		return idx, nil, nil
	}

	results := make(structs.CheckServiceNodes, 0, len(services))
	for _, sn := range services {
		// Retrieve the node.
		n, err := tx.First("nodes", "id", sn.Node)
		if err != nil {
			return 0, nil, fmt.Errorf("failed node lookup: %s", err)
		}
		if n == nil {
			return 0, nil, ErrMissingNode
		}
		node := n.(*structs.Node)

		// We need to return the checks specific to the given service
		// as well as the node itself. Unfortunately, memdb won't let
		// us use the index to do the latter query so we have to pull
		// them all and filter.
		var checks structs.HealthChecks
		iter, err := tx.Get("checks", "node", sn.Node)
		if err != nil {
			return 0, nil, err
		}
		for check := iter.Next(); check != nil; check = iter.Next() {
			hc := check.(*structs.HealthCheck)
			if hc.ServiceID == "" || hc.ServiceID == sn.ServiceID {
				checks = append(checks, hc)
			}
		}

		// Append to the results.
		results = append(results, structs.CheckServiceNode{
			Node:    node,
			Service: sn.ToNodeService(),
			Checks:  checks,
		})
	}

	return idx, results, nil
}

// NodeInfo is used to generate a dump of a single node. The dump includes
// all services and checks which are registered against the node.
func (s *StateStore) NodeInfo(node string) (uint64, structs.NodeDump, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("NodeInfo")...)

	// Query the node by the passed node
	nodes, err := tx.Get("nodes", "id", node)
	if err != nil {
		return 0, nil, fmt.Errorf("failed node lookup: %s", err)
	}
	return s.parseNodes(tx, idx, nodes)
}

// NodeDump is used to generate a dump of all nodes. This call is expensive
// as it has to query every node, service, and check. The response can also
// be quite large since there is currently no filtering applied.
func (s *StateStore) NodeDump() (uint64, structs.NodeDump, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("NodeDump")...)

	// Fetch all of the registered nodes
	nodes, err := tx.Get("nodes", "id")
	if err != nil {
		return 0, nil, fmt.Errorf("failed node lookup: %s", err)
	}
	return s.parseNodes(tx, idx, nodes)
}

// parseNodes takes an iterator over a set of nodes and returns a struct
// containing the nodes along with all of their associated services
// and/or health checks.
func (s *StateStore) parseNodes(tx *memdb.Txn, idx uint64,
	iter memdb.ResultIterator) (uint64, structs.NodeDump, error) {

	var results structs.NodeDump
	for n := iter.Next(); n != nil; n = iter.Next() {
		node := n.(*structs.Node)

		// Create the wrapped node
		dump := &structs.NodeInfo{
			Node:            node.Node,
			Address:         node.Address,
			TaggedAddresses: node.TaggedAddresses,
		}

		// Query the node services
		services, err := tx.Get("services", "node", node.Node)
		if err != nil {
			return 0, nil, fmt.Errorf("failed services lookup: %s", err)
		}
		for service := services.Next(); service != nil; service = services.Next() {
			ns := service.(*structs.ServiceNode).ToNodeService()
			dump.Services = append(dump.Services, ns)
		}

		// Query the node checks
		checks, err := tx.Get("checks", "node", node.Node)
		if err != nil {
			return 0, nil, fmt.Errorf("failed node lookup: %s", err)
		}
		for check := checks.Next(); check != nil; check = checks.Next() {
			hc := check.(*structs.HealthCheck)
			dump.Checks = append(dump.Checks, hc)
		}

		// Add the result to the slice
		results = append(results, dump)
	}
	return idx, results, nil
}

// SessionCreate is used to register a new session in the state store.
func (s *StateStore) SessionCreate(idx uint64, sess *structs.Session) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// This code is technically able to (incorrectly) update an existing
	// session but we never do that in practice. The upstream endpoint code
	// always adds a unique ID when doing a create operation so we never hit
	// an existing session again. It isn't worth the overhead to verify
	// that here, but it's worth noting that we should never do this in the
	// future.

	// Call the session creation
	if err := s.sessionCreateTxn(tx, idx, sess); err != nil {
		return err
	}

	tx.Commit()
	return nil
}

// sessionCreateTxn is the inner method used for creating session entries in
// an open transaction. Any health checks registered with the session will be
// checked for failing status. Returns any error encountered.
func (s *StateStore) sessionCreateTxn(tx *memdb.Txn, idx uint64, sess *structs.Session) error {
	// Check that we have a session ID
	if sess.ID == "" {
		return ErrMissingSessionID
	}

	// Verify the session behavior is valid
	switch sess.Behavior {
	case "":
		// Release by default to preserve backwards compatibility
		sess.Behavior = structs.SessionKeysRelease
	case structs.SessionKeysRelease:
	case structs.SessionKeysDelete:
	default:
		return fmt.Errorf("Invalid session behavior: %s", sess.Behavior)
	}

	// Assign the indexes. ModifyIndex likely will not be used but
	// we set it here anyways for sanity.
	sess.CreateIndex = idx
	sess.ModifyIndex = idx

	// Check that the node exists
	node, err := tx.First("nodes", "id", sess.Node)
	if err != nil {
		return fmt.Errorf("failed node lookup: %s", err)
	}
	if node == nil {
		return ErrMissingNode
	}

	// Go over the session checks and ensure they exist.
	for _, checkID := range sess.Checks {
		check, err := tx.First("checks", "id", sess.Node, string(checkID))
		if err != nil {
			return fmt.Errorf("failed check lookup: %s", err)
		}
		if check == nil {
			return fmt.Errorf("Missing check '%s' registration", checkID)
		}

		// Check that the check is not in critical state
		status := check.(*structs.HealthCheck).Status
		if status == structs.HealthCritical {
			return fmt.Errorf("Check '%s' is in %s state", checkID, status)
		}
	}

	// Insert the session
	if err := tx.Insert("sessions", sess); err != nil {
		return fmt.Errorf("failed inserting session: %s", err)
	}

	// Insert the check mappings
	for _, checkID := range sess.Checks {
		mapping := &sessionCheck{
			Node:    sess.Node,
			CheckID: checkID,
			Session: sess.ID,
		}
		if err := tx.Insert("session_checks", mapping); err != nil {
			return fmt.Errorf("failed inserting session check mapping: %s", err)
		}
	}

	// Update the index
	if err := tx.Insert("index", &IndexEntry{"sessions", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	tx.Defer(func() { s.tableWatches["sessions"].Notify() })
	return nil
}

// SessionGet is used to retrieve an active session from the state store.
func (s *StateStore) SessionGet(sessionID string) (uint64, *structs.Session, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("SessionGet")...)

	// Look up the session by its ID
	session, err := tx.First("sessions", "id", sessionID)
	if err != nil {
		return 0, nil, fmt.Errorf("failed session lookup: %s", err)
	}
	if session != nil {
		return idx, session.(*structs.Session), nil
	}
	return idx, nil, nil
}

// SessionList returns a slice containing all of the active sessions.
func (s *StateStore) SessionList() (uint64, structs.Sessions, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("SessionList")...)

	// Query all of the active sessions.
	sessions, err := tx.Get("sessions", "id")
	if err != nil {
		return 0, nil, fmt.Errorf("failed session lookup: %s", err)
	}

	// Go over the sessions and create a slice of them.
	var result structs.Sessions
	for session := sessions.Next(); session != nil; session = sessions.Next() {
		result = append(result, session.(*structs.Session))
	}
	return idx, result, nil
}

// NodeSessions returns a set of active sessions associated
// with the given node ID. The returned index is the highest
// index seen from the result set.
func (s *StateStore) NodeSessions(nodeID string) (uint64, structs.Sessions, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("NodeSessions")...)

	// Get all of the sessions which belong to the node
	sessions, err := tx.Get("sessions", "node", nodeID)
	if err != nil {
		return 0, nil, fmt.Errorf("failed session lookup: %s", err)
	}

	// Go over all of the sessions and return them as a slice
	var result structs.Sessions
	for session := sessions.Next(); session != nil; session = sessions.Next() {
		result = append(result, session.(*structs.Session))
	}
	return idx, result, nil
}

// SessionDestroy is used to remove an active session. This will
// implicitly invalidate the session and invoke the specified
// session destroy behavior.
func (s *StateStore) SessionDestroy(idx uint64, sessionID string) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// Call the session deletion.
	watches := NewDumbWatchManager(s.tableWatches)
	if err := s.deleteSessionTxn(tx, idx, watches, sessionID); err != nil {
		return err
	}

	tx.Defer(func() { watches.Notify() })
	tx.Commit()
	return nil
}

// deleteSessionTxn is the inner method, which is used to do the actual
// session deletion and handle session invalidation, watch triggers, etc.
func (s *StateStore) deleteSessionTxn(tx *memdb.Txn, idx uint64, watches *DumbWatchManager, sessionID string) error {
	// Look up the session.
	sess, err := tx.First("sessions", "id", sessionID)
	if err != nil {
		return fmt.Errorf("failed session lookup: %s", err)
	}
	if sess == nil {
		return nil
	}

	// Delete the session and write the new index.
	if err := tx.Delete("sessions", sess); err != nil {
		return fmt.Errorf("failed deleting session: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"sessions", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	// Enforce the max lock delay.
	session := sess.(*structs.Session)
	delay := session.LockDelay
	if delay > structs.MaxLockDelay {
		delay = structs.MaxLockDelay
	}

	// Snag the current now time so that all the expirations get calculated
	// the same way.
	now := time.Now()

	// Get an iterator over all of the keys with the given session.
	entries, err := tx.Get("kvs", "session", sessionID)
	if err != nil {
		return fmt.Errorf("failed kvs lookup: %s", err)
	}
	var kvs []interface{}
	for entry := entries.Next(); entry != nil; entry = entries.Next() {
		kvs = append(kvs, entry)
	}

	// Invalidate any held locks.
	switch session.Behavior {
	case structs.SessionKeysRelease:
		for _, obj := range kvs {
			// Note that we clone here since we are modifying the
			// returned object and want to make sure our set op
			// respects the transaction we are in.
			e := obj.(*structs.DirEntry).Clone()
			e.Session = ""
			if err := s.kvsSetTxn(tx, idx, e, true); err != nil {
				return fmt.Errorf("failed kvs update: %s", err)
			}

			// Apply the lock delay if present.
			if delay > 0 {
				s.lockDelay.SetExpiration(e.Key, now, delay)
			}
		}
	case structs.SessionKeysDelete:
		for _, obj := range kvs {
			e := obj.(*structs.DirEntry)
			if err := s.kvsDeleteTxn(tx, idx, e.Key); err != nil {
				return fmt.Errorf("failed kvs delete: %s", err)
			}

			// Apply the lock delay if present.
			if delay > 0 {
				s.lockDelay.SetExpiration(e.Key, now, delay)
			}
		}
	default:
		return fmt.Errorf("unknown session behavior %#v", session.Behavior)
	}

	// Delete any check mappings.
	mappings, err := tx.Get("session_checks", "session", sessionID)
	if err != nil {
		return fmt.Errorf("failed session checks lookup: %s", err)
	}
	{
		var objs []interface{}
		for mapping := mappings.Next(); mapping != nil; mapping = mappings.Next() {
			objs = append(objs, mapping)
		}

		// Do the delete in a separate loop so we don't trash the iterator.
		for _, obj := range objs {
			if err := tx.Delete("session_checks", obj); err != nil {
				return fmt.Errorf("failed deleting session check: %s", err)
			}
		}
	}

	// Delete any prepared queries.
	queries, err := tx.Get("prepared-queries", "session", sessionID)
	if err != nil {
		return fmt.Errorf("failed prepared query lookup: %s", err)
	}
	{
		var ids []string
		for wrapped := queries.Next(); wrapped != nil; wrapped = queries.Next() {
			ids = append(ids, toPreparedQuery(wrapped).ID)
		}

		// Do the delete in a separate loop so we don't trash the iterator.
		for _, id := range ids {
			if err := s.preparedQueryDeleteTxn(tx, idx, watches, id); err != nil {
				return fmt.Errorf("failed prepared query delete: %s", err)
			}
		}
	}

	watches.Arm("sessions")
	return nil
}

// ACLSet is used to insert an ACL rule into the state store.
func (s *StateStore) ACLSet(idx uint64, acl *structs.ACL) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// Call set on the ACL
	if err := s.aclSetTxn(tx, idx, acl); err != nil {
		return err
	}

	tx.Commit()
	return nil
}

// aclSetTxn is the inner method used to insert an ACL rule with the
// proper indexes into the state store.
func (s *StateStore) aclSetTxn(tx *memdb.Txn, idx uint64, acl *structs.ACL) error {
	// Check that the ID is set
	if acl.ID == "" {
		return ErrMissingACLID
	}

	// Check for an existing ACL
	existing, err := tx.First("acls", "id", acl.ID)
	if err != nil {
		return fmt.Errorf("failed acl lookup: %s", err)
	}

	// Set the indexes
	if existing != nil {
		acl.CreateIndex = existing.(*structs.ACL).CreateIndex
		acl.ModifyIndex = idx
	} else {
		acl.CreateIndex = idx
		acl.ModifyIndex = idx
	}

	// Insert the ACL
	if err := tx.Insert("acls", acl); err != nil {
		return fmt.Errorf("failed inserting acl: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"acls", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	tx.Defer(func() { s.tableWatches["acls"].Notify() })
	return nil
}

// ACLGet is used to look up an existing ACL by ID.
func (s *StateStore) ACLGet(aclID string) (uint64, *structs.ACL, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("ACLGet")...)

	// Query for the existing ACL
	acl, err := tx.First("acls", "id", aclID)
	if err != nil {
		return 0, nil, fmt.Errorf("failed acl lookup: %s", err)
	}
	if acl != nil {
		return idx, acl.(*structs.ACL), nil
	}
	return idx, nil, nil
}

// ACLList is used to list out all of the ACLs in the state store.
func (s *StateStore) ACLList() (uint64, structs.ACLs, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("ACLList")...)

	// Return the ACLs.
	acls, err := s.aclListTxn(tx)
	if err != nil {
		return 0, nil, fmt.Errorf("failed acl lookup: %s", err)
	}
	return idx, acls, nil
}

// aclListTxn is used to list out all of the ACLs in the state store. This is a
// function vs. a method so it can be called from the snapshotter.
func (s *StateStore) aclListTxn(tx *memdb.Txn) (structs.ACLs, error) {
	// Query all of the ACLs in the state store
	acls, err := tx.Get("acls", "id")
	if err != nil {
		return nil, fmt.Errorf("failed acl lookup: %s", err)
	}

	// Go over all of the ACLs and build the response
	var result structs.ACLs
	for acl := acls.Next(); acl != nil; acl = acls.Next() {
		a := acl.(*structs.ACL)
		result = append(result, a)
	}
	return result, nil
}

// ACLDelete is used to remove an existing ACL from the state store. If
// the ACL does not exist this is a no-op and no error is returned.
func (s *StateStore) ACLDelete(idx uint64, aclID string) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// Call the ACL delete
	if err := s.aclDeleteTxn(tx, idx, aclID); err != nil {
		return err
	}

	tx.Commit()
	return nil
}

// aclDeleteTxn is used to delete an ACL from the state store within
// an existing transaction.
func (s *StateStore) aclDeleteTxn(tx *memdb.Txn, idx uint64, aclID string) error {
	// Look up the existing ACL
	acl, err := tx.First("acls", "id", aclID)
	if err != nil {
		return fmt.Errorf("failed acl lookup: %s", err)
	}
	if acl == nil {
		return nil
	}

	// Delete the ACL from the state store and update indexes
	if err := tx.Delete("acls", acl); err != nil {
		return fmt.Errorf("failed deleting acl: %s", err)
	}
	if err := tx.Insert("index", &IndexEntry{"acls", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	tx.Defer(func() { s.tableWatches["acls"].Notify() })
	return nil
}

// CoordinateGetRaw queries for the coordinate of the given node. This is an
// unusual state store method because it just returns the raw coordinate or
// nil, none of the Raft or node information is returned. This hits the 90%
// internal-to-Consul use case for this data, and this isn't exposed via an
// endpoint, so it doesn't matter that the Raft info isn't available.
func (s *StateStore) CoordinateGetRaw(node string) (*coordinate.Coordinate, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Pull the full coordinate entry.
	coord, err := tx.First("coordinates", "id", node)
	if err != nil {
		return nil, fmt.Errorf("failed coordinate lookup: %s", err)
	}

	// Pick out just the raw coordinate.
	if coord != nil {
		return coord.(*structs.Coordinate).Coord, nil
	}
	return nil, nil
}

// Coordinates queries for all nodes with coordinates.
func (s *StateStore) Coordinates() (uint64, structs.Coordinates, error) {
	tx := s.db.Txn(false)
	defer tx.Abort()

	// Get the table index.
	idx := maxIndexTxn(tx, s.getWatchTables("Coordinates")...)

	// Pull all the coordinates.
	coords, err := tx.Get("coordinates", "id")
	if err != nil {
		return 0, nil, fmt.Errorf("failed coordinate lookup: %s", err)
	}
	var results structs.Coordinates
	for coord := coords.Next(); coord != nil; coord = coords.Next() {
		results = append(results, coord.(*structs.Coordinate))
	}
	return idx, results, nil
}

// CoordinateBatchUpdate processes a batch of coordinate updates and applies
// them in a single transaction.
func (s *StateStore) CoordinateBatchUpdate(idx uint64, updates structs.Coordinates) error {
	tx := s.db.Txn(true)
	defer tx.Abort()

	// Upsert the coordinates.
	for _, update := range updates {
		// Since the cleanup of coordinates is tied to deletion of
		// nodes, we silently drop any updates for nodes that we don't
		// know about. This might be possible during normal operation
		// if we happen to get a coordinate update for a node that
		// hasn't been able to add itself to the catalog yet. Since we
		// don't carefully sequence this, and since it will fix itself
		// on the next coordinate update from that node, we don't return
		// an error or log anything.
		node, err := tx.First("nodes", "id", update.Node)
		if err != nil {
			return fmt.Errorf("failed node lookup: %s", err)
		}
		if node == nil {
			continue
		}

		if err := tx.Insert("coordinates", update); err != nil {
			return fmt.Errorf("failed inserting coordinate: %s", err)
		}
	}

	// Update the index.
	if err := tx.Insert("index", &IndexEntry{"coordinates", idx}); err != nil {
		return fmt.Errorf("failed updating index: %s", err)
	}

	tx.Defer(func() { s.tableWatches["coordinates"].Notify() })
	tx.Commit()
	return nil
}
