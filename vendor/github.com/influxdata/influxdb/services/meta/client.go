package meta

import (
	"bytes"
	crand "crypto/rand"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/influxdata/influxdb"
	"github.com/influxdata/influxdb/influxql"

	"golang.org/x/crypto/bcrypt"
)

const (
	// SaltBytes is the number of bytes used for salts
	SaltBytes = 32

	metaFile = "meta.db"
)

var (
	// ErrServiceUnavailable is returned when the meta service is unavailable.
	ErrServiceUnavailable = errors.New("meta service unavailable")

	// ErrService is returned when the meta service returns an error.
	ErrService = errors.New("meta service error")
)

// Client is used to execute commands on and read data from
// a meta service cluster.
type Client struct {
	logger *log.Logger

	mu        sync.RWMutex
	closing   chan struct{}
	changed   chan struct{}
	cacheData *Data

	// Authentication cache.
	authCache map[string]authUser

	path string

	retentionAutoCreate bool
}

type authUser struct {
	bhash string
	salt  []byte
	hash  []byte
}

// NewClient returns a new *Client.
func NewClient(config *Config) *Client {
	return &Client{
		cacheData: &Data{
			ClusterID: uint64(rand.Int63()),
			Index:     1,
		},
		closing:             make(chan struct{}),
		changed:             make(chan struct{}),
		logger:              log.New(ioutil.Discard, "[metaclient] ", log.LstdFlags),
		authCache:           make(map[string]authUser, 0),
		path:                config.Dir,
		retentionAutoCreate: config.RetentionAutoCreate,
	}
}

// Open a connection to a meta service cluster.
func (c *Client) Open() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Try to load from disk
	if err := c.Load(); err != nil {
		return err
	}

	// If this is a brand new instance, persist to disk immediatly.
	if c.cacheData.Index == 1 {
		if err := snapshot(c.path, c.cacheData); err != nil {
			return err
		}
	}

	return nil
}

// Close the meta service cluster connection.
func (c *Client) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if t, ok := http.DefaultTransport.(*http.Transport); ok {
		t.CloseIdleConnections()
	}

	select {
	case <-c.closing:
		return nil
	default:
		close(c.closing)
	}

	return nil
}

// AcquireLease attempts to acquire the specified lease.
// TODO corylanou remove this for single node
func (c *Client) AcquireLease(name string) (*Lease, error) {
	l := Lease{
		Name:       name,
		Expiration: time.Now().Add(DefaultLeaseDuration),
	}
	return &l, nil
}

func (c *Client) data() *Data {
	c.mu.RLock()
	defer c.mu.RUnlock()
	data := c.cacheData.Clone()
	return data
}

// ClusterID returns the ID of the cluster it's connected to.
func (c *Client) ClusterID() uint64 {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.cacheData.ClusterID
}

// Database returns info for the requested database.
func (c *Client) Database(name string) *DatabaseInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	for _, d := range c.cacheData.Databases {
		if d.Name == name {
			return &d
		}
	}

	return nil
}

// Databases returns a list of all database infos.
func (c *Client) Databases() []DatabaseInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	dbs := c.cacheData.Databases
	if dbs == nil {
		return []DatabaseInfo{}
	}
	return dbs
}

// CreateDatabase creates a database or returns it if it already exists
func (c *Client) CreateDatabase(name string) (*DatabaseInfo, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if db := data.Database(name); db != nil {
		return db, nil
	}

	if err := data.CreateDatabase(name); err != nil {
		return nil, err
	}

	// create default retention policy
	if c.retentionAutoCreate {
		rpi := DefaultRetentionPolicyInfo()
		if err := data.CreateRetentionPolicy(name, rpi); err != nil {
			return nil, err
		}
		if err := data.SetDefaultRetentionPolicy(name, rpi.Name); err != nil {
			return nil, err
		}
	}

	db := data.Database(name)

	if err := c.commit(data); err != nil {
		return nil, err
	}

	return db, nil
}

// CreateDatabaseWithRetentionPolicy creates a database with the specified retention policy.
func (c *Client) CreateDatabaseWithRetentionPolicy(name string, spec *RetentionPolicySpec) (*DatabaseInfo, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if spec.Duration != nil && *spec.Duration < MinRetentionPolicyDuration && *spec.Duration != 0 {
		return nil, ErrRetentionPolicyDurationTooLow
	}

	db := data.Database(name)
	if db == nil {
		if err := data.CreateDatabase(name); err != nil {
			return nil, err
		}
		db = data.Database(name)
	}

	rpi := spec.NewRetentionPolicyInfo()
	if rp := db.RetentionPolicy(rpi.Name); rp == nil {
		if err := data.CreateRetentionPolicy(name, rpi); err != nil {
			return nil, err
		}
	} else if !spec.Matches(rp) {
		// Verify that the retention policy with this name matches
		// the one already created.
		return nil, ErrRetentionPolicyConflict
	}

	// If no default retention policy has been set, set it to the retention
	// policy we just created. If the default is different from what we are
	// trying to create, record it as a conflict and abandon with an error.
	if db.DefaultRetentionPolicy == "" {
		if err := data.SetDefaultRetentionPolicy(name, rpi.Name); err != nil {
			return nil, err
		}
	} else if rpi.Name != db.DefaultRetentionPolicy {
		return nil, ErrRetentionPolicyConflict
	}

	// Refresh the database info.
	db = data.Database(name)

	if err := c.commit(data); err != nil {
		return nil, err
	}

	return db, nil
}

// DropDatabase deletes a database.
func (c *Client) DropDatabase(name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if err := data.DropDatabase(name); err != nil {
		return err
	}

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

// CreateRetentionPolicy creates a retention policy on the specified database.
func (c *Client) CreateRetentionPolicy(database string, spec *RetentionPolicySpec) (*RetentionPolicyInfo, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if spec.Duration != nil && *spec.Duration < MinRetentionPolicyDuration && *spec.Duration != 0 {
		return nil, ErrRetentionPolicyDurationTooLow
	}

	rp := spec.NewRetentionPolicyInfo()
	if err := data.CreateRetentionPolicy(database, rp); err != nil {
		return nil, err
	}

	if err := c.commit(data); err != nil {
		return nil, err
	}

	return rp, nil
}

// RetentionPolicy returns the requested retention policy info.
func (c *Client) RetentionPolicy(database, name string) (rpi *RetentionPolicyInfo, err error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	db := c.cacheData.Database(database)
	if db == nil {
		return nil, influxdb.ErrDatabaseNotFound(database)
	}

	return db.RetentionPolicy(name), nil
}

// DropRetentionPolicy drops a retention policy from a database.
func (c *Client) DropRetentionPolicy(database, name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if err := data.DropRetentionPolicy(database, name); err != nil {
		return err
	}

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

// SetDefaultRetentionPolicy sets a database's default retention policy.
func (c *Client) SetDefaultRetentionPolicy(database, name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if err := data.SetDefaultRetentionPolicy(database, name); err != nil {
		return err
	}

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

// UpdateRetentionPolicy updates a retention policy.
func (c *Client) UpdateRetentionPolicy(database, name string, rpu *RetentionPolicyUpdate) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if err := data.UpdateRetentionPolicy(database, name, rpu); err != nil {
		return err
	}

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

func (c *Client) Users() []UserInfo {
	c.mu.RLock()
	defer c.mu.RUnlock()

	users := c.cacheData.Users

	if users == nil {
		return []UserInfo{}
	}
	return users
}

func (c *Client) User(name string) (*UserInfo, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	for _, u := range c.cacheData.Users {
		if u.Name == name {
			return &u, nil
		}
	}

	return nil, ErrUserNotFound
}

// bcryptCost is the cost associated with generating password with bcrypt.
// This setting is lowered during testing to improve test suite performance.
var bcryptCost = bcrypt.DefaultCost

// hashWithSalt returns a salted hash of password using salt
func (c *Client) hashWithSalt(salt []byte, password string) []byte {
	hasher := sha256.New()
	hasher.Write(salt)
	hasher.Write([]byte(password))
	return hasher.Sum(nil)
}

// saltedHash returns a salt and salted hash of password
func (c *Client) saltedHash(password string) (salt, hash []byte, err error) {
	salt = make([]byte, SaltBytes)
	if _, err := io.ReadFull(crand.Reader, salt); err != nil {
		return nil, nil, err
	}

	return salt, c.hashWithSalt(salt, password), nil
}

func (c *Client) CreateUser(name, password string, admin bool) (*UserInfo, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	// See if the user already exists.
	if u := data.User(name); u != nil {
		if err := bcrypt.CompareHashAndPassword([]byte(u.Hash), []byte(password)); err != nil || u.Admin != admin {
			return nil, ErrUserExists
		}
		return u, nil
	}

	// Hash the password before serializing it.
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcryptCost)
	if err != nil {
		return nil, err
	}

	if err := data.CreateUser(name, string(hash), admin); err != nil {
		return nil, err
	}

	u := data.User(name)

	if err := c.commit(data); err != nil {
		return nil, err
	}

	return u, nil
}

func (c *Client) UpdateUser(name, password string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	// Hash the password before serializing it.
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcryptCost)
	if err != nil {
		return err
	}

	if err := data.UpdateUser(name, string(hash)); err != nil {
		return nil
	}

	delete(c.authCache, name)

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

func (c *Client) DropUser(name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if err := data.DropUser(name); err != nil {
		return err
	}

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

func (c *Client) SetPrivilege(username, database string, p influxql.Privilege) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if err := data.SetPrivilege(username, database, p); err != nil {
		return err
	}

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

func (c *Client) SetAdminPrivilege(username string, admin bool) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if err := data.SetAdminPrivilege(username, admin); err != nil {
		return err
	}

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

func (c *Client) UserPrivileges(username string) (map[string]influxql.Privilege, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	p, err := c.cacheData.UserPrivileges(username)
	if err != nil {
		return nil, err
	}
	return p, nil
}

func (c *Client) UserPrivilege(username, database string) (*influxql.Privilege, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	p, err := c.cacheData.UserPrivilege(username, database)
	if err != nil {
		return nil, err
	}
	return p, nil
}

func (c *Client) AdminUserExists() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	for _, u := range c.cacheData.Users {
		if u.Admin {
			return true
		}
	}
	return false
}

func (c *Client) Authenticate(username, password string) (*UserInfo, error) {
	// Find user.
	c.mu.RLock()
	userInfo := c.cacheData.User(username)
	c.mu.RUnlock()
	if userInfo == nil {
		return nil, ErrUserNotFound
	}

	// Check the local auth cache first.
	c.mu.RLock()
	au, ok := c.authCache[username]
	c.mu.RUnlock()
	if ok {
		// verify the password using the cached salt and hash
		if bytes.Equal(c.hashWithSalt(au.salt, password), au.hash) {
			return userInfo, nil
		}

		// fall through to requiring a full bcrypt hash for invalid passwords
	}

	// Compare password with user hash.
	if err := bcrypt.CompareHashAndPassword([]byte(userInfo.Hash), []byte(password)); err != nil {
		return nil, ErrAuthenticate
	}

	// generate a salt and hash of the password for the cache
	salt, hashed, err := c.saltedHash(password)
	if err != nil {
		return nil, err
	}
	c.mu.Lock()
	c.authCache[username] = authUser{salt: salt, hash: hashed, bhash: userInfo.Hash}
	c.mu.Unlock()
	return userInfo, nil
}

func (c *Client) UserCount() int {
	c.mu.RLock()
	defer c.mu.RUnlock()

	return len(c.cacheData.Users)
}

// ShardIDs returns a list of all shard ids.
func (c *Client) ShardIDs() []uint64 {
	c.mu.RLock()

	var a []uint64
	for _, dbi := range c.cacheData.Databases {
		for _, rpi := range dbi.RetentionPolicies {
			for _, sgi := range rpi.ShardGroups {
				for _, si := range sgi.Shards {
					a = append(a, si.ID)
				}
			}
		}
	}
	c.mu.RUnlock()
	sort.Sort(uint64Slice(a))
	return a
}

// ShardGroupsByTimeRange returns a list of all shard groups on a database and policy that may contain data
// for the specified time range. Shard groups are sorted by start time.
func (c *Client) ShardGroupsByTimeRange(database, policy string, min, max time.Time) (a []ShardGroupInfo, err error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	// Find retention policy.
	rpi, err := c.cacheData.RetentionPolicy(database, policy)
	if err != nil {
		return nil, err
	} else if rpi == nil {
		return nil, influxdb.ErrRetentionPolicyNotFound(policy)
	}
	groups := make([]ShardGroupInfo, 0, len(rpi.ShardGroups))
	for _, g := range rpi.ShardGroups {
		if g.Deleted() || !g.Overlaps(min, max) {
			continue
		}
		groups = append(groups, g)
	}
	return groups, nil
}

// ShardsByTimeRange returns a slice of shards that may contain data in the time range.
func (c *Client) ShardsByTimeRange(sources influxql.Sources, tmin, tmax time.Time) (a []ShardInfo, err error) {
	m := make(map[*ShardInfo]struct{})
	for _, src := range sources {
		mm, ok := src.(*influxql.Measurement)
		if !ok {
			return nil, fmt.Errorf("invalid source type: %#v", src)
		}

		groups, err := c.ShardGroupsByTimeRange(mm.Database, mm.RetentionPolicy, tmin, tmax)
		if err != nil {
			return nil, err
		}
		for _, g := range groups {
			for i := range g.Shards {
				m[&g.Shards[i]] = struct{}{}
			}
		}
	}

	a = make([]ShardInfo, 0, len(m))
	for sh := range m {
		a = append(a, *sh)
	}

	return a, nil
}

// DropShard deletes a shard by ID.
func (c *Client) DropShard(id uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()
	data.DropShard(id)
	return c.commit(data)
}

// CreateShardGroup creates a shard group on a database and policy for a given timestamp.
func (c *Client) CreateShardGroup(database, policy string, timestamp time.Time) (*ShardGroupInfo, error) {
	// Check under a read-lock
	c.mu.RLock()
	if sg, _ := c.cacheData.ShardGroupByTimestamp(database, policy, timestamp); sg != nil {
		c.mu.RUnlock()
		return sg, nil
	}
	c.mu.RUnlock()

	c.mu.Lock()
	defer c.mu.Unlock()

	// Check again under the write lock
	data := c.cacheData.Clone()
	if sg, _ := data.ShardGroupByTimestamp(database, policy, timestamp); sg != nil {
		return sg, nil
	}

	sgi, err := createShardGroup(data, database, policy, timestamp)
	if err != nil {
		return nil, err
	}

	if err := c.commit(data); err != nil {
		return nil, err
	}

	return sgi, nil
}

func createShardGroup(data *Data, database, policy string, timestamp time.Time) (*ShardGroupInfo, error) {
	// It is the responsibility of the caller to check if it exists before calling this method.
	if sg, _ := data.ShardGroupByTimestamp(database, policy, timestamp); sg != nil {
		return nil, ErrShardGroupExists
	}

	if err := data.CreateShardGroup(database, policy, timestamp); err != nil {
		return nil, err
	}

	rpi, err := data.RetentionPolicy(database, policy)
	if err != nil {
		return nil, err
	} else if rpi == nil {
		return nil, errors.New("retention policy deleted after shard group created")
	}

	sgi := rpi.ShardGroupByTimestamp(timestamp)
	return sgi, nil
}

// DeleteShardGroup removes a shard group from a database and retention policy by id.
func (c *Client) DeleteShardGroup(database, policy string, id uint64) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if err := data.DeleteShardGroup(database, policy, id); err != nil {
		return err
	}

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

// PrecreateShardGroups creates shard groups whose endtime is before the 'to' time passed in, but
// is yet to expire before 'from'. This is to avoid the need for these shards to be created when data
// for the corresponding time range arrives. Shard creation involves Raft consensus, and precreation
// avoids taking the hit at write-time.
func (c *Client) PrecreateShardGroups(from, to time.Time) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	data := c.cacheData.Clone()
	var changed bool

	for _, di := range data.Databases {
		for _, rp := range di.RetentionPolicies {
			if len(rp.ShardGroups) == 0 {
				// No data was ever written to this group, or all groups have been deleted.
				continue
			}
			g := rp.ShardGroups[len(rp.ShardGroups)-1] // Get the last group in time.
			if !g.Deleted() && g.EndTime.Before(to) && g.EndTime.After(from) {
				// Group is not deleted, will end before the future time, but is still yet to expire.
				// This last check is important, so the system doesn't create shards groups wholly
				// in the past.

				// Create successive shard group.
				nextShardGroupTime := g.EndTime.Add(1 * time.Nanosecond)
				// if it already exists, continue
				if sg, _ := data.ShardGroupByTimestamp(di.Name, rp.Name, nextShardGroupTime); sg != nil {
					c.logger.Printf("shard group %d exists for database %s, retention policy %s", sg.ID, di.Name, rp.Name)
					continue
				}
				newGroup, err := createShardGroup(data, di.Name, rp.Name, nextShardGroupTime)
				if err != nil {
					c.logger.Printf("failed to precreate successive shard group for group %d: %s", g.ID, err.Error())
					continue
				}
				changed = true
				c.logger.Printf("new shard group %d successfully precreated for database %s, retention policy %s", newGroup.ID, di.Name, rp.Name)
			}
		}
	}

	if changed {
		if err := c.commit(data); err != nil {
			return err
		}
	}

	return nil
}

// ShardOwner returns the owning shard group info for a specific shard.
func (c *Client) ShardOwner(shardID uint64) (database, policy string, sgi *ShardGroupInfo) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	for _, dbi := range c.cacheData.Databases {
		for _, rpi := range dbi.RetentionPolicies {
			for _, g := range rpi.ShardGroups {
				if g.Deleted() {
					continue
				}

				for _, sh := range g.Shards {
					if sh.ID == shardID {
						database = dbi.Name
						policy = rpi.Name
						sgi = &g
						return
					}
				}
			}
		}
	}
	return
}

func (c *Client) CreateContinuousQuery(database, name, query string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if err := data.CreateContinuousQuery(database, name, query); err != nil {
		return err
	}

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

func (c *Client) DropContinuousQuery(database, name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if err := data.DropContinuousQuery(database, name); err != nil {
		return nil
	}

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

func (c *Client) CreateSubscription(database, rp, name, mode string, destinations []string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if err := data.CreateSubscription(database, rp, name, mode, destinations); err != nil {
		return err
	}

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

func (c *Client) DropSubscription(database, rp, name string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	data := c.cacheData.Clone()

	if err := data.DropSubscription(database, rp, name); err != nil {
		return err
	}

	if err := c.commit(data); err != nil {
		return err
	}

	return nil
}

func (c *Client) SetData(data *Data) error {
	c.mu.Lock()

	// reset the index so the commit will fire a change event
	c.cacheData.Index = 0

	// increment the index to force the changed channel to fire
	d := data.Clone()
	d.Index++

	if err := c.commit(d); err != nil {
		return err
	}

	c.mu.Unlock()

	return nil
}

func (c *Client) Data() Data {
	c.mu.RLock()
	defer c.mu.RUnlock()
	d := c.cacheData.Clone()
	return *d
}

// WaitForDataChanged will return a channel that will get closed when
// the metastore data has changed
func (c *Client) WaitForDataChanged() chan struct{} {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.changed
}

// commit assumes it is under a full lock
func (c *Client) commit(data *Data) error {
	data.Index++

	// try to write to disk before updating in memory
	if err := snapshot(c.path, data); err != nil {
		return err
	}

	// update in memory
	c.cacheData = data

	// close channels to signal changes
	close(c.changed)
	c.changed = make(chan struct{})

	return nil
}

func (c *Client) MarshalBinary() ([]byte, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.cacheData.MarshalBinary()
}

// SetLogOutput sets the writer to which all logs are written. It must not be
// called after Open is called.
func (c *Client) SetLogOutput(w io.Writer) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.logger = log.New(w, "[metaclient] ", log.LstdFlags)
}

func (c *Client) updateAuthCache() {
	// copy cached user info for still-present users
	newCache := make(map[string]authUser, len(c.authCache))

	for _, userInfo := range c.cacheData.Users {
		if cached, ok := c.authCache[userInfo.Name]; ok {
			if cached.bhash == userInfo.Hash {
				newCache[userInfo.Name] = cached
			}
		}
	}

	c.authCache = newCache
}

// snapshot will save the current meta data to disk
func snapshot(path string, data *Data) error {
	file := filepath.Join(path, metaFile)
	tmpFile := file + "tmp"

	f, err := os.Create(tmpFile)
	if err != nil {
		return err
	}
	defer f.Close()

	var d []byte
	if b, err := data.MarshalBinary(); err != nil {
		return err
	} else {
		d = b
	}

	if _, err := f.Write(d); err != nil {
		return err
	}

	if err = f.Sync(); err != nil {
		return err
	}

	//close file handle before renaming to support Windows
	if err = f.Close(); err != nil {
		return err
	}

	return renameFile(tmpFile, file)
}

// Load will save the current meta data from disk
func (c *Client) Load() error {
	file := filepath.Join(c.path, metaFile)

	f, err := os.Open(file)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	defer f.Close()

	data, err := ioutil.ReadAll(f)
	if err != nil {
		return err
	}

	if err := c.cacheData.UnmarshalBinary(data); err != nil {
		return err
	}
	return nil
}

type uint64Slice []uint64

func (a uint64Slice) Len() int           { return len(a) }
func (a uint64Slice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a uint64Slice) Less(i, j int) bool { return a[i] < a[j] }
