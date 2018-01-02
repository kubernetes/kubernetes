package container

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/go-connections/nat"
	"github.com/hashicorp/go-memdb"
	"github.com/sirupsen/logrus"
)

const (
	memdbContainersTable = "containers"
	memdbNamesTable      = "names"

	memdbIDIndex          = "id"
	memdbContainerIDIndex = "containerid"
)

var (
	// ErrNameReserved is an error which is returned when a name is requested to be reserved that already is reserved
	ErrNameReserved = errors.New("name is reserved")
	// ErrNameNotReserved is an error which is returned when trying to find a name that is not reserved
	ErrNameNotReserved = errors.New("name is not reserved")
)

// Snapshot is a read only view for Containers. It holds all information necessary to serve container queries in a
// versioned ACID in-memory store.
type Snapshot struct {
	types.Container

	// additional info queries need to filter on
	// preserve nanosec resolution for queries
	CreatedAt    time.Time
	StartedAt    time.Time
	Name         string
	Pid          int
	ExitCode     int
	Running      bool
	Paused       bool
	Managed      bool
	ExposedPorts nat.PortSet
	PortBindings nat.PortSet
	Health       string
	HostConfig   struct {
		Isolation string
	}
}

// nameAssociation associates a container id with a name.
type nameAssociation struct {
	// name is the name to associate. Note that name is the primary key
	// ("id" in memdb).
	name        string
	containerID string
}

// ViewDB provides an in-memory transactional (ACID) container Store
type ViewDB interface {
	Snapshot() View
	Save(*Container) error
	Delete(*Container) error

	ReserveName(name, containerID string) error
	ReleaseName(name string) error
}

// View can be used by readers to avoid locking
type View interface {
	All() ([]Snapshot, error)
	Get(id string) (*Snapshot, error)

	GetID(name string) (string, error)
	GetAllNames() map[string][]string
}

var schema = &memdb.DBSchema{
	Tables: map[string]*memdb.TableSchema{
		memdbContainersTable: {
			Name: memdbContainersTable,
			Indexes: map[string]*memdb.IndexSchema{
				memdbIDIndex: {
					Name:    memdbIDIndex,
					Unique:  true,
					Indexer: &containerByIDIndexer{},
				},
			},
		},
		memdbNamesTable: {
			Name: memdbNamesTable,
			Indexes: map[string]*memdb.IndexSchema{
				// Used for names, because "id" is the primary key in memdb.
				memdbIDIndex: {
					Name:    memdbIDIndex,
					Unique:  true,
					Indexer: &namesByNameIndexer{},
				},
				memdbContainerIDIndex: {
					Name:    memdbContainerIDIndex,
					Indexer: &namesByContainerIDIndexer{},
				},
			},
		},
	},
}

type memDB struct {
	store *memdb.MemDB
}

// NoSuchContainerError indicates that the container wasn't found in the
// database.
type NoSuchContainerError struct {
	id string
}

// Error satisfies the error interface.
func (e NoSuchContainerError) Error() string {
	return "no such container " + e.id
}

// NewViewDB provides the default implementation, with the default schema
func NewViewDB() (ViewDB, error) {
	store, err := memdb.NewMemDB(schema)
	if err != nil {
		return nil, err
	}
	return &memDB{store: store}, nil
}

// Snapshot provides a consistent read-only View of the database
func (db *memDB) Snapshot() View {
	return &memdbView{
		txn: db.store.Txn(false),
	}
}

func (db *memDB) withTxn(cb func(*memdb.Txn) error) error {
	txn := db.store.Txn(true)
	err := cb(txn)
	if err != nil {
		txn.Abort()
		return err
	}
	txn.Commit()
	return nil
}

// Save atomically updates the in-memory store state for a Container.
// Only read only (deep) copies of containers may be passed in.
func (db *memDB) Save(c *Container) error {
	return db.withTxn(func(txn *memdb.Txn) error {
		return txn.Insert(memdbContainersTable, c)
	})
}

// Delete removes an item by ID
func (db *memDB) Delete(c *Container) error {
	return db.withTxn(func(txn *memdb.Txn) error {
		view := &memdbView{txn: txn}
		names := view.getNames(c.ID)

		for _, name := range names {
			txn.Delete(memdbNamesTable, nameAssociation{name: name})
		}

		// Ignore error - the container may not actually exist in the
		// db, but we still need to clean up associated names.
		txn.Delete(memdbContainersTable, NewBaseContainer(c.ID, c.Root))
		return nil
	})
}

// ReserveName registers a container ID to a name
// ReserveName is idempotent
// Attempting to reserve a container ID to a name that already exists results in an `ErrNameReserved`
// A name reservation is globally unique
func (db *memDB) ReserveName(name, containerID string) error {
	return db.withTxn(func(txn *memdb.Txn) error {
		s, err := txn.First(memdbNamesTable, memdbIDIndex, name)
		if err != nil {
			return err
		}
		if s != nil {
			if s.(nameAssociation).containerID != containerID {
				return ErrNameReserved
			}
			return nil
		}

		if err := txn.Insert(memdbNamesTable, nameAssociation{name: name, containerID: containerID}); err != nil {
			return err
		}
		return nil
	})
}

// ReleaseName releases the reserved name
// Once released, a name can be reserved again
func (db *memDB) ReleaseName(name string) error {
	return db.withTxn(func(txn *memdb.Txn) error {
		if err := txn.Delete(memdbNamesTable, nameAssociation{name: name}); err != nil {
			return err
		}
		return nil
	})
}

type memdbView struct {
	txn *memdb.Txn
}

// All returns a all items in this snapshot. Returned objects must never be modified.
func (v *memdbView) All() ([]Snapshot, error) {
	var all []Snapshot
	iter, err := v.txn.Get(memdbContainersTable, memdbIDIndex)
	if err != nil {
		return nil, err
	}
	for {
		item := iter.Next()
		if item == nil {
			break
		}
		snapshot := v.transform(item.(*Container))
		all = append(all, *snapshot)
	}
	return all, nil
}

// Get returns an item by id. Returned objects must never be modified.
func (v *memdbView) Get(id string) (*Snapshot, error) {
	s, err := v.txn.First(memdbContainersTable, memdbIDIndex, id)
	if err != nil {
		return nil, err
	}
	if s == nil {
		return nil, NoSuchContainerError{id: id}
	}
	return v.transform(s.(*Container)), nil
}

// getNames lists all the reserved names for the given container ID.
func (v *memdbView) getNames(containerID string) []string {
	iter, err := v.txn.Get(memdbNamesTable, memdbContainerIDIndex, containerID)
	if err != nil {
		return nil
	}

	var names []string
	for {
		item := iter.Next()
		if item == nil {
			break
		}
		names = append(names, item.(nameAssociation).name)
	}

	return names
}

// GetID returns the container ID that the passed in name is reserved to.
func (v *memdbView) GetID(name string) (string, error) {
	s, err := v.txn.First(memdbNamesTable, memdbIDIndex, name)
	if err != nil {
		return "", err
	}
	if s == nil {
		return "", ErrNameNotReserved
	}
	return s.(nameAssociation).containerID, nil
}

// GetAllNames returns all registered names.
func (v *memdbView) GetAllNames() map[string][]string {
	iter, err := v.txn.Get(memdbNamesTable, memdbContainerIDIndex)
	if err != nil {
		return nil
	}

	out := make(map[string][]string)
	for {
		item := iter.Next()
		if item == nil {
			break
		}
		assoc := item.(nameAssociation)
		out[assoc.containerID] = append(out[assoc.containerID], assoc.name)
	}

	return out
}

// transform maps a (deep) copied Container object to what queries need.
// A lock on the Container is not held because these are immutable deep copies.
func (v *memdbView) transform(container *Container) *Snapshot {
	snapshot := &Snapshot{
		Container: types.Container{
			ID:      container.ID,
			Names:   v.getNames(container.ID),
			ImageID: container.ImageID.String(),
			Ports:   []types.Port{},
			Mounts:  container.GetMountPoints(),
			State:   container.State.StateString(),
			Status:  container.State.String(),
			Created: container.Created.Unix(),
		},
		CreatedAt:    container.Created,
		StartedAt:    container.StartedAt,
		Name:         container.Name,
		Pid:          container.Pid,
		Managed:      container.Managed,
		ExposedPorts: make(nat.PortSet),
		PortBindings: make(nat.PortSet),
		Health:       container.HealthString(),
		Running:      container.Running,
		Paused:       container.Paused,
		ExitCode:     container.ExitCode(),
	}

	if snapshot.Names == nil {
		// Dead containers will often have no name, so make sure the response isn't null
		snapshot.Names = []string{}
	}

	if container.HostConfig != nil {
		snapshot.Container.HostConfig.NetworkMode = string(container.HostConfig.NetworkMode)
		snapshot.HostConfig.Isolation = string(container.HostConfig.Isolation)
		for binding := range container.HostConfig.PortBindings {
			snapshot.PortBindings[binding] = struct{}{}
		}
	}

	if container.Config != nil {
		snapshot.Image = container.Config.Image
		snapshot.Labels = container.Config.Labels
		for exposed := range container.Config.ExposedPorts {
			snapshot.ExposedPorts[exposed] = struct{}{}
		}
	}

	if len(container.Args) > 0 {
		args := []string{}
		for _, arg := range container.Args {
			if strings.Contains(arg, " ") {
				args = append(args, fmt.Sprintf("'%s'", arg))
			} else {
				args = append(args, arg)
			}
		}
		argsAsString := strings.Join(args, " ")
		snapshot.Command = fmt.Sprintf("%s %s", container.Path, argsAsString)
	} else {
		snapshot.Command = container.Path
	}

	snapshot.Ports = []types.Port{}
	networks := make(map[string]*network.EndpointSettings)
	if container.NetworkSettings != nil {
		for name, netw := range container.NetworkSettings.Networks {
			if netw == nil || netw.EndpointSettings == nil {
				continue
			}
			networks[name] = &network.EndpointSettings{
				EndpointID:          netw.EndpointID,
				Gateway:             netw.Gateway,
				IPAddress:           netw.IPAddress,
				IPPrefixLen:         netw.IPPrefixLen,
				IPv6Gateway:         netw.IPv6Gateway,
				GlobalIPv6Address:   netw.GlobalIPv6Address,
				GlobalIPv6PrefixLen: netw.GlobalIPv6PrefixLen,
				MacAddress:          netw.MacAddress,
				NetworkID:           netw.NetworkID,
			}
			if netw.IPAMConfig != nil {
				networks[name].IPAMConfig = &network.EndpointIPAMConfig{
					IPv4Address: netw.IPAMConfig.IPv4Address,
					IPv6Address: netw.IPAMConfig.IPv6Address,
				}
			}
		}
		for port, bindings := range container.NetworkSettings.Ports {
			p, err := nat.ParsePort(port.Port())
			if err != nil {
				logrus.Warnf("invalid port map %+v", err)
				continue
			}
			if len(bindings) == 0 {
				snapshot.Ports = append(snapshot.Ports, types.Port{
					PrivatePort: uint16(p),
					Type:        port.Proto(),
				})
				continue
			}
			for _, binding := range bindings {
				h, err := nat.ParsePort(binding.HostPort)
				if err != nil {
					logrus.Warnf("invalid host port map %+v", err)
					continue
				}
				snapshot.Ports = append(snapshot.Ports, types.Port{
					PrivatePort: uint16(p),
					PublicPort:  uint16(h),
					Type:        port.Proto(),
					IP:          binding.HostIP,
				})
			}
		}
	}
	snapshot.NetworkSettings = &types.SummaryNetworkSettings{Networks: networks}

	return snapshot
}

// containerByIDIndexer is used to extract the ID field from Container types.
// memdb.StringFieldIndex can not be used since ID is a field from an embedded struct.
type containerByIDIndexer struct{}

// FromObject implements the memdb.SingleIndexer interface for Container objects
func (e *containerByIDIndexer) FromObject(obj interface{}) (bool, []byte, error) {
	c, ok := obj.(*Container)
	if !ok {
		return false, nil, fmt.Errorf("%T is not a Container", obj)
	}
	// Add the null character as a terminator
	v := c.ID + "\x00"
	return true, []byte(v), nil
}

// FromArgs implements the memdb.Indexer interface
func (e *containerByIDIndexer) FromArgs(args ...interface{}) ([]byte, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("must provide only a single argument")
	}
	arg, ok := args[0].(string)
	if !ok {
		return nil, fmt.Errorf("argument must be a string: %#v", args[0])
	}
	// Add the null character as a terminator
	arg += "\x00"
	return []byte(arg), nil
}

// namesByNameIndexer is used to index container name associations by name.
type namesByNameIndexer struct{}

func (e *namesByNameIndexer) FromObject(obj interface{}) (bool, []byte, error) {
	n, ok := obj.(nameAssociation)
	if !ok {
		return false, nil, fmt.Errorf(`%T does not have type "nameAssociation"`, obj)
	}

	// Add the null character as a terminator
	return true, []byte(n.name + "\x00"), nil
}

func (e *namesByNameIndexer) FromArgs(args ...interface{}) ([]byte, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("must provide only a single argument")
	}
	arg, ok := args[0].(string)
	if !ok {
		return nil, fmt.Errorf("argument must be a string: %#v", args[0])
	}
	// Add the null character as a terminator
	arg += "\x00"
	return []byte(arg), nil
}

// namesByContainerIDIndexer is used to index container names by container ID.
type namesByContainerIDIndexer struct{}

func (e *namesByContainerIDIndexer) FromObject(obj interface{}) (bool, []byte, error) {
	n, ok := obj.(nameAssociation)
	if !ok {
		return false, nil, fmt.Errorf(`%T does not have type "nameAssocation"`, obj)
	}

	// Add the null character as a terminator
	return true, []byte(n.containerID + "\x00"), nil
}

func (e *namesByContainerIDIndexer) FromArgs(args ...interface{}) ([]byte, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("must provide only a single argument")
	}
	arg, ok := args[0].(string)
	if !ok {
		return nil, fmt.Errorf("argument must be a string: %#v", args[0])
	}
	// Add the null character as a terminator
	arg += "\x00"
	return []byte(arg), nil
}
