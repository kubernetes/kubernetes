package internal

import (
	"time"

	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/services/meta"
)

// MetaClientMock is a mockable implementation of meta.MetaClient.
type MetaClientMock struct {
	CloseFn                             func() error
	CreateContinuousQueryFn             func(database, name, query string) error
	CreateDatabaseFn                    func(name string) (*meta.DatabaseInfo, error)
	CreateDatabaseWithRetentionPolicyFn func(name string, spec *meta.RetentionPolicySpec) (*meta.DatabaseInfo, error)
	CreateRetentionPolicyFn             func(database string, spec *meta.RetentionPolicySpec) (*meta.RetentionPolicyInfo, error)
	CreateShardGroupFn                  func(database, policy string, timestamp time.Time) (*meta.ShardGroupInfo, error)
	CreateSubscriptionFn                func(database, rp, name, mode string, destinations []string) error
	CreateUserFn                        func(name, password string, admin bool) (*meta.UserInfo, error)

	DatabaseFn  func(name string) *meta.DatabaseInfo
	DatabasesFn func() []meta.DatabaseInfo

	DataFn                func() meta.Data
	DeleteShardGroupFn    func(database string, policy string, id uint64) error
	DropContinuousQueryFn func(database, name string) error
	DropDatabaseFn        func(name string) error
	DropRetentionPolicyFn func(database, name string) error
	DropSubscriptionFn    func(database, rp, name string) error
	DropShardFn           func(id uint64) error
	DropUserFn            func(name string) error

	OpenFn func() error

	RetentionPolicyFn func(database, name string) (rpi *meta.RetentionPolicyInfo, err error)

	SetAdminPrivilegeFn         func(username string, admin bool) error
	SetDataFn                   func(*meta.Data) error
	SetDefaultRetentionPolicyFn func(database, name string) error
	SetPrivilegeFn              func(username, database string, p influxql.Privilege) error
	ShardsByTimeRangeFn         func(sources influxql.Sources, tmin, tmax time.Time) (a []meta.ShardInfo, err error)
	ShardOwnerFn                func(shardID uint64) (database, policy string, sgi *meta.ShardGroupInfo)
	UpdateRetentionPolicyFn     func(database, name string, rpu *meta.RetentionPolicyUpdate) error
	UpdateUserFn                func(name, password string) error
	UserPrivilegeFn             func(username, database string) (*influxql.Privilege, error)
	UserPrivilegesFn            func(username string) (map[string]influxql.Privilege, error)
	UsersFn                     func() []meta.UserInfo
}

func (c *MetaClientMock) Close() error {
	return c.CloseFn()
}

func (c *MetaClientMock) CreateContinuousQuery(database, name, query string) error {
	return c.CreateContinuousQueryFn(database, name, query)
}

func (c *MetaClientMock) CreateDatabase(name string) (*meta.DatabaseInfo, error) {
	return c.CreateDatabaseFn(name)
}

func (c *MetaClientMock) CreateDatabaseWithRetentionPolicy(name string, spec *meta.RetentionPolicySpec) (*meta.DatabaseInfo, error) {
	return c.CreateDatabaseWithRetentionPolicyFn(name, spec)
}

func (c *MetaClientMock) CreateRetentionPolicy(database string, spec *meta.RetentionPolicySpec) (*meta.RetentionPolicyInfo, error) {
	return c.CreateRetentionPolicyFn(database, spec)
}

func (c *MetaClientMock) CreateShardGroup(database, policy string, timestamp time.Time) (*meta.ShardGroupInfo, error) {
	return c.CreateShardGroupFn(database, policy, timestamp)
}

func (c *MetaClientMock) CreateSubscription(database, rp, name, mode string, destinations []string) error {
	return c.CreateSubscriptionFn(database, rp, name, mode, destinations)
}

func (c *MetaClientMock) CreateUser(name, password string, admin bool) (*meta.UserInfo, error) {
	return c.CreateUserFn(name, password, admin)
}

func (c *MetaClientMock) Database(name string) *meta.DatabaseInfo {
	return c.DatabaseFn(name)
}

func (c *MetaClientMock) Databases() []meta.DatabaseInfo {
	return c.DatabasesFn()
}

func (c *MetaClientMock) DeleteShardGroup(database string, policy string, id uint64) error {
	return c.DeleteShardGroup(database, policy, id)
}

func (c *MetaClientMock) DropContinuousQuery(database, name string) error {
	return c.DropContinuousQueryFn(database, name)
}

func (c *MetaClientMock) DropDatabase(name string) error {
	return c.DropDatabaseFn(name)
}

func (c *MetaClientMock) DropRetentionPolicy(database, name string) error {
	return c.DropRetentionPolicyFn(database, name)
}

func (c *MetaClientMock) DropShard(id uint64) error {
	return c.DropShardFn(id)
}

func (c *MetaClientMock) DropSubscription(database, rp, name string) error {
	return c.DropSubscriptionFn(database, rp, name)
}

func (c *MetaClientMock) DropUser(name string) error {
	return c.DropUserFn(name)
}

func (c *MetaClientMock) RetentionPolicy(database, name string) (rpi *meta.RetentionPolicyInfo, err error) {
	return c.RetentionPolicyFn(database, name)
}

func (c *MetaClientMock) SetAdminPrivilege(username string, admin bool) error {
	return c.SetAdminPrivilegeFn(username, admin)
}

func (c *MetaClientMock) SetDefaultRetentionPolicy(database, name string) error {
	return c.SetDefaultRetentionPolicyFn(database, name)
}

func (c *MetaClientMock) SetPrivilege(username, database string, p influxql.Privilege) error {
	return c.SetPrivilegeFn(username, database, p)
}

func (c *MetaClientMock) ShardsByTimeRange(sources influxql.Sources, tmin, tmax time.Time) (a []meta.ShardInfo, err error) {
	return c.ShardsByTimeRangeFn(sources, tmin, tmax)
}

func (c *MetaClientMock) ShardOwner(shardID uint64) (database, policy string, sgi *meta.ShardGroupInfo) {
	return c.ShardOwnerFn(shardID)
}

func (c *MetaClientMock) UpdateRetentionPolicy(database, name string, rpu *meta.RetentionPolicyUpdate) error {
	return c.UpdateRetentionPolicyFn(database, name, rpu)
}

func (c *MetaClientMock) UpdateUser(name, password string) error {
	return c.UpdateUserFn(name, password)
}

func (c *MetaClientMock) UserPrivilege(username, database string) (*influxql.Privilege, error) {
	return c.UserPrivilegeFn(username, database)
}

func (c *MetaClientMock) UserPrivileges(username string) (map[string]influxql.Privilege, error) {
	return c.UserPrivilegesFn(username)
}

func (c *MetaClientMock) Users() []meta.UserInfo { return c.UsersFn() }

func (c *MetaClientMock) Open() error                { return c.OpenFn() }
func (c *MetaClientMock) Data() meta.Data            { return c.DataFn() }
func (c *MetaClientMock) SetData(d *meta.Data) error { return c.SetDataFn(d) }
