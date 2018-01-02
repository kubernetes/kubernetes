package coordinator

import (
	"time"

	"github.com/influxdata/influxdb/influxql"
	"github.com/influxdata/influxdb/services/meta"
)

// MetaClient is an interface for accessing meta data.
type MetaClient interface {
	CreateContinuousQuery(database, name, query string) error
	CreateDatabase(name string) (*meta.DatabaseInfo, error)
	CreateDatabaseWithRetentionPolicy(name string, spec *meta.RetentionPolicySpec) (*meta.DatabaseInfo, error)
	CreateRetentionPolicy(database string, spec *meta.RetentionPolicySpec) (*meta.RetentionPolicyInfo, error)
	CreateSubscription(database, rp, name, mode string, destinations []string) error
	CreateUser(name, password string, admin bool) (*meta.UserInfo, error)
	Database(name string) *meta.DatabaseInfo
	Databases() []meta.DatabaseInfo
	DropShard(id uint64) error
	DropContinuousQuery(database, name string) error
	DropDatabase(name string) error
	DropRetentionPolicy(database, name string) error
	DropSubscription(database, rp, name string) error
	DropUser(name string) error
	RetentionPolicy(database, name string) (rpi *meta.RetentionPolicyInfo, err error)
	SetAdminPrivilege(username string, admin bool) error
	SetDefaultRetentionPolicy(database, name string) error
	SetPrivilege(username, database string, p influxql.Privilege) error
	ShardsByTimeRange(sources influxql.Sources, tmin, tmax time.Time) (a []meta.ShardInfo, err error)
	UpdateRetentionPolicy(database, name string, rpu *meta.RetentionPolicyUpdate) error
	UpdateUser(name, password string) error
	UserPrivilege(username, database string) (*influxql.Privilege, error)
	UserPrivileges(username string) (map[string]influxql.Privilege, error)
	Users() []meta.UserInfo
}
