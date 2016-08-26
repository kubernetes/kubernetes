package influxql

import (
	"bytes"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"
)

// DataType represents the primitive data types available in InfluxQL.
type DataType int

const (
	// Unknown primitive data type.
	Unknown DataType = 0
	// Float means the data type is a float
	Float = 1
	// Integer means the data type is a integer
	Integer = 2
	// Boolean means the data type is a boolean.
	Boolean = 3
	// String means the data type is a string of text.
	String = 4
	// Time means the data type is a time.
	Time = 5
	// Duration means the data type is a duration of time.
	Duration = 6
)

// InspectDataType returns the data type of a given value.
func InspectDataType(v interface{}) DataType {
	switch v.(type) {
	case float64:
		return Float
	case int64, int32, int:
		return Integer
	case bool:
		return Boolean
	case string:
		return String
	case time.Time:
		return Time
	case time.Duration:
		return Duration
	default:
		return Unknown
	}
}

func (d DataType) String() string {
	switch d {
	case Float:
		return "float"
	case Integer:
		return "integer"
	case Boolean:
		return "boolean"
	case String:
		return "string"
	case Time:
		return "time"
	case Duration:
		return "duration"
	}
	return "unknown"
}

// Node represents a node in the InfluxDB abstract syntax tree.
type Node interface {
	node()
	String() string
}

func (*Query) node()     {}
func (Statements) node() {}

func (*AlterRetentionPolicyStatement) node()  {}
func (*CreateContinuousQueryStatement) node() {}
func (*CreateDatabaseStatement) node()        {}
func (*CreateRetentionPolicyStatement) node() {}
func (*CreateUserStatement) node()            {}
func (*Distinct) node()                       {}
func (*DeleteStatement) node()                {}
func (*DropContinuousQueryStatement) node()   {}
func (*DropDatabaseStatement) node()          {}
func (*DropMeasurementStatement) node()       {}
func (*DropRetentionPolicyStatement) node()   {}
func (*DropSeriesStatement) node()            {}
func (*DropUserStatement) node()              {}
func (*GrantStatement) node()                 {}
func (*GrantAdminStatement) node()            {}
func (*RevokeStatement) node()                {}
func (*RevokeAdminStatement) node()           {}
func (*SelectStatement) node()                {}
func (*SetPasswordUserStatement) node()       {}
func (*ShowContinuousQueriesStatement) node() {}
func (*ShowGrantsForUserStatement) node()     {}
func (*ShowServersStatement) node()           {}
func (*ShowDatabasesStatement) node()         {}
func (*ShowFieldKeysStatement) node()         {}
func (*ShowRetentionPoliciesStatement) node() {}
func (*ShowMeasurementsStatement) node()      {}
func (*ShowSeriesStatement) node()            {}
func (*ShowStatsStatement) node()             {}
func (*ShowDiagnosticsStatement) node()       {}
func (*ShowTagKeysStatement) node()           {}
func (*ShowTagValuesStatement) node()         {}
func (*ShowUsersStatement) node()             {}

func (*BinaryExpr) node()      {}
func (*BooleanLiteral) node()  {}
func (*Call) node()            {}
func (*Dimension) node()       {}
func (Dimensions) node()       {}
func (*DurationLiteral) node() {}
func (*Field) node()           {}
func (Fields) node()           {}
func (*Measurement) node()     {}
func (Measurements) node()     {}
func (*nilLiteral) node()      {}
func (*NumberLiteral) node()   {}
func (*ParenExpr) node()       {}
func (*RegexLiteral) node()    {}
func (*SortField) node()       {}
func (SortFields) node()       {}
func (Sources) node()          {}
func (*StringLiteral) node()   {}
func (*Target) node()          {}
func (*TimeLiteral) node()     {}
func (*VarRef) node()          {}
func (*Wildcard) node()        {}

// Query represents a collection of ordered statements.
type Query struct {
	Statements Statements
}

// String returns a string representation of the query.
func (q *Query) String() string { return q.Statements.String() }

// Statements represents a list of statements.
type Statements []Statement

// String returns a string representation of the statements.
func (a Statements) String() string {
	var str []string
	for _, stmt := range a {
		str = append(str, stmt.String())
	}
	return strings.Join(str, ";\n")
}

// Statement represents a single command in InfluxQL.
type Statement interface {
	Node
	stmt()
	RequiredPrivileges() ExecutionPrivileges
}

// HasDefaultDatabase provides an interface to get the default database from a Statement.
type HasDefaultDatabase interface {
	Node
	stmt()
	DefaultDatabase() string
}

// ExecutionPrivilege is a privilege required for a user to execute
// a statement on a database or resource.
type ExecutionPrivilege struct {
	// Admin privilege required.
	Admin bool

	// Name of the database.
	Name string

	// Database privilege required.
	Privilege Privilege
}

// ExecutionPrivileges is a list of privileges required to execute a statement.
type ExecutionPrivileges []ExecutionPrivilege

func (*AlterRetentionPolicyStatement) stmt()  {}
func (*CreateContinuousQueryStatement) stmt() {}
func (*CreateDatabaseStatement) stmt()        {}
func (*CreateRetentionPolicyStatement) stmt() {}
func (*CreateUserStatement) stmt()            {}
func (*DeleteStatement) stmt()                {}
func (*DropContinuousQueryStatement) stmt()   {}
func (*DropDatabaseStatement) stmt()          {}
func (*DropMeasurementStatement) stmt()       {}
func (*DropRetentionPolicyStatement) stmt()   {}
func (*DropSeriesStatement) stmt()            {}
func (*DropUserStatement) stmt()              {}
func (*GrantStatement) stmt()                 {}
func (*GrantAdminStatement) stmt()            {}
func (*ShowContinuousQueriesStatement) stmt() {}
func (*ShowGrantsForUserStatement) stmt()     {}
func (*ShowServersStatement) stmt()           {}
func (*ShowDatabasesStatement) stmt()         {}
func (*ShowFieldKeysStatement) stmt()         {}
func (*ShowMeasurementsStatement) stmt()      {}
func (*ShowRetentionPoliciesStatement) stmt() {}
func (*ShowSeriesStatement) stmt()            {}
func (*ShowStatsStatement) stmt()             {}
func (*ShowDiagnosticsStatement) stmt()       {}
func (*ShowTagKeysStatement) stmt()           {}
func (*ShowTagValuesStatement) stmt()         {}
func (*ShowUsersStatement) stmt()             {}
func (*RevokeStatement) stmt()                {}
func (*RevokeAdminStatement) stmt()           {}
func (*SelectStatement) stmt()                {}
func (*SetPasswordUserStatement) stmt()       {}

// Expr represents an expression that can be evaluated to a value.
type Expr interface {
	Node
	expr()
}

func (*BinaryExpr) expr()      {}
func (*BooleanLiteral) expr()  {}
func (*Call) expr()            {}
func (*Distinct) expr()        {}
func (*DurationLiteral) expr() {}
func (*nilLiteral) expr()      {}
func (*NumberLiteral) expr()   {}
func (*ParenExpr) expr()       {}
func (*RegexLiteral) expr()    {}
func (*StringLiteral) expr()   {}
func (*TimeLiteral) expr()     {}
func (*VarRef) expr()          {}
func (*Wildcard) expr()        {}

// Source represents a source of data for a statement.
type Source interface {
	Node
	source()
}

func (*Measurement) source() {}

// Sources represents a list of sources.
type Sources []Source

// String returns a string representation of a Sources array.
func (a Sources) String() string {
	var buf bytes.Buffer

	ubound := len(a) - 1
	for i, src := range a {
		_, _ = buf.WriteString(src.String())
		if i < ubound {
			_, _ = buf.WriteString(", ")
		}
	}

	return buf.String()
}

// SortField represents a field to sort results by.
type SortField struct {
	// Name of the field
	Name string

	// Sort order.
	Ascending bool
}

// String returns a string representation of a sort field
func (field *SortField) String() string {
	var buf bytes.Buffer
	if field.Name == "" {
		_, _ = buf.WriteString(field.Name)
		_, _ = buf.WriteString(" ")
	}
	if field.Ascending {
		_, _ = buf.WriteString("ASC")
	} else {
		_, _ = buf.WriteString("DESC")
	}
	return buf.String()
}

// SortFields represents an ordered list of ORDER BY fields
type SortFields []*SortField

// String returns a string representation of sort fields
func (a SortFields) String() string {
	fields := make([]string, 0, len(a))
	for _, field := range a {
		fields = append(fields, field.String())
	}
	return strings.Join(fields, ", ")
}

// CreateDatabaseStatement represents a command for creating a new database.
type CreateDatabaseStatement struct {
	// Name of the database to be created.
	Name string
}

// String returns a string representation of the create database statement.
func (s *CreateDatabaseStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("CREATE DATABASE ")
	_, _ = buf.WriteString(s.Name)
	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute a CreateDatabaseStatement.
func (s *CreateDatabaseStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// DropDatabaseStatement represents a command to drop a database.
type DropDatabaseStatement struct {
	// Name of the database to be dropped.
	Name string
}

// String returns a string representation of the drop database statement.
func (s *DropDatabaseStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("DROP DATABASE ")
	_, _ = buf.WriteString(s.Name)
	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute a DropDatabaseStatement.
func (s *DropDatabaseStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// DropRetentionPolicyStatement represents a command to drop a retention policy from a database.
type DropRetentionPolicyStatement struct {
	// Name of the policy to drop.
	Name string

	// Name of the database to drop the policy from.
	Database string
}

// String returns a string representation of the drop retention policy statement.
func (s *DropRetentionPolicyStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("DROP RETENTION POLICY ")
	_, _ = buf.WriteString(s.Name)
	_, _ = buf.WriteString(" ON ")
	_, _ = buf.WriteString(s.Database)
	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute a DropRetentionPolicyStatement.
func (s *DropRetentionPolicyStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: false, Name: s.Database, Privilege: WritePrivilege}}
}

// CreateUserStatement represents a command for creating a new user.
type CreateUserStatement struct {
	// Name of the user to be created.
	Name string

	// User's password.
	Password string

	// User's admin privilege.
	Admin bool
}

// String returns a string representation of the create user statement.
func (s *CreateUserStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("CREATE USER ")
	_, _ = buf.WriteString(s.Name)
	_, _ = buf.WriteString(" WITH PASSWORD ")
	_, _ = buf.WriteString("[REDACTED]")
	if s.Admin {
		_, _ = buf.WriteString(" WITH ALL PRIVILEGES")
	}
	return buf.String()
}

// RequiredPrivileges returns the privilege(s) required to execute a CreateUserStatement.
func (s *CreateUserStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// DropUserStatement represents a command for dropping a user.
type DropUserStatement struct {
	// Name of the user to drop.
	Name string
}

// String returns a string representation of the drop user statement.
func (s *DropUserStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("DROP USER ")
	_, _ = buf.WriteString(s.Name)
	return buf.String()
}

// RequiredPrivileges returns the privilege(s) required to execute a DropUserStatement.
func (s *DropUserStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// Privilege is a type of action a user can be granted the right to use.
type Privilege int

const (
	// NoPrivileges means no privileges required / granted / revoked.
	NoPrivileges Privilege = iota
	// ReadPrivilege means read privilege required / granted / revoked.
	ReadPrivilege
	// WritePrivilege means write privilege required / granted / revoked.
	WritePrivilege
	// AllPrivileges means all privileges required / granted / revoked.
	AllPrivileges
)

// NewPrivilege returns an initialized *Privilege.
func NewPrivilege(p Privilege) *Privilege { return &p }

// String returns a string representation of a Privilege.
func (p Privilege) String() string {
	switch p {
	case NoPrivileges:
		return "NO PRIVILEGES"
	case ReadPrivilege:
		return "READ"
	case WritePrivilege:
		return "WRITE"
	case AllPrivileges:
		return "ALL PRIVILEGES"
	}
	return ""
}

// GrantStatement represents a command for granting a privilege.
type GrantStatement struct {
	// The privilege to be granted.
	Privilege Privilege

	// Database to grant the privilege to.
	On string

	// Who to grant the privilege to.
	User string
}

// String returns a string representation of the grant statement.
func (s *GrantStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("GRANT ")
	_, _ = buf.WriteString(s.Privilege.String())
	_, _ = buf.WriteString(" ON ")
	_, _ = buf.WriteString(s.On)
	_, _ = buf.WriteString(" TO ")
	_, _ = buf.WriteString(s.User)
	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute a GrantStatement.
func (s *GrantStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// GrantAdminStatement represents a command for granting admin privilege.
type GrantAdminStatement struct {
	// Who to grant the privilege to.
	User string
}

// String returns a string representation of the grant admin statement.
func (s *GrantAdminStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("GRANT ALL PRIVILEGES TO ")
	_, _ = buf.WriteString(s.User)
	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute a GrantAdminStatement.
func (s *GrantAdminStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// SetPasswordUserStatement represents a command for changing user password.
type SetPasswordUserStatement struct {
	// Plain Password
	Password string

	// Who to grant the privilege to.
	Name string
}

// String returns a string representation of the set password statement.
func (s *SetPasswordUserStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("SET PASSWORD FOR ")
	_, _ = buf.WriteString(s.Name)
	_, _ = buf.WriteString(" = ")
	_, _ = buf.WriteString("[REDACTED]")
	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute a SetPasswordUserStatement.
func (s *SetPasswordUserStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// RevokeStatement represents a command to revoke a privilege from a user.
type RevokeStatement struct {
	// The privilege to be revoked.
	Privilege Privilege

	// Database to revoke the privilege from.
	On string

	// Who to revoke privilege from.
	User string
}

// String returns a string representation of the revoke statement.
func (s *RevokeStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("REVOKE ")
	_, _ = buf.WriteString(s.Privilege.String())
	_, _ = buf.WriteString(" ON ")
	_, _ = buf.WriteString(s.On)
	_, _ = buf.WriteString(" FROM ")
	_, _ = buf.WriteString(s.User)
	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute a RevokeStatement.
func (s *RevokeStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// RevokeAdminStatement represents a command to revoke admin privilege from a user.
type RevokeAdminStatement struct {
	// Who to revoke admin privilege from.
	User string
}

// String returns a string representation of the revoke admin statement.
func (s *RevokeAdminStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("REVOKE ALL PRIVILEGES FROM ")
	_, _ = buf.WriteString(s.User)
	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute a RevokeAdminStatement.
func (s *RevokeAdminStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// CreateRetentionPolicyStatement represents a command to create a retention policy.
type CreateRetentionPolicyStatement struct {
	// Name of policy to create.
	Name string

	// Name of database this policy belongs to.
	Database string

	// Duration data written to this policy will be retained.
	Duration time.Duration

	// Replication factor for data written to this policy.
	Replication int

	// Should this policy be set as default for the database?
	Default bool
}

// String returns a string representation of the create retention policy.
func (s *CreateRetentionPolicyStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("CREATE RETENTION POLICY ")
	_, _ = buf.WriteString(s.Name)
	_, _ = buf.WriteString(" ON ")
	_, _ = buf.WriteString(s.Database)
	_, _ = buf.WriteString(" DURATION ")
	_, _ = buf.WriteString(FormatDuration(s.Duration))
	_, _ = buf.WriteString(" REPLICATION ")
	_, _ = buf.WriteString(strconv.Itoa(s.Replication))
	if s.Default {
		_, _ = buf.WriteString(" DEFAULT")
	}
	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute a CreateRetentionPolicyStatement.
func (s *CreateRetentionPolicyStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// AlterRetentionPolicyStatement represents a command to alter an existing retention policy.
type AlterRetentionPolicyStatement struct {
	// Name of policy to alter.
	Name string

	// Name of the database this policy belongs to.
	Database string

	// Duration data written to this policy will be retained.
	Duration *time.Duration

	// Replication factor for data written to this policy.
	Replication *int

	// Should this policy be set as defalut for the database?
	Default bool
}

// String returns a string representation of the alter retention policy statement.
func (s *AlterRetentionPolicyStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("ALTER RETENTION POLICY ")
	_, _ = buf.WriteString(s.Name)
	_, _ = buf.WriteString(" ON ")
	_, _ = buf.WriteString(s.Database)

	if s.Duration != nil {
		_, _ = buf.WriteString(" DURATION ")
		_, _ = buf.WriteString(FormatDuration(*s.Duration))
	}

	if s.Replication != nil {
		_, _ = buf.WriteString(" REPLICATION ")
		_, _ = buf.WriteString(strconv.Itoa(*s.Replication))
	}

	if s.Default {
		_, _ = buf.WriteString(" DEFAULT")
	}

	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute an AlterRetentionPolicyStatement.
func (s *AlterRetentionPolicyStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

type FillOption int

const (
	// NullFill means that empty aggregate windows will just have null values.
	NullFill FillOption = iota
	// NoFill means that empty aggregate windows will be purged from the result.
	NoFill
	// NumberFill means that empty aggregate windows will be filled with the given number
	NumberFill
	// PreviousFill means that empty aggregate windows will be filled with whatever the previous aggregate window had
	PreviousFill
)

// SelectStatement represents a command for extracting data from the database.
type SelectStatement struct {
	// Expressions returned from the selection.
	Fields Fields

	// Target (destination) for the result of the select.
	Target *Target

	// Expressions used for grouping the selection.
	Dimensions Dimensions

	// Data sources that fields are extracted from.
	Sources Sources

	// An expression evaluated on data point.
	Condition Expr

	// Fields to sort results by
	SortFields SortFields

	// Maximum number of rows to be returned. Unlimited if zero.
	Limit int

	// Returns rows starting at an offset from the first row.
	Offset int

	// Maxiumum number of series to be returned. Unlimited if zero.
	SLimit int

	// Returns series starting at an offset from the first one.
	SOffset int

	// memoize the group by interval
	groupByInterval time.Duration

	// if it's a query for raw data values (i.e. not an aggregate)
	IsRawQuery bool

	// What fill option the select statement uses, if any
	Fill FillOption

	// The value to fill empty aggregate buckets with, if any
	FillValue interface{}
}

// HasDerivative returns true if one of the function calls in the statement is a
// derivative aggregate
func (s *SelectStatement) HasDerivative() bool {
	for _, f := range s.FunctionCalls() {
		if strings.HasSuffix(f.Name, "derivative") {
			return true
		}
	}
	return false
}

// IsSimpleDerivative return true if one of the function call is a derivative function with a
// variable ref as the first arg
func (s *SelectStatement) IsSimpleDerivative() bool {
	for _, f := range s.FunctionCalls() {
		if strings.HasSuffix(f.Name, "derivative") {
			// it's nested if the first argument is an aggregate function
			if _, ok := f.Args[0].(*VarRef); ok {
				return true
			}
		}
	}
	return false
}

// Clone returns a deep copy of the statement.
func (s *SelectStatement) Clone() *SelectStatement {
	clone := &SelectStatement{
		Fields:     make(Fields, 0, len(s.Fields)),
		Dimensions: make(Dimensions, 0, len(s.Dimensions)),
		Sources:    cloneSources(s.Sources),
		SortFields: make(SortFields, 0, len(s.SortFields)),
		Condition:  CloneExpr(s.Condition),
		Limit:      s.Limit,
		Offset:     s.Offset,
		SLimit:     s.SLimit,
		SOffset:    s.SOffset,
		Fill:       s.Fill,
		FillValue:  s.FillValue,
		IsRawQuery: s.IsRawQuery,
	}
	if s.Target != nil {
		clone.Target = &Target{
			Measurement: &Measurement{
				Database:        s.Target.Measurement.Database,
				RetentionPolicy: s.Target.Measurement.RetentionPolicy,
				Name:            s.Target.Measurement.Name,
				Regex:           CloneRegexLiteral(s.Target.Measurement.Regex),
			},
		}
	}
	for _, f := range s.Fields {
		clone.Fields = append(clone.Fields, &Field{Expr: CloneExpr(f.Expr), Alias: f.Alias})
	}
	for _, d := range s.Dimensions {
		clone.Dimensions = append(clone.Dimensions, &Dimension{Expr: CloneExpr(d.Expr)})
	}
	for _, f := range s.SortFields {
		clone.SortFields = append(clone.SortFields, &SortField{Name: f.Name, Ascending: f.Ascending})
	}
	return clone
}

func cloneSources(sources Sources) Sources {
	clone := make(Sources, 0, len(sources))
	for _, s := range sources {
		clone = append(clone, cloneSource(s))
	}
	return clone
}

func cloneSource(s Source) Source {
	if s == nil {
		return nil
	}

	switch s := s.(type) {
	case *Measurement:
		m := &Measurement{Database: s.Database, RetentionPolicy: s.RetentionPolicy, Name: s.Name}
		if s.Regex != nil {
			m.Regex = &RegexLiteral{Val: regexp.MustCompile(s.Regex.Val.String())}
		}
		return m
	default:
		panic("unreachable")
	}
}

// RewriteWildcards returns the re-written form of the select statement. Any wildcard query
// fields are replaced with the supplied fields, and any wildcard GROUP BY fields are replaced
// with the supplied dimensions.
func (s *SelectStatement) RewriteWildcards(fields Fields, dimensions Dimensions) *SelectStatement {
	other := s.Clone()
	selectWildcard, groupWildcard := false, false

	// Rewrite all wildcard query fields
	rwFields := make(Fields, 0, len(s.Fields))
	for _, f := range s.Fields {
		switch f.Expr.(type) {
		case *Wildcard:
			// Sort wildcard fields for consistent output
			sort.Sort(fields)
			rwFields = append(rwFields, fields...)
			selectWildcard = true
		default:
			rwFields = append(rwFields, f)
		}
	}
	other.Fields = rwFields

	// Rewrite all wildcard GROUP BY fields
	rwDimensions := make(Dimensions, 0, len(s.Dimensions))
	for _, d := range s.Dimensions {
		switch d.Expr.(type) {
		case *Wildcard:
			rwDimensions = append(rwDimensions, dimensions...)
			groupWildcard = true
		default:
			rwDimensions = append(rwDimensions, d)
		}
	}

	if selectWildcard && !groupWildcard {
		rwDimensions = append(rwDimensions, dimensions...)
	}
	other.Dimensions = rwDimensions

	return other
}

// RewriteDistinct rewrites the expression to be a call for map/reduce to work correctly
// This method assumes all validation has passed
func (s *SelectStatement) RewriteDistinct() {
	for i, f := range s.Fields {
		if d, ok := f.Expr.(*Distinct); ok {
			s.Fields[i].Expr = d.NewCall()
			s.IsRawQuery = false
		}
	}
}

// String returns a string representation of the select statement.
func (s *SelectStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("SELECT ")
	_, _ = buf.WriteString(s.Fields.String())

	if s.Target != nil {
		_, _ = buf.WriteString(" ")
		_, _ = buf.WriteString(s.Target.String())
	}
	if len(s.Sources) > 0 {
		_, _ = buf.WriteString(" FROM ")
		_, _ = buf.WriteString(s.Sources.String())
	}
	if s.Condition != nil {
		_, _ = buf.WriteString(" WHERE ")
		_, _ = buf.WriteString(s.Condition.String())
	}
	if len(s.Dimensions) > 0 {
		_, _ = buf.WriteString(" GROUP BY ")
		_, _ = buf.WriteString(s.Dimensions.String())
	}
	switch s.Fill {
	case NoFill:
		_, _ = buf.WriteString(" fill(none)")
	case NumberFill:
		_, _ = buf.WriteString(fmt.Sprintf(" fill(%v)", s.FillValue))
	case PreviousFill:
		_, _ = buf.WriteString(" fill(previous)")
	}
	if len(s.SortFields) > 0 {
		_, _ = buf.WriteString(" ORDER BY ")
		_, _ = buf.WriteString(s.SortFields.String())
	}
	if s.Limit > 0 {
		_, _ = fmt.Fprintf(&buf, " LIMIT %d", s.Limit)
	}
	if s.Offset > 0 {
		_, _ = buf.WriteString(" OFFSET ")
		_, _ = buf.WriteString(strconv.Itoa(s.Offset))
	}
	if s.SLimit > 0 {
		_, _ = fmt.Fprintf(&buf, " SLIMIT %d", s.SLimit)
	}
	if s.SOffset > 0 {
		_, _ = fmt.Fprintf(&buf, " SOFFSET %d", s.SOffset)
	}
	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute the SelectStatement.
func (s *SelectStatement) RequiredPrivileges() ExecutionPrivileges {
	ep := ExecutionPrivileges{{Admin: false, Name: "", Privilege: ReadPrivilege}}

	if s.Target != nil {
		p := ExecutionPrivilege{Admin: false, Name: s.Target.Measurement.Database, Privilege: WritePrivilege}
		ep = append(ep, p)
	}
	return ep
}

// OnlyTimeDimensions returns true if the statement has a where clause with only time constraints
func (s *SelectStatement) OnlyTimeDimensions() bool {
	return s.walkForTime(s.Condition)
}

// walkForTime is called by the OnlyTimeDimensions method to walk the where clause to determine if
// the only things specified are based on time
func (s *SelectStatement) walkForTime(node Node) bool {
	switch n := node.(type) {
	case *BinaryExpr:
		if n.Op == AND || n.Op == OR {
			return s.walkForTime(n.LHS) && s.walkForTime(n.RHS)
		}
		if ref, ok := n.LHS.(*VarRef); ok && strings.ToLower(ref.Val) == "time" {
			return true
		}
		return false
	case *ParenExpr:
		// walk down the tree
		return s.walkForTime(n.Expr)
	default:
		return false
	}
}

// HasWildcard returns whether or not the select statement has at least 1 wildcard
func (s *SelectStatement) HasWildcard() bool {
	for _, f := range s.Fields {
		_, ok := f.Expr.(*Wildcard)
		if ok {
			return true
		}
	}

	for _, d := range s.Dimensions {
		_, ok := d.Expr.(*Wildcard)
		if ok {
			return true
		}
	}

	return false
}

// hasTimeDimensions returns whether or not the select statement has at least 1
// where condition with time as the condition
func (s *SelectStatement) hasTimeDimensions(node Node) bool {
	switch n := node.(type) {
	case *BinaryExpr:
		if n.Op == AND || n.Op == OR {
			return s.hasTimeDimensions(n.LHS) || s.hasTimeDimensions(n.RHS)
		}
		if ref, ok := n.LHS.(*VarRef); ok && strings.ToLower(ref.Val) == "time" {
			return true
		}
		return false
	case *ParenExpr:
		// walk down the tree
		return s.hasTimeDimensions(n.Expr)
	default:
		return false
	}
}

func (s *SelectStatement) validate(tr targetRequirement) error {
	if err := s.validateDistinct(); err != nil {
		return err
	}

	if err := s.validateCountDistinct(); err != nil {
		return err
	}

	if err := s.validateAggregates(tr); err != nil {
		return err
	}

	if err := s.validateDerivative(); err != nil {
		return err
	}

	return nil
}

func (s *SelectStatement) validateAggregates(tr targetRequirement) error {
	// First, determine if specific calls have at least one and only one argument
	for _, f := range s.Fields {
		if c, ok := f.Expr.(*Call); ok {
			switch c.Name {
			case "derivative", "non_negative_derivative":
				if min, max, got := 1, 2, len(c.Args); got > max || got < min {
					return fmt.Errorf("invalid number of arguments for %s, expected at least %d but no more than %d, got %d", c.Name, min, max, got)
				}
			case "percentile":
				if exp, got := 2, len(c.Args); got != exp {
					return fmt.Errorf("invalid number of arguments for %s, expected %d, got %d", c.Name, exp, got)
				}
			default:
				if exp, got := 1, len(c.Args); got != exp {
					return fmt.Errorf("invalid number of arguments for %s, expected %d, got %d", c.Name, exp, got)
				}
			}
		}
	}

	// Now, check that we have valid duration and where clauses for aggregates

	// fetch the group by duration
	groupByDuration, _ := s.GroupByInterval()

	// If we have a group by interval, but no aggregate function, it's an invalid statement
	if s.IsRawQuery && groupByDuration > 0 {
		return fmt.Errorf("GROUP BY requires at least one aggregate function")
	}

	// If we have an aggregate function with a group by time without a where clause, it's an invalid statement
	if tr == targetNotRequired { // ignore create continuous query statements
		if !s.IsRawQuery && groupByDuration > 0 && !s.hasTimeDimensions(s.Condition) {
			return fmt.Errorf("aggregate functions with GROUP BY time require a WHERE time clause")
		}
	}
	return nil
}

func (s *SelectStatement) HasDistinct() bool {
	// determine if we have a call named distinct
	for _, f := range s.Fields {
		switch c := f.Expr.(type) {
		case *Call:
			if c.Name == "distinct" {
				return true
			}
		case *Distinct:
			return true
		}
	}
	return false
}

func (s *SelectStatement) validateDistinct() error {
	if !s.HasDistinct() {
		return nil
	}

	if len(s.Fields) > 1 {
		return fmt.Errorf("aggregate function distinct() can not be combined with other functions or fields")
	}

	switch c := s.Fields[0].Expr.(type) {
	case *Call:
		if len(c.Args) == 0 {
			return fmt.Errorf("distinct function requires at least one argument")
		}

		if len(c.Args) != 1 {
			return fmt.Errorf("distinct function can only have one argument")
		}
	}
	return nil
}

func (s *SelectStatement) HasCountDistinct() bool {
	for _, f := range s.Fields {
		if c, ok := f.Expr.(*Call); ok {
			if c.Name == "count" {
				for _, a := range c.Args {
					if _, ok := a.(*Distinct); ok {
						return true
					}
					if c, ok := a.(*Call); ok {
						if c.Name == "distinct" {
							return true
						}
					}
				}
			}
		}
	}
	return false
}

func (s *SelectStatement) validateCountDistinct() error {
	if !s.HasCountDistinct() {
		return nil
	}

	valid := func(e Expr) bool {
		c, ok := e.(*Call)
		if !ok {
			return true
		}
		if c.Name != "count" {
			return true
		}
		for _, a := range c.Args {
			if _, ok := a.(*Distinct); ok {
				return len(c.Args) == 1
			}
			if d, ok := a.(*Call); ok {
				if d.Name == "distinct" {
					return len(d.Args) == 1
				}
			}
		}
		return true
	}

	for _, f := range s.Fields {
		if !valid(f.Expr) {
			return fmt.Errorf("count(distinct <field>) can only have one argument")
		}
	}

	return nil
}

func (s *SelectStatement) validateDerivative() error {
	if !s.HasDerivative() {
		return nil
	}

	// If a derivative is requested, it must be the only field in the query. We don't support
	// multiple fields in combination w/ derivaties yet.
	if len(s.Fields) != 1 {
		return fmt.Errorf("derivative cannot be used with other fields")
	}

	aggr := s.FunctionCalls()
	if len(aggr) != 1 {
		return fmt.Errorf("derivative cannot be used with other fields")
	}

	// Derivative requires two arguments
	derivativeCall := aggr[0]
	if len(derivativeCall.Args) == 0 {
		return fmt.Errorf("derivative requires a field argument")
	}

	// First arg must be a field or aggr over a field e.g. (mean(field))
	_, callOk := derivativeCall.Args[0].(*Call)
	_, varOk := derivativeCall.Args[0].(*VarRef)

	if !(callOk || varOk) {
		return fmt.Errorf("derivative requires a field argument")
	}

	// If a duration arg is pased, make sure it's a duration
	if len(derivativeCall.Args) == 2 {
		// Second must be a duration .e.g (1h)
		if _, ok := derivativeCall.Args[1].(*DurationLiteral); !ok {
			return fmt.Errorf("derivative requires a duration argument")
		}
	}

	return nil
}

// GroupByIterval extracts the time interval, if specified.
func (s *SelectStatement) GroupByInterval() (time.Duration, error) {
	// return if we've already pulled it out
	if s.groupByInterval != 0 {
		return s.groupByInterval, nil
	}

	// Ignore if there are no dimensions.
	if len(s.Dimensions) == 0 {
		return 0, nil
	}

	for _, d := range s.Dimensions {
		if call, ok := d.Expr.(*Call); ok && call.Name == "time" {
			// Make sure there is exactly one argument.
			if len(call.Args) != 1 {
				return 0, errors.New("time dimension expected one argument")
			}

			// Ensure the argument is a duration.
			lit, ok := call.Args[0].(*DurationLiteral)
			if !ok {
				return 0, errors.New("time dimension must have one duration argument")
			}
			s.groupByInterval = lit.Val
			return lit.Val, nil
		}
	}
	return 0, nil
}

// SetTimeRange sets the start and end time of the select statement to [start, end). i.e. start inclusive, end exclusive.
// This is used commonly for continuous queries so the start and end are in buckets.
func (s *SelectStatement) SetTimeRange(start, end time.Time) error {
	cond := fmt.Sprintf("time >= '%s' AND time < '%s'", start.UTC().Format(time.RFC3339Nano), end.UTC().Format(time.RFC3339Nano))
	if s.Condition != nil {
		cond = fmt.Sprintf("%s AND %s", s.rewriteWithoutTimeDimensions(), cond)
	}

	expr, err := NewParser(strings.NewReader(cond)).ParseExpr()
	if err != nil {
		return err
	}

	// fold out any previously replaced time dimensios and set the condition
	s.Condition = Reduce(expr, nil)

	return nil
}

// rewriteWithoutTimeDimensions will remove any WHERE time... clauses from the select statement
// This is necessary when setting an explicit time range to override any that previously existed.
func (s *SelectStatement) rewriteWithoutTimeDimensions() string {
	n := RewriteFunc(s.Condition, func(n Node) Node {
		switch n := n.(type) {
		case *BinaryExpr:
			if n.LHS.String() == "time" {
				return &BooleanLiteral{Val: true}
			}
			return n
		case *Call:
			return &BooleanLiteral{Val: true}
		default:
			return n
		}
	})

	return n.String()
}

/*

BinaryExpr

SELECT mean(xxx.value) + avg(yyy.value) FROM xxx JOIN yyy WHERE xxx.host = 123

from xxx where host = 123
select avg(value) from yyy where host = 123

SELECT xxx.value FROM xxx WHERE xxx.host = 123
SELECT yyy.value FROM yyy

---

SELECT MEAN(xxx.value) + MEAN(cpu.load.value)
FROM xxx JOIN yyy
GROUP BY host
WHERE (xxx.region == "uswest" OR yyy.region == "uswest") AND xxx.otherfield == "XXX"

select * from (
	select mean + mean from xxx join yyy
	group by time(5m), host
) (xxx.region == "uswest" OR yyy.region == "uswest") AND xxx.otherfield == "XXX"

(seriesIDS for xxx.region = 'uswest' union seriesIDs for yyy.regnion = 'uswest') | seriesIDS xxx.otherfield = 'XXX'

WHERE xxx.region == "uswest" AND xxx.otherfield == "XXX"
WHERE yyy.region == "uswest"


*/

// Substatement returns a single-series statement for a given variable reference.
func (s *SelectStatement) Substatement(ref *VarRef) (*SelectStatement, error) {
	// Copy dimensions and properties to new statement.
	other := &SelectStatement{
		Fields:     Fields{{Expr: ref}},
		Dimensions: s.Dimensions,
		Limit:      s.Limit,
		Offset:     s.Offset,
		SortFields: s.SortFields,
	}

	// If there is only one series source then return it with the whole condition.
	if len(s.Sources) == 1 {
		other.Sources = s.Sources
		other.Condition = s.Condition
		return other, nil
	}

	// Find the matching source.
	name := MatchSource(s.Sources, ref.Val)
	if name == "" {
		return nil, fmt.Errorf("field source not found: %s", ref.Val)
	}
	other.Sources = append(other.Sources, &Measurement{Name: name})

	// Filter out conditions.
	if s.Condition != nil {
		other.Condition = filterExprBySource(name, s.Condition)
	}

	return other, nil
}

// NamesInWhere returns the field and tag names (idents) referenced in the where clause
func (s *SelectStatement) NamesInWhere() []string {
	var a []string
	if s.Condition != nil {
		a = walkNames(s.Condition)
	}
	return a
}

// NamesInSelect returns the field and tag names (idents) in the select clause
func (s *SelectStatement) NamesInSelect() []string {
	var a []string

	for _, f := range s.Fields {
		a = append(a, walkNames(f.Expr)...)
	}

	return a
}

// walkNames will walk the Expr and return the database fields
func walkNames(exp Expr) []string {
	switch expr := exp.(type) {
	case *VarRef:
		return []string{expr.Val}
	case *Call:
		if len(expr.Args) == 0 {
			return nil
		}
		lit, ok := expr.Args[0].(*VarRef)
		if !ok {
			return nil
		}

		return []string{lit.Val}
	case *BinaryExpr:
		var ret []string
		ret = append(ret, walkNames(expr.LHS)...)
		ret = append(ret, walkNames(expr.RHS)...)
		return ret
	case *ParenExpr:
		return walkNames(expr.Expr)
	}

	return nil
}

// FunctionCalls returns the Call objects from the query
func (s *SelectStatement) FunctionCalls() []*Call {
	var a []*Call
	for _, f := range s.Fields {
		a = append(a, walkFunctionCalls(f.Expr)...)
	}
	return a
}

// walkFunctionCalls walks the Field of a query for any function calls made
func walkFunctionCalls(exp Expr) []*Call {
	switch expr := exp.(type) {
	case *VarRef:
		return nil
	case *Call:
		return []*Call{expr}
	case *BinaryExpr:
		var ret []*Call
		ret = append(ret, walkFunctionCalls(expr.LHS)...)
		ret = append(ret, walkFunctionCalls(expr.RHS)...)
		return ret
	case *ParenExpr:
		return walkFunctionCalls(expr.Expr)
	}

	return nil
}

// filters an expression to exclude expressions unrelated to a source.
func filterExprBySource(name string, expr Expr) Expr {
	switch expr := expr.(type) {
	case *VarRef:
		if !strings.HasPrefix(expr.Val, name) {
			return nil
		}

	case *BinaryExpr:
		lhs := filterExprBySource(name, expr.LHS)
		rhs := filterExprBySource(name, expr.RHS)

		// If an expr is logical then return either LHS/RHS or both.
		// If an expr is arithmetic or comparative then require both sides.
		if expr.Op == AND || expr.Op == OR {
			if lhs == nil && rhs == nil {
				return nil
			} else if lhs != nil && rhs == nil {
				return lhs
			} else if lhs == nil && rhs != nil {
				return rhs
			}
		} else {
			if lhs == nil || rhs == nil {
				return nil
			}
		}
		return &BinaryExpr{Op: expr.Op, LHS: lhs, RHS: rhs}

	case *ParenExpr:
		exp := filterExprBySource(name, expr.Expr)
		if exp == nil {
			return nil
		}
		return &ParenExpr{Expr: exp}
	}
	return expr
}

// MatchSource returns the source name that matches a field name.
// Returns a blank string if no sources match.
func MatchSource(sources Sources, name string) string {
	for _, src := range sources {
		switch src := src.(type) {
		case *Measurement:
			if strings.HasPrefix(name, src.Name) {
				return src.Name
			}
		}
	}
	return ""
}

// Target represents a target (destination) policy, measurement, and DB.
type Target struct {
	// Measurement to write into.
	Measurement *Measurement
}

// String returns a string representation of the Target.
func (t *Target) String() string {
	if t == nil {
		return ""
	}

	var buf bytes.Buffer
	_, _ = buf.WriteString("INTO ")
	_, _ = buf.WriteString(t.Measurement.String())

	return buf.String()
}

// DeleteStatement represents a command for removing data from the database.
type DeleteStatement struct {
	// Data source that values are removed from.
	Source Source

	// An expression evaluated on data point.
	Condition Expr
}

// String returns a string representation of the delete statement.
func (s *DeleteStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("DELETE ")
	_, _ = buf.WriteString(s.Source.String())
	if s.Condition != nil {
		_, _ = buf.WriteString(" WHERE ")
		_, _ = buf.WriteString(s.Condition.String())
	}
	return s.String()
}

// RequiredPrivileges returns the privilege required to execute a DeleteStatement.
func (s *DeleteStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: false, Name: "", Privilege: WritePrivilege}}
}

// ShowSeriesStatement represents a command for listing series in the database.
type ShowSeriesStatement struct {
	// Measurement(s) the series are listed for.
	Sources Sources

	// An expression evaluated on a series name or tag.
	Condition Expr

	// Fields to sort results by
	SortFields SortFields

	// Maximum number of rows to be returned.
	// Unlimited if zero.
	Limit int

	// Returns rows starting at an offset from the first row.
	Offset int
}

// String returns a string representation of the list series statement.
func (s *ShowSeriesStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("SHOW SERIES")

	if s.Sources != nil {
		_, _ = buf.WriteString(" FROM ")
		_, _ = buf.WriteString(s.Sources.String())
	}

	if s.Condition != nil {
		_, _ = buf.WriteString(" WHERE ")
		_, _ = buf.WriteString(s.Condition.String())
	}
	if len(s.SortFields) > 0 {
		_, _ = buf.WriteString(" ORDER BY ")
		_, _ = buf.WriteString(s.SortFields.String())
	}
	if s.Limit > 0 {
		_, _ = buf.WriteString(" LIMIT ")
		_, _ = buf.WriteString(strconv.Itoa(s.Limit))
	}
	if s.Offset > 0 {
		_, _ = buf.WriteString(" OFFSET ")
		_, _ = buf.WriteString(strconv.Itoa(s.Offset))
	}
	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute a ShowSeriesStatement.
func (s *ShowSeriesStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: false, Name: "", Privilege: ReadPrivilege}}
}

// DropSeriesStatement represents a command for removing a series from the database.
type DropSeriesStatement struct {
	// Data source that fields are extracted from (optional)
	Sources Sources

	// An expression evaluated on data point (optional)
	Condition Expr
}

// String returns a string representation of the drop series statement.
func (s *DropSeriesStatement) String() string {
	var buf bytes.Buffer
	buf.WriteString("DROP SERIES")

	if s.Sources != nil {
		buf.WriteString(" FROM ")
		buf.WriteString(s.Sources.String())
	}
	if s.Condition != nil {
		buf.WriteString(" WHERE ")
		buf.WriteString(s.Condition.String())
	}

	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute a DropSeriesStatement.
func (s DropSeriesStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: false, Name: "", Privilege: WritePrivilege}}
}

// ShowContinuousQueriesStatement represents a command for listing continuous queries.
type ShowContinuousQueriesStatement struct{}

// String returns a string representation of the list continuous queries statement.
func (s *ShowContinuousQueriesStatement) String() string { return "SHOW CONTINUOUS QUERIES" }

// RequiredPrivileges returns the privilege required to execute a ShowContinuousQueriesStatement.
func (s *ShowContinuousQueriesStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: false, Name: "", Privilege: ReadPrivilege}}
}

// ShowGrantsForUserStatement represents a command for listing user privileges.
type ShowGrantsForUserStatement struct {
	// Name of the user to display privileges.
	Name string
}

// String returns a string representation of the show grants for user.
func (s *ShowGrantsForUserStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("SHOW GRANTS FOR ")
	_, _ = buf.WriteString(s.Name)

	return buf.String()
}

// RequiredPrivileges returns the privilege required to execute a ShowGrantsForUserStatement
func (s *ShowGrantsForUserStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// ShowServersStatement represents a command for listing all servers.
type ShowServersStatement struct{}

// String returns a string representation of the show servers command.
func (s *ShowServersStatement) String() string { return "SHOW SERVERS" }

// RequiredPrivileges returns the privilege required to execute a ShowServersStatement
func (s *ShowServersStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// ShowDatabasesStatement represents a command for listing all databases in the cluster.
type ShowDatabasesStatement struct{}

// String returns a string representation of the list databases command.
func (s *ShowDatabasesStatement) String() string { return "SHOW DATABASES" }

// RequiredPrivileges returns the privilege required to execute a ShowDatabasesStatement
func (s *ShowDatabasesStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// CreateContinuousQueryStatement represents a command for creating a continuous query.
type CreateContinuousQueryStatement struct {
	// Name of the continuous query to be created.
	Name string

	// Name of the database to create the continuous query on.
	Database string

	// Source of data (SELECT statement).
	Source *SelectStatement
}

// String returns a string representation of the statement.
func (s *CreateContinuousQueryStatement) String() string {
	return fmt.Sprintf("CREATE CONTINUOUS QUERY %s ON %s BEGIN %s END", QuoteIdent(s.Name), QuoteIdent(s.Database), s.Source.String())
}

// DefaultDatabase returns the default database from the statement.
func (s *CreateContinuousQueryStatement) DefaultDatabase() string {
	return s.Database
}

// RequiredPrivileges returns the privilege required to execute a CreateContinuousQueryStatement.
func (s *CreateContinuousQueryStatement) RequiredPrivileges() ExecutionPrivileges {
	ep := ExecutionPrivileges{{Admin: false, Name: s.Database, Privilege: ReadPrivilege}}

	// Selecting into a database that's different from the source?
	if s.Source.Target.Measurement.Database != "" {
		// Change source database privilege requirement to read.
		ep[0].Privilege = ReadPrivilege

		// Add destination database privilege requirement and set it to write.
		p := ExecutionPrivilege{
			Admin:     false,
			Name:      s.Source.Target.Measurement.Database,
			Privilege: WritePrivilege,
		}
		ep = append(ep, p)
	}

	return ep
}

// DropContinuousQueryStatement represents a command for removing a continuous query.
type DropContinuousQueryStatement struct {
	Name     string
	Database string
}

// String returns a string representation of the statement.
func (s *DropContinuousQueryStatement) String() string {
	return fmt.Sprintf("DROP CONTINUOUS QUERY %s", s.Name)
}

// RequiredPrivileges returns the privilege(s) required to execute a DropContinuousQueryStatement
func (s *DropContinuousQueryStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: false, Name: "", Privilege: WritePrivilege}}
}

// ShowMeasurementsStatement represents a command for listing measurements.
type ShowMeasurementsStatement struct {
	// An expression evaluated on data point.
	Condition Expr

	// Fields to sort results by
	SortFields SortFields

	// Maximum number of rows to be returned.
	// Unlimited if zero.
	Limit int

	// Returns rows starting at an offset from the first row.
	Offset int
}

// String returns a string representation of the statement.
func (s *ShowMeasurementsStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("SHOW MEASUREMENTS")

	if s.Condition != nil {
		_, _ = buf.WriteString(" WHERE ")
		_, _ = buf.WriteString(s.Condition.String())
	}
	if len(s.SortFields) > 0 {
		_, _ = buf.WriteString(" ORDER BY ")
		_, _ = buf.WriteString(s.SortFields.String())
	}
	if s.Limit > 0 {
		_, _ = buf.WriteString(" LIMIT ")
		_, _ = buf.WriteString(strconv.Itoa(s.Limit))
	}
	if s.Offset > 0 {
		_, _ = buf.WriteString(" OFFSET ")
		_, _ = buf.WriteString(strconv.Itoa(s.Offset))
	}
	return buf.String()
}

// RequiredPrivileges returns the privilege(s) required to execute a ShowMeasurementsStatement
func (s *ShowMeasurementsStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: false, Name: "", Privilege: ReadPrivilege}}
}

// DropMeasurementStatement represents a command to drop a measurement.
type DropMeasurementStatement struct {
	// Name of the measurement to be dropped.
	Name string
}

// String returns a string representation of the drop measurement statement.
func (s *DropMeasurementStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("DROP MEASUREMENT ")
	_, _ = buf.WriteString(s.Name)
	return buf.String()
}

// RequiredPrivileges returns the privilege(s) required to execute a DropMeasurementStatement
func (s *DropMeasurementStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// ShowRetentionPoliciesStatement represents a command for listing retention policies.
type ShowRetentionPoliciesStatement struct {
	// Name of the database to list policies for.
	Database string
}

// String returns a string representation of a ShowRetentionPoliciesStatement.
func (s *ShowRetentionPoliciesStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("SHOW RETENTION POLICIES ")
	_, _ = buf.WriteString(s.Database)
	return buf.String()
}

// RequiredPrivileges returns the privilege(s) required to execute a ShowRetentionPoliciesStatement
func (s *ShowRetentionPoliciesStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: false, Name: "", Privilege: ReadPrivilege}}
}

// ShowRetentionPoliciesStatement represents a command for displaying stats for a given server.
type ShowStatsStatement struct {
	// Hostname or IP of the server for stats.
	Host string
}

// String returns a string representation of a ShowStatsStatement.
func (s *ShowStatsStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("SHOW STATS ")
	if s.Host != "" {
		_, _ = buf.WriteString(s.Host)
	}
	return buf.String()
}

// RequiredPrivileges returns the privilege(s) required to execute a ShowStatsStatement
func (s *ShowStatsStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// ShowDiagnosticsStatement represents a command for show node diagnostics.
type ShowDiagnosticsStatement struct{}

// String returns a string representation of the ShowDiagnosticsStatement.
func (s *ShowDiagnosticsStatement) String() string { return "SHOW DIAGNOSTICS" }

// RequiredPrivileges returns the privilege required to execute a ShowDiagnosticsStatement
func (s *ShowDiagnosticsStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// ShowTagKeysStatement represents a command for listing tag keys.
type ShowTagKeysStatement struct {
	// Data sources that fields are extracted from.
	Sources Sources

	// An expression evaluated on data point.
	Condition Expr

	// Fields to sort results by
	SortFields SortFields

	// Maximum number of rows to be returned.
	// Unlimited if zero.
	Limit int

	// Returns rows starting at an offset from the first row.
	Offset int
}

// String returns a string representation of the statement.
func (s *ShowTagKeysStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("SHOW TAG KEYS")

	if s.Sources != nil {
		_, _ = buf.WriteString(" FROM ")
		_, _ = buf.WriteString(s.Sources.String())
	}
	if s.Condition != nil {
		_, _ = buf.WriteString(" WHERE ")
		_, _ = buf.WriteString(s.Condition.String())
	}
	if len(s.SortFields) > 0 {
		_, _ = buf.WriteString(" ORDER BY ")
		_, _ = buf.WriteString(s.SortFields.String())
	}
	if s.Limit > 0 {
		_, _ = buf.WriteString(" LIMIT ")
		_, _ = buf.WriteString(strconv.Itoa(s.Limit))
	}
	if s.Offset > 0 {
		_, _ = buf.WriteString(" OFFSET ")
		_, _ = buf.WriteString(strconv.Itoa(s.Offset))
	}
	return buf.String()
}

// RequiredPrivileges returns the privilege(s) required to execute a ShowTagKeysStatement
func (s *ShowTagKeysStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: false, Name: "", Privilege: ReadPrivilege}}
}

// ShowTagValuesStatement represents a command for listing tag values.
type ShowTagValuesStatement struct {
	// Data source that fields are extracted from.
	Sources Sources

	// Tag key(s) to pull values from.
	TagKeys []string

	// An expression evaluated on data point.
	Condition Expr

	// Fields to sort results by
	SortFields SortFields

	// Maximum number of rows to be returned.
	// Unlimited if zero.
	Limit int

	// Returns rows starting at an offset from the first row.
	Offset int
}

// String returns a string representation of the statement.
func (s *ShowTagValuesStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("SHOW TAG VALUES")

	if s.Sources != nil {
		_, _ = buf.WriteString(" FROM ")
		_, _ = buf.WriteString(s.Sources.String())
	}
	if s.Condition != nil {
		_, _ = buf.WriteString(" WHERE ")
		_, _ = buf.WriteString(s.Condition.String())
	}
	if len(s.SortFields) > 0 {
		_, _ = buf.WriteString(" ORDER BY ")
		_, _ = buf.WriteString(s.SortFields.String())
	}
	if s.Limit > 0 {
		_, _ = buf.WriteString(" LIMIT ")
		_, _ = buf.WriteString(strconv.Itoa(s.Limit))
	}
	if s.Offset > 0 {
		_, _ = buf.WriteString(" OFFSET ")
		_, _ = buf.WriteString(strconv.Itoa(s.Offset))
	}
	return buf.String()
}

// RequiredPrivileges returns the privilege(s) required to execute a ShowTagValuesStatement
func (s *ShowTagValuesStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: false, Name: "", Privilege: ReadPrivilege}}
}

// ShowUsersStatement represents a command for listing users.
type ShowUsersStatement struct{}

// String returns a string representation of the ShowUsersStatement.
func (s *ShowUsersStatement) String() string {
	return "SHOW USERS"
}

// RequiredPrivileges returns the privilege(s) required to execute a ShowUsersStatement
func (s *ShowUsersStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: true, Name: "", Privilege: AllPrivileges}}
}

// ShowFieldKeysStatement represents a command for listing field keys.
type ShowFieldKeysStatement struct {
	// Data sources that fields are extracted from.
	Sources Sources

	// Fields to sort results by
	SortFields SortFields

	// Maximum number of rows to be returned.
	// Unlimited if zero.
	Limit int

	// Returns rows starting at an offset from the first row.
	Offset int
}

// String returns a string representation of the statement.
func (s *ShowFieldKeysStatement) String() string {
	var buf bytes.Buffer
	_, _ = buf.WriteString("SHOW FIELD KEYS")

	if s.Sources != nil {
		_, _ = buf.WriteString(" FROM ")
		_, _ = buf.WriteString(s.Sources.String())
	}
	if len(s.SortFields) > 0 {
		_, _ = buf.WriteString(" ORDER BY ")
		_, _ = buf.WriteString(s.SortFields.String())
	}
	if s.Limit > 0 {
		_, _ = buf.WriteString(" LIMIT ")
		_, _ = buf.WriteString(strconv.Itoa(s.Limit))
	}
	if s.Offset > 0 {
		_, _ = buf.WriteString(" OFFSET ")
		_, _ = buf.WriteString(strconv.Itoa(s.Offset))
	}
	return buf.String()
}

// RequiredPrivileges returns the privilege(s) required to execute a ShowFieldKeysStatement
func (s *ShowFieldKeysStatement) RequiredPrivileges() ExecutionPrivileges {
	return ExecutionPrivileges{{Admin: false, Name: "", Privilege: ReadPrivilege}}
}

// Fields represents a list of fields.
type Fields []*Field

// String returns a string representation of the fields.
func (a Fields) String() string {
	var str []string
	for _, f := range a {
		str = append(str, f.String())
	}
	return strings.Join(str, ", ")
}

// Field represents an expression retrieved from a select statement.
type Field struct {
	Expr  Expr
	Alias string
}

// Name returns the name of the field. Returns alias, if set.
// Otherwise uses the function name or variable name.
func (f *Field) Name() string {
	// Return alias, if set.
	if f.Alias != "" {
		return f.Alias
	}

	// Return the function name or variable name, if available.
	switch expr := f.Expr.(type) {
	case *Call:
		return expr.Name
	case *VarRef:
		return expr.Val
	}

	// Otherwise return a blank name.
	return ""
}

// String returns a string representation of the field.
func (f *Field) String() string {
	str := f.Expr.String()

	switch f.Expr.(type) {
	case *VarRef:
		quoted := false
		// Escape any double-quotes in the field
		if strings.Contains(str, `"`) {
			str = strings.Replace(str, `"`, `\"`, -1)
			quoted = true
		}

		// Escape any single-quotes in the field
		if strings.Contains(str, `'`) {
			quoted = true
		}

		// Double-quote field names with spaces or that were previously escaped
		if strings.Contains(str, " ") || quoted {
			str = fmt.Sprintf("\"%s\"", str)
		}
	}

	if f.Alias == "" {
		return str
	}
	return fmt.Sprintf("%s AS %s", str, fmt.Sprintf(`"%s"`, f.Alias))
}

// Sort Interface for Fields
func (f Fields) Len() int           { return len(f) }
func (f Fields) Less(i, j int) bool { return f[i].Name() < f[j].Name() }
func (f Fields) Swap(i, j int)      { f[i], f[j] = f[j], f[i] }

// Dimensions represents a list of dimensions.
type Dimensions []*Dimension

// String returns a string representation of the dimensions.
func (a Dimensions) String() string {
	var str []string
	for _, d := range a {
		str = append(str, d.String())
	}
	return strings.Join(str, ", ")
}

// Normalize returns the interval and tag dimensions separately.
// Returns 0 if no time interval is specified.
// Returns an error if multiple time dimensions exist or if non-VarRef dimensions are specified.
func (a Dimensions) Normalize() (time.Duration, []string, error) {
	var dur time.Duration
	var tags []string

	for _, dim := range a {
		switch expr := dim.Expr.(type) {
		case *Call:
			// Ensure the call is time() and it only has one duration argument.
			// If we already have a duration
			if expr.Name != "time" {
				return 0, nil, errors.New("only time() calls allowed in dimensions")
			} else if len(expr.Args) != 1 {
				return 0, nil, errors.New("time dimension expected one argument")
			} else if lit, ok := expr.Args[0].(*DurationLiteral); !ok {
				return 0, nil, errors.New("time dimension must have one duration argument")
			} else if dur != 0 {
				return 0, nil, errors.New("multiple time dimensions not allowed")
			} else {
				dur = lit.Val
			}

		case *VarRef:
			tags = append(tags, expr.Val)

		default:
			return 0, nil, errors.New("only time and tag dimensions allowed")
		}
	}

	return dur, tags, nil
}

// Dimension represents an expression that a select statement is grouped by.
type Dimension struct {
	Expr Expr
}

// String returns a string representation of the dimension.
func (d *Dimension) String() string { return d.Expr.String() }

// Measurements represents a list of measurements.
type Measurements []*Measurement

// String returns a string representation of the measurements.
func (a Measurements) String() string {
	var str []string
	for _, m := range a {
		str = append(str, m.String())
	}
	return strings.Join(str, ", ")
}

// Measurement represents a single measurement used as a datasource.
type Measurement struct {
	Database        string
	RetentionPolicy string
	Name            string
	Regex           *RegexLiteral
}

// String returns a string representation of the measurement.
func (m *Measurement) String() string {
	var buf bytes.Buffer
	if m.Database != "" {
		_, _ = buf.WriteString(`"`)
		_, _ = buf.WriteString(m.Database)
		_, _ = buf.WriteString(`".`)
	}

	if m.RetentionPolicy != "" {
		_, _ = buf.WriteString(`"`)
		_, _ = buf.WriteString(m.RetentionPolicy)
		_, _ = buf.WriteString(`"`)
	}

	if m.Database != "" || m.RetentionPolicy != "" {
		_, _ = buf.WriteString(`.`)
	}

	if m.Name != "" {
		_, _ = buf.WriteString(QuoteIdent(m.Name))
	} else if m.Regex != nil {
		_, _ = buf.WriteString(m.Regex.String())
	}

	return buf.String()
}

// VarRef represents a reference to a variable.
type VarRef struct {
	Val string
}

// String returns a string representation of the variable reference.
func (r *VarRef) String() string { return r.Val }

// Call represents a function call.
type Call struct {
	Name string
	Args []Expr
}

// String returns a string representation of the call.
func (c *Call) String() string {
	// Join arguments.
	var str []string
	for _, arg := range c.Args {
		str = append(str, arg.String())
	}

	// Write function name and args.
	return fmt.Sprintf("%s(%s)", c.Name, strings.Join(str, ", "))
}

// Distinct represents a DISTINCT expression.
type Distinct struct {
	// Identifier following DISTINCT
	Val string
}

// String returns a string representation of the expression.
func (d *Distinct) String() string {
	return fmt.Sprintf("DISTINCT %s", d.Val)
}

// NewCall returns a new call expression from this expressions.
func (d *Distinct) NewCall() *Call {
	return &Call{
		Name: "distinct",
		Args: []Expr{
			&VarRef{Val: d.Val},
		},
	}
}

// NumberLiteral represents a numeric literal.
type NumberLiteral struct {
	Val float64
}

// String returns a string representation of the literal.
func (l *NumberLiteral) String() string { return strconv.FormatFloat(l.Val, 'f', 3, 64) }

// BooleanLiteral represents a boolean literal.
type BooleanLiteral struct {
	Val bool
}

// String returns a string representation of the literal.
func (l *BooleanLiteral) String() string {
	if l.Val {
		return "true"
	}
	return "false"
}

// isTrueLiteral returns true if the expression is a literal "true" value.
func isTrueLiteral(expr Expr) bool {
	if expr, ok := expr.(*BooleanLiteral); ok {
		return expr.Val == true
	}
	return false
}

// isFalseLiteral returns true if the expression is a literal "false" value.
func isFalseLiteral(expr Expr) bool {
	if expr, ok := expr.(*BooleanLiteral); ok {
		return expr.Val == false
	}
	return false
}

// StringLiteral represents a string literal.
type StringLiteral struct {
	Val string
}

// String returns a string representation of the literal.
func (l *StringLiteral) String() string { return QuoteString(l.Val) }

// TimeLiteral represents a point-in-time literal.
type TimeLiteral struct {
	Val time.Time
}

// String returns a string representation of the literal.
func (l *TimeLiteral) String() string {
	return `'` + l.Val.UTC().Format(time.RFC3339Nano) + `'`
}

// DurationLiteral represents a duration literal.
type DurationLiteral struct {
	Val time.Duration
}

// String returns a string representation of the literal.
func (l *DurationLiteral) String() string { return FormatDuration(l.Val) }

// nilLiteral represents a nil literal.
// This is not available to the query language itself. It's only used internally.
type nilLiteral struct{}

// String returns a string representation of the literal.
func (l *nilLiteral) String() string { return `nil` }

// BinaryExpr represents an operation between two expressions.
type BinaryExpr struct {
	Op  Token
	LHS Expr
	RHS Expr
}

// String returns a string representation of the binary expression.
func (e *BinaryExpr) String() string {
	return fmt.Sprintf("%s %s %s", e.LHS.String(), e.Op.String(), e.RHS.String())
}

// ParenExpr represents a parenthesized expression.
type ParenExpr struct {
	Expr Expr
}

// String returns a string representation of the parenthesized expression.
func (e *ParenExpr) String() string { return fmt.Sprintf("(%s)", e.Expr.String()) }

// RegexLiteral represents a regular expression.
type RegexLiteral struct {
	Val *regexp.Regexp
}

// String returns a string representation of the literal.
func (r *RegexLiteral) String() string {
	if r.Val != nil {
		return fmt.Sprintf("/%s/", r.Val.String())
	}
	return ""
}

// CloneRegexLiteral returns a clone of the RegexLiteral.
func CloneRegexLiteral(r *RegexLiteral) *RegexLiteral {
	if r == nil {
		return nil
	}

	clone := &RegexLiteral{}
	if r.Val != nil {
		clone.Val = regexp.MustCompile(r.Val.String())
	}

	return clone
}

// Wildcard represents a wild card expression.
type Wildcard struct{}

// String returns a string representation of the wildcard.
func (e *Wildcard) String() string { return "*" }

// CloneExpr returns a deep copy of the expression.
func CloneExpr(expr Expr) Expr {
	if expr == nil {
		return nil
	}
	switch expr := expr.(type) {
	case *BinaryExpr:
		return &BinaryExpr{Op: expr.Op, LHS: CloneExpr(expr.LHS), RHS: CloneExpr(expr.RHS)}
	case *BooleanLiteral:
		return &BooleanLiteral{Val: expr.Val}
	case *Call:
		args := make([]Expr, len(expr.Args))
		for i, arg := range expr.Args {
			args[i] = CloneExpr(arg)
		}
		return &Call{Name: expr.Name, Args: args}
	case *Distinct:
		return &Distinct{Val: expr.Val}
	case *DurationLiteral:
		return &DurationLiteral{Val: expr.Val}
	case *NumberLiteral:
		return &NumberLiteral{Val: expr.Val}
	case *ParenExpr:
		return &ParenExpr{Expr: CloneExpr(expr.Expr)}
	case *RegexLiteral:
		return &RegexLiteral{Val: expr.Val}
	case *StringLiteral:
		return &StringLiteral{Val: expr.Val}
	case *TimeLiteral:
		return &TimeLiteral{Val: expr.Val}
	case *VarRef:
		return &VarRef{Val: expr.Val}
	case *Wildcard:
		return &Wildcard{}
	}
	panic("unreachable")
}

// TimeRange returns the minimum and maximum times specified by an expression.
// Returns zero times if there is no bound.
func TimeRange(expr Expr) (min, max time.Time) {
	WalkFunc(expr, func(n Node) {
		if n, ok := n.(*BinaryExpr); ok {
			// Extract literal expression & operator on LHS.
			// Check for "time" on the left-hand side first.
			// Otherwise check for for the right-hand side and flip the operator.
			value, op := timeExprValue(n.LHS, n.RHS), n.Op
			if value.IsZero() {
				if value = timeExprValue(n.RHS, n.LHS); value.IsZero() {
					return
				} else if op == LT {
					op = GT
				} else if op == LTE {
					op = GTE
				} else if op == GT {
					op = LT
				} else if op == GTE {
					op = LTE
				}
			}

			// Update the min/max depending on the operator.
			// The GT & LT update the value by +/- 1ns not make them "not equal".
			switch op {
			case GT:
				if min.IsZero() || value.After(min) {
					min = value.Add(time.Nanosecond)
				}
			case GTE:
				if min.IsZero() || value.After(min) {
					min = value
				}
			case LT:
				if max.IsZero() || value.Before(max) {
					max = value.Add(-time.Nanosecond)
				}
			case LTE:
				if max.IsZero() || value.Before(max) {
					max = value
				}
			case EQ:
				if min.IsZero() || value.After(min) {
					min = value
				}
				if max.IsZero() || value.Before(max) {
					max = value
				}
			}
		}
	})
	return
}

// TimeRange returns the minimum and maximum times, as epoch nano, specified by
// and expression. If there is no lower bound, the start of the epoch is returned
// for minimum. If there is no higher bound, now is returned for maximum.
func TimeRangeAsEpochNano(expr Expr) (min, max int64) {
	tmin, tmax := TimeRange(expr)
	if tmin.IsZero() {
		min = time.Unix(0, 0).UnixNano()
	} else {
		min = tmin.UnixNano()
	}
	if tmax.IsZero() {
		max = time.Now().UnixNano()
	} else {
		max = tmax.UnixNano()
	}
	return
}

// timeExprValue returns the time literal value of a "time == <TimeLiteral>" expression.
// Returns zero time if the expression is not a time expression.
func timeExprValue(ref Expr, lit Expr) time.Time {
	if ref, ok := ref.(*VarRef); ok && strings.ToLower(ref.Val) == "time" {
		switch lit := lit.(type) {
		case *TimeLiteral:
			return lit.Val
		case *DurationLiteral:
			return time.Unix(0, int64(lit.Val)).UTC()
		}
	}
	return time.Time{}
}

// Visitor can be called by Walk to traverse an AST hierarchy.
// The Visit() function is called once per node.
type Visitor interface {
	Visit(Node) Visitor
}

// Walk traverses a node hierarchy in depth-first order.
func Walk(v Visitor, node Node) {
	if node == nil {
		return
	}

	if v = v.Visit(node); v == nil {
		return
	}

	switch n := node.(type) {
	case *BinaryExpr:
		Walk(v, n.LHS)
		Walk(v, n.RHS)

	case *Call:
		for _, expr := range n.Args {
			Walk(v, expr)
		}

	case *CreateContinuousQueryStatement:
		Walk(v, n.Source)

	case *Dimension:
		Walk(v, n.Expr)

	case Dimensions:
		for _, c := range n {
			Walk(v, c)
		}

	case *Field:
		Walk(v, n.Expr)

	case Fields:
		for _, c := range n {
			Walk(v, c)
		}

	case *ParenExpr:
		Walk(v, n.Expr)

	case *Query:
		Walk(v, n.Statements)

	case *SelectStatement:
		Walk(v, n.Fields)
		Walk(v, n.Target)
		Walk(v, n.Dimensions)
		Walk(v, n.Sources)
		Walk(v, n.Condition)
		Walk(v, n.SortFields)

	case *ShowSeriesStatement:
		Walk(v, n.Sources)
		Walk(v, n.Condition)

	case *ShowTagKeysStatement:
		Walk(v, n.Sources)
		Walk(v, n.Condition)
		Walk(v, n.SortFields)

	case *ShowTagValuesStatement:
		Walk(v, n.Sources)
		Walk(v, n.Condition)
		Walk(v, n.SortFields)

	case *ShowFieldKeysStatement:
		Walk(v, n.Sources)
		Walk(v, n.SortFields)

	case SortFields:
		for _, sf := range n {
			Walk(v, sf)
		}

	case Sources:
		for _, s := range n {
			Walk(v, s)
		}

	case Statements:
		for _, s := range n {
			Walk(v, s)
		}

	case *Target:
		if n != nil {
			Walk(v, n.Measurement)
		}
	}
}

// WalkFunc traverses a node hierarchy in depth-first order.
func WalkFunc(node Node, fn func(Node)) {
	Walk(walkFuncVisitor(fn), node)
}

type walkFuncVisitor func(Node)

func (fn walkFuncVisitor) Visit(n Node) Visitor { fn(n); return fn }

// Rewriter can be called by Rewrite to replace nodes in the AST hierarchy.
// The Rewrite() function is called once per node.
type Rewriter interface {
	Rewrite(Node) Node
}

// Rewrite recursively invokes the rewriter to replace each node.
// Nodes are traversed depth-first and rewritten from leaf to root.
func Rewrite(r Rewriter, node Node) Node {
	switch n := node.(type) {
	case *Query:
		n.Statements = Rewrite(r, n.Statements).(Statements)

	case Statements:
		for i, s := range n {
			n[i] = Rewrite(r, s).(Statement)
		}

	case *SelectStatement:
		n.Fields = Rewrite(r, n.Fields).(Fields)
		n.Dimensions = Rewrite(r, n.Dimensions).(Dimensions)
		n.Sources = Rewrite(r, n.Sources).(Sources)
		n.Condition = Rewrite(r, n.Condition).(Expr)

	case Fields:
		for i, f := range n {
			n[i] = Rewrite(r, f).(*Field)
		}

	case *Field:
		n.Expr = Rewrite(r, n.Expr).(Expr)

	case Dimensions:
		for i, d := range n {
			n[i] = Rewrite(r, d).(*Dimension)
		}

	case *Dimension:
		n.Expr = Rewrite(r, n.Expr).(Expr)

	case *BinaryExpr:
		n.LHS = Rewrite(r, n.LHS).(Expr)
		n.RHS = Rewrite(r, n.RHS).(Expr)

	case *ParenExpr:
		n.Expr = Rewrite(r, n.Expr).(Expr)

	case *Call:
		for i, expr := range n.Args {
			n.Args[i] = Rewrite(r, expr).(Expr)
		}
	}

	return r.Rewrite(node)
}

// RewriteFunc rewrites a node hierarchy.
func RewriteFunc(node Node, fn func(Node) Node) Node {
	return Rewrite(rewriterFunc(fn), node)
}

type rewriterFunc func(Node) Node

func (fn rewriterFunc) Rewrite(n Node) Node { return fn(n) }

// Eval evaluates expr against a map.
func Eval(expr Expr, m map[string]interface{}) interface{} {
	if expr == nil {
		return nil
	}

	switch expr := expr.(type) {
	case *BinaryExpr:
		return evalBinaryExpr(expr, m)
	case *BooleanLiteral:
		return expr.Val
	case *NumberLiteral:
		return expr.Val
	case *ParenExpr:
		return Eval(expr.Expr, m)
	case *StringLiteral:
		return expr.Val
	case *VarRef:
		return m[expr.Val]
	default:
		return nil
	}
}

func evalBinaryExpr(expr *BinaryExpr, m map[string]interface{}) interface{} {
	lhs := Eval(expr.LHS, m)
	rhs := Eval(expr.RHS, m)

	// Evaluate if both sides are simple types.
	switch lhs := lhs.(type) {
	case bool:
		rhs, _ := rhs.(bool)
		switch expr.Op {
		case AND:
			return lhs && rhs
		case OR:
			return lhs || rhs
		case EQ:
			return lhs == rhs
		case NEQ:
			return lhs != rhs
		}
	case float64:
		rhs, _ := rhs.(float64)
		switch expr.Op {
		case EQ:
			return lhs == rhs
		case NEQ:
			return lhs != rhs
		case LT:
			return lhs < rhs
		case LTE:
			return lhs <= rhs
		case GT:
			return lhs > rhs
		case GTE:
			return lhs >= rhs
		case ADD:
			return lhs + rhs
		case SUB:
			return lhs - rhs
		case MUL:
			return lhs * rhs
		case DIV:
			if rhs == 0 {
				return float64(0)
			}
			return lhs / rhs
		}
	case int64:
		// we parse all number literals as float 64, so we have to convert from
		// an interface to the float64, then cast to an int64 for comparison
		rhsf, _ := rhs.(float64)
		rhs := int64(rhsf)
		switch expr.Op {
		case EQ:
			return lhs == rhs
		case NEQ:
			return lhs != rhs
		case LT:
			return lhs < rhs
		case LTE:
			return lhs <= rhs
		case GT:
			return lhs > rhs
		case GTE:
			return lhs >= rhs
		case ADD:
			return lhs + rhs
		case SUB:
			return lhs - rhs
		case MUL:
			return lhs * rhs
		case DIV:
			if rhs == 0 {
				return int64(0)
			}
			return lhs / rhs
		}
	case string:
		rhs, _ := rhs.(string)
		switch expr.Op {
		case EQ:
			return lhs == rhs
		case NEQ:
			return lhs != rhs
		}
	}
	return nil
}

// Reduce evaluates expr using the available values in valuer.
// References that don't exist in valuer are ignored.
func Reduce(expr Expr, valuer Valuer) Expr {
	expr = reduce(expr, valuer)

	// Unwrap parens at top level.
	if expr, ok := expr.(*ParenExpr); ok {
		return expr.Expr
	}
	return expr
}

func reduce(expr Expr, valuer Valuer) Expr {
	if expr == nil {
		return nil
	}

	switch expr := expr.(type) {
	case *BinaryExpr:
		return reduceBinaryExpr(expr, valuer)
	case *Call:
		return reduceCall(expr, valuer)
	case *ParenExpr:
		return reduceParenExpr(expr, valuer)
	case *VarRef:
		return reduceVarRef(expr, valuer)
	default:
		return CloneExpr(expr)
	}
}

func reduceBinaryExpr(expr *BinaryExpr, valuer Valuer) Expr {
	// Reduce both sides first.
	op := expr.Op
	lhs := reduce(expr.LHS, valuer)
	rhs := reduce(expr.RHS, valuer)

	// Do not evaluate if one side is nil.
	if lhs == nil || rhs == nil {
		return &BinaryExpr{LHS: lhs, RHS: rhs, Op: expr.Op}
	}

	// If we have a logical operator (AND, OR) and one side is a boolean literal
	// then we need to have special handling.
	if op == AND {
		if isFalseLiteral(lhs) || isFalseLiteral(rhs) {
			return &BooleanLiteral{Val: false}
		} else if isTrueLiteral(lhs) {
			return rhs
		} else if isTrueLiteral(rhs) {
			return lhs
		}
	} else if op == OR {
		if isTrueLiteral(lhs) || isTrueLiteral(rhs) {
			return &BooleanLiteral{Val: true}
		} else if isFalseLiteral(lhs) {
			return rhs
		} else if isFalseLiteral(rhs) {
			return lhs
		}
	}

	// Evaluate if both sides are simple types.
	switch lhs := lhs.(type) {
	case *BooleanLiteral:
		return reduceBinaryExprBooleanLHS(op, lhs, rhs)
	case *DurationLiteral:
		return reduceBinaryExprDurationLHS(op, lhs, rhs)
	case *nilLiteral:
		return reduceBinaryExprNilLHS(op, lhs, rhs)
	case *NumberLiteral:
		return reduceBinaryExprNumberLHS(op, lhs, rhs)
	case *StringLiteral:
		return reduceBinaryExprStringLHS(op, lhs, rhs)
	case *TimeLiteral:
		return reduceBinaryExprTimeLHS(op, lhs, rhs)
	default:
		return &BinaryExpr{Op: op, LHS: lhs, RHS: rhs}
	}
}

func reduceBinaryExprBooleanLHS(op Token, lhs *BooleanLiteral, rhs Expr) Expr {
	switch rhs := rhs.(type) {
	case *BooleanLiteral:
		switch op {
		case EQ:
			return &BooleanLiteral{Val: lhs.Val == rhs.Val}
		case NEQ:
			return &BooleanLiteral{Val: lhs.Val != rhs.Val}
		case AND:
			return &BooleanLiteral{Val: lhs.Val && rhs.Val}
		case OR:
			return &BooleanLiteral{Val: lhs.Val || rhs.Val}
		}
	case *nilLiteral:
		return &BooleanLiteral{Val: false}
	}
	return &BinaryExpr{Op: op, LHS: lhs, RHS: rhs}
}

func reduceBinaryExprDurationLHS(op Token, lhs *DurationLiteral, rhs Expr) Expr {
	switch rhs := rhs.(type) {
	case *DurationLiteral:
		switch op {
		case ADD:
			return &DurationLiteral{Val: lhs.Val + rhs.Val}
		case SUB:
			return &DurationLiteral{Val: lhs.Val - rhs.Val}
		case EQ:
			return &BooleanLiteral{Val: lhs.Val == rhs.Val}
		case NEQ:
			return &BooleanLiteral{Val: lhs.Val != rhs.Val}
		case GT:
			return &BooleanLiteral{Val: lhs.Val > rhs.Val}
		case GTE:
			return &BooleanLiteral{Val: lhs.Val >= rhs.Val}
		case LT:
			return &BooleanLiteral{Val: lhs.Val < rhs.Val}
		case LTE:
			return &BooleanLiteral{Val: lhs.Val <= rhs.Val}
		}
	case *NumberLiteral:
		switch op {
		case MUL:
			return &DurationLiteral{Val: lhs.Val * time.Duration(rhs.Val)}
		case DIV:
			if rhs.Val == 0 {
				return &DurationLiteral{Val: 0}
			}
			return &DurationLiteral{Val: lhs.Val / time.Duration(rhs.Val)}
		}
	case *TimeLiteral:
		switch op {
		case ADD:
			return &TimeLiteral{Val: rhs.Val.Add(lhs.Val)}
		}
	case *nilLiteral:
		return &BooleanLiteral{Val: false}
	}
	return &BinaryExpr{Op: op, LHS: lhs, RHS: rhs}
}

func reduceBinaryExprNilLHS(op Token, lhs *nilLiteral, rhs Expr) Expr {
	switch op {
	case EQ, NEQ:
		return &BooleanLiteral{Val: false}
	}
	return &BinaryExpr{Op: op, LHS: lhs, RHS: rhs}
}

func reduceBinaryExprNumberLHS(op Token, lhs *NumberLiteral, rhs Expr) Expr {
	switch rhs := rhs.(type) {
	case *NumberLiteral:
		switch op {
		case ADD:
			return &NumberLiteral{Val: lhs.Val + rhs.Val}
		case SUB:
			return &NumberLiteral{Val: lhs.Val - rhs.Val}
		case MUL:
			return &NumberLiteral{Val: lhs.Val * rhs.Val}
		case DIV:
			if rhs.Val == 0 {
				return &NumberLiteral{Val: 0}
			}
			return &NumberLiteral{Val: lhs.Val / rhs.Val}
		case EQ:
			return &BooleanLiteral{Val: lhs.Val == rhs.Val}
		case NEQ:
			return &BooleanLiteral{Val: lhs.Val != rhs.Val}
		case GT:
			return &BooleanLiteral{Val: lhs.Val > rhs.Val}
		case GTE:
			return &BooleanLiteral{Val: lhs.Val >= rhs.Val}
		case LT:
			return &BooleanLiteral{Val: lhs.Val < rhs.Val}
		case LTE:
			return &BooleanLiteral{Val: lhs.Val <= rhs.Val}
		}
	case *nilLiteral:
		return &BooleanLiteral{Val: false}
	}
	return &BinaryExpr{Op: op, LHS: lhs, RHS: rhs}
}

func reduceBinaryExprStringLHS(op Token, lhs *StringLiteral, rhs Expr) Expr {
	switch rhs := rhs.(type) {
	case *StringLiteral:
		switch op {
		case EQ:
			return &BooleanLiteral{Val: lhs.Val == rhs.Val}
		case NEQ:
			return &BooleanLiteral{Val: lhs.Val != rhs.Val}
		case ADD:
			return &StringLiteral{Val: lhs.Val + rhs.Val}
		}
	case *nilLiteral:
		switch op {
		case EQ, NEQ:
			return &BooleanLiteral{Val: false}
		}
	}
	return &BinaryExpr{Op: op, LHS: lhs, RHS: rhs}
}

func reduceBinaryExprTimeLHS(op Token, lhs *TimeLiteral, rhs Expr) Expr {
	switch rhs := rhs.(type) {
	case *DurationLiteral:
		switch op {
		case ADD:
			return &TimeLiteral{Val: lhs.Val.Add(rhs.Val)}
		case SUB:
			return &TimeLiteral{Val: lhs.Val.Add(-rhs.Val)}
		}
	case *TimeLiteral:
		switch op {
		case SUB:
			return &DurationLiteral{Val: lhs.Val.Sub(rhs.Val)}
		case EQ:
			return &BooleanLiteral{Val: lhs.Val.Equal(rhs.Val)}
		case NEQ:
			return &BooleanLiteral{Val: !lhs.Val.Equal(rhs.Val)}
		case GT:
			return &BooleanLiteral{Val: lhs.Val.After(rhs.Val)}
		case GTE:
			return &BooleanLiteral{Val: lhs.Val.After(rhs.Val) || lhs.Val.Equal(rhs.Val)}
		case LT:
			return &BooleanLiteral{Val: lhs.Val.Before(rhs.Val)}
		case LTE:
			return &BooleanLiteral{Val: lhs.Val.Before(rhs.Val) || lhs.Val.Equal(rhs.Val)}
		}
	case *nilLiteral:
		return &BooleanLiteral{Val: false}
	}
	return &BinaryExpr{Op: op, LHS: lhs, RHS: rhs}
}

func reduceCall(expr *Call, valuer Valuer) Expr {
	// Evaluate "now()" if valuer is set.
	if expr.Name == "now" && len(expr.Args) == 0 && valuer != nil {
		if v, ok := valuer.Value("now()"); ok {
			v, _ := v.(time.Time)
			return &TimeLiteral{Val: v}
		}
	}

	// Otherwise reduce arguments.
	args := make([]Expr, len(expr.Args))
	for i, arg := range expr.Args {
		args[i] = reduce(arg, valuer)
	}
	return &Call{Name: expr.Name, Args: args}
}

func reduceParenExpr(expr *ParenExpr, valuer Valuer) Expr {
	subexpr := reduce(expr.Expr, valuer)
	if subexpr, ok := subexpr.(*BinaryExpr); ok {
		return &ParenExpr{Expr: subexpr}
	}
	return subexpr
}

func reduceVarRef(expr *VarRef, valuer Valuer) Expr {
	// Ignore if there is no valuer.
	if valuer == nil {
		return &VarRef{Val: expr.Val}
	}

	// Retrieve the value of the ref.
	// Ignore if the value doesn't exist.
	v, ok := valuer.Value(expr.Val)
	if !ok {
		return &VarRef{Val: expr.Val}
	}

	// Return the value as a literal.
	switch v := v.(type) {
	case bool:
		return &BooleanLiteral{Val: v}
	case time.Duration:
		return &DurationLiteral{Val: v}
	case float64:
		return &NumberLiteral{Val: v}
	case string:
		return &StringLiteral{Val: v}
	case time.Time:
		return &TimeLiteral{Val: v}
	default:
		return &nilLiteral{}
	}
}

// Valuer is the interface that wraps the Value() method.
//
// Value returns the value and existence flag for a given key.
type Valuer interface {
	Value(key string) (interface{}, bool)
}

// nowValuer returns only the value for "now()".
type NowValuer struct {
	Now time.Time
}

func (v *NowValuer) Value(key string) (interface{}, bool) {
	if key == "now()" {
		return v.Now, true
	}
	return nil, false
}
