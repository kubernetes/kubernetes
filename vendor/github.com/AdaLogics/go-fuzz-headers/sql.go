package gofuzzheaders

import (
	"fmt"
	"strings"
)

// returns a keyword by index
func getKeyword(f *ConsumeFuzzer) (string, error) {
	index, err := f.GetInt()
	if err != nil {
		return keywords[0], err
	}
	for i, k := range keywords {
		if i == index {
			return k, nil
		}
	}
	return keywords[0], fmt.Errorf("Could not get a kw")
}

// Simple utility function to check if a string
// slice contains a string.
func containsString(s []string, e string) bool {
	for _, a := range s {
		if a == e {
			return true
		}
	}
	return false
}

// These keywords are used specifically for fuzzing Vitess
var keywords = []string{
	"accessible", "action", "add", "after", "against", "algorithm",
	"all", "alter", "always", "analyze", "and", "as", "asc", "asensitive",
	"auto_increment", "avg_row_length", "before", "begin", "between",
	"bigint", "binary", "_binary", "_utf8mb4", "_utf8", "_latin1", "bit",
	"blob", "bool", "boolean", "both", "by", "call", "cancel", "cascade",
	"cascaded", "case", "cast", "channel", "change", "char", "character",
	"charset", "check", "checksum", "coalesce", "code", "collate", "collation",
	"column", "columns", "comment", "committed", "commit", "compact", "complete",
	"compressed", "compression", "condition", "connection", "constraint", "continue",
	"convert", "copy", "cume_dist", "substr", "substring", "create", "cross",
	"csv", "current_date", "current_time", "current_timestamp", "current_user",
	"cursor", "data", "database", "databases", "day", "day_hour", "day_microsecond",
	"day_minute", "day_second", "date", "datetime", "dec", "decimal", "declare",
	"default", "definer", "delay_key_write", "delayed", "delete", "dense_rank",
	"desc", "describe", "deterministic", "directory", "disable", "discard",
	"disk", "distinct", "distinctrow", "div", "double", "do", "drop", "dumpfile",
	"duplicate", "dynamic", "each", "else", "elseif", "empty", "enable",
	"enclosed", "encryption", "end", "enforced", "engine", "engines", "enum",
	"error", "escape", "escaped", "event", "exchange", "exclusive", "exists",
	"exit", "explain", "expansion", "export", "extended", "extract", "false",
	"fetch", "fields", "first", "first_value", "fixed", "float", "float4",
	"float8", "flush", "for", "force", "foreign", "format", "from", "full",
	"fulltext", "function", "general", "generated", "geometry", "geometrycollection",
	"get", "global", "gtid_executed", "grant", "group", "grouping", "groups",
	"group_concat", "having", "header", "high_priority", "hosts", "hour", "hour_microsecond",
	"hour_minute", "hour_second", "if", "ignore", "import", "in", "index", "indexes",
	"infile", "inout", "inner", "inplace", "insensitive", "insert", "insert_method",
	"int", "int1", "int2", "int3", "int4", "int8", "integer", "interval",
	"into", "io_after_gtids", "is", "isolation", "iterate", "invoker", "join",
	"json", "json_table", "key", "keys", "keyspaces", "key_block_size", "kill", "lag",
	"language", "last", "last_value", "last_insert_id", "lateral", "lead", "leading",
	"leave", "left", "less", "level", "like", "limit", "linear", "lines",
	"linestring", "load", "local", "localtime", "localtimestamp", "lock", "logs",
	"long", "longblob", "longtext", "loop", "low_priority", "manifest",
	"master_bind", "match", "max_rows", "maxvalue", "mediumblob", "mediumint",
	"mediumtext", "memory", "merge", "microsecond", "middleint", "min_rows", "minute",
	"minute_microsecond", "minute_second", "mod", "mode", "modify", "modifies",
	"multilinestring", "multipoint", "multipolygon", "month", "name",
	"names", "natural", "nchar", "next", "no", "none", "not", "no_write_to_binlog",
	"nth_value", "ntile", "null", "numeric", "of", "off", "offset", "on",
	"only", "open", "optimize", "optimizer_costs", "option", "optionally",
	"or", "order", "out", "outer", "outfile", "over", "overwrite", "pack_keys",
	"parser", "partition", "partitioning", "password", "percent_rank", "plugins",
	"point", "polygon", "precision", "primary", "privileges", "processlist",
	"procedure", "query", "quarter", "range", "rank", "read", "reads", "read_write",
	"real", "rebuild", "recursive", "redundant", "references", "regexp", "relay",
	"release", "remove", "rename", "reorganize", "repair", "repeat", "repeatable",
	"replace", "require", "resignal", "restrict", "return", "retry", "revert",
	"revoke", "right", "rlike", "rollback", "row", "row_format", "row_number",
	"rows", "s3", "savepoint", "schema", "schemas", "second", "second_microsecond",
	"security", "select", "sensitive", "separator", "sequence", "serializable",
	"session", "set", "share", "shared", "show", "signal", "signed", "slow",
	"smallint", "spatial", "specific", "sql", "sqlexception", "sqlstate",
	"sqlwarning", "sql_big_result", "sql_cache", "sql_calc_found_rows",
	"sql_no_cache", "sql_small_result", "ssl", "start", "starting",
	"stats_auto_recalc", "stats_persistent", "stats_sample_pages", "status",
	"storage", "stored", "straight_join", "stream", "system", "vstream",
	"table", "tables", "tablespace", "temporary", "temptable", "terminated",
	"text", "than", "then", "time", "timestamp", "timestampadd", "timestampdiff",
	"tinyblob", "tinyint", "tinytext", "to", "trailing", "transaction", "tree",
	"traditional", "trigger", "triggers", "true", "truncate", "uncommitted",
	"undefined", "undo", "union", "unique", "unlock", "unsigned", "update",
	"upgrade", "usage", "use", "user", "user_resources", "using", "utc_date",
	"utc_time", "utc_timestamp", "validation", "values", "variables", "varbinary",
	"varchar", "varcharacter", "varying", "vgtid_executed", "virtual", "vindex",
	"vindexes", "view", "vitess", "vitess_keyspaces", "vitess_metadata",
	"vitess_migration", "vitess_migrations", "vitess_replication_status",
	"vitess_shards", "vitess_tablets", "vschema", "warnings", "when",
	"where", "while", "window", "with", "without", "work", "write", "xor",
	"year", "year_month", "zerofill"}

// Keywords that could get an additional keyword
var needCustomString = []string{
	"DISTINCTROW", "FROM", // Select keywords:
	"GROUP BY", "HAVING", "WINDOW",
	"FOR",
	"ORDER BY", "LIMIT",
	"INTO", "PARTITION", "AS", // Insert Keywords:
	"ON DUPLICATE KEY UPDATE",
	"WHERE", "LIMIT", // Delete keywords
	"INFILE", "INTO TABLE", "CHARACTER SET", // Load keywords
	"TERMINATED BY", "ENCLOSED BY",
	"ESCAPED BY", "STARTING BY",
	"TERMINATED BY", "STARTING BY",
	"IGNORE",
	"VALUE", "VALUES", // Replace tokens
	"SET",                                   // Update tokens
	"ENGINE =",                              // Drop tokens
	"DEFINER =", "ON SCHEDULE", "RENAME TO", // Alter tokens
	"COMMENT", "DO", "INITIAL_SIZE = ", "OPTIONS",
}

var alterTableTokens = [][]string{
	{"CUSTOM_FUZZ_STRING"},
	{"CUSTOM_ALTTER_TABLE_OPTIONS"},
	{"PARTITION_OPTIONS_FOR_ALTER_TABLE"},
}

var alterTokens = [][]string{
	{"DATABASE", "SCHEMA", "DEFINER = ", "EVENT", "FUNCTION", "INSTANCE",
		"LOGFILE GROUP", "PROCEDURE", "SERVER"},
	{"CUSTOM_FUZZ_STRING"},
	{"ON SCHEDULE", "ON COMPLETION PRESERVE", "ON COMPLETION NOT PRESERVE",
		"ADD UNDOFILE", "OPTIONS"},
	{"RENAME TO", "INITIAL_SIZE = "},
	{"ENABLE", "DISABLE", "DISABLE ON SLAVE", "ENGINE"},
	{"COMMENT"},
	{"DO"},
}

var setTokens = [][]string{
	{"CHARACTER SET", "CHARSET", "CUSTOM_FUZZ_STRING", "NAMES"},
	{"CUSTOM_FUZZ_STRING", "DEFAULT", "="},
	{"CUSTOM_FUZZ_STRING"},
}

var dropTokens = [][]string{
	{"TEMPORARY", "UNDO"},
	{"DATABASE", "SCHEMA", "EVENT", "INDEX", "LOGFILE GROUP",
		"PROCEDURE", "FUNCTION", "SERVER", "SPATIAL REFERENCE SYSTEM",
		"TABLE", "TABLESPACE", "TRIGGER", "VIEW"},
	{"IF EXISTS"},
	{"CUSTOM_FUZZ_STRING"},
	{"ON", "ENGINE = ", "RESTRICT", "CASCADE"},
}

var renameTokens = [][]string{
	{"TABLE"},
	{"CUSTOM_FUZZ_STRING"},
	{"TO"},
	{"CUSTOM_FUZZ_STRING"},
}

var truncateTokens = [][]string{
	{"TABLE"},
	{"CUSTOM_FUZZ_STRING"},
}

var createTokens = [][]string{
	{"OR REPLACE", "TEMPORARY", "UNDO"}, // For create spatial reference system
	{"UNIQUE", "FULLTEXT", "SPATIAL", "ALGORITHM = UNDEFINED", "ALGORITHM = MERGE",
		"ALGORITHM = TEMPTABLE"},
	{"DATABASE", "SCHEMA", "EVENT", "FUNCTION", "INDEX", "LOGFILE GROUP",
		"PROCEDURE", "SERVER", "SPATIAL REFERENCE SYSTEM", "TABLE", "TABLESPACE",
		"TRIGGER", "VIEW"},
	{"IF NOT EXISTS"},
	{"CUSTOM_FUZZ_STRING"},
}

var updateTokens = [][]string{
	{"LOW_PRIORITY"},
	{"IGNORE"},
	{"SET"},
	{"WHERE"},
	{"ORDER BY"},
	{"LIMIT"},
}
var replaceTokens = [][]string{
	{"LOW_PRIORITY", "DELAYED"},
	{"INTO"},
	{"PARTITION"},
	{"CUSTOM_FUZZ_STRING"},
	{"VALUES", "VALUE"},
}
var loadTokens = [][]string{
	{"DATA"},
	{"LOW_PRIORITY", "CONCURRENT", "LOCAL"},
	{"INFILE"},
	{"REPLACE", "IGNORE"},
	{"INTO TABLE"},
	{"PARTITION"},
	{"CHARACTER SET"},
	{"FIELDS", "COLUMNS"},
	{"TERMINATED BY"},
	{"OPTIONALLY"},
	{"ENCLOSED BY"},
	{"ESCAPED BY"},
	{"LINES"},
	{"STARTING BY"},
	{"TERMINATED BY"},
	{"IGNORE"},
	{"LINES", "ROWS"},
	{"CUSTOM_FUZZ_STRING"},
}

// These Are everything that comes after "INSERT"
var insertTokens = [][]string{
	{"LOW_PRIORITY", "DELAYED", "HIGH_PRIORITY", "IGNORE"},
	{"INTO"},
	{"PARTITION"},
	{"CUSTOM_FUZZ_STRING"},
	{"AS"},
	{"ON DUPLICATE KEY UPDATE"},
}

// These are everything that comes after "SELECT"
var selectTokens = [][]string{
	{"*", "CUSTOM_FUZZ_STRING", "DISTINCTROW"},
	{"HIGH_PRIORITY"},
	{"STRAIGHT_JOIN"},
	{"SQL_SMALL_RESULT", "SQL_BIG_RESULT", "SQL_BUFFER_RESULT"},
	{"SQL_NO_CACHE", "SQL_CALC_FOUND_ROWS"},
	{"CUSTOM_FUZZ_STRING"},
	{"FROM"},
	{"WHERE"},
	{"GROUP BY"},
	{"HAVING"},
	{"WINDOW"},
	{"ORDER BY"},
	{"LIMIT"},
	{"CUSTOM_FUZZ_STRING"},
	{"FOR"},
}

// These are everything that comes after "DELETE"
var deleteTokens = [][]string{
	{"LOW_PRIORITY", "QUICK", "IGNORE", "FROM", "AS"},
	{"PARTITION"},
	{"WHERE"},
	{"ORDER BY"},
	{"LIMIT"},
}

var alter_table_options = []string{
	"ADD", "COLUMN", "FIRST", "AFTER", "INDEX", "KEY", "FULLTEXT", "SPATIAL",
	"CONSTRAINT", "UNIQUE", "FOREIGN KEY", "CHECK", "ENFORCED", "DROP", "ALTER",
	"NOT", "INPLACE", "COPY", "SET", "VISIBLE", "INVISIBLE", "DEFAULT", "CHANGE",
	"CHARACTER SET", "COLLATE", "DISABLE", "ENABLE", "KEYS", "TABLESPACE", "LOCK",
	"FORCE", "MODIFY", "SHARED", "EXCLUSIVE", "NONE", "ORDER BY", "RENAME COLUMN",
	"AS", "=", "ASC", "DESC", "WITH", "WITHOUT", "VALIDATION", "ADD PARTITION",
	"DROP PARTITION", "DISCARD PARTITION", "IMPORT PARTITION", "TRUNCATE PARTITION",
	"COALESCE PARTITION", "REORGANIZE PARTITION", "EXCHANGE PARTITION",
	"ANALYZE PARTITION", "CHECK PARTITION", "OPTIMIZE PARTITION", "REBUILD PARTITION",
	"REPAIR PARTITION", "REMOVE PARTITIONING", "USING", "BTREE", "HASH", "COMMENT",
	"KEY_BLOCK_SIZE", "WITH PARSER", "AUTOEXTEND_SIZE", "AUTO_INCREMENT", "AVG_ROW_LENGTH",
	"CHECKSUM", "INSERT_METHOD", "ROW_FORMAT", "DYNAMIC", "FIXED", "COMPRESSED", "REDUNDANT",
	"COMPACT", "SECONDARY_ENGINE_ATTRIBUTE", "STATS_AUTO_RECALC", "STATS_PERSISTENT",
	"STATS_SAMPLE_PAGES", "ZLIB", "LZ4", "ENGINE_ATTRIBUTE", "KEY_BLOCK_SIZE", "MAX_ROWS",
	"MIN_ROWS", "PACK_KEYS", "PASSWORD", "COMPRESSION", "CONNECTION", "DIRECTORY",
	"DELAY_KEY_WRITE", "ENCRYPTION", "STORAGE", "DISK", "MEMORY", "UNION"}

// Creates an 'alter table' statement. 'alter table' is an exception
// in that it has its own function. The majority of statements
// are created by 'createStmt()'.
func createAlterTableStmt(f *ConsumeFuzzer) (string, error) {
	var stmt strings.Builder
	stmt.WriteString("ALTER TABLE ")
	maxArgs, err := f.GetInt()
	if err != nil {
		return "", err
	}
	maxArgs = maxArgs % 30
	if maxArgs == 0 {
		return "", fmt.Errorf("Could not create alter table stmt")
	}
	for i := 0; i < maxArgs; i++ {
		// Calculate if we get existing token or custom string
		tokenType, err := f.GetInt()
		if err != nil {
			return "", err
		}
		if tokenType%4 == 1 {
			customString, err := f.GetString()
			if err != nil {
				return "", err
			}
			stmt.WriteString(fmt.Sprintf(" %s", customString))
		} else {
			tokenIndex, err := f.GetInt()
			if err != nil {
				return "", err
			}
			stmt.WriteString(fmt.Sprintf(" %s", alter_table_options[tokenIndex%len(alter_table_options)]))
		}
	}
	return stmt.String(), nil
}

func chooseToken(tokens []string, f *ConsumeFuzzer) (string, error) {
	index, err := f.GetInt()
	if err != nil {
		return "", err
	}
	var token strings.Builder
	token.WriteString(fmt.Sprintf(" %s", tokens[index%len(tokens)]))
	if token.String() == "CUSTOM_FUZZ_STRING" {
		customFuzzString, err := f.GetString()
		if err != nil {
			return "", err
		}
		return customFuzzString, nil
	}

	// Check if token requires an argument
	if containsString(needCustomString, token.String()) {
		customFuzzString, err := f.GetString()
		if err != nil {
			return "", err
		}
		token.WriteString(fmt.Sprintf(" %s", customFuzzString))
	}
	return token.String(), nil
}

var stmtTypes = map[string][][]string{
	"DELETE":      deleteTokens,
	"INSERT":      insertTokens,
	"SELECT":      selectTokens,
	"LOAD":        loadTokens,
	"REPLACE":     replaceTokens,
	"CREATE":      createTokens,
	"DROP":        dropTokens,
	"RENAME":      renameTokens,
	"TRUNCATE":    truncateTokens,
	"SET":         setTokens,
	"ALTER":       alterTokens,
	"ALTER TABLE": alterTableTokens, // ALTER TABLE has its own set of tokens
}

var stmtTypeEnum = map[int]string{
	0:  "DELETE",
	1:  "INSERT",
	2:  "SELECT",
	3:  "LOAD",
	4:  "REPLACE",
	5:  "CREATE",
	6:  "DROP",
	7:  "RENAME",
	8:  "TRUNCATE",
	9:  "SET",
	10: "ALTER",
	11: "ALTER TABLE",
}

func createStmt(f *ConsumeFuzzer) (string, error) {
	stmtIndex, err := f.GetInt()
	if err != nil {
		return "", err
	}
	stmtIndex = stmtIndex % len(stmtTypes)

	queryType := stmtTypeEnum[stmtIndex]
	tokens := stmtTypes[queryType]

	// We have custom creator for ALTER TABLE
	if queryType == "ALTER TABLE" {
		query, err := createAlterTableStmt(f)
		if err != nil {
			return "", err
		}
		return query, nil
	}

	// Here we are creating a query that is not
	// an 'alter table' query. For available
	// queries, see "stmtTypes"

	// First specify the first query keyword:
	var query strings.Builder
	query.WriteString(queryType)

	// Next create the args for the
	queryArgs, err := createStmtArgs(tokens, f)
	if err != nil {
		return "", err
	}
	query.WriteString(fmt.Sprintf(" %s", queryArgs))
	return query.String(), nil
}

// Creates the arguments of a statements. In a select statement
// that would be everything after "select".
func createStmtArgs(tokenslice [][]string, f *ConsumeFuzzer) (string, error) {
	var query strings.Builder
	var token strings.Builder

	// We go through the tokens in the tokenslice,
	// create the respective token and add it to
	// "query"
	for _, tokens := range tokenslice {

		// For extra randomization, the fuzzer can
		// choose to not include this token.
		includeThisToken, err := f.GetBool()
		if err != nil {
			return "", err
		}
		if !includeThisToken {
			continue
		}

		// There may be several tokens to choose from:
		if len(tokens) > 1 {
			chosenToken, err := chooseToken(tokens, f)
			if err != nil {
				return "", err
			}
			query.WriteString(fmt.Sprintf(" %s", chosenToken))
		} else {
			token.WriteString(tokens[0])

			// In case the token is "CUSTOM_FUZZ_STRING"
			// we will then create a non-structured string
			if token.String() == "CUSTOM_FUZZ_STRING" {
				customFuzzString, err := f.GetString()
				if err != nil {
					return "", err
				}
				query.WriteString(fmt.Sprintf(" %s", customFuzzString))
				continue
			}

			// Check if token requires an argument.
			// Tokens that take an argument can be found
			// in 'needCustomString'. If so, we add a
			// non-structured string to the token.
			if containsString(needCustomString, token.String()) {
				customFuzzString, err := f.GetString()
				if err != nil {
					return "", err
				}
				token.WriteString(fmt.Sprintf(" %s", customFuzzString))
			}
			query.WriteString(fmt.Sprintf(" %s", token.String()))
		}
	}
	return query.String(), nil
}

// Creates a semi-structured query. It creates a string
// that is a combination of the keywords and random strings.
func createQuery(f *ConsumeFuzzer) (string, error) {
	queryLen, err := f.GetInt()
	if err != nil {
		return "", err
	}
	maxLen := queryLen % 60
	if maxLen == 0 {
		return "", fmt.Errorf("Could not create a query")
	}
	var query strings.Builder
	for i := 0; i < maxLen; i++ {
		// Get a new token:
		useKeyword, err := f.GetBool()
		if err != nil {
			return "", err
		}
		if useKeyword {
			keyword, err := getKeyword(f)
			if err != nil {
				return "", err
			}
			query.WriteString(fmt.Sprintf(" %s", keyword))
		} else {
			customString, err := f.GetString()
			if err != nil {
				return "", err
			}
			query.WriteString(fmt.Sprintf(" %s", customString))
		}
	}
	if query.String() == "" {
		return "", fmt.Errorf("Could not create a query")
	}
	return query.String(), nil
}

// This is the API that users will interact with.
// Usage:
// f := NewConsumer(data)
// sqlString, err := f.GetSQLString()
func (f *ConsumeFuzzer) GetSQLString() (string, error) {
	var query string
	veryStructured, err := f.GetBool()
	if err != nil {
		return "", err
	}
	if veryStructured {
		query, err = createStmt(f)
		if err != nil {
			return "", err
		}
	} else {
		query, err = createQuery(f)
		if err != nil {
			return "", err
		}
	}
	return query, nil
}
