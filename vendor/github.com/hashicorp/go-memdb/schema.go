package memdb

import "fmt"

// DBSchema contains the full database schema used for MemDB
type DBSchema struct {
	Tables map[string]*TableSchema
}

// Validate is used to validate the database schema
func (s *DBSchema) Validate() error {
	if s == nil {
		return fmt.Errorf("missing schema")
	}
	if len(s.Tables) == 0 {
		return fmt.Errorf("no tables defined")
	}
	for name, table := range s.Tables {
		if name != table.Name {
			return fmt.Errorf("table name mis-match for '%s'", name)
		}
		if err := table.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// TableSchema contains the schema for a single table
type TableSchema struct {
	Name    string
	Indexes map[string]*IndexSchema
}

// Validate is used to validate the table schema
func (s *TableSchema) Validate() error {
	if s.Name == "" {
		return fmt.Errorf("missing table name")
	}
	if len(s.Indexes) == 0 {
		return fmt.Errorf("missing table schemas for '%s'", s.Name)
	}
	if _, ok := s.Indexes["id"]; !ok {
		return fmt.Errorf("must have id index")
	}
	if !s.Indexes["id"].Unique {
		return fmt.Errorf("id index must be unique")
	}
	for name, index := range s.Indexes {
		if name != index.Name {
			return fmt.Errorf("index name mis-match for '%s'", name)
		}
		if err := index.Validate(); err != nil {
			return err
		}
	}
	return nil
}

// IndexSchema contains the schema for an index
type IndexSchema struct {
	Name         string
	AllowMissing bool
	Unique       bool
	Indexer      Indexer
}

func (s *IndexSchema) Validate() error {
	if s.Name == "" {
		return fmt.Errorf("missing index name")
	}
	if s.Indexer == nil {
		return fmt.Errorf("missing index function for '%s'", s.Name)
	}
	return nil
}
