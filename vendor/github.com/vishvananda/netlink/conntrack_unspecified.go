// +build !linux

package netlink

// ConntrackTableType Conntrack table for the netlink operation
type ConntrackTableType uint8

// InetFamily Family type
type InetFamily uint8

// ConntrackFlow placeholder
type ConntrackFlow struct{}

// CustomConntrackFilter placeholder
type CustomConntrackFilter struct{}

// ConntrackFilter placeholder
type ConntrackFilter struct{}

// ConntrackTableList returns the flow list of a table of a specific family
// conntrack -L [table] [options]          List conntrack or expectation table
func ConntrackTableList(table ConntrackTableType, family InetFamily) ([]*ConntrackFlow, error) {
	return nil, ErrNotImplemented
}

// ConntrackTableFlush flushes all the flows of a specified table
// conntrack -F [table]            Flush table
// The flush operation applies to all the family types
func ConntrackTableFlush(table ConntrackTableType) error {
	return ErrNotImplemented
}

// ConntrackDeleteFilter deletes entries on the specified table on the base of the filter
// conntrack -D [table] parameters         Delete conntrack or expectation
//
// Deprecated: use [ConntrackDeleteFilter] instead.
func ConntrackDeleteFilter(table ConntrackTableType, family InetFamily, filter *ConntrackFilter) (uint, error) {
	return 0, ErrNotImplemented
}

// ConntrackDeleteFilters deletes entries on the specified table matching any of the specified filters
// conntrack -D [table] parameters         Delete conntrack or expectation
func ConntrackDeleteFilters(table ConntrackTableType, family InetFamily, filters ...CustomConntrackFilter) (uint, error) {
	return 0, ErrNotImplemented
}

// ConntrackTableList returns the flow list of a table of a specific family using the netlink handle passed
// conntrack -L [table] [options]          List conntrack or expectation table
func (h *Handle) ConntrackTableList(table ConntrackTableType, family InetFamily) ([]*ConntrackFlow, error) {
	return nil, ErrNotImplemented
}

// ConntrackTableFlush flushes all the flows of a specified table using the netlink handle passed
// conntrack -F [table]            Flush table
// The flush operation applies to all the family types
func (h *Handle) ConntrackTableFlush(table ConntrackTableType) error {
	return ErrNotImplemented
}

// ConntrackDeleteFilter deletes entries on the specified table on the base of the filter using the netlink handle passed
// conntrack -D [table] parameters         Delete conntrack or expectation
//
// Deprecated: use [Handle.ConntrackDeleteFilters] instead.
func (h *Handle) ConntrackDeleteFilter(table ConntrackTableType, family InetFamily, filter *ConntrackFilter) (uint, error) {
	return 0, ErrNotImplemented
}

// ConntrackDeleteFilters deletes entries on the specified table matching any of the specified filters using the netlink handle passed
// conntrack -D [table] parameters         Delete conntrack or expectation
func (h *Handle) ConntrackDeleteFilters(table ConntrackTableType, family InetFamily, filters ...CustomConntrackFilter) (uint, error) {
	return 0, ErrNotImplemented
}
