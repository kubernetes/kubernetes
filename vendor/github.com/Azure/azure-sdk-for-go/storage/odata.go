package storage

// MetadataLevel determines if operations should return a paylod,
// and it level of detail.
type MetadataLevel string

// This consts are meant to help with Odata supported operations
const (
	OdataTypeSuffix = "@odata.type"

	// Types

	OdataBinary   = "Edm.Binary"
	OdataDateTime = "Edm.DateTime"
	OdataGUID     = "Edm.Guid"
	OdataInt64    = "Edm.Int64"

	// Query options

	OdataFilter  = "$filter"
	OdataOrderBy = "$orderby"
	OdataTop     = "$top"
	OdataSkip    = "$skip"
	OdataCount   = "$count"
	OdataExpand  = "$expand"
	OdataSelect  = "$select"
	OdataSearch  = "$search"

	EmptyPayload    MetadataLevel = ""
	NoMetadata      MetadataLevel = "application/json;odata=nometadata"
	MinimalMetadata MetadataLevel = "application/json;odata=minimalmetadata"
	FullMetadata    MetadataLevel = "application/json;odata=fullmetadata"
)
