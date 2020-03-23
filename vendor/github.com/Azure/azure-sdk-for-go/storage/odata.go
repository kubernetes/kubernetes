package storage

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

// MetadataLevel determines if operations should return a paylod,
// and it level of detail.
type MetadataLevel string

// This consts are meant to help with Odata supported operations
const (
	OdataTypeSuffix = "@odata.type"

	// Types

	OdataBinary   = "Edm.Binary"
	OdataDateTime = "Edm.DateTime"
	OdataDouble   = "Edm.Double"
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
