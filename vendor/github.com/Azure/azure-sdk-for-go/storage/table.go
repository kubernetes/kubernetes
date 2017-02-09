package storage

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// TableServiceClient contains operations for Microsoft Azure Table Storage
// Service.
type TableServiceClient struct {
	client Client
	auth   authentication
}

// AzureTable is the typedef of the Azure Table name
type AzureTable string

const (
	tablesURIPath = "/Tables"
)

type createTableRequest struct {
	TableName string `json:"TableName"`
}

// TableAccessPolicy are used for SETTING table policies
type TableAccessPolicy struct {
	ID         string
	StartTime  time.Time
	ExpiryTime time.Time
	CanRead    bool
	CanAppend  bool
	CanUpdate  bool
	CanDelete  bool
}

func pathForTable(table AzureTable) string { return fmt.Sprintf("%s", table) }

func (c *TableServiceClient) getStandardHeaders() map[string]string {
	return map[string]string{
		"x-ms-version":   "2015-02-21",
		"x-ms-date":      currentTimeRfc1123Formatted(),
		"Accept":         "application/json;odata=nometadata",
		"Accept-Charset": "UTF-8",
		"Content-Type":   "application/json",
		userAgentHeader:  c.client.userAgent,
	}
}

// QueryTables returns the tables created in the
// *TableServiceClient storage account.
func (c *TableServiceClient) QueryTables() ([]AzureTable, error) {
	uri := c.client.getEndpoint(tableServiceName, tablesURIPath, url.Values{})

	headers := c.getStandardHeaders()
	headers["Content-Length"] = "0"

	resp, err := c.client.execInternalJSON(http.MethodGet, uri, headers, nil, c.auth)
	if err != nil {
		return nil, err
	}
	defer resp.body.Close()

	if err := checkRespCode(resp.statusCode, []int{http.StatusOK}); err != nil {
		return nil, err
	}

	buf := new(bytes.Buffer)
	if _, err := buf.ReadFrom(resp.body); err != nil {
		return nil, err
	}

	var respArray queryTablesResponse
	if err := json.Unmarshal(buf.Bytes(), &respArray); err != nil {
		return nil, err
	}

	s := make([]AzureTable, len(respArray.TableName))
	for i, elem := range respArray.TableName {
		s[i] = AzureTable(elem.TableName)
	}

	return s, nil
}

// CreateTable creates the table given the specific
// name. This function fails if the name is not compliant
// with the specification or the tables already exists.
func (c *TableServiceClient) CreateTable(table AzureTable) error {
	uri := c.client.getEndpoint(tableServiceName, tablesURIPath, url.Values{})

	headers := c.getStandardHeaders()

	req := createTableRequest{TableName: string(table)}
	buf := new(bytes.Buffer)

	if err := json.NewEncoder(buf).Encode(req); err != nil {
		return err
	}

	headers["Content-Length"] = fmt.Sprintf("%d", buf.Len())

	resp, err := c.client.execInternalJSON(http.MethodPost, uri, headers, buf, c.auth)

	if err != nil {
		return err
	}
	defer resp.body.Close()

	if err := checkRespCode(resp.statusCode, []int{http.StatusCreated}); err != nil {
		return err
	}

	return nil
}

// DeleteTable deletes the table given the specific
// name. This function fails if the table is not present.
// Be advised: DeleteTable deletes all the entries
// that may be present.
func (c *TableServiceClient) DeleteTable(table AzureTable) error {
	uri := c.client.getEndpoint(tableServiceName, tablesURIPath, url.Values{})
	uri += fmt.Sprintf("('%s')", string(table))

	headers := c.getStandardHeaders()

	headers["Content-Length"] = "0"

	resp, err := c.client.execInternalJSON(http.MethodDelete, uri, headers, nil, c.auth)

	if err != nil {
		return err
	}
	defer resp.body.Close()

	if err := checkRespCode(resp.statusCode, []int{http.StatusNoContent}); err != nil {
		return err

	}
	return nil
}

// SetTablePermissions sets up table ACL permissions as per REST details https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/Set-Table-ACL
func (c *TableServiceClient) SetTablePermissions(table AzureTable, policies []TableAccessPolicy, timeout uint) (err error) {
	params := url.Values{"comp": {"acl"}}

	if timeout > 0 {
		params.Add("timeout", fmt.Sprint(timeout))
	}

	uri := c.client.getEndpoint(tableServiceName, string(table), params)
	headers := c.client.getStandardHeaders()

	body, length, err := generateTableACLPayload(policies)
	if err != nil {
		return err
	}
	headers["Content-Length"] = fmt.Sprintf("%v", length)

	resp, err := c.client.execInternalJSON(http.MethodPut, uri, headers, body, c.auth)
	if err != nil {
		return err
	}
	defer resp.body.Close()

	if err := checkRespCode(resp.statusCode, []int{http.StatusNoContent}); err != nil {
		return err
	}
	return nil
}

func generateTableACLPayload(policies []TableAccessPolicy) (io.Reader, int, error) {
	sil := SignedIdentifiers{
		SignedIdentifiers: []SignedIdentifier{},
	}
	for _, tap := range policies {
		permission := generateTablePermissions(&tap)
		signedIdentifier := convertAccessPolicyToXMLStructs(tap.ID, tap.StartTime, tap.ExpiryTime, permission)
		sil.SignedIdentifiers = append(sil.SignedIdentifiers, signedIdentifier)
	}
	return xmlMarshal(sil)
}

// GetTablePermissions gets the table ACL permissions, as per REST details https://docs.microsoft.com/en-us/rest/api/storageservices/fileservices/get-table-acl
func (c *TableServiceClient) GetTablePermissions(table AzureTable, timeout int) (permissionResponse []TableAccessPolicy, err error) {
	params := url.Values{"comp": {"acl"}}

	if timeout > 0 {
		params.Add("timeout", strconv.Itoa(timeout))
	}

	uri := c.client.getEndpoint(tableServiceName, string(table), params)
	headers := c.client.getStandardHeaders()
	resp, err := c.client.execInternalJSON(http.MethodGet, uri, headers, nil, c.auth)
	if err != nil {
		return nil, err
	}
	defer resp.body.Close()

	if err = checkRespCode(resp.statusCode, []int{http.StatusOK}); err != nil {
		return nil, err
	}

	var ap AccessPolicy
	err = xmlUnmarshal(resp.body, &ap.SignedIdentifiersList)
	if err != nil {
		return nil, err
	}
	out := updateTableAccessPolicy(ap)
	return out, nil
}

func updateTableAccessPolicy(ap AccessPolicy) []TableAccessPolicy {
	out := []TableAccessPolicy{}
	for _, policy := range ap.SignedIdentifiersList.SignedIdentifiers {
		tap := TableAccessPolicy{
			ID:         policy.ID,
			StartTime:  policy.AccessPolicy.StartTime,
			ExpiryTime: policy.AccessPolicy.ExpiryTime,
		}
		tap.CanRead = updatePermissions(policy.AccessPolicy.Permission, "r")
		tap.CanAppend = updatePermissions(policy.AccessPolicy.Permission, "a")
		tap.CanUpdate = updatePermissions(policy.AccessPolicy.Permission, "u")
		tap.CanDelete = updatePermissions(policy.AccessPolicy.Permission, "d")

		out = append(out, tap)
	}
	return out
}

func generateTablePermissions(tap *TableAccessPolicy) (permissions string) {
	// generate the permissions string (raud).
	// still want the end user API to have bool flags.
	permissions = ""

	if tap.CanRead {
		permissions += "r"
	}

	if tap.CanAppend {
		permissions += "a"
	}

	if tap.CanUpdate {
		permissions += "u"
	}

	if tap.CanDelete {
		permissions += "d"
	}
	return permissions
}
