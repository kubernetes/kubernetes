package dnspod

import (
	"fmt"
)

type Record struct {
	ID            string `json:"id,omitempty"`
	Name          string `json:"name,omitempty"`
	Line          string `json:"line,omitempty"`
	LineID        string `json:"line_id,omitempty"`
	Type          string `json:"type,omitempty"`
	TTL           string `json:"ttl,omitempty"`
	Value         string `json:"value,omitempty"`
	MX            string `json:"mx,omitempty"`
	Enabled       string `json:"enabled,omitempty"`
	Status        string `json:"status,omitempty"`
	MonitorStatus string `json:"monitor_status,omitempty"`
	Remark        string `json:"remark,omitempty"`
	UpdateOn      string `json:"updated_on,omitempty"`
	UseAQB        string `json:"use_aqb,omitempty"`
}

type recordsWrapper struct {
	Status  Status     `json:"status"`
	Info    DomainInfo `json:"info"`
	Records []Record   `json:"records"`
}

type recordWrapper struct {
	Status Status     `json:"status"`
	Info   DomainInfo `json:"info"`
	Record Record     `json:"record"`
}

// recordAction generates the resource path for given record that belongs to a domain.
func recordAction(action string) string {
	if len(action) > 0 {
		return fmt.Sprintf("Record.%s", action)
	}
	return "Record.List"
}

// List the domain records.
//
// dnspod API docs: https://www.dnspod.cn/docs/records.html#record-list
func (s *DomainsService) ListRecords(domain string, recordName string) ([]Record, *Response, error) {
	path := recordAction("List")

	payload := newPayLoad(s.client.CommonParams)

	payload.Add("domain_id", domain)

	if recordName != "" {
		payload.Add("sub_domain", recordName)
	}

	wrappedRecords := recordsWrapper{}

	res, err := s.client.post(path, payload, &wrappedRecords)
	if err != nil {
		return []Record{}, res, err
	}

	if wrappedRecords.Status.Code != "1" {
		return wrappedRecords.Records, nil, fmt.Errorf("Could not get domains: %s", wrappedRecords.Status.Message)
	}

	records := []Record{}
	for _, record := range wrappedRecords.Records {
		records = append(records, record)
	}

	return records, res, nil
}

// CreateRecord creates a domain record.
//
// dnspod API docs: https://www.dnspod.cn/docs/records.html#record-create
func (s *DomainsService) CreateRecord(domain string, recordAttributes Record) (Record, *Response, error) {
	path := recordAction("Create")

	payload := newPayLoad(s.client.CommonParams)

	payload.Add("domain_id", domain)

	if recordAttributes.Name != "" {
		payload.Add("sub_domain", recordAttributes.Name)
	}

	if recordAttributes.Type != "" {
		payload.Add("record_type", recordAttributes.Type)
	}

	if recordAttributes.Line != "" {
		payload.Add("record_line", recordAttributes.Line)
	}

	if recordAttributes.LineID != "" {
		payload.Add("record_line_id", recordAttributes.LineID)
	}

	if recordAttributes.Value != "" {
		payload.Add("value", recordAttributes.Value)
	}

	if recordAttributes.MX != "" {
		payload.Add("mx", recordAttributes.MX)
	}

	if recordAttributes.TTL != "" {
		payload.Add("ttl", recordAttributes.TTL)
	}

	if recordAttributes.Status != "" {
		payload.Add("status", recordAttributes.Status)
	}

	returnedRecord := recordWrapper{}

	res, err := s.client.post(path, payload, &returnedRecord)
	if err != nil {
		return Record{}, res, err
	}

	if returnedRecord.Status.Code != "1" {
		return returnedRecord.Record, nil, fmt.Errorf("Could not get domains: %s", returnedRecord.Status.Message)
	}

	return returnedRecord.Record, res, nil
}

// GetRecord fetches the domain record.
//
// dnspod API docs: https://www.dnspod.cn/docs/records.html#record-info
func (s *DomainsService) GetRecord(domain string, recordID string) (Record, *Response, error) {
	path := recordAction("Info")

	payload := newPayLoad(s.client.CommonParams)

	payload.Add("domain_id", domain)
	payload.Add("record_id", recordID)

	returnedRecord := recordWrapper{}

	res, err := s.client.post(path, payload, &returnedRecord)
	if err != nil {
		return Record{}, res, err
	}

	if returnedRecord.Status.Code != "1" {
		return returnedRecord.Record, nil, fmt.Errorf("Could not get domains: %s", returnedRecord.Status.Message)
	}

	return returnedRecord.Record, res, nil
}

// UpdateRecord updates a domain record.
//
// dnspod API docs: https://www.dnspod.cn/docs/records.html#record-modify
func (s *DomainsService) UpdateRecord(domain string, recordID string, recordAttributes Record) (Record, *Response, error) {
	path := recordAction("Modify")

	payload := newPayLoad(s.client.CommonParams)

	payload.Add("domain_id", domain)

	if recordAttributes.Name != "" {
		payload.Add("sub_domain", recordAttributes.Name)
	}

	if recordAttributes.Type != "" {
		payload.Add("record_type", recordAttributes.Type)
	}

	if recordAttributes.Line != "" {
		payload.Add("record_line", recordAttributes.Line)
	}

	if recordAttributes.LineID != "" {
		payload.Add("record_line_id", recordAttributes.LineID)
	}

	if recordAttributes.Value != "" {
		payload.Add("value", recordAttributes.Value)
	}

	if recordAttributes.MX != "" {
		payload.Add("mx", recordAttributes.MX)
	}

	if recordAttributes.TTL != "" {
		payload.Add("ttl", recordAttributes.TTL)
	}

	if recordAttributes.Status != "" {
		payload.Add("status", recordAttributes.Status)
	}

	returnedRecord := recordWrapper{}

	res, err := s.client.post(path, payload, &returnedRecord)
	if err != nil {
		return Record{}, res, err
	}

	if returnedRecord.Status.Code != "1" {
		return returnedRecord.Record, nil, fmt.Errorf("Could not get domains: %s", returnedRecord.Status.Message)
	}

	return returnedRecord.Record, res, nil
}

// DeleteRecord deletes a domain record.
//
// dnspod API docs: https://www.dnspod.cn/docs/records.html#record-remove
func (s *DomainsService) DeleteRecord(domain string, recordID string) (*Response, error) {
	path := recordAction("Remove")

	payload := newPayLoad(s.client.CommonParams)

	payload.Add("domain_id", domain)
	payload.Add("record_id", recordID)

	returnedRecord := recordWrapper{}

	res, err := s.client.post(path, payload, &returnedRecord)
	if err != nil {
		return res, err
	}

	if returnedRecord.Status.Code != "1" {
		return nil, fmt.Errorf("Could not get domains: %s", returnedRecord.Status.Message)
	}

	return res, nil
}
