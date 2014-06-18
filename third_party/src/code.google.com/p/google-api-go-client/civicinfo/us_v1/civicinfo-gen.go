// Package civicinfo provides access to the Google Civic Information API.
//
// See https://developers.google.com/civic-information
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/civicinfo/us_v1"
//   ...
//   civicinfoService, err := civicinfo.New(oauthHttpClient)
package civicinfo

import (
	"bytes"
	"code.google.com/p/google-api-go-client/googleapi"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// Always reference these packages, just in case the auto-generated code
// below doesn't.
var _ = bytes.NewBuffer
var _ = strconv.Itoa
var _ = fmt.Sprintf
var _ = json.NewDecoder
var _ = io.Copy
var _ = url.Parse
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace

const apiId = "civicinfo:us_v1"
const apiName = "civicinfo"
const apiVersion = "us_v1"
const basePath = "https://www.googleapis.com/civicinfo/us_v1/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Divisions = NewDivisionsService(s)
	s.Elections = NewElectionsService(s)
	s.Representatives = NewRepresentativesService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Divisions *DivisionsService

	Elections *ElectionsService

	Representatives *RepresentativesService
}

func NewDivisionsService(s *Service) *DivisionsService {
	rs := &DivisionsService{s: s}
	return rs
}

type DivisionsService struct {
	s *Service
}

func NewElectionsService(s *Service) *ElectionsService {
	rs := &ElectionsService{s: s}
	return rs
}

type ElectionsService struct {
	s *Service
}

func NewRepresentativesService(s *Service) *RepresentativesService {
	rs := &RepresentativesService{s: s}
	return rs
}

type RepresentativesService struct {
	s *Service
}

type AdministrationRegion struct {
	// ElectionAdministrationBody: The election administration body for this
	// area.
	ElectionAdministrationBody *AdministrativeBody `json:"electionAdministrationBody,omitempty"`

	// Id: An ID for this object. IDs may change in future requests and
	// should not be cached. Access to this field requires special access
	// that can be requested from the Request more link on the Quotas page.
	Id string `json:"id,omitempty"`

	// Local_jurisdiction: The city or county that provides election
	// information for this voter. This object can have the same elements as
	// state.
	Local_jurisdiction *AdministrationRegion `json:"local_jurisdiction,omitempty"`

	// Name: The name of the jurisdiction.
	Name string `json:"name,omitempty"`

	// Sources: A list of sources for this area. If multiple sources are
	// listed the data has been aggregated from those sources.
	Sources []*Source `json:"sources,omitempty"`
}

type AdministrativeBody struct {
	// AbsenteeVotingInfoUrl: A URL provided by this administrative body for
	// information on absentee voting.
	AbsenteeVotingInfoUrl string `json:"absenteeVotingInfoUrl,omitempty"`

	// BallotInfoUrl: A URL provided by this administrative body to give
	// contest information to the voter.
	BallotInfoUrl string `json:"ballotInfoUrl,omitempty"`

	// CorrespondenceAddress: The mailing address of this administrative
	// body.
	CorrespondenceAddress *SimpleAddressType `json:"correspondenceAddress,omitempty"`

	// ElectionInfoUrl: A URL provided by this administrative body for
	// looking up general election information.
	ElectionInfoUrl string `json:"electionInfoUrl,omitempty"`

	// ElectionOfficials: The election officials for this election
	// administrative body.
	ElectionOfficials []*ElectionOfficial `json:"electionOfficials,omitempty"`

	// ElectionRegistrationConfirmationUrl: A URL provided by this
	// administrative body for confirming that the voter is registered to
	// vote.
	ElectionRegistrationConfirmationUrl string `json:"electionRegistrationConfirmationUrl,omitempty"`

	// ElectionRegistrationUrl: A URL provided by this administrative body
	// for looking up how to register to vote.
	ElectionRegistrationUrl string `json:"electionRegistrationUrl,omitempty"`

	// ElectionRulesUrl: A URL provided by this administrative body
	// describing election rules to the voter.
	ElectionRulesUrl string `json:"electionRulesUrl,omitempty"`

	// HoursOfOperation: A description of the hours of operation for this
	// administrative body.
	HoursOfOperation string `json:"hoursOfOperation,omitempty"`

	// Name: The name of this election administrative body.
	Name string `json:"name,omitempty"`

	// PhysicalAddress: The physical address of this administrative body.
	PhysicalAddress *SimpleAddressType `json:"physicalAddress,omitempty"`

	// Voter_services: A description of the services this administrative
	// body may provide.
	Voter_services []string `json:"voter_services,omitempty"`

	// VotingLocationFinderUrl: A URL provided by this administrative body
	// for looking up where to vote.
	VotingLocationFinderUrl string `json:"votingLocationFinderUrl,omitempty"`
}

type Candidate struct {
	// CandidateUrl: The URL for the candidate's campaign web site.
	CandidateUrl string `json:"candidateUrl,omitempty"`

	// Channels: A list of known (social) media channels for this candidate.
	Channels []*Channel `json:"channels,omitempty"`

	// Email: The email address for the candidate's campaign.
	Email string `json:"email,omitempty"`

	// Name: The candidate's name.
	Name string `json:"name,omitempty"`

	// OrderOnBallot: The order the candidate appears on the ballot for this
	// contest.
	OrderOnBallot int64 `json:"orderOnBallot,omitempty,string"`

	// Party: The full name of the party the candidate is a member of.
	Party string `json:"party,omitempty"`

	// Phone: The voice phone number for the candidate's campaign office.
	Phone string `json:"phone,omitempty"`

	// PhotoUrl: A URL for a photo of the candidate.
	PhotoUrl string `json:"photoUrl,omitempty"`
}

type Channel struct {
	// Id: The unique public identifier for the candidate's channel.
	Id string `json:"id,omitempty"`

	// Type: The type of channel. The following is a list of types of
	// channels, but is not exhaustive. More channel types may be added at a
	// later time. One of: GooglePlus, YouTube, Facebook, Twitter
	Type string `json:"type,omitempty"`
}

type Contest struct {
	// BallotPlacement: A number specifying the position of this contest on
	// the voter's ballot.
	BallotPlacement int64 `json:"ballotPlacement,omitempty,string"`

	// Candidates: The candidate choices for this contest.
	Candidates []*Candidate `json:"candidates,omitempty"`

	// District: Information about the electoral district that this contest
	// is in.
	District *ElectoralDistrict `json:"district,omitempty"`

	// ElectorateSpecifications: A description of any additional eligibility
	// requirements for voting in this contest.
	ElectorateSpecifications string `json:"electorateSpecifications,omitempty"`

	// Id: An ID for this object. IDs may change in future requests and
	// should not be cached. Access to this field requires special access
	// that can be requested from the Request more link on the Quotas page.
	Id string `json:"id,omitempty"`

	// Level: The level of office for this contest. One of: federal, state,
	// county, city, other
	Level string `json:"level,omitempty"`

	// NumberElected: The number of candidates that will be elected to
	// office in this contest.
	NumberElected int64 `json:"numberElected,omitempty,string"`

	// NumberVotingFor: The number of candidates that a voter may vote for
	// in this contest.
	NumberVotingFor int64 `json:"numberVotingFor,omitempty,string"`

	// Office: The name of the office for this contest.
	Office string `json:"office,omitempty"`

	// PrimaryParty: If this is a partisan election, the name of the party
	// it is for.
	PrimaryParty string `json:"primaryParty,omitempty"`

	// ReferendumSubtitle: A brief description of the referendum. This field
	// is only populated for contests of type 'Referendum'.
	ReferendumSubtitle string `json:"referendumSubtitle,omitempty"`

	// ReferendumTitle: The title of the referendum (e.g. 'Proposition 42').
	// This field is only populated for contests of type 'Referendum'.
	ReferendumTitle string `json:"referendumTitle,omitempty"`

	// ReferendumUrl: A link to the referendum. This field is only populated
	// for contests of type 'Referendum'.
	ReferendumUrl string `json:"referendumUrl,omitempty"`

	// Sources: A list of sources for this contest. If multiple sources are
	// listed, the data has been aggregated from those sources.
	Sources []*Source `json:"sources,omitempty"`

	// Special: "Yes" or "No" depending on whether this a contest being held
	// outside the normal election cycle.
	Special string `json:"special,omitempty"`

	// Type: The type of contest. Usually this will be 'General', 'Primary',
	// or 'Run-off' for contests with candidates. For referenda this will be
	// 'Referendum'.
	Type string `json:"type,omitempty"`
}

type DivisionSearchResponse struct {
	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "civicinfo#divisionSearchResponse".
	Kind string `json:"kind,omitempty"`

	Results []*DivisionSearchResult `json:"results,omitempty"`

	// Status: The result of the request. One of: success,
	// addressUnparseable, noAddressParameter, internalLookupFailure
	Status string `json:"status,omitempty"`
}

type DivisionSearchResult struct {
	// Name: The name of the division.
	Name string `json:"name,omitempty"`

	// OcdId: The unique Open Civic Data identifier for this division.
	OcdId string `json:"ocdId,omitempty"`
}

type Election struct {
	// ElectionDay: Day of the election in YYYY-MM-DD format.
	ElectionDay string `json:"electionDay,omitempty"`

	// Id: The unique ID of this election.
	Id int64 `json:"id,omitempty,string"`

	// Name: A displayable name for the election.
	Name string `json:"name,omitempty"`
}

type ElectionOfficial struct {
	// EmailAddress: The email address of the election official.
	EmailAddress string `json:"emailAddress,omitempty"`

	// FaxNumber: The fax number of the election official.
	FaxNumber string `json:"faxNumber,omitempty"`

	// Name: The full name of the election official.
	Name string `json:"name,omitempty"`

	// OfficePhoneNumber: The office phone number of the election official.
	OfficePhoneNumber string `json:"officePhoneNumber,omitempty"`

	// Title: The title of the election official.
	Title string `json:"title,omitempty"`
}

type ElectionsQueryResponse struct {
	// Elections: A list of available elections
	Elections []*Election `json:"elections,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "civicinfo#electionsQueryResponse".
	Kind string `json:"kind,omitempty"`
}

type ElectoralDistrict struct {
	// Id: An identifier for this district, relative to its scope. For
	// example, the 34th State Senate district would have id "34" and a
	// scope of stateUpper.
	Id string `json:"id,omitempty"`

	// Name: The name of the district.
	Name string `json:"name,omitempty"`

	// Scope: The geographic scope of this district. If unspecified the
	// district's geography is not known. One of: national, statewide,
	// congressional, stateUpper, stateLower, countywide, judicial,
	// schoolBoard, cityWide, township, countyCouncil, cityCouncil, ward,
	// special
	Scope string `json:"scope,omitempty"`
}

type GeographicDivision struct {
	// Name: The name of the division.
	Name string `json:"name,omitempty"`

	// OfficeIds: List of keys in the offices object, one for each office
	// elected from this division. Will only be present if includeOffices
	// was true (or absent) in the request.
	OfficeIds []string `json:"officeIds,omitempty"`

	// Scope: The geographic scope of the division. If unspecified, the
	// division's geography is not known. One of: national, statewide,
	// congressional, stateUpper, stateLower, countywide, judicial,
	// schoolBoard, cityWide, township, countyCouncil, cityCouncil, ward,
	// special
	Scope string `json:"scope,omitempty"`
}

type Office struct {
	// Level: The level of this elected office. One of: federal, state,
	// county, city, other
	Level string `json:"level,omitempty"`

	// Name: The human-readable name of the office.
	Name string `json:"name,omitempty"`

	// OfficialIds: List of people who presently hold the office.
	OfficialIds []string `json:"officialIds,omitempty"`

	// Sources: A list of sources for this office. If multiple sources are
	// listed, the data has been aggregated from those sources.
	Sources []*Source `json:"sources,omitempty"`
}

type Official struct {
	// Address: Addresses at which to contact the official.
	Address []*SimpleAddressType `json:"address,omitempty"`

	// Channels: A list of known (social) media channels for this official.
	Channels []*Channel `json:"channels,omitempty"`

	// Emails: The direct email addresses for the official.
	Emails []string `json:"emails,omitempty"`

	// Name: The official's name.
	Name string `json:"name,omitempty"`

	// Party: The full name of the party the official belongs to.
	Party string `json:"party,omitempty"`

	// Phones: The official's public contact phone numbers.
	Phones []string `json:"phones,omitempty"`

	// PhotoUrl: A URL for a photo of the official.
	PhotoUrl string `json:"photoUrl,omitempty"`

	// Urls: The official's public website URLs.
	Urls []string `json:"urls,omitempty"`
}

type PollingLocation struct {
	// Address: The address of the location
	Address *SimpleAddressType `json:"address,omitempty"`

	// EndDate: The last date that this early vote site may be used. This
	// field is not populated for polling locations.
	EndDate string `json:"endDate,omitempty"`

	// Id: An ID for this object. IDs may change in future requests and
	// should not be cached. Access to this field requires special access
	// that can be requested from the Request more link on the Quotas page.
	Id string `json:"id,omitempty"`

	// Name: The name of the early vote site. This field is not populated
	// for polling locations.
	Name string `json:"name,omitempty"`

	// Notes: Notes about this location (e.g. accessibility ramp or entrance
	// to use)
	Notes string `json:"notes,omitempty"`

	// PollingHours: A description of when this location is open.
	PollingHours string `json:"pollingHours,omitempty"`

	// Sources: A list of sources for this location. If multiple sources are
	// listed the data has been aggregated from those sources.
	Sources []*Source `json:"sources,omitempty"`

	// StartDate: The first date that this early vote site may be used. This
	// field is not populated for polling locations.
	StartDate string `json:"startDate,omitempty"`

	// VoterServices: The services provided by this early vote site. This
	// field is not populated for polling locations.
	VoterServices string `json:"voterServices,omitempty"`
}

type RepresentativeInfoRequest struct {
	// Address: The address to look up. May only be specified if the field
	// ocdId is not given in the URL.
	Address string `json:"address,omitempty"`
}

type RepresentativeInfoResponse struct {
	// Divisions: Political geographic divisions that contain the requested
	// address.
	Divisions *RepresentativeInfoResponseDivisions `json:"divisions,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "civicinfo#representativeInfoResponse".
	Kind string `json:"kind,omitempty"`

	// NormalizedInput: The normalized version of the requested address
	NormalizedInput *SimpleAddressType `json:"normalizedInput,omitempty"`

	// Offices: Elected offices referenced by the divisions listed above.
	// Will only be present if includeOffices was true in the request.
	Offices *RepresentativeInfoResponseOffices `json:"offices,omitempty"`

	// Officials: Officials holding the offices listed above. Will only be
	// present if includeOffices was true in the request.
	Officials *RepresentativeInfoResponseOfficials `json:"officials,omitempty"`

	// Status: The result of the request. One of: success,
	// noStreetSegmentFound, addressUnparseable, noAddressParameter,
	// multipleStreetSegmentsFound, electionOver, electionUnknown,
	// internalLookupFailure, RequestedBothAddressAndOcdId
	Status string `json:"status,omitempty"`
}

type RepresentativeInfoResponseDivisions struct {
}

type RepresentativeInfoResponseOffices struct {
}

type RepresentativeInfoResponseOfficials struct {
}

type SimpleAddressType struct {
	// City: The city or town for the address.
	City string `json:"city,omitempty"`

	// Line1: The street name and number of this address.
	Line1 string `json:"line1,omitempty"`

	// Line2: The second line the address, if needed.
	Line2 string `json:"line2,omitempty"`

	// Line3: The third line of the address, if needed.
	Line3 string `json:"line3,omitempty"`

	// LocationName: The name of the location.
	LocationName string `json:"locationName,omitempty"`

	// State: The US two letter state abbreviation of the address.
	State string `json:"state,omitempty"`

	// Zip: The US Postal Zip Code of the address.
	Zip string `json:"zip,omitempty"`
}

type Source struct {
	// Name: The name of the data source.
	Name string `json:"name,omitempty"`

	// Official: Whether this data comes from an official government source.
	Official bool `json:"official,omitempty"`
}

type VoterInfoRequest struct {
	// Address: The registered address of the voter to look up.
	Address string `json:"address,omitempty"`
}

type VoterInfoResponse struct {
	// Contests: Contests that will appear on the voter's ballot
	Contests []*Contest `json:"contests,omitempty"`

	// EarlyVoteSites: Locations where the voter is eligible to vote early,
	// prior to election day
	EarlyVoteSites []*PollingLocation `json:"earlyVoteSites,omitempty"`

	// Election: The election that was queried.
	Election *Election `json:"election,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "civicinfo#voterInfoResponse".
	Kind string `json:"kind,omitempty"`

	// NormalizedInput: The normalized version of the requested address
	NormalizedInput *SimpleAddressType `json:"normalizedInput,omitempty"`

	// PollingLocations: Locations where the voter is eligible to vote on
	// election day. For states with mail-in voting only, these locations
	// will be nearby drop box locations. Drop box locations are free to the
	// voter and may be used instead of placing the ballot in the mail.
	PollingLocations []*PollingLocation `json:"pollingLocations,omitempty"`

	// State: Local Election Information for the state that the voter votes
	// in. For the US, there will only be one element in this array.
	State []*AdministrationRegion `json:"state,omitempty"`

	// Status: The result of the request. One of: success,
	// noStreetSegmentFound, addressUnparseable, noAddressParameter,
	// multipleStreetSegmentsFound, electionOver, electionUnknown,
	// internalLookupFailure
	Status string `json:"status,omitempty"`
}

// method id "civicinfo.divisions.search":

type DivisionsSearchCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// Search: Searches for political divisions by their natural name or OCD
// ID.
func (r *DivisionsService) Search() *DivisionsSearchCall {
	c := &DivisionsSearchCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// Query sets the optional parameter "query": The search query. Queries
// can cover any parts of a OCD ID or a human readable division name.
// All words given in the query are treated as required patterns. In
// addition to that, most query operators of the Apache Lucene library
// are supported. See
// http://lucene.apache.org/core/2_9_4/queryparsersyntax.html
func (c *DivisionsSearchCall) Query(query string) *DivisionsSearchCall {
	c.opt_["query"] = query
	return c
}

func (c *DivisionsSearchCall) Do() (*DivisionSearchResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["query"]; ok {
		params.Set("query", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "representatives/division_search")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(DivisionSearchResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Searches for political divisions by their natural name or OCD ID.",
	//   "httpMethod": "GET",
	//   "id": "civicinfo.divisions.search",
	//   "parameters": {
	//     "query": {
	//       "description": "The search query. Queries can cover any parts of a OCD ID or a human readable division name. All words given in the query are treated as required patterns. In addition to that, most query operators of the Apache Lucene library are supported. See http://lucene.apache.org/core/2_9_4/queryparsersyntax.html",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "representatives/division_search",
	//   "response": {
	//     "$ref": "DivisionSearchResponse"
	//   }
	// }

}

// method id "civicinfo.elections.electionQuery":

type ElectionsElectionQueryCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// ElectionQuery: List of available elections to query.
func (r *ElectionsService) ElectionQuery() *ElectionsElectionQueryCall {
	c := &ElectionsElectionQueryCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *ElectionsElectionQueryCall) Do() (*ElectionsQueryResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "elections")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(ElectionsQueryResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List of available elections to query.",
	//   "httpMethod": "GET",
	//   "id": "civicinfo.elections.electionQuery",
	//   "path": "elections",
	//   "response": {
	//     "$ref": "ElectionsQueryResponse"
	//   }
	// }

}

// method id "civicinfo.elections.voterInfoQuery":

type ElectionsVoterInfoQueryCall struct {
	s                *Service
	electionId       int64
	voterinforequest *VoterInfoRequest
	opt_             map[string]interface{}
}

// VoterInfoQuery: Looks up information relevant to a voter based on the
// voter's registered address.
func (r *ElectionsService) VoterInfoQuery(electionId int64, voterinforequest *VoterInfoRequest) *ElectionsVoterInfoQueryCall {
	c := &ElectionsVoterInfoQueryCall{s: r.s, opt_: make(map[string]interface{})}
	c.electionId = electionId
	c.voterinforequest = voterinforequest
	return c
}

// OfficialOnly sets the optional parameter "officialOnly": If set to
// true, only data from official state sources will be returned.
func (c *ElectionsVoterInfoQueryCall) OfficialOnly(officialOnly bool) *ElectionsVoterInfoQueryCall {
	c.opt_["officialOnly"] = officialOnly
	return c
}

func (c *ElectionsVoterInfoQueryCall) Do() (*VoterInfoResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.voterinforequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["officialOnly"]; ok {
		params.Set("officialOnly", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "voterinfo/{electionId}/lookup")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{electionId}", strconv.FormatInt(c.electionId, 10), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(VoterInfoResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Looks up information relevant to a voter based on the voter's registered address.",
	//   "httpMethod": "POST",
	//   "id": "civicinfo.elections.voterInfoQuery",
	//   "parameterOrder": [
	//     "electionId"
	//   ],
	//   "parameters": {
	//     "electionId": {
	//       "description": "The unique ID of the election to look up. A list of election IDs can be obtained at https://www.googleapis.com/civicinfo/{version}/elections",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "officialOnly": {
	//       "default": "false",
	//       "description": "If set to true, only data from official state sources will be returned.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "voterinfo/{electionId}/lookup",
	//   "request": {
	//     "$ref": "VoterInfoRequest"
	//   },
	//   "response": {
	//     "$ref": "VoterInfoResponse"
	//   }
	// }

}

// method id "civicinfo.representatives.representativeInfoQuery":

type RepresentativesRepresentativeInfoQueryCall struct {
	s                         *Service
	representativeinforequest *RepresentativeInfoRequest
	opt_                      map[string]interface{}
}

// RepresentativeInfoQuery: Looks up political geography and
// (optionally) representative information based on an address.
func (r *RepresentativesService) RepresentativeInfoQuery(representativeinforequest *RepresentativeInfoRequest) *RepresentativesRepresentativeInfoQueryCall {
	c := &RepresentativesRepresentativeInfoQueryCall{s: r.s, opt_: make(map[string]interface{})}
	c.representativeinforequest = representativeinforequest
	return c
}

// IncludeOffices sets the optional parameter "includeOffices": Whether
// to return information about offices and officials. If false, only the
// top-level district information will be returned.
func (c *RepresentativesRepresentativeInfoQueryCall) IncludeOffices(includeOffices bool) *RepresentativesRepresentativeInfoQueryCall {
	c.opt_["includeOffices"] = includeOffices
	return c
}

// OcdId sets the optional parameter "ocdId": The division to look up.
// May only be specified if the address field is not given in the
// request body.
func (c *RepresentativesRepresentativeInfoQueryCall) OcdId(ocdId string) *RepresentativesRepresentativeInfoQueryCall {
	c.opt_["ocdId"] = ocdId
	return c
}

func (c *RepresentativesRepresentativeInfoQueryCall) Do() (*RepresentativeInfoResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.representativeinforequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["includeOffices"]; ok {
		params.Set("includeOffices", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["ocdId"]; ok {
		params.Set("ocdId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "representatives/lookup")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(RepresentativeInfoResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Looks up political geography and (optionally) representative information based on an address.",
	//   "httpMethod": "POST",
	//   "id": "civicinfo.representatives.representativeInfoQuery",
	//   "parameters": {
	//     "includeOffices": {
	//       "default": "true",
	//       "description": "Whether to return information about offices and officials. If false, only the top-level district information will be returned.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "ocdId": {
	//       "description": "The division to look up. May only be specified if the address field is not given in the request body.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "representatives/lookup",
	//   "request": {
	//     "$ref": "RepresentativeInfoRequest"
	//   },
	//   "response": {
	//     "$ref": "RepresentativeInfoResponse"
	//   }
	// }

}
