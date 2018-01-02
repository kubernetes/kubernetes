// Package qpxexpress provides access to the QPX Express API.
//
// See http://developers.google.com/qpx-express
//
// Usage example:
//
//   import "google.golang.org/api/qpxexpress/v1"
//   ...
//   qpxexpressService, err := qpxexpress.New(oauthHttpClient)
package qpxexpress // import "google.golang.org/api/qpxexpress/v1"

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	context "golang.org/x/net/context"
	ctxhttp "golang.org/x/net/context/ctxhttp"
	gensupport "google.golang.org/api/gensupport"
	googleapi "google.golang.org/api/googleapi"
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
var _ = gensupport.MarshalJSON
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace
var _ = context.Canceled
var _ = ctxhttp.Do

const apiId = "qpxExpress:v1"
const apiName = "qpxExpress"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/qpxExpress/v1/trips/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Trips = NewTripsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Trips *TripsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewTripsService(s *Service) *TripsService {
	rs := &TripsService{s: s}
	return rs
}

type TripsService struct {
	s *Service
}

// AircraftData: The make, model, and type of an aircraft.
type AircraftData struct {
	// Code: The aircraft code. For example, for a Boeing 777 the code would
	// be 777.
	Code string `json:"code,omitempty"`

	// Kind: Identifies this as an aircraftData object. Value: the fixed
	// string qpxexpress#aircraftData
	Kind string `json:"kind,omitempty"`

	// Name: The name of an aircraft, for example Boeing 777.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Code") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AircraftData) MarshalJSON() ([]byte, error) {
	type noMethod AircraftData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// AirportData: An airport.
type AirportData struct {
	// City: The city code an airport is located in. For example, for JFK
	// airport, this is NYC.
	City string `json:"city,omitempty"`

	// Code: An airport's code. For example, for Boston Logan airport, this
	// is BOS.
	Code string `json:"code,omitempty"`

	// Kind: Identifies this as an airport object. Value: the fixed string
	// qpxexpress#airportData.
	Kind string `json:"kind,omitempty"`

	// Name: The name of an airport. For example, for airport BOS the name
	// is "Boston Logan International".
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "City") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "City") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *AirportData) MarshalJSON() ([]byte, error) {
	type noMethod AirportData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BagDescriptor: Information about an item of baggage.
type BagDescriptor struct {
	// CommercialName: Provides the commercial name for an optional service.
	CommercialName string `json:"commercialName,omitempty"`

	// Count: How many of this type of bag will be checked on this flight.
	Count int64 `json:"count,omitempty"`

	// Description: A description of the baggage.
	Description []string `json:"description,omitempty"`

	// Kind: Identifies this as a baggage object. Value: the fixed string
	// qpxexpress#bagDescriptor.
	Kind string `json:"kind,omitempty"`

	// Subcode: The standard IATA subcode used to identify this optional
	// service.
	Subcode string `json:"subcode,omitempty"`

	// ForceSendFields is a list of field names (e.g. "CommercialName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "CommercialName") to
	// include in API requests with the JSON null value. By default, fields
	// with empty values are omitted from API requests. However, any field
	// with an empty value appearing in NullFields will be sent to the
	// server as null. It is an error if a field in this list has a
	// non-empty value. This may be used to include null fields in Patch
	// requests.
	NullFields []string `json:"-"`
}

func (s *BagDescriptor) MarshalJSON() ([]byte, error) {
	type noMethod BagDescriptor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CarrierData: Information about a carrier (ie. an airline, bus line,
// railroad, etc) that might be useful to display to an end-user.
type CarrierData struct {
	// Code: The IATA designator of a carrier (airline, etc). For example,
	// for American Airlines, the code is AA.
	Code string `json:"code,omitempty"`

	// Kind: Identifies this as a kind of carrier (ie. an airline, bus line,
	// railroad, etc). Value: the fixed string qpxexpress#carrierData.
	Kind string `json:"kind,omitempty"`

	// Name: The long, full name of a carrier. For example: American
	// Airlines.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Code") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CarrierData) MarshalJSON() ([]byte, error) {
	type noMethod CarrierData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CityData: Information about a city that might be useful to an
// end-user; typically the city of an airport.
type CityData struct {
	// Code: The IATA character ID of a city. For example, for Boston this
	// is BOS.
	Code string `json:"code,omitempty"`

	// Country: The two-character country code of the country the city is
	// located in. For example, US for the United States of America.
	Country string `json:"country,omitempty"`

	// Kind: Identifies this as a city, typically with one or more airports.
	// Value: the fixed string qpxexpress#cityData.
	Kind string `json:"kind,omitempty"`

	// Name: The full name of a city. An example would be: New York.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Code") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Code") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CityData) MarshalJSON() ([]byte, error) {
	type noMethod CityData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// Data: Detailed information about components found in the solutions of
// this response, including a trip's airport, city, taxes, airline, and
// aircraft.
type Data struct {
	// Aircraft: The aircraft that is flying between an origin and
	// destination.
	Aircraft []*AircraftData `json:"aircraft,omitempty"`

	// Airport: The airport of an origin or destination.
	Airport []*AirportData `json:"airport,omitempty"`

	// Carrier: The airline carrier of the aircraft flying between an origin
	// and destination. Allowed values are IATA carrier codes.
	Carrier []*CarrierData `json:"carrier,omitempty"`

	// City: The city that is either the origin or destination of part of a
	// trip.
	City []*CityData `json:"city,omitempty"`

	// Kind: Identifies this as QPX Express response resource, including a
	// trip's airport, city, taxes, airline, and aircraft. Value: the fixed
	// string qpxexpress#data.
	Kind string `json:"kind,omitempty"`

	// Tax: The taxes due for flying between an origin and a destination.
	Tax []*TaxData `json:"tax,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Aircraft") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Aircraft") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Data) MarshalJSON() ([]byte, error) {
	type noMethod Data
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FareInfo: Complete information about a fare used in the solution to a
// low-fare search query. In the airline industry a fare is a price an
// airline charges for one-way travel between two points. A fare
// typically contains a carrier code, two city codes, a price, and a
// fare basis. (A fare basis is a one-to-eight character alphanumeric
// code used to identify a fare.)
type FareInfo struct {
	BasisCode string `json:"basisCode,omitempty"`

	// Carrier: The carrier of the aircraft or other vehicle commuting
	// between two points.
	Carrier string `json:"carrier,omitempty"`

	// Destination: The city code of the city the trip ends at.
	Destination string `json:"destination,omitempty"`

	// Id: A unique identifier of the fare.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this as a fare object. Value: the fixed string
	// qpxexpress#fareInfo.
	Kind string `json:"kind,omitempty"`

	// Origin: The city code of the city the trip begins at.
	Origin string `json:"origin,omitempty"`

	// Private: Whether this is a private fare, for example one offered only
	// to select customers rather than the general public.
	Private bool `json:"private,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BasisCode") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BasisCode") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FareInfo) MarshalJSON() ([]byte, error) {
	type noMethod FareInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FlightInfo: A flight is a sequence of legs with the same airline
// carrier and flight number. (A leg is the smallest unit of travel, in
// the case of a flight a takeoff immediately followed by a landing at
// two set points on a particular carrier with a particular flight
// number.) The naive view is that a flight is scheduled travel of an
// aircraft between two points, with possibly intermediate stops, but
// carriers will frequently list flights that require a change of
// aircraft between legs.
type FlightInfo struct {
	Carrier string `json:"carrier,omitempty"`

	// Number: The flight number.
	Number string `json:"number,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Carrier") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Carrier") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FlightInfo) MarshalJSON() ([]byte, error) {
	type noMethod FlightInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// FreeBaggageAllowance: Information about free baggage allowed on one
// segment of a trip.
type FreeBaggageAllowance struct {
	// BagDescriptor: A representation of a type of bag, such as an ATPCo
	// subcode, Commercial Name, or other description.
	BagDescriptor []*BagDescriptor `json:"bagDescriptor,omitempty"`

	// Kilos: The maximum number of kilos all the free baggage together may
	// weigh.
	Kilos int64 `json:"kilos,omitempty"`

	// KilosPerPiece: The maximum number of kilos any one piece of baggage
	// may weigh.
	KilosPerPiece int64 `json:"kilosPerPiece,omitempty"`

	// Kind: Identifies this as free baggage object, allowed on one segment
	// of a trip. Value: the fixed string qpxexpress#freeBaggageAllowance.
	Kind string `json:"kind,omitempty"`

	// Pieces: The number of free pieces of baggage allowed.
	Pieces int64 `json:"pieces,omitempty"`

	// Pounds: The number of pounds of free baggage allowed.
	Pounds int64 `json:"pounds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BagDescriptor") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BagDescriptor") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *FreeBaggageAllowance) MarshalJSON() ([]byte, error) {
	type noMethod FreeBaggageAllowance
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// LegInfo: Information about a leg. (A leg is the smallest unit of
// travel, in the case of a flight a takeoff immediately followed by a
// landing at two set points on a particular carrier with a particular
// flight number.)
type LegInfo struct {
	// Aircraft: The aircraft (or bus, ferry, railcar, etc) travelling
	// between the two points of this leg.
	Aircraft string `json:"aircraft,omitempty"`

	// ArrivalTime: The scheduled time of arrival at the destination of the
	// leg, local to the point of arrival.
	ArrivalTime string `json:"arrivalTime,omitempty"`

	// ChangePlane: Whether you have to change planes following this leg.
	// Only applies to the next leg.
	ChangePlane bool `json:"changePlane,omitempty"`

	// ConnectionDuration: Duration of a connection following this leg, in
	// minutes.
	ConnectionDuration int64 `json:"connectionDuration,omitempty"`

	// DepartureTime: The scheduled departure time of the leg, local to the
	// point of departure.
	DepartureTime string `json:"departureTime,omitempty"`

	// Destination: The leg destination as a city and airport.
	Destination string `json:"destination,omitempty"`

	// DestinationTerminal: The terminal the flight is scheduled to arrive
	// at.
	DestinationTerminal string `json:"destinationTerminal,omitempty"`

	// Duration: The scheduled travelling time from the origin to the
	// destination.
	Duration int64 `json:"duration,omitempty"`

	// Id: An identifier that uniquely identifies this leg in the solution.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this as a leg object. A leg is the smallest unit of
	// travel, in the case of a flight a takeoff immediately followed by a
	// landing at two set points on a particular carrier with a particular
	// flight number. Value: the fixed string qpxexpress#legInfo.
	Kind string `json:"kind,omitempty"`

	// Meal: A simple, general description of the meal(s) served on the
	// flight, for example: "Hot meal".
	Meal string `json:"meal,omitempty"`

	// Mileage: The number of miles in this leg.
	Mileage int64 `json:"mileage,omitempty"`

	// OnTimePerformance: In percent, the published on time performance on
	// this leg.
	OnTimePerformance int64 `json:"onTimePerformance,omitempty"`

	// OperatingDisclosure: Department of Transportation disclosure
	// information on the actual operator of a flight in a code share. (A
	// code share refers to a marketing agreement between two carriers,
	// where one carrier will list in its schedules (and take bookings for)
	// flights that are actually operated by another carrier.)
	OperatingDisclosure string `json:"operatingDisclosure,omitempty"`

	// Origin: The leg origin as a city and airport.
	Origin string `json:"origin,omitempty"`

	// OriginTerminal: The terminal the flight is scheduled to depart from.
	OriginTerminal string `json:"originTerminal,omitempty"`

	// Secure: Whether passenger information must be furnished to the United
	// States Transportation Security Administration (TSA) prior to
	// departure.
	Secure bool `json:"secure,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Aircraft") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Aircraft") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *LegInfo) MarshalJSON() ([]byte, error) {
	type noMethod LegInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PassengerCounts: The number and type of passengers. Unfortunately the
// definition of an infant, child, adult, and senior citizen varies
// across carriers and reservation systems.
type PassengerCounts struct {
	// AdultCount: The number of passengers that are adults.
	AdultCount int64 `json:"adultCount,omitempty"`

	// ChildCount: The number of passengers that are children.
	ChildCount int64 `json:"childCount,omitempty"`

	// InfantInLapCount: The number of passengers that are infants
	// travelling in the lap of an adult.
	InfantInLapCount int64 `json:"infantInLapCount,omitempty"`

	// InfantInSeatCount: The number of passengers that are infants each
	// assigned a seat.
	InfantInSeatCount int64 `json:"infantInSeatCount,omitempty"`

	// Kind: Identifies this as a passenger count object, representing the
	// number of passengers. Value: the fixed string
	// qpxexpress#passengerCounts.
	Kind string `json:"kind,omitempty"`

	// SeniorCount: The number of passengers that are senior citizens.
	SeniorCount int64 `json:"seniorCount,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AdultCount") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "AdultCount") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PassengerCounts) MarshalJSON() ([]byte, error) {
	type noMethod PassengerCounts
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PricingInfo: The price of one or more travel segments. The currency
// used to purchase tickets is usually determined by the sale/ticketing
// city or the sale/ticketing country, unless none are specified, in
// which case it defaults to that of the journey origin country.
type PricingInfo struct {
	// BaseFareTotal: The total fare in the base fare currency (the currency
	// of the country of origin). This element is only present when the
	// sales currency and the currency of the country of commencement are
	// different.
	BaseFareTotal string `json:"baseFareTotal,omitempty"`

	// Fare: The fare used to price one or more segments.
	Fare []*FareInfo `json:"fare,omitempty"`

	// FareCalculation: The horizontal fare calculation. This is a field on
	// a ticket that displays all of the relevant items that go into the
	// calculation of the fare.
	FareCalculation string `json:"fareCalculation,omitempty"`

	// Kind: Identifies this as a pricing object, representing the price of
	// one or more travel segments. Value: the fixed string
	// qpxexpress#pricingInfo.
	Kind string `json:"kind,omitempty"`

	// LatestTicketingTime: The latest ticketing time for this pricing
	// assuming the reservation occurs at ticketing time and there is no
	// change in fares/rules. The time is local to the point of sale (POS).
	LatestTicketingTime string `json:"latestTicketingTime,omitempty"`

	// Passengers: The number of passengers to which this price applies.
	Passengers *PassengerCounts `json:"passengers,omitempty"`

	// Ptc: The passenger type code for this pricing. An alphanumeric code
	// used by a carrier to restrict fares to certain categories of
	// passenger. For instance, a fare might be valid only for senior
	// citizens.
	Ptc string `json:"ptc,omitempty"`

	// Refundable: Whether the fares on this pricing are refundable.
	Refundable bool `json:"refundable,omitempty"`

	// SaleFareTotal: The total fare in the sale or equivalent currency.
	SaleFareTotal string `json:"saleFareTotal,omitempty"`

	// SaleTaxTotal: The taxes in the sale or equivalent currency.
	SaleTaxTotal string `json:"saleTaxTotal,omitempty"`

	// SaleTotal: Total per-passenger price (fare and tax) in the sale or
	// equivalent currency.
	SaleTotal string `json:"saleTotal,omitempty"`

	// SegmentPricing: The per-segment price and baggage information.
	SegmentPricing []*SegmentPricing `json:"segmentPricing,omitempty"`

	// Tax: The taxes used to calculate the tax total per ticket.
	Tax []*TaxInfo `json:"tax,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BaseFareTotal") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BaseFareTotal") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PricingInfo) MarshalJSON() ([]byte, error) {
	type noMethod PricingInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SegmentInfo: Details of a segment of a flight; a segment is one or
// more consecutive legs on the same flight. For example a hypothetical
// flight ZZ001, from DFW to OGG, would have one segment with two legs:
// DFW to HNL (leg 1), HNL to OGG (leg 2), and DFW to OGG (legs 1 and
// 2).
type SegmentInfo struct {
	// BookingCode: The booking code or class for this segment.
	BookingCode string `json:"bookingCode,omitempty"`

	// BookingCodeCount: The number of seats available in this booking code
	// on this segment.
	BookingCodeCount int64 `json:"bookingCodeCount,omitempty"`

	// Cabin: The cabin booked for this segment.
	Cabin string `json:"cabin,omitempty"`

	// ConnectionDuration: In minutes, the duration of the connection
	// following this segment.
	ConnectionDuration int64 `json:"connectionDuration,omitempty"`

	// Duration: The duration of the flight segment in minutes.
	Duration int64 `json:"duration,omitempty"`

	// Flight: The flight this is a segment of.
	Flight *FlightInfo `json:"flight,omitempty"`

	// Id: An id uniquely identifying the segment in the solution.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this as a segment object. A segment is one or more
	// consecutive legs on the same flight. For example a hypothetical
	// flight ZZ001, from DFW to OGG, could have one segment with two legs:
	// DFW to HNL (leg 1), HNL to OGG (leg 2). Value: the fixed string
	// qpxexpress#segmentInfo.
	Kind string `json:"kind,omitempty"`

	// Leg: The legs composing this segment.
	Leg []*LegInfo `json:"leg,omitempty"`

	// MarriedSegmentGroup: The solution-based index of a segment in a
	// married segment group. Married segments can only be booked together.
	// For example, an airline might report a certain booking code as sold
	// out from Boston to Pittsburgh, but as available as part of two
	// married segments Boston to Chicago connecting through Pittsburgh. For
	// example content of this field, consider the round-trip flight ZZ1
	// PHX-PHL ZZ2 PHL-CLT ZZ3 CLT-PHX. This has three segments, with the
	// two outbound ones (ZZ1 ZZ2) married. In this case, the two outbound
	// segments belong to married segment group 0, and the return segment
	// belongs to married segment group 1.
	MarriedSegmentGroup string `json:"marriedSegmentGroup,omitempty"`

	// SubjectToGovernmentApproval: Whether the operation of this segment
	// remains subject to government approval.
	SubjectToGovernmentApproval bool `json:"subjectToGovernmentApproval,omitempty"`

	// ForceSendFields is a list of field names (e.g. "BookingCode") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "BookingCode") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SegmentInfo) MarshalJSON() ([]byte, error) {
	type noMethod SegmentInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SegmentPricing: The price of this segment.
type SegmentPricing struct {
	// FareId: A segment identifier unique within a single solution. It is
	// used to refer to different parts of the same solution.
	FareId string `json:"fareId,omitempty"`

	// FreeBaggageOption: Details of the free baggage allowance on this
	// segment.
	FreeBaggageOption []*FreeBaggageAllowance `json:"freeBaggageOption,omitempty"`

	// Kind: Identifies this as a segment pricing object, representing the
	// price of this segment. Value: the fixed string
	// qpxexpress#segmentPricing.
	Kind string `json:"kind,omitempty"`

	// SegmentId: Unique identifier in the response of this segment.
	SegmentId string `json:"segmentId,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FareId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "FareId") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SegmentPricing) MarshalJSON() ([]byte, error) {
	type noMethod SegmentPricing
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SliceInfo: Information about a slice. A slice represents a
// traveller's intent, the portion of a low-fare search corresponding to
// a traveler's request to get between two points. One-way journeys are
// generally expressed using 1 slice, round-trips using 2. For example,
// if a traveler specifies the following trip in a user interface:
// | Origin | Destination | Departure Date | | BOS | LAX | March 10,
// 2007 | | LAX | SYD | March 17, 2007 | | SYD | BOS | March 22, 2007
// |
// then this is a three slice trip.
type SliceInfo struct {
	// Duration: The duration of the slice in minutes.
	Duration int64 `json:"duration,omitempty"`

	// Kind: Identifies this as a slice object. A slice represents a
	// traveller's intent, the portion of a low-fare search corresponding to
	// a traveler's request to get between two points. One-way journeys are
	// generally expressed using 1 slice, round-trips using 2. Value: the
	// fixed string qpxexpress#sliceInfo.
	Kind string `json:"kind,omitempty"`

	// Segment: The segment(s) constituting the slice.
	Segment []*SegmentInfo `json:"segment,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Duration") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Duration") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SliceInfo) MarshalJSON() ([]byte, error) {
	type noMethod SliceInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// SliceInput: Criteria a desired slice must satisfy.
type SliceInput struct {
	// Alliance: Slices with only the carriers in this alliance should be
	// returned; do not use this field with permittedCarrier. Allowed values
	// are ONEWORLD, SKYTEAM, and STAR.
	Alliance string `json:"alliance,omitempty"`

	// Date: Departure date in YYYY-MM-DD format.
	Date string `json:"date,omitempty"`

	// Destination: Airport or city IATA designator of the destination.
	Destination string `json:"destination,omitempty"`

	// Kind: Identifies this as a slice input object, representing the
	// criteria a desired slice must satisfy. Value: the fixed string
	// qpxexpress#sliceInput.
	Kind string `json:"kind,omitempty"`

	// MaxConnectionDuration: The longest connection between two legs, in
	// minutes, you are willing to accept.
	MaxConnectionDuration int64 `json:"maxConnectionDuration,omitempty"`

	// MaxStops: The maximum number of stops you are willing to accept in
	// this slice.
	MaxStops int64 `json:"maxStops,omitempty"`

	// Origin: Airport or city IATA designator of the origin.
	Origin string `json:"origin,omitempty"`

	// PermittedCarrier: A list of 2-letter IATA airline designators. Slices
	// with only these carriers should be returned.
	PermittedCarrier []string `json:"permittedCarrier,omitempty"`

	// PermittedDepartureTime: Slices must depart in this time of day range,
	// local to the point of departure.
	PermittedDepartureTime *TimeOfDayRange `json:"permittedDepartureTime,omitempty"`

	// PreferredCabin: Prefer solutions that book in this cabin for this
	// slice. Allowed values are COACH, PREMIUM_COACH, BUSINESS, and FIRST.
	PreferredCabin string `json:"preferredCabin,omitempty"`

	// ProhibitedCarrier: A list of 2-letter IATA airline designators.
	// Exclude slices that use these carriers.
	ProhibitedCarrier []string `json:"prohibitedCarrier,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Alliance") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Alliance") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *SliceInput) MarshalJSON() ([]byte, error) {
	type noMethod SliceInput
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TaxData: Tax data.
type TaxData struct {
	// Id: An identifier uniquely identifying a tax in a response.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this as a tax data object, representing some tax.
	// Value: the fixed string qpxexpress#taxData.
	Kind string `json:"kind,omitempty"`

	// Name: The name of a tax.
	Name string `json:"name,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TaxData) MarshalJSON() ([]byte, error) {
	type noMethod TaxData
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TaxInfo: Tax information.
type TaxInfo struct {
	// ChargeType: Whether this is a government charge or a carrier
	// surcharge.
	ChargeType string `json:"chargeType,omitempty"`

	// Code: The code to enter in the ticket's tax box.
	Code string `json:"code,omitempty"`

	// Country: For government charges, the country levying the charge.
	Country string `json:"country,omitempty"`

	// Id: Identifier uniquely identifying this tax in a response. Not
	// present for unnamed carrier surcharges.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this as a tax information object. Value: the fixed
	// string qpxexpress#taxInfo.
	Kind string `json:"kind,omitempty"`

	// SalePrice: The price of the tax in the sales or equivalent currency.
	SalePrice string `json:"salePrice,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ChargeType") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "ChargeType") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TaxInfo) MarshalJSON() ([]byte, error) {
	type noMethod TaxInfo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TimeOfDayRange: Two times in a single day defining a time range.
type TimeOfDayRange struct {
	// EarliestTime: The earliest time of day in HH:MM format.
	EarliestTime string `json:"earliestTime,omitempty"`

	// Kind: Identifies this as a time of day range object, representing two
	// times in a single day defining a time range. Value: the fixed string
	// qpxexpress#timeOfDayRange.
	Kind string `json:"kind,omitempty"`

	// LatestTime: The latest time of day in HH:MM format.
	LatestTime string `json:"latestTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "EarliestTime") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "EarliestTime") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TimeOfDayRange) MarshalJSON() ([]byte, error) {
	type noMethod TimeOfDayRange
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TripOption: Trip information.
type TripOption struct {
	// Id: Identifier uniquely identifying this trip in a response.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this as a trip information object. Value: the fixed
	// string qpxexpress#tripOption.
	Kind string `json:"kind,omitempty"`

	// Pricing: Per passenger pricing information.
	Pricing []*PricingInfo `json:"pricing,omitempty"`

	// SaleTotal: The total price for all passengers on the trip, in the
	// form of a currency followed by an amount, e.g. USD253.35.
	SaleTotal string `json:"saleTotal,omitempty"`

	// Slice: The slices that make up this trip's itinerary.
	Slice []*SliceInfo `json:"slice,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TripOption) MarshalJSON() ([]byte, error) {
	type noMethod TripOption
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TripOptionsRequest: A QPX Express search request, which will yield
// one or more solutions.
type TripOptionsRequest struct {
	// MaxPrice: Do not return solutions that cost more than this price. The
	// alphabetical part of the price is in ISO 4217. The format, in regex,
	// is [A-Z]{3}\d+(\.\d+)? Example: $102.07
	MaxPrice string `json:"maxPrice,omitempty"`

	// Passengers: Counts for each passenger type in the request.
	Passengers *PassengerCounts `json:"passengers,omitempty"`

	// Refundable: Return only solutions with refundable fares.
	Refundable bool `json:"refundable,omitempty"`

	// SaleCountry: IATA country code representing the point of sale. This
	// determines the "equivalent amount paid" currency for the ticket.
	SaleCountry string `json:"saleCountry,omitempty"`

	// Slice: The slices that make up the itinerary of this trip. A slice
	// represents a traveler's intent, the portion of a low-fare search
	// corresponding to a traveler's request to get between two points.
	// One-way journeys are generally expressed using one slice, round-trips
	// using two. An example of a one slice trip with three segments might
	// be BOS-SYD, SYD-LAX, LAX-BOS if the traveler only stopped in SYD and
	// LAX just long enough to change planes.
	Slice []*SliceInput `json:"slice,omitempty"`

	// Solutions: The number of solutions to return, maximum 500.
	Solutions int64 `json:"solutions,omitempty"`

	// TicketingCountry: IATA country code representing the point of
	// ticketing.
	TicketingCountry string `json:"ticketingCountry,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxPrice") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "MaxPrice") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TripOptionsRequest) MarshalJSON() ([]byte, error) {
	type noMethod TripOptionsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TripOptionsResponse: A QPX Express search response.
type TripOptionsResponse struct {
	// Data: Informational data global to list of solutions.
	Data *Data `json:"data,omitempty"`

	// Kind: Identifies this as a QPX Express trip response object, which
	// consists of zero or more solutions. Value: the fixed string
	// qpxexpress#tripOptions.
	Kind string `json:"kind,omitempty"`

	// RequestId: An identifier uniquely identifying this response.
	RequestId string `json:"requestId,omitempty"`

	// TripOption: A list of priced itinerary solutions to the QPX Express
	// query.
	TripOption []*TripOption `json:"tripOption,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Data") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Data") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TripOptionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod TripOptionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TripsSearchRequest: A QPX Express search request.
type TripsSearchRequest struct {
	// Request: A QPX Express search request. Required values are at least
	// one adult or senior passenger, an origin, a destination, and a date.
	Request *TripOptionsRequest `json:"request,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Request") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Request") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TripsSearchRequest) MarshalJSON() ([]byte, error) {
	type noMethod TripsSearchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// TripsSearchResponse: A QPX Express search response.
type TripsSearchResponse struct {
	// Kind: Identifies this as a QPX Express API search response resource.
	// Value: the fixed string qpxExpress#tripsSearch.
	Kind string `json:"kind,omitempty"`

	// Trips: All possible solutions to the QPX Express search request.
	Trips *TripOptionsResponse `json:"trips,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Kind") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Kind") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *TripsSearchResponse) MarshalJSON() ([]byte, error) {
	type noMethod TripsSearchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "qpxExpress.trips.search":

type TripsSearchCall struct {
	s                  *Service
	tripssearchrequest *TripsSearchRequest
	urlParams_         gensupport.URLParams
	ctx_               context.Context
	header_            http.Header
}

// Search: Returns a list of flights.
func (r *TripsService) Search(tripssearchrequest *TripsSearchRequest) *TripsSearchCall {
	c := &TripsSearchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.tripssearchrequest = tripssearchrequest
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TripsSearchCall) Fields(s ...googleapi.Field) *TripsSearchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TripsSearchCall) Context(ctx context.Context) *TripsSearchCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *TripsSearchCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *TripsSearchCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.tripssearchrequest)
	if err != nil {
		return nil, err
	}
	reqHeaders.Set("Content-Type", "application/json")
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "search")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header = reqHeaders
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "qpxExpress.trips.search" call.
// Exactly one of *TripsSearchResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *TripsSearchResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TripsSearchCall) Do(opts ...googleapi.CallOption) (*TripsSearchResponse, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &TripsSearchResponse{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of flights.",
	//   "httpMethod": "POST",
	//   "id": "qpxExpress.trips.search",
	//   "path": "search",
	//   "request": {
	//     "$ref": "TripsSearchRequest"
	//   },
	//   "response": {
	//     "$ref": "TripsSearchResponse"
	//   }
	// }

}
