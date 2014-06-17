// Package qpxexpress provides access to the QPX Express API.
//
// See http://developers.google.com/qpx-express
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/qpxexpress/v1"
//   ...
//   qpxexpressService, err := qpxexpress.New(oauthHttpClient)
package qpxexpress

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
	client   *http.Client
	BasePath string // API endpoint base URL

	Trips *TripsService
}

func NewTripsService(s *Service) *TripsService {
	rs := &TripsService{s: s}
	return rs
}

type TripsService struct {
	s *Service
}

type AircraftData struct {
	// Code: The aircraft code. For example, for a Boeing 777 the code would
	// be 777.
	Code string `json:"code,omitempty"`

	// Kind: Identifies this as an aircraftData object. Value: the fixed
	// string qpxexpress#aircraftData
	Kind string `json:"kind,omitempty"`

	// Name: The name of an aircraft, for example Boeing 777.
	Name string `json:"name,omitempty"`
}

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
}

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
}

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
}

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
}

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
}

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
}

type FlightInfo struct {
	Carrier string `json:"carrier,omitempty"`

	// Number: The flight number.
	Number string `json:"number,omitempty"`
}

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
}

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
}

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
}

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
}

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
}

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
}

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
}

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
}

type TaxData struct {
	// Id: An identifier uniquely identifying a tax in a response.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this as a tax data object, representing some tax.
	// Value: the fixed string qpxexpress#taxData.
	Kind string `json:"kind,omitempty"`

	// Name: The name of a tax.
	Name string `json:"name,omitempty"`
}

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
}

type TimeOfDayRange struct {
	// EarliestTime: The earliest time of day in HH:MM format.
	EarliestTime string `json:"earliestTime,omitempty"`

	// Kind: Identifies this as a time of day range object, representing two
	// times in a single day defining a time range. Value: the fixed string
	// qpxexpress#timeOfDayRange.
	Kind string `json:"kind,omitempty"`

	// LatestTime: The latest time of day in HH:MM format.
	LatestTime string `json:"latestTime,omitempty"`
}

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
}

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
}

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
}

type TripsSearchRequest struct {
	// Request: A QPX Express search request. Required values are at least
	// one adult or senior passenger, an origin, a destination, and a date.
	Request *TripOptionsRequest `json:"request,omitempty"`
}

type TripsSearchResponse struct {
	// Kind: Identifies this as a QPX Express API search response resource.
	// Value: the fixed string qpxExpress#tripsSearch.
	Kind string `json:"kind,omitempty"`

	// Trips: All possible solutions to the QPX Express search request.
	Trips *TripOptionsResponse `json:"trips,omitempty"`
}

// method id "qpxExpress.trips.search":

type TripsSearchCall struct {
	s                  *Service
	tripssearchrequest *TripsSearchRequest
	opt_               map[string]interface{}
}

// Search: Returns a list of flights.
func (r *TripsService) Search(tripssearchrequest *TripsSearchRequest) *TripsSearchCall {
	c := &TripsSearchCall{s: r.s, opt_: make(map[string]interface{})}
	c.tripssearchrequest = tripssearchrequest
	return c
}

func (c *TripsSearchCall) Do() (*TripsSearchResponse, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.tripssearchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "search")
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
	ret := new(TripsSearchResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
