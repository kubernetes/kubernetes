package stripe

import (
	"time"
)

// Customer encapsulates details about a Customer registered in Stripe.
//
// see https://stripe.com/docs/api#customer_object
type Customer struct {
	Id           string        `json:"id"`
	Desc         string        `json:"description,omitempty"`
	Email        string        `json:"email,omitempty"`
	Created      int64         `json:"created"`
	Balance      int64         `json:"account_balance"`
	Delinquent   bool          `json:"delinquent"`
	Cards        CardData      `json:"cards,omitempty"`
	Discount     *Discount     `json:"discount,omitempty"`
	Subscription *Subscription `json:"subscription,omitempty"`
	Livemode     bool          `json:"livemode"`
	DefaultCard  string        `json:"default_card"`
}

type CardData struct {
	Object string  `json:"object"`
	Count  int     `json:"count"`
	Url    string  `json:"url"`
	Data   []*Card `json:"data"`
}

// Credit Card Types accepted by the Stripe API.
const (
	AmericanExpress = "American Express"
	DinersClub      = "Diners Club"
	Discover        = "Discover"
	JCB             = "JCB"
	MasterCard      = "MasterCard"
	Visa            = "Visa"
	UnknownCard     = "Unknown"
)

// Card represents details about a Credit Card entered into Stripe.
type Card struct {
	Id                string `json:"id"`
	Name              string `json:"name,omitempty"`
	Type              string `json:"type"`
	ExpMonth          int    `json:"exp_month"`
	ExpYear           int    `json:"exp_year"`
	Last4             string `json:"last4"`
	Fingerprint       string `json:"fingerprint"`
	Country           string `json:"country,omitempty"`
	AddrUess1         string `json:"address_line1,omitempty"`
	Address2          string `json:"address_line2,omitempty"`
	AddressCountry    string `json:"address_country,omitempty"`
	AddressState      string `json:"address_state,omitempty"`
	AddressZip        string `json:"address_zip,omitempty"`
	AddressCity       string `json:"address_city"`
	AddressLine1Check string `json:"address_line1_check,omitempty"`
	AddressZipCheck   string `json:"address_zip_check,omitempty"`
	CVCCheck          string `json:"cvc_check,omitempty"`
}

// Discount represents the actual application of a coupon to a particular
// customer.
//
// see https://stripe.com/docs/api#discount_object
type Discount struct {
	Id       string  `json:"id"`
	Customer string  `json:"customer"`
	Start    int64   `json:"start"`
	End      int64   `json:"end"`
	Coupon   *Coupon `json:"coupon"`
}

// Coupon represents percent-off discount you might want to apply to a customer.
//
// see https://stripe.com/docs/api#coupon_object
type Coupon struct {
	Id               string `json:"id"`
	Duration         string `json:"duration"`
	PercentOff       int    `json:"percent_off"`
	DurationInMonths int    `json:"duration_in_months,omitempty"`
	MaxRedemptions   int    `json:"max_redemptions,omitempty"`
	RedeemBy         int64  `json:"redeem_by,omitempty"`
	TimesRedeemed    int    `json:"times_redeemed,omitempty"`
	Livemode         bool   `json:"livemode"`
}

// Subscription Statuses
const (
	SubscriptionTrialing = "trialing"
	SubscriptionActive   = "active"
	SubscriptionPastDue  = "past_due"
	SubscriptionCanceled = "canceled"
	SubscriptionUnpaid   = "unpaid"
)

// Subscriptions represents a recurring charge a customer's card.
//
// see https://stripe.com/docs/api#subscription_object
type Subscription struct {
	Customer           string `json:"customer"`
	Status             string `json:"status"`
	Plan               *Plan  `json:"plan"`
	Start              int64  `json:"start"`
	EndedAt            int64  `json:"ended_at"`
	CurrentPeriodStart int64  `json:"current_period_start"`
	CurrentPeriodEnd   int64  `json:"current_period_end"`
	TrialStart         int64  `json:"trial_start"`
	TrialEnd           int64  `json:"trial_end"`
	CanceledAt         int64  `json:"canceled_at"`
	CancelAtPeriodEnd  bool   `json:"cancel_at_period_end"`
	Quantity           int64  `json"quantity"`
}

// Plan holds details about pricing information for different products and
// feature levels on your site. For example, you might have a $10/month plan
// for basic features and a different $20/month plan for premium features.
//
// see https://stripe.com/docs/api#plan_object
type Plan struct {
	Id              string `json:"id"`
	Name            string `json:"name"`
	Amount          int64  `json:"amount"`
	Interval        string `json:"interval"`
	IntervalCount   int    `json:"interval_count"`
	Currency        string `json:"currency"`
	TrialPeriodDays int    `json:"trial_period_days"`
	Livemode        bool   `json:"livemode"`
}

func NewCustomer() *Customer {

	return &Customer{
		Id:         "hooN5ne7ug",
		Desc:       "A very nice customer.",
		Email:      "customer@example.com",
		Created:    time.Now().UnixNano(),
		Balance:    10,
		Delinquent: false,
		Cards: CardData{
			Object: "A92F4CFE-8B6B-4176-873E-887AC0D120EB",
			Count:  1,
			Url:    "https://stripe.example.com/card/A92F4CFE-8B6B-4176-873E-887AC0D120EB",
			Data: []*Card{
				&Card{
					Name:        "John Smith",
					Id:          "7526EC97-A0B6-47B2-AAE5-17443626A116",
					Fingerprint: "4242424242424242",
					ExpYear:     time.Now().Year() + 1,
					ExpMonth:    1,
				},
			},
		},
		Discount: &Discount{
			Id:       "Ee9ieZ8zie",
			Customer: "hooN5ne7ug",
			Start:    time.Now().UnixNano(),
			End:      time.Now().UnixNano(),
			Coupon: &Coupon{
				Id:               "ieQuo5Aiph",
				Duration:         "2m",
				PercentOff:       10,
				DurationInMonths: 2,
				MaxRedemptions:   1,
				RedeemBy:         time.Now().UnixNano(),
				TimesRedeemed:    1,
				Livemode:         true,
			},
		},
		Subscription: &Subscription{
			Customer: "hooN5ne7ug",
			Status:   SubscriptionActive,
			Plan: &Plan{
				Id:              "gaiyeLua5u",
				Name:            "Great Plan (TM)",
				Amount:          10,
				Interval:        "monthly",
				IntervalCount:   3,
				Currency:        "USD",
				TrialPeriodDays: 15,
				Livemode:        true,
			},
			Start:              time.Now().UnixNano(),
			EndedAt:            0,
			CurrentPeriodStart: time.Now().UnixNano(),
			CurrentPeriodEnd:   time.Now().UnixNano(),
			TrialStart:         time.Now().UnixNano(),
			TrialEnd:           time.Now().UnixNano(),
			CanceledAt:         0,
			CancelAtPeriodEnd:  false,
			Quantity:           2,
		},
		Livemode:    true,
		DefaultCard: "7526EC97-A0B6-47B2-AAE5-17443626A116",
	}
}
