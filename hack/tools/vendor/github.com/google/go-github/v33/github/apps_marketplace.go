// Copyright 2017 The go-github AUTHORS. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package github

import (
	"context"
	"fmt"
)

// MarketplaceService handles communication with the marketplace related
// methods of the GitHub API.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps#marketplace
type MarketplaceService struct {
	client *Client
	// Stubbed controls whether endpoints that return stubbed data are used
	// instead of production endpoints. Stubbed data is fake data that's useful
	// for testing your GitHub Apps. Stubbed data is hard-coded and will not
	// change based on actual subscriptions.
	//
	// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps#testing-with-stubbed-endpoints
	Stubbed bool
}

// MarketplacePlan represents a GitHub Apps Marketplace Listing Plan.
type MarketplacePlan struct {
	URL                 *string `json:"url,omitempty"`
	AccountsURL         *string `json:"accounts_url,omitempty"`
	ID                  *int64  `json:"id,omitempty"`
	Name                *string `json:"name,omitempty"`
	Description         *string `json:"description,omitempty"`
	MonthlyPriceInCents *int    `json:"monthly_price_in_cents,omitempty"`
	YearlyPriceInCents  *int    `json:"yearly_price_in_cents,omitempty"`
	// The pricing model for this listing.  Can be one of "flat-rate", "per-unit", or "free".
	PriceModel *string   `json:"price_model,omitempty"`
	UnitName   *string   `json:"unit_name,omitempty"`
	Bullets    *[]string `json:"bullets,omitempty"`
	// State can be one of the values "draft" or "published".
	State        *string `json:"state,omitempty"`
	HasFreeTrial *bool   `json:"has_free_trial,omitempty"`
}

// MarketplacePurchase represents a GitHub Apps Marketplace Purchase.
type MarketplacePurchase struct {
	// BillingCycle can be one of the values "yearly", "monthly" or nil.
	BillingCycle    *string                 `json:"billing_cycle,omitempty"`
	NextBillingDate *Timestamp              `json:"next_billing_date,omitempty"`
	UnitCount       *int                    `json:"unit_count,omitempty"`
	Plan            *MarketplacePlan        `json:"plan,omitempty"`
	Account         *MarketplacePlanAccount `json:"account,omitempty"`
	OnFreeTrial     *bool                   `json:"on_free_trial,omitempty"`
	FreeTrialEndsOn *Timestamp              `json:"free_trial_ends_on,omitempty"`
}

// MarketplacePendingChange represents a pending change to a GitHub Apps Marketplace Plan.
type MarketplacePendingChange struct {
	EffectiveDate *Timestamp       `json:"effective_date,omitempty"`
	UnitCount     *int             `json:"unit_count,omitempty"`
	ID            *int64           `json:"id,omitempty"`
	Plan          *MarketplacePlan `json:"plan,omitempty"`
}

// MarketplacePlanAccount represents a GitHub Account (user or organization) on a specific plan.
type MarketplacePlanAccount struct {
	URL                      *string                   `json:"url,omitempty"`
	Type                     *string                   `json:"type,omitempty"`
	ID                       *int64                    `json:"id,omitempty"`
	NodeID                   *string                   `json:"node_id,omitempty"`
	Login                    *string                   `json:"login,omitempty"`
	Email                    *string                   `json:"email,omitempty"`
	OrganizationBillingEmail *string                   `json:"organization_billing_email,omitempty"`
	MarketplacePurchase      *MarketplacePurchase      `json:"marketplace_purchase,omitempty"`
	MarketplacePendingChange *MarketplacePendingChange `json:"marketplace_pending_change,omitempty"`
}

// ListPlans lists all plans for your Marketplace listing.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps#list-plans
func (s *MarketplaceService) ListPlans(ctx context.Context, opts *ListOptions) ([]*MarketplacePlan, *Response, error) {
	uri := s.marketplaceURI("plans")
	u, err := addOptions(uri, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var plans []*MarketplacePlan
	resp, err := s.client.Do(ctx, req, &plans)
	if err != nil {
		return nil, resp, err
	}

	return plans, resp, nil
}

// ListPlanAccountsForPlan lists all GitHub accounts (user or organization) on a specific plan.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#list-all-github-accounts-user-or-organization-on-a-specific-plan
func (s *MarketplaceService) ListPlanAccountsForPlan(ctx context.Context, planID int64, opts *ListOptions) ([]*MarketplacePlanAccount, *Response, error) {
	uri := s.marketplaceURI(fmt.Sprintf("plans/%v/accounts", planID))
	u, err := addOptions(uri, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var accounts []*MarketplacePlanAccount
	resp, err := s.client.Do(ctx, req, &accounts)
	if err != nil {
		return nil, resp, err
	}

	return accounts, resp, nil
}

// ListPlanAccountsForAccount lists all GitHub accounts (user or organization) associated with an account.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#check-if-a-github-account-is-associated-with-any-marketplace-listing
func (s *MarketplaceService) ListPlanAccountsForAccount(ctx context.Context, accountID int64, opts *ListOptions) ([]*MarketplacePlanAccount, *Response, error) {
	uri := s.marketplaceURI(fmt.Sprintf("accounts/%v", accountID))
	u, err := addOptions(uri, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var accounts []*MarketplacePlanAccount
	resp, err := s.client.Do(ctx, req, &accounts)
	if err != nil {
		return nil, resp, err
	}

	return accounts, resp, nil
}

// ListMarketplacePurchasesForUser lists all GitHub marketplace purchases made by a user.
//
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#list-subscriptions-for-the-authenticated-user-stubbed
// GitHub API docs: https://docs.github.com/en/free-pro-team@latest/rest/reference/apps/#list-subscriptions-for-the-authenticated-user
func (s *MarketplaceService) ListMarketplacePurchasesForUser(ctx context.Context, opts *ListOptions) ([]*MarketplacePurchase, *Response, error) {
	uri := "user/marketplace_purchases"
	if s.Stubbed {
		uri = "user/marketplace_purchases/stubbed"
	}

	u, err := addOptions(uri, opts)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", u, nil)
	if err != nil {
		return nil, nil, err
	}

	var purchases []*MarketplacePurchase
	resp, err := s.client.Do(ctx, req, &purchases)
	if err != nil {
		return nil, resp, err
	}
	return purchases, resp, nil
}

func (s *MarketplaceService) marketplaceURI(endpoint string) string {
	url := "marketplace_listing"
	if s.Stubbed {
		url = "marketplace_listing/stubbed"
	}
	return url + "/" + endpoint
}
