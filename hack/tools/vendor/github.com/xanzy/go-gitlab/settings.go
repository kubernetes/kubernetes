//
// Copyright 2021, Sander van Harmelen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package gitlab

import "time"

// SettingsService handles communication with the application SettingsService
// related methods of the GitLab API.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/settings.html
type SettingsService struct {
	client *Client
}

// Settings represents the GitLab application settings.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/settings.html
type Settings struct {
	ID                                        int               `json:"id"`
	CreatedAt                                 *time.Time        `json:"created_at"`
	UpdatedAt                                 *time.Time        `json:"updated_at"`
	AdminNotificationEmail                    string            `json:"admin_notification_email"`
	AfterSignOutPath                          string            `json:"after_sign_out_path"`
	AfterSignUpText                           string            `json:"after_sign_up_text"`
	AkismetAPIKey                             string            `json:"akismet_api_key"`
	AkismetEnabled                            bool              `json:"akismet_enabled"`
	AllowGroupOwnersToManageLDAP              bool              `json:"allow_group_owners_to_manage_ldap"`
	AllowLocalRequestsFromHooksAndServices    bool              `json:"allow_local_requests_from_hooks_and_services"`
	AllowLocalRequestsFromSystemHooks         bool              `json:"allow_local_requests_from_system_hooks"`
	AllowLocalRequestsFromWebHooksAndServices bool              `json:"allow_local_requests_from_web_hooks_and_services"`
	ArchiveBuildsInHumanReadable              string            `json:"archive_builds_in_human_readable"`
	AssetProxyEnabled                         bool              `json:"asset_proxy_enabled"`
	AssetProxySecretKey                       string            `json:"asset_proxy_secret_key"`
	AssetProxyURL                             string            `json:"asset_proxy_url"`
	AssetProxyWhitelist                       []string          `json:"asset_proxy_whitelist"`
	AuthorizedKeysEnabled                     bool              `json:"authorized_keys_enabled_enabled"`
	AutoDevOpsDomain                          string            `json:"auto_devops_domain"`
	AutoDevOpsEnabled                         bool              `json:"auto_devops_enabled"`
	CheckNamespacePlan                        bool              `json:"check_namespace_plan"`
	CommitEmailHostname                       string            `json:"commit_email_hostname"`
	ContainerRegistryTokenExpireDelay         int               `json:"container_registry_token_expire_delay"`
	DefaultArtifactsExpireIn                  string            `json:"default_artifacts_expire_in"`
	DefaultBranchProtection                   int               `json:"default_branch_protection"`
	DefaultGroupVisibility                    VisibilityValue   `json:"default_group_visibility"`
	DefaultProjectCreation                    int               `json:"default_project_creation"`
	DefaultProjectsLimit                      int               `json:"default_projects_limit"`
	DefaultProjectVisibility                  VisibilityValue   `json:"default_project_visibility"`
	DefaultSnippetVisibility                  VisibilityValue   `json:"default_snippet_visibility"`
	DiffMaxPatchBytes                         int               `json:"diff_max_patch_bytes"`
	DisabledOauthSignInSources                []string          `json:"disabled_oauth_sign_in_sources"`
	DNSRebindingProtectionEnabled             bool              `json:"dns_rebinding_protection_enabled"`
	DomainBlacklist                           []string          `json:"domain_blacklist"`
	DomainBlacklistEnabled                    bool              `json:"domain_blacklist_enabled"`
	DomainWhitelist                           []string          `json:"domain_whitelist"`
	DSAKeyRestriction                         int               `json:"dsa_key_restriction"`
	ECDSAKeyRestriction                       int               `json:"ecdsa_key_restriction"`
	Ed25519KeyRestriction                     int               `json:"ed25519_key_restriction"`
	ElasticsearchAWSAccessKey                 string            `json:"elasticsearch_aws_access_key"`
	ElasticsearchAWS                          bool              `json:"elasticsearch_aws"`
	ElasticsearchAWSRegion                    string            `json:"elasticsearch_aws_region"`
	ElasticsearchAWSSecretAccessKey           string            `json:"elasticsearch_aws_secret_access_key"`
	ElasticsearchIndexing                     bool              `json:"elasticsearch_indexing"`
	ElasticsearchLimitIndexing                bool              `json:"elasticsearch_limit_indexing"`
	ElasticsearchNamespaceIDs                 []int             `json:"elasticsearch_namespace_ids"`
	ElasticsearchProjectIDs                   []int             `json:"elasticsearch_project_ids"`
	ElasticsearchSearch                       bool              `json:"elasticsearch_search"`
	ElasticsearchURL                          []string          `json:"elasticsearch_url"`
	EmailAdditionalText                       string            `json:"email_additional_text"`
	EmailAuthorInBody                         bool              `json:"email_author_in_body"`
	EnabledGitAccessProtocol                  string            `json:"enabled_git_access_protocol"`
	EnforceTerms                              bool              `json:"enforce_terms"`
	ExternalAuthClientCert                    string            `json:"external_auth_client_cert"`
	ExternalAuthClientKeyPass                 string            `json:"external_auth_client_key_pass"`
	ExternalAuthClientKey                     string            `json:"external_auth_client_key"`
	ExternalAuthorizationServiceDefaultLabel  string            `json:"external_authorization_service_default_label"`
	ExternalAuthorizationServiceEnabled       bool              `json:"external_authorization_service_enabled"`
	ExternalAuthorizationServiceTimeout       float64           `json:"external_authorization_service_timeout"`
	ExternalAuthorizationServiceURL           string            `json:"external_authorization_service_url"`
	FileTemplateProjectID                     int               `json:"file_template_project_id"`
	FirstDayOfWeek                            int               `json:"first_day_of_week"`
	GeoNodeAllowedIPs                         string            `json:"geo_node_allowed_ips"`
	GeoStatusTimeout                          int               `json:"geo_status_timeout"`
	GitalyTimeoutDefault                      int               `json:"gitaly_timeout_default"`
	GitalyTimeoutFast                         int               `json:"gitaly_timeout_fast"`
	GitalyTimeoutMedium                       int               `json:"gitaly_timeout_medium"`
	GrafanaEnabled                            bool              `json:"grafana_enabled"`
	GrafanaURL                                string            `json:"grafana_url"`
	GravatarEnabled                           bool              `json:"gravatar_enabled"`
	HashedStorageEnabled                      bool              `json:"hashed_storage_enabled"`
	HelpPageHideCommercialContent             bool              `json:"help_page_hide_commercial_content"`
	HelpPageSupportURL                        string            `json:"help_page_support_url"`
	HelpPageText                              string            `json:"help_page_text"`
	HelpText                                  string            `json:"help_text"`
	HideThirdPartyOffers                      bool              `json:"hide_third_party_offers"`
	HomePageURL                               string            `json:"home_page_url"`
	HousekeepingBitmapsEnabled                bool              `json:"housekeeping_bitmaps_enabled"`
	HousekeepingEnabled                       bool              `json:"housekeeping_enabled"`
	HousekeepingFullRepackPeriod              int               `json:"housekeeping_full_repack_period"`
	HousekeepingGcPeriod                      int               `json:"housekeeping_gc_period"`
	HousekeepingIncrementalRepackPeriod       int               `json:"housekeeping_incremental_repack_period"`
	HTMLEmailsEnabled                         bool              `json:"html_emails_enabled"`
	ImportSources                             []string          `json:"import_sources"`
	InstanceStatisticsVisibilityPrivate       bool              `json:"instance_statistics_visibility_private"`
	LocalMarkdownVersion                      int               `json:"local_markdown_version"`
	MaxArtifactsSize                          int               `json:"max_artifacts_size"`
	MaxAttachmentSize                         int               `json:"max_attachment_size"`
	MaxPagesSize                              int               `json:"max_pages_size"`
	MetricsEnabled                            bool              `json:"metrics_enabled"`
	MetricsHost                               string            `json:"metrics_host"`
	MetricsMethodCallThreshold                int               `json:"metrics_method_call_threshold"`
	MetricsPacketSize                         int               `json:"metrics_packet_size"`
	MetricsPoolSize                           int               `json:"metrics_pool_size"`
	MetricsPort                               int               `json:"metrics_port"`
	MetricsSampleInterval                     int               `json:"metrics_sample_interval"`
	MetricsTimeout                            int               `json:"metrics_timeout"`
	MirrorAvailable                           bool              `json:"mirror_available"`
	MirrorCapacityThreshold                   int               `json:"mirror_capacity_threshold"`
	MirrorMaxCapacity                         int               `json:"mirror_max_capacity"`
	MirrorMaxDelay                            int               `json:"mirror_max_delay"`
	OutboundLocalRequestsWhitelist            []string          `json:"outbound_local_requests_whitelist"`
	PagesDomainVerificationEnabled            bool              `json:"pages_domain_verification_enabled"`
	PasswordAuthenticationEnabledForGit       bool              `json:"password_authentication_enabled_for_git"`
	PasswordAuthenticationEnabledForWeb       bool              `json:"password_authentication_enabled_for_web"`
	PerformanceBarAllowedGroupID              string            `json:"performance_bar_allowed_group_id"`
	PerformanceBarAllowedGroupPath            string            `json:"performance_bar_allowed_group_path"`
	PerformanceBarEnabled                     bool              `json:"performance_bar_enabled"`
	PlantumlEnabled                           bool              `json:"plantuml_enabled"`
	PlantumlURL                               string            `json:"plantuml_url"`
	PollingIntervalMultiplier                 float64           `json:"polling_interval_multiplier,string"`
	ProjectExportEnabled                      bool              `json:"project_export_enabled"`
	PrometheusMetricsEnabled                  bool              `json:"prometheus_metrics_enabled"`
	ProtectedCIVariables                      bool              `json:"protected_ci_variables"`
	PseudonymizerEnabled                      bool              `json:"psedonymizer_enabled"`
	PushEventHooksLimit                       int               `json:"push_event_hooks_limit"`
	PushEventActivitiesLimit                  int               `json:"push_event_activities_limit"`
	RecaptchaEnabled                          bool              `json:"recaptcha_enabled"`
	RecaptchaPrivateKey                       string            `json:"recaptcha_private_key"`
	RecaptchaSiteKey                          string            `json:"recaptcha_site_key"`
	ReceiveMaxInputSize                       int               `json:"receive_max_input_size"`
	RepositoryChecksEnabled                   bool              `json:"repository_checks_enabled"`
	RepositorySizeLimit                       int               `json:"repository_size_limit"`
	RepositoryStorages                        []string          `json:"repository_storages"`
	RequireTwoFactorAuthentication            bool              `json:"require_two_factor_authentication"`
	RestrictedVisibilityLevels                []VisibilityValue `json:"restricted_visibility_levels"`
	RsaKeyRestriction                         int               `json:"rsa_key_restriction"`
	SendUserConfirmationEmail                 bool              `json:"send_user_confirmation_email"`
	SessionExpireDelay                        int               `json:"session_expire_delay"`
	SharedRunnersEnabled                      bool              `json:"shared_runners_enabled"`
	SharedRunnersMinutes                      int               `json:"shared_runners_minutes"`
	SharedRunnersText                         string            `json:"shared_runners_text"`
	SignInText                                string            `json:"sign_in_text"`
	SignupEnabled                             bool              `json:"signup_enabled"`
	SlackAppEnabled                           bool              `json:"slack_app_enabled"`
	SlackAppID                                string            `json:"slack_app_id"`
	SlackAppSecret                            string            `json:"slack_app_secret"`
	SlackAppVerificationToken                 string            `json:"slack_app_verification_token"`
	SnowplowCollectorHostname                 string            `json:"snowplow_collector_hostname"`
	SnowplowCookieDomain                      string            `json:"snowplow_cookie_domain"`
	SnowplowEnabled                           bool              `json:"snowplow_enabled"`
	SnowplowSiteID                            string            `json:"snowplow_site_id"`
	TerminalMaxSessionTime                    int               `json:"terminal_max_session_time"`
	Terms                                     string            `json:"terms"`
	ThrottleAuthenticatedAPIEnabled           bool              `json:"throttle_authenticated_api_enabled"`
	ThrottleAuthenticatedAPIPeriodInSeconds   int               `json:"throttle_authenticated_api_period_in_seconds"`
	ThrottleAuthenticatedAPIRequestsPerPeriod int               `json:"throttle_authenticated_api_requests_per_period"`
	ThrottleAuthenticatedWebEnabled           bool              `json:"throttle_authenticated_web_enabled"`
	ThrottleAuthenticatedWebPeriodInSeconds   int               `json:"throttle_authenticated_web_period_in_seconds"`
	ThrottleAuthenticatedWebRequestsPerPeriod int               `json:"throttle_authenticated_web_requests_per_period"`
	ThrottleUnauthenticatedEnabled            bool              `json:"throttle_unauthenticated_enabled"`
	ThrottleUnauthenticatedPeriodInSeconds    int               `json:"throttle_unauthenticated_period_in_seconds"`
	ThrottleUnauthenticatedRequestsPerPeriod  int               `json:"throttle_unauthenticated_requests_per_period"`
	TimeTrackingLimitToHours                  bool              `json:"time_tracking_limit_to_hours"`
	TwoFactorGracePeriod                      int               `json:"two_factor_grace_period"`
	UniqueIPsLimitEnabled                     bool              `json:"unique_ips_limit_enabled"`
	UniqueIPsLimitPerUser                     int               `json:"unique_ips_limit_per_user"`
	UniqueIPsLimitTimeWindow                  int               `json:"unique_ips_limit_time_window"`
	UsagePingEnabled                          bool              `json:"usage_ping_enabled"`
	UserDefaultExternal                       bool              `json:"user_default_external"`
	UserDefaultInternalRegex                  string            `json:"user_default_internal_regex"`
	UserOauthApplications                     bool              `json:"user_oauth_applications"`
	UserShowAddSSHKeyMessage                  bool              `json:"user_show_add_ssh_key_message"`
	VersionCheckEnabled                       bool              `json:"version_check_enabled"`
	WebIDEClientsidePreviewEnabled            bool              `json:"web_ide_clientside_preview_enabled"`
}

func (s Settings) String() string {
	return Stringify(s)
}

// GetSettings gets the current application settings.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/settings.html#get-current-application.settings
func (s *SettingsService) GetSettings(options ...RequestOptionFunc) (*Settings, *Response, error) {
	req, err := s.client.NewRequest("GET", "application/settings", nil, options)
	if err != nil {
		return nil, nil, err
	}

	as := new(Settings)
	resp, err := s.client.Do(req, as)
	if err != nil {
		return nil, resp, err
	}

	return as, resp, err
}

// UpdateSettingsOptions represents the available UpdateSettings() options.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/settings.html#change-application.settings
type UpdateSettingsOptions struct {
	AdminNotificationEmail                    *string           `url:"admin_notification_email,omitempty" json:"admin_notification_email,omitempty"`
	AfterSignOutPath                          *string           `url:"after_sign_out_path,omitempty" json:"after_sign_out_path,omitempty"`
	AfterSignUpText                           *string           `url:"after_sign_up_text,omitempty" json:"after_sign_up_text,omitempty"`
	AkismetAPIKey                             *string           `url:"akismet_api_key,omitempty" json:"akismet_api_key,omitempty"`
	AkismetEnabled                            *bool             `url:"akismet_enabled,omitempty" json:"akismet_enabled,omitempty"`
	AllowGroupOwnersToManageLDAP              *bool             `url:"allow_group_owners_to_manage_ldap,omitempty" json:"allow_group_owners_to_manage_ldap,omitempty"`
	AllowLocalRequestsFromHooksAndServices    *bool             `url:"allow_local_requests_from_hooks_and_services,omitempty" json:"allow_local_requests_from_hooks_and_services,omitempty"`
	AllowLocalRequestsFromSystemHooks         *bool             `url:"allow_local_requests_from_system_hooks,omitempty" json:"allow_local_requests_from_system_hooks,omitempty"`
	AllowLocalRequestsFromWebHooksAndServices *bool             `url:"allow_local_requests_from_web_hooks_and_services,omitempty" json:"allow_local_requests_from_web_hooks_and_services,omitempty"`
	ArchiveBuildsInHumanReadable              *string           `url:"archive_builds_in_human_readable,omitempty" json:"archive_builds_in_human_readable,omitempty"`
	AssetProxyEnabled                         *bool             `url:"asset_proxy_enabled,omitempty" json:"asset_proxy_enabled,omitempty"`
	AssetProxySecretKey                       *string           `url:"asset_proxy_secret_key,omitempty" json:"asset_proxy_secret_key,omitempty"`
	AssetProxyURL                             *string           `url:"asset_proxy_url,omitempty" json:"asset_proxy_url,omitempty"`
	AssetProxyWhitelist                       []string          `url:"asset_proxy_whitelist,omitempty" json:"asset_proxy_whitelist,omitempty"`
	AuthorizedKeysEnabled                     *bool             `url:"authorized_keys_enabled,omitempty" json:"authorized_keys_enabled,omitempty"`
	AutoDevOpsDomain                          *string           `url:"auto_devops_domain,omitempty" json:"auto_devops_domain,omitempty"`
	AutoDevOpsEnabled                         *bool             `url:"auto_devops_enabled,omitempty" json:"auto_devops_enabled,omitempty"`
	CheckNamespacePlan                        *bool             `url:"check_namespace_plan,omitempty" json:"check_namespace_plan,omitempty"`
	CommitEmailHostname                       *string           `url:"commit_email_hostname,omitempty" json:"commit_email_hostname,omitempty"`
	ContainerRegistryTokenExpireDelay         *int              `url:"container_registry_token_expire_delay,omitempty" json:"container_registry_token_expire_delay,omitempty"`
	DefaultArtifactsExpireIn                  *string           `url:"default_artifacts_expire_in,omitempty" json:"default_artifacts_expire_in,omitempty"`
	DefaultBranchProtection                   *int              `url:"default_branch_protection,omitempty" json:"default_branch_protection,omitempty"`
	DefaultGroupVisibility                    *VisibilityValue  `url:"default_group_visibility,omitempty" json:"default_group_visibility,omitempty"`
	DefaultProjectCreation                    *int              `url:"default_project_creation,omitempty" json:"default_project_creation,omitempty"`
	DefaultProjectsLimit                      *int              `url:"default_projects_limit,omitempty" json:"default_projects_limit,omitempty"`
	DefaultProjectVisibility                  *VisibilityValue  `url:"default_project_visibility,omitempty" json:"default_project_visibility,omitempty"`
	DefaultSnippetVisibility                  *VisibilityValue  `url:"default_snippet_visibility,omitempty" json:"default_snippet_visibility,omitempty"`
	DiffMaxPatchBytes                         *int              `url:"diff_max_patch_bytes,omitempty" json:"diff_max_patch_bytes,omitempty"`
	DisabledOauthSignInSources                []string          `url:"disabled_oauth_sign_in_sources,omitempty" json:"disabled_oauth_sign_in_sources,omitempty"`
	DNSRebindingProtectionEnabled             *bool             `url:"dns_rebinding_protection_enabled,omitempty" json:"dns_rebinding_protection_enabled,omitempty"`
	DomainBlacklist                           []string          `url:"domain_blacklist,omitempty" json:"domain_blacklist,omitempty"`
	DomainBlacklistEnabled                    *bool             `url:"domain_blacklist_enabled,omitempty" json:"domain_blacklist_enabled,omitempty"`
	DomainWhitelist                           []string          `url:"domain_whitelist,omitempty" json:"domain_whitelist,omitempty"`
	DSAKeyRestriction                         *int              `url:"dsa_key_restriction,omitempty" json:"dsa_key_restriction,omitempty"`
	ECDSAKeyRestriction                       *int              `url:"ecdsa_key_restriction,omitempty" json:"ecdsa_key_restriction,omitempty"`
	Ed25519KeyRestriction                     *int              `url:"ed25519_key_restriction,omitempty" json:"ed25519_key_restriction,omitempty"`
	ElasticsearchAWSAccessKey                 *string           `url:"elasticsearch_aws_access_key,omitempty" json:"elasticsearch_aws_access_key,omitempty"`
	ElasticsearchAWS                          *bool             `url:"elasticsearch_aws,omitempty" json:"elasticsearch_aws,omitempty"`
	ElasticsearchAWSRegion                    *string           `url:"elasticsearch_aws_region,omitempty" json:"elasticsearch_aws_region,omitempty"`
	ElasticsearchAWSSecretAccessKey           *string           `url:"elasticsearch_aws_secret_access_key,omitempty" json:"elasticsearch_aws_secret_access_key,omitempty"`
	ElasticsearchIndexing                     *bool             `url:"elasticsearch_indexing,omitempty" json:"elasticsearch_indexing,omitempty"`
	ElasticsearchLimitIndexing                *bool             `url:"elasticsearch_limit_indexing,omitempty" json:"elasticsearch_limit_indexing,omitempty"`
	ElasticsearchNamespaceIDs                 []int             `url:"elasticsearch_namespace_ids,omitempty" json:"elasticsearch_namespace_ids,omitempty"`
	ElasticsearchProjectIDs                   []int             `url:"elasticsearch_project_ids,omitempty" json:"elasticsearch_project_ids,omitempty"`
	ElasticsearchSearch                       *bool             `url:"elasticsearch_search,omitempty" json:"elasticsearch_search,omitempty"`
	ElasticsearchURL                          *string           `url:"elasticsearch_url,omitempty" json:"elasticsearch_url,omitempty"`
	EmailAdditionalText                       *string           `url:"email_additional_text,omitempty" json:"email_additional_text,omitempty"`
	EmailAuthorInBody                         *bool             `url:"email_author_in_body,omitempty" json:"email_author_in_body,omitempty"`
	EnabledGitAccessProtocol                  *string           `url:"enabled_git_access_protocol,omitempty" json:"enabled_git_access_protocol,omitempty"`
	EnforceTerms                              *bool             `url:"enforce_terms,omitempty" json:"enforce_terms,omitempty"`
	ExternalAuthClientCert                    *string           `url:"external_auth_client_cert,omitempty" json:"external_auth_client_cert,omitempty"`
	ExternalAuthClientKeyPass                 *string           `url:"external_auth_client_key_pass,omitempty" json:"external_auth_client_key_pass,omitempty"`
	ExternalAuthClientKey                     *string           `url:"external_auth_client_key,omitempty" json:"external_auth_client_key,omitempty"`
	ExternalAuthorizationServiceDefaultLabel  *string           `url:"external_authorization_service_default_label,omitempty" json:"external_authorization_service_default_label,omitempty"`
	ExternalAuthorizationServiceEnabled       *bool             `url:"external_authorization_service_enabled,omitempty" json:"external_authorization_service_enabled,omitempty"`
	ExternalAuthorizationServiceTimeout       *float64          `url:"external_authorization_service_timeout,omitempty" json:"external_authorization_service_timeout,omitempty"`
	ExternalAuthorizationServiceURL           *string           `url:"external_authorization_service_url,omitempty" json:"external_authorization_service_url,omitempty"`
	FileTemplateProjectID                     *int              `url:"file_template_project_id,omitempty" json:"file_template_project_id,omitempty"`
	FirstDayOfWeek                            *int              `url:"first_day_of_week,omitempty" json:"first_day_of_week,omitempty"`
	GeoNodeAllowedIPs                         *string           `url:"geo_node_allowed_ips,omitempty" json:"geo_node_allowed_ips,omitempty"`
	GeoStatusTimeout                          *int              `url:"geo_status_timeout,omitempty" json:"geo_status_timeout,omitempty"`
	GitalyTimeoutDefault                      *int              `url:"gitaly_timeout_default,omitempty" json:"gitaly_timeout_default,omitempty"`
	GitalyTimeoutFast                         *int              `url:"gitaly_timeout_fast,omitempty" json:"gitaly_timeout_fast,omitempty"`
	GitalyTimeoutMedium                       *int              `url:"gitaly_timeout_medium,omitempty" json:"gitaly_timeout_medium,omitempty"`
	GrafanaEnabled                            *bool             `url:"grafana_enabled,omitempty" json:"grafana_enabled,omitempty"`
	GrafanaURL                                *string           `url:"grafana_url,omitempty" json:"grafana_url,omitempty"`
	GravatarEnabled                           *bool             `url:"gravatar_enabled,omitempty" json:"gravatar_enabled,omitempty"`
	HashedStorageEnabled                      *bool             `url:"hashed_storage_enabled,omitempty" json:"hashed_storage_enabled,omitempty"`
	HelpPageHideCommercialContent             *bool             `url:"help_page_hide_commercial_content,omitempty" json:"help_page_hide_commercial_content,omitempty"`
	HelpPageSupportURL                        *string           `url:"help_page_support_url,omitempty" json:"help_page_support_url,omitempty"`
	HelpPageText                              *string           `url:"help_page_text,omitempty" json:"help_page_text,omitempty"`
	HelpText                                  *string           `url:"help_text,omitempty" json:"help_text,omitempty"`
	HideThirdPartyOffers                      *bool             `url:"hide_third_party_offers,omitempty" json:"hide_third_party_offers,omitempty"`
	HomePageURL                               *string           `url:"home_page_url,omitempty" json:"home_page_url,omitempty"`
	HousekeepingBitmapsEnabled                *bool             `url:"housekeeping_bitmaps_enabled,omitempty" json:"housekeeping_bitmaps_enabled,omitempty"`
	HousekeepingEnabled                       *bool             `url:"housekeeping_enabled,omitempty" json:"housekeeping_enabled,omitempty"`
	HousekeepingFullRepackPeriod              *int              `url:"housekeeping_full_repack_period,omitempty" json:"housekeeping_full_repack_period,omitempty"`
	HousekeepingGcPeriod                      *int              `url:"housekeeping_gc_period,omitempty" json:"housekeeping_gc_period,omitempty"`
	HousekeepingIncrementalRepackPeriod       *int              `url:"housekeeping_incremental_repack_period,omitempty" json:"housekeeping_incremental_repack_period,omitempty"`
	HTMLEmailsEnabled                         *bool             `url:"html_emails_enabled,omitempty" json:"html_emails_enabled,omitempty"`
	ImportSources                             []string          `url:"import_sources,omitempty" json:"import_sources,omitempty"`
	InstanceStatisticsVisibilityPrivate       *bool             `url:"instance_statistics_visibility_private,omitempty" json:"instance_statistics_visibility_private,omitempty"`
	LocalMarkdownVersion                      *int              `url:"local_markdown_version,omitempty" json:"local_markdown_version,omitempty"`
	MaxArtifactsSize                          *int              `url:"max_artifacts_size,omitempty" json:"max_artifacts_size,omitempty"`
	MaxAttachmentSize                         *int              `url:"max_attachment_size,omitempty" json:"max_attachment_size,omitempty"`
	MaxPagesSize                              *int              `url:"max_pages_size,omitempty" json:"max_pages_size,omitempty"`
	MetricsEnabled                            *bool             `url:"metrics_enabled,omitempty" json:"metrics_enabled,omitempty"`
	MetricsHost                               *string           `url:"metrics_host,omitempty" json:"metrics_host,omitempty"`
	MetricsMethodCallThreshold                *int              `url:"metrics_method_call_threshold,omitempty" json:"metrics_method_call_threshold,omitempty"`
	MetricsPacketSize                         *int              `url:"metrics_packet_size,omitempty" json:"metrics_packet_size,omitempty"`
	MetricsPoolSize                           *int              `url:"metrics_pool_size,omitempty" json:"metrics_pool_size,omitempty"`
	MetricsPort                               *int              `url:"metrics_port,omitempty" json:"metrics_port,omitempty"`
	MetricsSampleInterval                     *int              `url:"metrics_sample_interval,omitempty" json:"metrics_sample_interval,omitempty"`
	MetricsTimeout                            *int              `url:"metrics_timeout,omitempty" json:"metrics_timeout,omitempty"`
	MirrorAvailable                           *bool             `url:"mirror_available,omitempty" json:"mirror_available,omitempty"`
	MirrorCapacityThreshold                   *int              `url:"mirror_capacity_threshold,omitempty" json:"mirror_capacity_threshold,omitempty"`
	MirrorMaxCapacity                         *int              `url:"mirror_max_capacity,omitempty" json:"mirror_max_capacity,omitempty"`
	MirrorMaxDelay                            *int              `url:"mirror_max_delay,omitempty" json:"mirror_max_delay,omitempty"`
	OutboundLocalRequestsWhitelist            []string          `url:"outbound_local_requests_whitelist,omitempty" json:"outbound_local_requests_whitelist,omitempty"`
	PagesDomainVerificationEnabled            *bool             `url:"pages_domain_verification_enabled,omitempty" json:"pages_domain_verification_enabled,omitempty"`
	PasswordAuthenticationEnabledForGit       *bool             `url:"password_authentication_enabled_for_git,omitempty" json:"password_authentication_enabled_for_git,omitempty"`
	PasswordAuthenticationEnabledForWeb       *bool             `url:"password_authentication_enabled_for_web,omitempty" json:"password_authentication_enabled_for_web,omitempty"`
	PerformanceBarAllowedGroupID              *string           `url:"performance_bar_allowed_group_id,omitempty" json:"performance_bar_allowed_group_id,omitempty"`
	PerformanceBarAllowedGroupPath            *string           `url:"performance_bar_allowed_group_path,omitempty" json:"performance_bar_allowed_group_path,omitempty"`
	PerformanceBarEnabled                     *bool             `url:"performance_bar_enabled,omitempty" json:"performance_bar_enabled,omitempty"`
	PlantumlEnabled                           *bool             `url:"plantuml_enabled,omitempty" json:"plantuml_enabled,omitempty"`
	PlantumlURL                               *string           `url:"plantuml_url,omitempty" json:"plantuml_url,omitempty"`
	PollingIntervalMultiplier                 *float64          `url:"polling_interval_multiplier,omitempty" json:"polling_interval_multiplier,omitempty"`
	ProjectExportEnabled                      *bool             `url:"project_export_enabled,omitempty" json:"project_export_enabled,omitempty"`
	PrometheusMetricsEnabled                  *bool             `url:"prometheus_metrics_enabled,omitempty" json:"prometheus_metrics_enabled,omitempty"`
	ProtectedCIVariables                      *bool             `url:"protected_ci_variables,omitempty" json:"protected_ci_variables,omitempty"`
	PseudonymizerEnabled                      *bool             `url:"psedonymizer_enabled,omitempty" json:"psedonymizer_enabled,omitempty"`
	PushEventHooksLimit                       *int              `url:"push_event_hooks_limit,omitempty" json:"push_event_hooks_limit,omitempty"`
	PushEventActivitiesLimit                  *int              `url:"push_event_activities_limit,omitempty" json:"push_event_activities_limit,omitempty"`
	RecaptchaEnabled                          *bool             `url:"recaptcha_enabled,omitempty" json:"recaptcha_enabled,omitempty"`
	RecaptchaPrivateKey                       *string           `url:"recaptcha_private_key,omitempty" json:"recaptcha_private_key,omitempty"`
	RecaptchaSiteKey                          *string           `url:"recaptcha_site_key,omitempty" json:"recaptcha_site_key,omitempty"`
	ReceiveMaxInputSize                       *int              `url:"receive_max_input_size,omitempty" json:"receive_max_input_size,omitempty"`
	RepositoryChecksEnabled                   *bool             `url:"repository_checks_enabled,omitempty" json:"repository_checks_enabled,omitempty"`
	RepositorySizeLimit                       *int              `url:"repository_size_limit,omitempty" json:"repository_size_limit,omitempty"`
	RepositoryStorages                        []string          `url:"repository_storages,omitempty" json:"repository_storages,omitempty"`
	RequireTwoFactorAuthentication            *bool             `url:"require_two_factor_authentication,omitempty" json:"require_two_factor_authentication,omitempty"`
	RestrictedVisibilityLevels                []VisibilityValue `url:"restricted_visibility_levels,omitempty" json:"restricted_visibility_levels,omitempty"`
	RsaKeyRestriction                         *int              `url:"rsa_key_restriction,omitempty" json:"rsa_key_restriction,omitempty"`
	SendUserConfirmationEmail                 *bool             `url:"send_user_confirmation_email,omitempty" json:"send_user_confirmation_email,omitempty"`
	SessionExpireDelay                        *int              `url:"session_expire_delay,omitempty" json:"session_expire_delay,omitempty"`
	SharedRunnersEnabled                      *bool             `url:"shared_runners_enabled,omitempty" json:"shared_runners_enabled,omitempty"`
	SharedRunnersMinutes                      *int              `url:"shared_runners_minutes,omitempty" json:"shared_runners_minutes,omitempty"`
	SharedRunnersText                         *string           `url:"shared_runners_text,omitempty" json:"shared_runners_text,omitempty"`
	SignInText                                *string           `url:"sign_in_text,omitempty" json:"sign_in_text,omitempty"`
	SignupEnabled                             *bool             `url:"signup_enabled,omitempty" json:"signup_enabled,omitempty"`
	SlackAppEnabled                           *bool             `url:"slack_app_enabled,omitempty" json:"slack_app_enabled,omitempty"`
	SlackAppID                                *string           `url:"slack_app_id,omitempty" json:"slack_app_id,omitempty"`
	SlackAppSecret                            *string           `url:"slack_app_secret,omitempty" json:"slack_app_secret,omitempty"`
	SlackAppVerificationToken                 *string           `url:"slack_app_verification_token,omitempty" json:"slack_app_verification_token,omitempty"`
	SnowplowCollectorHostname                 *string           `url:"snowplow_collector_hostname,omitempty" json:"snowplow_collector_hostname,omitempty"`
	SnowplowCookieDomain                      *string           `url:"snowplow_cookie_domain,omitempty" json:"snowplow_cookie_domain,omitempty"`
	SnowplowEnabled                           *bool             `url:"snowplow_enabled,omitempty" json:"snowplow_enabled,omitempty"`
	SnowplowSiteID                            *string           `url:"snowplow_site_id,omitempty" json:"snowplow_site_id,omitempty"`
	TerminalMaxSessionTime                    *int              `url:"terminal_max_session_time,omitempty" json:"terminal_max_session_time,omitempty"`
	Terms                                     *string           `url:"terms,omitempty" json:"terms,omitempty"`
	ThrottleAuthenticatedAPIEnabled           *bool             `url:"throttle_authenticated_api_enabled,omitempty" json:"throttle_authenticated_api_enabled,omitempty"`
	ThrottleAuthenticatedAPIPeriodInSeconds   *int              `url:"throttle_authenticated_api_period_in_seconds,omitempty" json:"throttle_authenticated_api_period_in_seconds,omitempty"`
	ThrottleAuthenticatedAPIRequestsPerPeriod *int              `url:"throttle_authenticated_api_requests_per_period,omitempty" json:"throttle_authenticated_api_requests_per_period,omitempty"`
	ThrottleAuthenticatedWebEnabled           *bool             `url:"throttle_authenticated_web_enabled,omitempty" json:"throttle_authenticated_web_enabled,omitempty"`
	ThrottleAuthenticatedWebPeriodInSeconds   *int              `url:"throttle_authenticated_web_period_in_seconds,omitempty" json:"throttle_authenticated_web_period_in_seconds,omitempty"`
	ThrottleAuthenticatedWebRequestsPerPeriod *int              `url:"throttle_authenticated_web_requests_per_period,omitempty" json:"throttle_authenticated_web_requests_per_period,omitempty"`
	ThrottleUnauthenticatedEnabled            *bool             `url:"throttle_unauthenticated_enabled,omitempty" json:"throttle_unauthenticated_enabled,omitempty"`
	ThrottleUnauthenticatedPeriodInSeconds    *int              `url:"throttle_unauthenticated_period_in_seconds,omitempty" json:"throttle_unauthenticated_period_in_seconds,omitempty"`
	ThrottleUnauthenticatedRequestsPerPeriod  *int              `url:"throttle_unauthenticated_requests_per_period,omitempty" json:"throttle_unauthenticated_requests_per_period,omitempty"`
	TimeTrackingLimitToHours                  *bool             `url:"time_tracking_limit_to_hours,omitempty" json:"time_tracking_limit_to_hours,omitempty"`
	TwoFactorGracePeriod                      *int              `url:"two_factor_grace_period,omitempty" json:"two_factor_grace_period,omitempty"`
	UniqueIPsLimitEnabled                     *bool             `url:"unique_ips_limit_enabled,omitempty" json:"unique_ips_limit_enabled,omitempty"`
	UniqueIPsLimitPerUser                     *int              `url:"unique_ips_limit_per_user,omitempty" json:"unique_ips_limit_per_user,omitempty"`
	UniqueIPsLimitTimeWindow                  *int              `url:"unique_ips_limit_time_window,omitempty" json:"unique_ips_limit_time_window,omitempty"`
	UsagePingEnabled                          *bool             `url:"usage_ping_enabled,omitempty" json:"usage_ping_enabled,omitempty"`
	UserDefaultExternal                       *bool             `url:"user_default_external,omitempty" json:"user_default_external,omitempty"`
	UserDefaultInternalRegex                  *string           `url:"user_default_internal_regex,omitempty" json:"user_default_internal_regex,omitempty"`
	UserOauthApplications                     *bool             `url:"user_oauth_applications,omitempty" json:"user_oauth_applications,omitempty"`
	UserShowAddSSHKeyMessage                  *bool             `url:"user_show_add_ssh_key_message,omitempty" json:"user_show_add_ssh_key_message,omitempty"`
	VersionCheckEnabled                       *bool             `url:"version_check_enabled,omitempty" json:"version_check_enabled,omitempty"`
	WebIDEClientsidePreviewEnabled            *bool             `url:"web_ide_clientside_preview_enabled,omitempty" json:"web_ide_clientside_preview_enabled,omitempty"`
}

// UpdateSettings updates the application settings.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/settings.html#change-application.settings
func (s *SettingsService) UpdateSettings(opt *UpdateSettingsOptions, options ...RequestOptionFunc) (*Settings, *Response, error) {
	req, err := s.client.NewRequest("PUT", "application/settings", opt, options)
	if err != nil {
		return nil, nil, err
	}

	as := new(Settings)
	resp, err := s.client.Do(req, as)
	if err != nil {
		return nil, resp, err
	}

	return as, resp, err
}
