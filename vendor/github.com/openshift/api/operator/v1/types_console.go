package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	configv1 "github.com/openshift/api/config/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Console provides a means to configure an operator to manage the console.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Console struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	// +kubebuilder:validation:Required
	// +required
	Spec ConsoleSpec `json:"spec,omitempty"`
	// +optional
	Status ConsoleStatus `json:"status,omitempty"`
}

// ConsoleSpec is the specification of the desired behavior of the Console.
type ConsoleSpec struct {
	OperatorSpec `json:",inline"`
	// customization is used to optionally provide a small set of
	// customization options to the web console.
	// +optional
	Customization ConsoleCustomization `json:"customization"`
	// providers contains configuration for using specific service providers.
	Providers ConsoleProviders `json:"providers"`
	// route contains hostname and secret reference that contains the serving certificate.
	// If a custom route is specified, a new route will be created with the
	// provided hostname, under which console will be available.
	// In case of custom hostname uses the default routing suffix of the cluster,
	// the Secret specification for a serving certificate will not be needed.
	// In case of custom hostname points to an arbitrary domain, manual DNS configurations steps are necessary.
	// The default console route will be maintained to reserve the default hostname
	// for console if the custom route is removed.
	// If not specified, default route will be used.
	// DEPRECATED
	// +optional
	Route ConsoleConfigRoute `json:"route"`
	// plugins defines a list of enabled console plugin names.
	// +optional
	Plugins []string `json:"plugins,omitempty"`
}

// ConsoleConfigRoute holds information on external route access to console.
// DEPRECATED
type ConsoleConfigRoute struct {
	// hostname is the desired custom domain under which console will be available.
	Hostname string `json:"hostname"`
	// secret points to secret in the openshift-config namespace that contains custom
	// certificate and key and needs to be created manually by the cluster admin.
	// Referenced Secret is required to contain following key value pairs:
	// - "tls.crt" - to specifies custom certificate
	// - "tls.key" - to specifies private key of the custom certificate
	// If the custom hostname uses the default routing suffix of the cluster,
	// the Secret specification for a serving certificate will not be needed.
	// +optional
	Secret configv1.SecretNameReference `json:"secret"`
}

// ConsoleStatus defines the observed status of the Console.
type ConsoleStatus struct {
	OperatorStatus `json:",inline"`
}

// ConsoleProviders defines a list of optional additional providers of
// functionality to the console.
type ConsoleProviders struct {
	// statuspage contains ID for statuspage.io page that provides status info about.
	// +optional
	Statuspage *StatuspageProvider `json:"statuspage,omitempty"`
}

// StatuspageProvider provides identity for statuspage account.
type StatuspageProvider struct {
	// pageID is the unique ID assigned by Statuspage for your page. This must be a public page.
	PageID string `json:"pageID"`
}

// ConsoleCustomization defines a list of optional configuration for the console UI.
type ConsoleCustomization struct {
	// brand is the default branding of the web console which can be overridden by
	// providing the brand field.  There is a limited set of specific brand options.
	// This field controls elements of the console such as the logo.
	// Invalid value will prevent a console rollout.
	Brand Brand `json:"brand,omitempty"`
	// documentationBaseURL links to external documentation are shown in various sections
	// of the web console.  Providing documentationBaseURL will override the default
	// documentation URL.
	// Invalid value will prevent a console rollout.
	// +kubebuilder:validation:Pattern=`^$|^((https):\/\/?)[^\s()<>]+(?:\([\w\d]+\)|([^[:punct:]\s]|\/?))\/$`
	DocumentationBaseURL string `json:"documentationBaseURL,omitempty"`
	// customProductName is the name that will be displayed in page titles, logo alt text, and the about dialog
	// instead of the normal OpenShift product name.
	// +optional
	CustomProductName string `json:"customProductName,omitempty"`
	// customLogoFile replaces the default OpenShift logo in the masthead and about dialog. It is a reference to a
	// ConfigMap in the openshift-config namespace. This can be created with a command like
	// 'oc create configmap custom-logo --from-file=/path/to/file -n openshift-config'.
	// Image size must be less than 1 MB due to constraints on the ConfigMap size.
	// The ConfigMap key should include a file extension so that the console serves the file
	// with the correct MIME type.
	// Recommended logo specifications:
	// Dimensions: Max height of 68px and max width of 200px
	// SVG format preferred
	// +optional
	CustomLogoFile configv1.ConfigMapFileReference `json:"customLogoFile,omitempty"`
	// developerCatalog allows to configure the shown developer catalog categories.
	// +kubebuilder:validation:Optional
	// +optional
	DeveloperCatalog DeveloperConsoleCatalogCustomization `json:"developerCatalog,omitempty"`
	// projectAccess allows customizing the available list of ClusterRoles in the Developer perspective
	// Project access page which can be used by a project admin to specify roles to other users and
	// restrict access within the project. If set, the list will replace the default ClusterRole options.
	// +kubebuilder:validation:Optional
	// +optional
	ProjectAccess ProjectAccess `json:"projectAccess,omitempty"`
	// quickStarts allows customization of available ConsoleQuickStart resources in console.
	// +kubebuilder:validation:Optional
	// +optional
	QuickStarts QuickStarts `json:"quickStarts,omitempty"`
	// addPage allows customizing actions on the Add page in developer perspective.
	// +kubebuilder:validation:Optional
	// +optional
	AddPage AddPage `json:"addPage,omitempty"`
}

// ProjectAccess contains options for project access roles
type ProjectAccess struct {
	// availableClusterRoles is the list of ClusterRole names that are assignable to users
	// through the project access tab.
	// +kubebuilder:validation:Optional
	// +optional
	AvailableClusterRoles []string `json:"availableClusterRoles,omitempty"`
}

// DeveloperConsoleCatalogCustomization allow cluster admin to configure developer catalog.
type DeveloperConsoleCatalogCustomization struct {
	// categories which are shown in the developer catalog.
	// +kubebuilder:validation:Optional
	// +optional
	Categories []DeveloperConsoleCatalogCategory `json:"categories,omitempty"`
}

// DeveloperConsoleCatalogCategoryMeta are the key identifiers of a developer catalog category.
type DeveloperConsoleCatalogCategoryMeta struct {
	// ID is an identifier used in the URL to enable deep linking in console.
	// ID is required and must have 1-32 URL safe (A-Z, a-z, 0-9, - and _) characters.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=32
	// +kubebuilder:validation:Pattern=`^[A-Za-z0-9-_]+$`
	// +required
	ID string `json:"id"`
	// label defines a category display label. It is required and must have 1-64 characters.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=64
	// +required
	Label string `json:"label"`
	// tags is a list of strings that will match the category. A selected category
	// show all items which has at least one overlapping tag between category and item.
	// +kubebuilder:validation:Optional
	// +optional
	Tags []string `json:"tags,omitempty"`
}

// DeveloperConsoleCatalogCategory for the developer console catalog.
type DeveloperConsoleCatalogCategory struct {
	// defines top level category ID, label and filter tags.
	DeveloperConsoleCatalogCategoryMeta `json:",inline"`
	// subcategories defines a list of child categories.
	// +kubebuilder:validation:Optional
	// +optional
	Subcategories []DeveloperConsoleCatalogCategoryMeta `json:"subcategories,omitempty"`
}

// QuickStarts allow cluster admins to customize available ConsoleQuickStart resources.
type QuickStarts struct {
	// disabled is a list of ConsoleQuickStart resource names that are not shown to users.
	// +kubebuilder:validation:Optional
	// +optional
	Disabled []string `json:"disabled,omitempty"`
}

// AddPage allows customizing actions on the Add page in developer perspective.
type AddPage struct {
	// disabledActions is a list of actions that are not shown to users.
	// Each action in the list is represented by its ID.
	// +kubebuilder:validation:Optional
	// +kubebuilder:validation:MinItems=1
	// +optional
	DisabledActions []string `json:"disabledActions,omitempty"`
}

// Brand is a specific supported brand within the console.
// +kubebuilder:validation:Pattern=`^$|^(ocp|origin|okd|dedicated|online|azure)$`
type Brand string

const (
	// Branding for OpenShift
	BrandOpenShift Brand = "openshift"
	// Branding for The Origin Community Distribution of Kubernetes
	BrandOKD Brand = "okd"
	// Branding for OpenShift Online
	BrandOnline Brand = "online"
	// Branding for OpenShift Container Platform
	BrandOCP Brand = "ocp"
	// Branding for OpenShift Dedicated
	BrandDedicated Brand = "dedicated"
	// Branding for Azure Red Hat OpenShift
	BrandAzure Brand = "azure"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type ConsoleList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata"`

	Items []Console `json:"items"`
}
