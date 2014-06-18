// Package shopping provides access to the Search API For Shopping.
//
// See https://developers.google.com/shopping-search/v1/getting_started
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/shopping/v1"
//   ...
//   shoppingService, err := shopping.New(oauthHttpClient)
package shopping

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

const apiId = "shopping:v1"
const apiName = "shopping"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/shopping/search/v1/"

// OAuth2 scopes used by this API.
const (
	// View your product data
	ShoppingapiScope = "https://www.googleapis.com/auth/shoppingapi"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client}
	s.Products = NewProductsService(s)
	return s, nil
}

type Service struct {
	client *http.Client

	Products *ProductsService
}

func NewProductsService(s *Service) *ProductsService {
	rs := &ProductsService{s: s}
	return rs
}

type ProductsService struct {
	s *Service
}

type Product struct {
	// Categories: List of categories for product.
	Categories []*ShoppingModelCategoryJsonV1 `json:"categories,omitempty"`

	// Debug: Google internal.
	Debug *ShoppingModelDebugJsonV1 `json:"debug,omitempty"`

	// Id: Id of product.
	Id string `json:"id,omitempty"`

	// Kind: The kind of item, always shopping#product.
	Kind string `json:"kind,omitempty"`

	// Product: Product.
	Product *ShoppingModelProductJsonV1 `json:"product,omitempty"`

	// Recommendations: Recommendations for product.
	Recommendations []*ShoppingModelRecommendationsJsonV1 `json:"recommendations,omitempty"`

	// RequestId: Unique identifier for this request.
	RequestId string `json:"requestId,omitempty"`

	// SelfLink: Self link of product when generated for a lookup request.
	// Self link of product when generated for a search request.
	SelfLink string `json:"selfLink,omitempty"`
}

type Products struct {
	// Categories: List of categories.
	Categories []*ShoppingModelCategoryJsonV1 `json:"categories,omitempty"`

	// CategoryRecommendations: Recommendations for category.
	CategoryRecommendations []*ShoppingModelRecommendationsJsonV1 `json:"categoryRecommendations,omitempty"`

	// CurrentItemCount: Current item count.
	CurrentItemCount int64 `json:"currentItemCount,omitempty"`

	// Debug: Google internal.
	Debug *ShoppingModelDebugJsonV1 `json:"debug,omitempty"`

	// Etag: Etag of feed.
	Etag string `json:"etag,omitempty"`

	// Extras: List of extras.
	Extras *ShoppingModelExtrasJsonV1 `json:"extras,omitempty"`

	// Facets: List of facets.
	Facets []*ProductsFacets `json:"facets,omitempty"`

	// Id: Id of feed.
	Id string `json:"id,omitempty"`

	// Items: List of returned products.
	Items []*Product `json:"items,omitempty"`

	// ItemsPerPage: Number of items per page of results.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: The fixed string "shopping#products". The kind of feed
	// returned.
	Kind string `json:"kind,omitempty"`

	// NextLink: Next link of feed.
	NextLink string `json:"nextLink,omitempty"`

	// PreviousLink: Previous link of feed.
	PreviousLink string `json:"previousLink,omitempty"`

	// Promotions: List of promotions.
	Promotions []*ProductsPromotions `json:"promotions,omitempty"`

	// Redirects: Redirects.
	Redirects []string `json:"redirects,omitempty"`

	// RequestId: Unique identifier for this request.
	RequestId string `json:"requestId,omitempty"`

	// SelfLink: Self link of feed.
	SelfLink string `json:"selfLink,omitempty"`

	// Spelling: Spelling.
	Spelling *ProductsSpelling `json:"spelling,omitempty"`

	// StartIndex: 1-based index of the first item in the search results.
	StartIndex int64 `json:"startIndex,omitempty"`

	// Stores: List of returned stores.
	Stores []*ProductsStores `json:"stores,omitempty"`

	// TotalItems: Total number of search results.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type ProductsFacets struct {
	// Buckets: List of Buckets within facet.
	Buckets []*ProductsFacetsBuckets `json:"buckets,omitempty"`

	// Count: Number of products matching the query that have a value for
	// the facet's property or attribute.
	Count int64 `json:"count,omitempty"`

	// DisplayName: Display name of facet.
	DisplayName string `json:"displayName,omitempty"`

	// Name: Name of the facet's attribute (omitted for property facets).
	Name string `json:"name,omitempty"`

	// Property: Property of facet (omitted for attribute facets).
	Property string `json:"property,omitempty"`

	// Type: Type of facet's attribute (omitted for property facets, one of:
	// text, bool, int, float).
	Type string `json:"type,omitempty"`

	// Unit: Unit of the facet's property or attribute (omitted if the
	// facet's property or attribute has no unit).
	Unit string `json:"unit,omitempty"`
}

type ProductsFacetsBuckets struct {
	// Count: Number of products matching the query that have a value for
	// the facet's property or attribute that matches the bucket.
	Count int64 `json:"count,omitempty"`

	// Max: Upper bound of the bucket (omitted for value buckets or if the
	// range has no upper bound).
	Max interface{} `json:"max,omitempty"`

	// MaxExclusive: Whether the upper bound of the bucket is exclusive
	// (omitted for value buckets or if the range has no upper bound).
	MaxExclusive bool `json:"maxExclusive,omitempty"`

	// Min: Lower bound of the bucket (omitted for value buckets or if the
	// range has no lower bound).
	Min interface{} `json:"min,omitempty"`

	// MinExclusive: Whether the lower bound of the bucket is exclusive
	// (omitted for value buckets or if the range has no lower bound).
	MinExclusive bool `json:"minExclusive,omitempty"`

	// Value: Value of the bucket (omitted for range buckets).
	Value interface{} `json:"value,omitempty"`
}

type ProductsPromotions struct {
	// CustomFields: List of custom fields of promotion.
	CustomFields []*ProductsPromotionsCustomFields `json:"customFields,omitempty"`

	// CustomHtml: Custom HTML of promotion (omitted if type is not custom).
	CustomHtml string `json:"customHtml,omitempty"`

	// Description: Description of promotion (omitted if type is not
	// standard).
	Description string `json:"description,omitempty"`

	// DestLink: Link to promotion (omitted if type is not standard).
	DestLink string `json:"destLink,omitempty"`

	// ImageLink: Link to promotion image (omitted if type is not standard).
	ImageLink string `json:"imageLink,omitempty"`

	// Name: Name of promotion (omitted if type is not standard).
	Name string `json:"name,omitempty"`

	// Product: Product of promotion (omitted if type is not product).
	Product *ShoppingModelProductJsonV1 `json:"product,omitempty"`

	// Type: Type of promotion (one of: standard, product, custom).
	Type string `json:"type,omitempty"`
}

type ProductsPromotionsCustomFields struct {
	// Name: Name of field.
	Name string `json:"name,omitempty"`

	// Value: Value of field.
	Value string `json:"value,omitempty"`
}

type ProductsSpelling struct {
	// Suggestion: Suggestion for spelling.
	Suggestion string `json:"suggestion,omitempty"`
}

type ProductsStores struct {
	// Address: Address of store.
	Address string `json:"address,omitempty"`

	// Location: Location of store.
	Location string `json:"location,omitempty"`

	// Name: Name of merchant.
	Name string `json:"name,omitempty"`

	// StoreCode: Merchant-supplied store code.
	StoreCode string `json:"storeCode,omitempty"`

	// StoreId: Id of store.
	StoreId string `json:"storeId,omitempty"`

	// StoreName: Name of store.
	StoreName string `json:"storeName,omitempty"`

	// Telephone: Telephone number of store.
	Telephone string `json:"telephone,omitempty"`
}

type ShoppingModelCategoryJsonV1 struct {
	// Id: Id of category.
	Id string `json:"id,omitempty"`

	// Parents: Ids of the parents of the category.
	Parents []string `json:"parents,omitempty"`

	// ShortName: Short name of category.
	ShortName string `json:"shortName,omitempty"`

	// Url: URL of category.
	Url string `json:"url,omitempty"`
}

type ShoppingModelDebugJsonV1 struct {
	// BackendTimes: Google internal
	BackendTimes []*ShoppingModelDebugJsonV1BackendTimes `json:"backendTimes,omitempty"`

	// ElapsedMillis: Google internal.
	ElapsedMillis int64 `json:"elapsedMillis,omitempty,string"`

	// FacetsRequest: Google internal.
	FacetsRequest string `json:"facetsRequest,omitempty"`

	// FacetsResponse: Google internal.
	FacetsResponse string `json:"facetsResponse,omitempty"`

	// RdcResponse: Google internal.
	RdcResponse string `json:"rdcResponse,omitempty"`

	// RecommendedItemsRequest: Google internal.
	RecommendedItemsRequest string `json:"recommendedItemsRequest,omitempty"`

	// RecommendedItemsResponse: Google internal.
	RecommendedItemsResponse string `json:"recommendedItemsResponse,omitempty"`

	// SearchRequest: Google internal.
	SearchRequest string `json:"searchRequest,omitempty"`

	// SearchResponse: Google internal.
	SearchResponse string `json:"searchResponse,omitempty"`
}

type ShoppingModelDebugJsonV1BackendTimes struct {
	// ElapsedMillis: Google internal
	ElapsedMillis int64 `json:"elapsedMillis,omitempty,string"`

	// HostName: Google internal
	HostName string `json:"hostName,omitempty"`

	// Name: Google internal
	Name string `json:"name,omitempty"`

	// ServerMillis: Google internal
	ServerMillis int64 `json:"serverMillis,omitempty,string"`
}

type ShoppingModelExtrasJsonV1 struct {
	FacetRules []*ShoppingModelExtrasJsonV1FacetRules `json:"facetRules,omitempty"`

	// RankingRules: Names of boost (ranking) rules applicable to this
	// search.
	RankingRules []*ShoppingModelExtrasJsonV1RankingRules `json:"rankingRules,omitempty"`
}

type ShoppingModelExtrasJsonV1FacetRules struct {
	Name string `json:"name,omitempty"`
}

type ShoppingModelExtrasJsonV1RankingRules struct {
	Name string `json:"name,omitempty"`
}

type ShoppingModelProductJsonV1 struct {
	// Attributes: Attributes of product (available only with a cx source).
	Attributes []*ShoppingModelProductJsonV1Attributes `json:"attributes,omitempty"`

	// Author: Author of product.
	Author *ShoppingModelProductJsonV1Author `json:"author,omitempty"`

	// Brand: Brand of product.
	Brand string `json:"brand,omitempty"`

	// Categories: Categories of product according to the selected taxonomy,
	// omitted if no taxonomy is selected.
	Categories []string `json:"categories,omitempty"`

	// Condition: Condition of product (one of: new, refurbished, used).
	Condition string `json:"condition,omitempty"`

	// Country: ISO 3166 code of target country of product.
	Country string `json:"country,omitempty"`

	// CreationTime: RFC 3339 formatted creation time and date of product.
	CreationTime string `json:"creationTime,omitempty"`

	// Description: Description of product.
	Description string `json:"description,omitempty"`

	// GoogleId: Google id of product.
	GoogleId uint64 `json:"googleId,omitempty,string"`

	// Gtin: The first GTIN of the product. Deprecated in favor of "gtins".
	Gtin string `json:"gtin,omitempty"`

	// Gtins: List of all the product's GTINs (in GTIN-14 format).
	Gtins []string `json:"gtins,omitempty"`

	// Images: Images of product.
	Images []*ShoppingModelProductJsonV1Images `json:"images,omitempty"`

	// Internal16: Google Internal. Attribute names are deliberately vague.
	Internal16 *ShoppingModelProductJsonV1Internal16 `json:"internal16,omitempty"`

	// Inventories: Inventories of product.
	Inventories []*ShoppingModelProductJsonV1Inventories `json:"inventories,omitempty"`

	// Language: BCP 47 language tag of language of product.
	Language string `json:"language,omitempty"`

	// Link: Link to product.
	Link string `json:"link,omitempty"`

	// ModificationTime: RFC 3339 formatted modification time and date of
	// product.
	ModificationTime string `json:"modificationTime,omitempty"`

	// Mpns: List of all the product's MPNs.
	Mpns []string `json:"mpns,omitempty"`

	// ProvidedId: Merchant-provided id of product (available only with a cx
	// source).
	ProvidedId string `json:"providedId,omitempty"`

	// QueryMatched: Whether this product matched the user query. Only set
	// for the variant offers (if any) attached to a product offer.
	QueryMatched bool `json:"queryMatched,omitempty"`

	// Score: Google Internal
	Score float64 `json:"score,omitempty"`

	// Title: Title of product.
	Title string `json:"title,omitempty"`

	// TotalMatchingVariants: The number of variant offers returned that
	// matched the query.
	TotalMatchingVariants int64 `json:"totalMatchingVariants,omitempty"`

	// Variants: A list of variant offers associated with this product.
	Variants []*ShoppingModelProductJsonV1Variants `json:"variants,omitempty"`
}

type ShoppingModelProductJsonV1Attributes struct {
	// DisplayName: Display Name of prodct attribute.
	DisplayName string `json:"displayName,omitempty"`

	// Name: Name of product attribute.
	Name string `json:"name,omitempty"`

	// Type: Type of product attribute (one of: text, bool, int, float,
	// dateRange, url).
	Type string `json:"type,omitempty"`

	// Unit: Unit of product attribute.
	Unit string `json:"unit,omitempty"`

	// Value: Value of product attribute.
	Value interface{} `json:"value,omitempty"`
}

type ShoppingModelProductJsonV1Author struct {
	// AccountId: Account id of product author.
	AccountId uint64 `json:"accountId,omitempty,string"`

	// Name: Name of product author.
	Name string `json:"name,omitempty"`
}

type ShoppingModelProductJsonV1Images struct {
	// Link: Link to product image.
	Link string `json:"link,omitempty"`

	Status string `json:"status,omitempty"`

	// Thumbnails: Thumbnails of product image.
	Thumbnails []*ShoppingModelProductJsonV1ImagesThumbnails `json:"thumbnails,omitempty"`
}

type ShoppingModelProductJsonV1ImagesThumbnails struct {
	// Content: Content of thumbnail (only available for the first thumbnail
	// of the top results if SAYT is enabled).
	Content string `json:"content,omitempty"`

	// Height: Height of thumbnail (omitted if not specified in the
	// request).
	Height int64 `json:"height,omitempty"`

	// Link: Link to thumbnail.
	Link string `json:"link,omitempty"`

	// Width: Width of thumbnail (omitted if not specified in the request).
	Width int64 `json:"width,omitempty"`
}

type ShoppingModelProductJsonV1Internal16 struct {
	Length int64 `json:"length,omitempty"`

	Number int64 `json:"number,omitempty"`

	Size int64 `json:"size,omitempty,string"`
}

type ShoppingModelProductJsonV1Inventories struct {
	// Availability: Availability of product inventory.
	Availability string `json:"availability,omitempty"`

	// Channel: Channel of product inventory (one of: online, local).
	Channel string `json:"channel,omitempty"`

	// Currency: Currency of product inventory (an ISO 4217 alphabetic
	// code).
	Currency string `json:"currency,omitempty"`

	// Distance: Distance of product inventory.
	Distance float64 `json:"distance,omitempty"`

	// DistanceUnit: Distance unit of product inventory.
	DistanceUnit string `json:"distanceUnit,omitempty"`

	// InstallmentMonths: Number of months for installment price.
	InstallmentMonths int64 `json:"installmentMonths,omitempty"`

	// InstallmentPrice: Installment price of product inventory.
	InstallmentPrice float64 `json:"installmentPrice,omitempty"`

	// OriginalPrice: Original price of product inventory. Only returned for
	// products that are on sale.
	OriginalPrice float64 `json:"originalPrice,omitempty"`

	// Price: Price of product inventory.
	Price float64 `json:"price,omitempty"`

	// SaleEndDate: Sale end date.
	SaleEndDate string `json:"saleEndDate,omitempty"`

	// SalePrice: Sale price of product inventory.
	SalePrice float64 `json:"salePrice,omitempty"`

	// SaleStartDate: Sale start date.
	SaleStartDate string `json:"saleStartDate,omitempty"`

	// Shipping: Shipping cost of product inventory.
	Shipping float64 `json:"shipping,omitempty"`

	// StoreId: Store ID of product inventory.
	StoreId string `json:"storeId,omitempty"`

	// Tax: Tax of product inventory.
	Tax float64 `json:"tax,omitempty"`
}

type ShoppingModelProductJsonV1Variants struct {
	// Variant: The detailed offer data for a particular variant offer.
	Variant *ShoppingModelProductJsonV1 `json:"variant,omitempty"`
}

type ShoppingModelRecommendationsJsonV1 struct {
	// RecommendationList: List of recommendations.
	RecommendationList []*ShoppingModelRecommendationsJsonV1RecommendationList `json:"recommendationList,omitempty"`

	// Type: Type of recommendation list (for offer-based recommendations,
	// one of: all, purchaseToPurchase, visitToVisit, visitToPurchase,
	// relatedItems, visuallySimilarItems; for category-based
	// recommendations, one of: all, categoryMostVisited,
	// categoryBestSeller).
	Type string `json:"type,omitempty"`
}

type ShoppingModelRecommendationsJsonV1RecommendationList struct {
	// Product: Recommended product.
	Product *ShoppingModelProductJsonV1 `json:"product,omitempty"`
}

// method id "shopping.products.get":

type ProductsGetCall struct {
	s             *Service
	source        string
	accountId     int64
	productIdType string
	productId     string
	opt_          map[string]interface{}
}

// Get: Returns a single product
func (r *ProductsService) Get(source string, accountId int64, productIdType string, productId string) *ProductsGetCall {
	c := &ProductsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.source = source
	c.accountId = accountId
	c.productIdType = productIdType
	c.productId = productId
	return c
}

// AttributeFilter sets the optional parameter "attributeFilter": Comma
// separated list of attributes to return
func (c *ProductsGetCall) AttributeFilter(attributeFilter string) *ProductsGetCall {
	c.opt_["attributeFilter"] = attributeFilter
	return c
}

// CategoriesEnabled sets the optional parameter "categories.enabled":
// Whether to return category information
func (c *ProductsGetCall) CategoriesEnabled(categoriesEnabled bool) *ProductsGetCall {
	c.opt_["categories.enabled"] = categoriesEnabled
	return c
}

// CategoriesInclude sets the optional parameter "categories.include":
// Category specification
func (c *ProductsGetCall) CategoriesInclude(categoriesInclude string) *ProductsGetCall {
	c.opt_["categories.include"] = categoriesInclude
	return c
}

// CategoriesUseGcsConfig sets the optional parameter
// "categories.useGcsConfig": This parameter is currently ignored
func (c *ProductsGetCall) CategoriesUseGcsConfig(categoriesUseGcsConfig bool) *ProductsGetCall {
	c.opt_["categories.useGcsConfig"] = categoriesUseGcsConfig
	return c
}

// Location sets the optional parameter "location": Location used to
// determine tax and shipping
func (c *ProductsGetCall) Location(location string) *ProductsGetCall {
	c.opt_["location"] = location
	return c
}

// RecommendationsEnabled sets the optional parameter
// "recommendations.enabled": Whether to return recommendation
// information
func (c *ProductsGetCall) RecommendationsEnabled(recommendationsEnabled bool) *ProductsGetCall {
	c.opt_["recommendations.enabled"] = recommendationsEnabled
	return c
}

// RecommendationsInclude sets the optional parameter
// "recommendations.include": Recommendation specification
func (c *ProductsGetCall) RecommendationsInclude(recommendationsInclude string) *ProductsGetCall {
	c.opt_["recommendations.include"] = recommendationsInclude
	return c
}

// RecommendationsUseGcsConfig sets the optional parameter
// "recommendations.useGcsConfig": This parameter is currently ignored
func (c *ProductsGetCall) RecommendationsUseGcsConfig(recommendationsUseGcsConfig bool) *ProductsGetCall {
	c.opt_["recommendations.useGcsConfig"] = recommendationsUseGcsConfig
	return c
}

// Taxonomy sets the optional parameter "taxonomy": Merchant taxonomy
func (c *ProductsGetCall) Taxonomy(taxonomy string) *ProductsGetCall {
	c.opt_["taxonomy"] = taxonomy
	return c
}

// Thumbnails sets the optional parameter "thumbnails": Thumbnail
// specification
func (c *ProductsGetCall) Thumbnails(thumbnails string) *ProductsGetCall {
	c.opt_["thumbnails"] = thumbnails
	return c
}

func (c *ProductsGetCall) Do() (*Product, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["attributeFilter"]; ok {
		params.Set("attributeFilter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["categories.enabled"]; ok {
		params.Set("categories.enabled", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["categories.include"]; ok {
		params.Set("categories.include", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["categories.useGcsConfig"]; ok {
		params.Set("categories.useGcsConfig", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["location"]; ok {
		params.Set("location", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["recommendations.enabled"]; ok {
		params.Set("recommendations.enabled", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["recommendations.include"]; ok {
		params.Set("recommendations.include", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["recommendations.useGcsConfig"]; ok {
		params.Set("recommendations.useGcsConfig", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["taxonomy"]; ok {
		params.Set("taxonomy", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["thumbnails"]; ok {
		params.Set("thumbnails", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/shopping/search/v1/", "{source}/products/{accountId}/{productIdType}/{productId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{source}", url.QueryEscape(c.source), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountId}", strconv.FormatInt(c.accountId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{productIdType}", url.QueryEscape(c.productIdType), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{productId}", url.QueryEscape(c.productId), 1)
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
	ret := new(Product)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a single product",
	//   "httpMethod": "GET",
	//   "id": "shopping.products.get",
	//   "parameterOrder": [
	//     "source",
	//     "accountId",
	//     "productIdType",
	//     "productId"
	//   ],
	//   "parameters": {
	//     "accountId": {
	//       "description": "Merchant center account id",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "attributeFilter": {
	//       "description": "Comma separated list of attributes to return",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "categories.enabled": {
	//       "description": "Whether to return category information",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "categories.include": {
	//       "description": "Category specification",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "categories.useGcsConfig": {
	//       "description": "This parameter is currently ignored",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "location": {
	//       "description": "Location used to determine tax and shipping",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "productId": {
	//       "description": "Id of product",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "productIdType": {
	//       "description": "Type of productId",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "recommendations.enabled": {
	//       "description": "Whether to return recommendation information",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "recommendations.include": {
	//       "description": "Recommendation specification",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "recommendations.useGcsConfig": {
	//       "description": "This parameter is currently ignored",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "source": {
	//       "description": "Query source",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "taxonomy": {
	//       "description": "Merchant taxonomy",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "thumbnails": {
	//       "description": "Thumbnail specification",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{source}/products/{accountId}/{productIdType}/{productId}",
	//   "response": {
	//     "$ref": "Product"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/shoppingapi"
	//   ]
	// }

}

// method id "shopping.products.list":

type ProductsListCall struct {
	s      *Service
	source string
	opt_   map[string]interface{}
}

// List: Returns a list of products and content modules
func (r *ProductsService) List(source string) *ProductsListCall {
	c := &ProductsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.source = source
	return c
}

// AttributeFilter sets the optional parameter "attributeFilter": Comma
// separated list of attributes to return
func (c *ProductsListCall) AttributeFilter(attributeFilter string) *ProductsListCall {
	c.opt_["attributeFilter"] = attributeFilter
	return c
}

// Availability sets the optional parameter "availability": Comma
// separated list of availabilities (outOfStock, limited, inStock,
// backOrder, preOrder, onDisplayToOrder) to return
func (c *ProductsListCall) Availability(availability string) *ProductsListCall {
	c.opt_["availability"] = availability
	return c
}

// BoostBy sets the optional parameter "boostBy": Boosting specification
func (c *ProductsListCall) BoostBy(boostBy string) *ProductsListCall {
	c.opt_["boostBy"] = boostBy
	return c
}

// CategoriesEnabled sets the optional parameter "categories.enabled":
// Whether to return category information
func (c *ProductsListCall) CategoriesEnabled(categoriesEnabled bool) *ProductsListCall {
	c.opt_["categories.enabled"] = categoriesEnabled
	return c
}

// CategoriesInclude sets the optional parameter "categories.include":
// Category specification
func (c *ProductsListCall) CategoriesInclude(categoriesInclude string) *ProductsListCall {
	c.opt_["categories.include"] = categoriesInclude
	return c
}

// CategoriesUseGcsConfig sets the optional parameter
// "categories.useGcsConfig": This parameter is currently ignored
func (c *ProductsListCall) CategoriesUseGcsConfig(categoriesUseGcsConfig bool) *ProductsListCall {
	c.opt_["categories.useGcsConfig"] = categoriesUseGcsConfig
	return c
}

// CategoryRecommendationsCategory sets the optional parameter
// "categoryRecommendations.category": Category for which to retrieve
// recommendations
func (c *ProductsListCall) CategoryRecommendationsCategory(categoryRecommendationsCategory string) *ProductsListCall {
	c.opt_["categoryRecommendations.category"] = categoryRecommendationsCategory
	return c
}

// CategoryRecommendationsEnabled sets the optional parameter
// "categoryRecommendations.enabled": Whether to return category
// recommendation information
func (c *ProductsListCall) CategoryRecommendationsEnabled(categoryRecommendationsEnabled bool) *ProductsListCall {
	c.opt_["categoryRecommendations.enabled"] = categoryRecommendationsEnabled
	return c
}

// CategoryRecommendationsInclude sets the optional parameter
// "categoryRecommendations.include": Category recommendation
// specification
func (c *ProductsListCall) CategoryRecommendationsInclude(categoryRecommendationsInclude string) *ProductsListCall {
	c.opt_["categoryRecommendations.include"] = categoryRecommendationsInclude
	return c
}

// CategoryRecommendationsUseGcsConfig sets the optional parameter
// "categoryRecommendations.useGcsConfig": This parameter is currently
// ignored
func (c *ProductsListCall) CategoryRecommendationsUseGcsConfig(categoryRecommendationsUseGcsConfig bool) *ProductsListCall {
	c.opt_["categoryRecommendations.useGcsConfig"] = categoryRecommendationsUseGcsConfig
	return c
}

// Channels sets the optional parameter "channels": Channels
// specification
func (c *ProductsListCall) Channels(channels string) *ProductsListCall {
	c.opt_["channels"] = channels
	return c
}

// ClickTracking sets the optional parameter "clickTracking": Whether to
// add a click tracking parameter to offer URLs
func (c *ProductsListCall) ClickTracking(clickTracking bool) *ProductsListCall {
	c.opt_["clickTracking"] = clickTracking
	return c
}

// Country sets the optional parameter "country": Country restriction
// (ISO 3166)
func (c *ProductsListCall) Country(country string) *ProductsListCall {
	c.opt_["country"] = country
	return c
}

// CrowdBy sets the optional parameter "crowdBy": Crowding specification
func (c *ProductsListCall) CrowdBy(crowdBy string) *ProductsListCall {
	c.opt_["crowdBy"] = crowdBy
	return c
}

// Currency sets the optional parameter "currency": Currency restriction
// (ISO 4217)
func (c *ProductsListCall) Currency(currency string) *ProductsListCall {
	c.opt_["currency"] = currency
	return c
}

// ExtrasEnabled sets the optional parameter "extras.enabled": Whether
// to return extra information.
func (c *ProductsListCall) ExtrasEnabled(extrasEnabled bool) *ProductsListCall {
	c.opt_["extras.enabled"] = extrasEnabled
	return c
}

// ExtrasInfo sets the optional parameter "extras.info": What extra
// information to return.
func (c *ProductsListCall) ExtrasInfo(extrasInfo string) *ProductsListCall {
	c.opt_["extras.info"] = extrasInfo
	return c
}

// FacetsDiscover sets the optional parameter "facets.discover": Facets
// to discover
func (c *ProductsListCall) FacetsDiscover(facetsDiscover string) *ProductsListCall {
	c.opt_["facets.discover"] = facetsDiscover
	return c
}

// FacetsEnabled sets the optional parameter "facets.enabled": Whether
// to return facet information
func (c *ProductsListCall) FacetsEnabled(facetsEnabled bool) *ProductsListCall {
	c.opt_["facets.enabled"] = facetsEnabled
	return c
}

// FacetsInclude sets the optional parameter "facets.include": Facets to
// include (applies when useGcsConfig == false)
func (c *ProductsListCall) FacetsInclude(facetsInclude string) *ProductsListCall {
	c.opt_["facets.include"] = facetsInclude
	return c
}

// FacetsIncludeEmptyBuckets sets the optional parameter
// "facets.includeEmptyBuckets": Return empty facet buckets.
func (c *ProductsListCall) FacetsIncludeEmptyBuckets(facetsIncludeEmptyBuckets bool) *ProductsListCall {
	c.opt_["facets.includeEmptyBuckets"] = facetsIncludeEmptyBuckets
	return c
}

// FacetsUseGcsConfig sets the optional parameter "facets.useGcsConfig":
// Whether to return facet information as configured in the GCS account
func (c *ProductsListCall) FacetsUseGcsConfig(facetsUseGcsConfig bool) *ProductsListCall {
	c.opt_["facets.useGcsConfig"] = facetsUseGcsConfig
	return c
}

// Language sets the optional parameter "language": Language restriction
// (BCP 47)
func (c *ProductsListCall) Language(language string) *ProductsListCall {
	c.opt_["language"] = language
	return c
}

// Location sets the optional parameter "location": Location used to
// determine tax and shipping
func (c *ProductsListCall) Location(location string) *ProductsListCall {
	c.opt_["location"] = location
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *ProductsListCall) MaxResults(maxResults int64) *ProductsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// MaxVariants sets the optional parameter "maxVariants": Maximum number
// of variant results to return per result
func (c *ProductsListCall) MaxVariants(maxVariants int64) *ProductsListCall {
	c.opt_["maxVariants"] = maxVariants
	return c
}

// PromotionsEnabled sets the optional parameter "promotions.enabled":
// Whether to return promotion information
func (c *ProductsListCall) PromotionsEnabled(promotionsEnabled bool) *ProductsListCall {
	c.opt_["promotions.enabled"] = promotionsEnabled
	return c
}

// PromotionsUseGcsConfig sets the optional parameter
// "promotions.useGcsConfig": Whether to return promotion information as
// configured in the GCS account
func (c *ProductsListCall) PromotionsUseGcsConfig(promotionsUseGcsConfig bool) *ProductsListCall {
	c.opt_["promotions.useGcsConfig"] = promotionsUseGcsConfig
	return c
}

// Q sets the optional parameter "q": Search query
func (c *ProductsListCall) Q(q string) *ProductsListCall {
	c.opt_["q"] = q
	return c
}

// RankBy sets the optional parameter "rankBy": Ranking specification
func (c *ProductsListCall) RankBy(rankBy string) *ProductsListCall {
	c.opt_["rankBy"] = rankBy
	return c
}

// RedirectsEnabled sets the optional parameter "redirects.enabled":
// Whether to return redirect information
func (c *ProductsListCall) RedirectsEnabled(redirectsEnabled bool) *ProductsListCall {
	c.opt_["redirects.enabled"] = redirectsEnabled
	return c
}

// RedirectsUseGcsConfig sets the optional parameter
// "redirects.useGcsConfig": Whether to return redirect information as
// configured in the GCS account
func (c *ProductsListCall) RedirectsUseGcsConfig(redirectsUseGcsConfig bool) *ProductsListCall {
	c.opt_["redirects.useGcsConfig"] = redirectsUseGcsConfig
	return c
}

// RestrictBy sets the optional parameter "restrictBy": Restriction
// specification
func (c *ProductsListCall) RestrictBy(restrictBy string) *ProductsListCall {
	c.opt_["restrictBy"] = restrictBy
	return c
}

// SpellingEnabled sets the optional parameter "spelling.enabled":
// Whether to return spelling suggestions
func (c *ProductsListCall) SpellingEnabled(spellingEnabled bool) *ProductsListCall {
	c.opt_["spelling.enabled"] = spellingEnabled
	return c
}

// SpellingUseGcsConfig sets the optional parameter
// "spelling.useGcsConfig": This parameter is currently ignored
func (c *ProductsListCall) SpellingUseGcsConfig(spellingUseGcsConfig bool) *ProductsListCall {
	c.opt_["spelling.useGcsConfig"] = spellingUseGcsConfig
	return c
}

// StartIndex sets the optional parameter "startIndex": Index (1-based)
// of first product to return
func (c *ProductsListCall) StartIndex(startIndex int64) *ProductsListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

// Taxonomy sets the optional parameter "taxonomy": Taxonomy name
func (c *ProductsListCall) Taxonomy(taxonomy string) *ProductsListCall {
	c.opt_["taxonomy"] = taxonomy
	return c
}

// Thumbnails sets the optional parameter "thumbnails": Image thumbnails
// specification
func (c *ProductsListCall) Thumbnails(thumbnails string) *ProductsListCall {
	c.opt_["thumbnails"] = thumbnails
	return c
}

// UseCase sets the optional parameter "useCase": One of
// CommerceSearchUseCase, ShoppingApiUseCase
func (c *ProductsListCall) UseCase(useCase string) *ProductsListCall {
	c.opt_["useCase"] = useCase
	return c
}

func (c *ProductsListCall) Do() (*Products, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["attributeFilter"]; ok {
		params.Set("attributeFilter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["availability"]; ok {
		params.Set("availability", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["boostBy"]; ok {
		params.Set("boostBy", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["categories.enabled"]; ok {
		params.Set("categories.enabled", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["categories.include"]; ok {
		params.Set("categories.include", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["categories.useGcsConfig"]; ok {
		params.Set("categories.useGcsConfig", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["categoryRecommendations.category"]; ok {
		params.Set("categoryRecommendations.category", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["categoryRecommendations.enabled"]; ok {
		params.Set("categoryRecommendations.enabled", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["categoryRecommendations.include"]; ok {
		params.Set("categoryRecommendations.include", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["categoryRecommendations.useGcsConfig"]; ok {
		params.Set("categoryRecommendations.useGcsConfig", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["channels"]; ok {
		params.Set("channels", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["clickTracking"]; ok {
		params.Set("clickTracking", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["country"]; ok {
		params.Set("country", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["crowdBy"]; ok {
		params.Set("crowdBy", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["currency"]; ok {
		params.Set("currency", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["extras.enabled"]; ok {
		params.Set("extras.enabled", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["extras.info"]; ok {
		params.Set("extras.info", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["facets.discover"]; ok {
		params.Set("facets.discover", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["facets.enabled"]; ok {
		params.Set("facets.enabled", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["facets.include"]; ok {
		params.Set("facets.include", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["facets.includeEmptyBuckets"]; ok {
		params.Set("facets.includeEmptyBuckets", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["facets.useGcsConfig"]; ok {
		params.Set("facets.useGcsConfig", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["language"]; ok {
		params.Set("language", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["location"]; ok {
		params.Set("location", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxVariants"]; ok {
		params.Set("maxVariants", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["promotions.enabled"]; ok {
		params.Set("promotions.enabled", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["promotions.useGcsConfig"]; ok {
		params.Set("promotions.useGcsConfig", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["q"]; ok {
		params.Set("q", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["rankBy"]; ok {
		params.Set("rankBy", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["redirects.enabled"]; ok {
		params.Set("redirects.enabled", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["redirects.useGcsConfig"]; ok {
		params.Set("redirects.useGcsConfig", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["restrictBy"]; ok {
		params.Set("restrictBy", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["spelling.enabled"]; ok {
		params.Set("spelling.enabled", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["spelling.useGcsConfig"]; ok {
		params.Set("spelling.useGcsConfig", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["taxonomy"]; ok {
		params.Set("taxonomy", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["thumbnails"]; ok {
		params.Set("thumbnails", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["useCase"]; ok {
		params.Set("useCase", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/shopping/search/v1/", "{source}/products")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{source}", url.QueryEscape(c.source), 1)
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
	ret := new(Products)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of products and content modules",
	//   "httpMethod": "GET",
	//   "id": "shopping.products.list",
	//   "parameterOrder": [
	//     "source"
	//   ],
	//   "parameters": {
	//     "attributeFilter": {
	//       "description": "Comma separated list of attributes to return",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "availability": {
	//       "description": "Comma separated list of availabilities (outOfStock, limited, inStock, backOrder, preOrder, onDisplayToOrder) to return",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "boostBy": {
	//       "description": "Boosting specification",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "categories.enabled": {
	//       "description": "Whether to return category information",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "categories.include": {
	//       "description": "Category specification",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "categories.useGcsConfig": {
	//       "description": "This parameter is currently ignored",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "categoryRecommendations.category": {
	//       "description": "Category for which to retrieve recommendations",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "categoryRecommendations.enabled": {
	//       "description": "Whether to return category recommendation information",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "categoryRecommendations.include": {
	//       "description": "Category recommendation specification",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "categoryRecommendations.useGcsConfig": {
	//       "description": "This parameter is currently ignored",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "channels": {
	//       "description": "Channels specification",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "clickTracking": {
	//       "description": "Whether to add a click tracking parameter to offer URLs",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "country": {
	//       "description": "Country restriction (ISO 3166)",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "crowdBy": {
	//       "description": "Crowding specification",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "currency": {
	//       "description": "Currency restriction (ISO 4217)",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "extras.enabled": {
	//       "description": "Whether to return extra information.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "extras.info": {
	//       "description": "What extra information to return.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "facets.discover": {
	//       "description": "Facets to discover",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "facets.enabled": {
	//       "description": "Whether to return facet information",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "facets.include": {
	//       "description": "Facets to include (applies when useGcsConfig == false)",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "facets.includeEmptyBuckets": {
	//       "description": "Return empty facet buckets.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "facets.useGcsConfig": {
	//       "description": "Whether to return facet information as configured in the GCS account",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "language": {
	//       "description": "Language restriction (BCP 47)",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "location": {
	//       "description": "Location used to determine tax and shipping",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "maxVariants": {
	//       "description": "Maximum number of variant results to return per result",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "promotions.enabled": {
	//       "description": "Whether to return promotion information",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "promotions.useGcsConfig": {
	//       "description": "Whether to return promotion information as configured in the GCS account",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "q": {
	//       "description": "Search query",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "rankBy": {
	//       "description": "Ranking specification",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "redirects.enabled": {
	//       "description": "Whether to return redirect information",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "redirects.useGcsConfig": {
	//       "description": "Whether to return redirect information as configured in the GCS account",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "restrictBy": {
	//       "description": "Restriction specification",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "source": {
	//       "description": "Query source",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "spelling.enabled": {
	//       "description": "Whether to return spelling suggestions",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "spelling.useGcsConfig": {
	//       "description": "This parameter is currently ignored",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "startIndex": {
	//       "description": "Index (1-based) of first product to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "taxonomy": {
	//       "description": "Taxonomy name",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "thumbnails": {
	//       "description": "Image thumbnails specification",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "useCase": {
	//       "description": "One of CommerceSearchUseCase, ShoppingApiUseCase",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "{source}/products",
	//   "response": {
	//     "$ref": "Products"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/shoppingapi"
	//   ]
	// }

}
